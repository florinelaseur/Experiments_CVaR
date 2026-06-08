# How does Sobol UNIFORM sampling (no Gaussian mean/covariance) behave with the
# adequacy cuts as the accept test? This script reports:
#   * uniform acceptance rate over [0, ub]   (fraction of draws that clear the cuts)
#   * gaussian-around-μ* acceptance rate      (for comparison)
#   * cut→LP precision of accepted UNIFORM samples (fraction actually LP-feasible
#     across the selected scenarios) — i.e. how "probable" the cut-passing uniform
#     pool really is for the solver.
#
# Usage: julia --project=. ScenarioReduction/test_uniform_adequacy_sampling.jl

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev="227a80f7907e2c7178edb0697874cfb6666ad644")

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using JuMP: JuMP
using CSV: CSV
using DataFrames
using TOML: TOML
using Random

include(joinpath("..", "utils", "functions.jl"))
include(joinpath("..", "utils", "constants.jl"))
include(joinpath(@__DIR__, "src", "utils.jl"))

const TARGET = 256   # cut-passing samples to collect per sampler
const N_LP   = 48    # accepted uniform samples to LP-validate

# Single-scenario warm-startable LP model (is_seasonal off), from original-id profiles.
function build_scenario_model(scenario_id, all_profiles_df, input_data_path, solver_sym)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)
    sp = filter(row -> row.scenario == scenario_id, all_profiles_df)
    sp[!, :scenario] .= 1
    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, sp, "profiles_wide")
    DuckDB.query(conn, "CREATE OR REPLACE TABLE stochastic_scenario AS SELECT 1 AS scenario, 1.0 AS probability")
    TC.transform_wide_to_long!(conn, "profiles_wide", "profiles"; exclude_columns=["scenario", "milestone_year", "timestep"])
    layout = TC.ProfilesTableLayout(; year=:milestone_year, cols_to_groupby=[:milestone_year, :scenario])
    TC.dummy_cluster!(conn; layout=layout)
    TEM.populate_with_defaults!(conn)
    DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    TEM.create_internal_tables!(conn)
    variables = TEM.compute_variables_indices(conn)
    constraints = TEM.compute_constraints_indices(conn)
    profiles = TEM.prepare_profiles_structure(conn)
    model, _ = TEM.create_model(conn, variables, constraints, profiles)
    optimizer, parameters = get_solver_parameters(solver_sym)
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in parameters]...)
    JuMP.set_silent(model)
    solver_sym == :Gurobi && JuMP.set_optimizer_attribute(model, "Method", 1)
    return (model=model, variables=variables, capacity_lookup=build_capacity_lookup(conn))
end

function lp_feasible!(m, x, solver_sym, first_solve)
    fix_variables_from_sample(m.variables, :assets_investment, x; m.capacity_lookup)
    JuMP.optimize!(m.model)
    feas = JuMP.termination_status(m.model) == JuMP.OPTIMAL
    if first_solve[] && solver_sym == :Gurobi
        JuMP.set_optimizer_attribute(m.model, "Presolve", 0)
        first_solve[] = false
    end
    return feas
end

function main()
    config = TOML.parsefile(joinpath("..", "config.toml"))
    input_data_path = joinpath("..", config["simulation"]["input_data"])
    n_scenarios = config["simulation"]["number_of_scenarios"]
    solver_sym = Symbol(first(config["simulation"]["solvers"]))

    Random.seed!(19990907)
    all_profiles_df =
        CSV.read(joinpath("..", "create-scenarios", "profiles-wide-all-scenarios.csv"), DataFrame)
    selected = sort(unique(get_scenario_set(all_profiles_df, n_scenarios).scenario))
    println("Selected scenarios: $selected")

    bounds = load_investment_bounds()
    cov = investment_covariance()
    lb = zeros(length(INVESTABLE_ASSETS))
    ub = Vector{Float64}(bounds.ub)

    conn0 = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn0, input_data_path)
    params = read_adequacy_params(conn0)
    cuts = [build_adequacy_cuts(all_profiles_df, s, params) for s in selected]
    accept = x -> adequacy_verdict(x, cuts).passed
    println("Adequacy cuts: ", [(s, length(c.b)) for (s, c) in zip(selected, cuts)])

    optimizer, parameters = get_solver_parameters(solver_sym)
    res_c = feasibility_center(cuts, bounds.mean, ub; optimizer=optimizer, optimizer_parameters=parameters)
    center = res_c.center === nothing ? Vector{Float64}(bounds.mean) : Vector{Float64}(res_c.center)
    println("feasibility_center LP=$(res_c.status); μ* shift MW=",
            res_c.shift === nothing ? "n/a" : round.(res_c.shift; digits=1))

    # --- Uniform reject-to-target over [0, ub] (no mean) ---
    ru = scrambled_sobol_uniform_reject_to_target(TARGET, lb, ub; accept=accept, seed=1)
    acc_u = size(ru.samples, 2) / ru.n_drawn

    # --- Gaussian reject-to-target around μ* (for comparison) ---
    Σ = shrink_covariance(cov.matrix; α=0.2)
    rg = sobol_gaussian_reject_to_target(TARGET, center, Σ; accept=accept, ub=ub, seed=1)
    acc_g = size(rg.samples, 2) / rg.n_drawn

    # --- LP-validate a subset of the accepted UNIFORM samples ---
    models = Dict(s => build_scenario_model(s, all_profiles_df, input_data_path, solver_sym) for s in selected)
    nval = min(N_LP, size(ru.samples, 2))
    lp_ok = 0
    first_solve = Ref(true)
    for j in 1:nval
        x = Vector(ru.samples[:, j])
        all(lp_feasible!(models[s], x, solver_sym, first_solve) for s in selected) && (lp_ok += 1)
    end

    println("\n" * "#"^78)
    println("# UNIFORM-vs-GAUSSIAN ADEQUACY SAMPLING (target=$TARGET, scenarios $selected)")
    println("#"^78)
    println("  UNIFORM  [0,ub], no mean : acceptance $(round(100*acc_u; digits=2))%  ($(size(ru.samples,2)) kept from $(ru.n_drawn) draws)")
    println("  GAUSSIAN around μ*       : acceptance $(round(100*acc_g; digits=2))%  ($(size(rg.samples,2)) kept from $(rg.n_drawn) draws)")
    println("  Uniform cut→LP precision : $lp_ok/$nval LP-feasible across $selected  ($(round(100*lp_ok/nval; digits=1))%)")
    println("  (cut is a NECESSARY condition; <100% precision = samples that pass cuts but fail")
    println("   the full LP on storage/ramping/H2 — the LP remains the final check.)")
    return nothing
end

main()
