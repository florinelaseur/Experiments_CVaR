# Quantify the SD-loop infeasibility: how many of the actual Sobol-Gaussian
# samples (centered on bounds.mean) can serve each selected scenario, and jointly.
#
# Conclusion under test: the infeasibility is genuine capacity-adequacy screening
# (the sample cloud is centered on an under-provisioned mean for this scenario
# pair), NOT a code/mapping/seasonal bug.
#
# Builds one warm-started model per scenario (is_seasonal=false; it is irrelevant
# here) and re-fixes each sample. Joint feasibility = feasible in BOTH (operations
# decompose per scenario given the fixed shared investment).
#
# Usage: julia --project=. ScenarioReduction/old_scripts/quantify_infeasibility.jl

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
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

include(joinpath(@__DIR__, "..", "..", "utils", "functions.jl"))
include(joinpath(@__DIR__, "..", "..", "utils", "constants.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))

const N_SAMPLES = 128   # subset of one Sobol sequence (full sequence is 512)

# Build one warm-startable single-scenario model (is_seasonal forced off).
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
    if solver_sym == :Gurobi
        JuMP.set_optimizer_attribute(model, "Method", 1)  # dual simplex (warm-startable)
    end
    return (conn=conn, variables=variables, model=model, capacity_lookup=build_capacity_lookup(conn))
end

function feasible_mask(m, samples, solver_sym)
    n = size(samples, 2)
    mask = falses(n)
    presolve_off = false
    for j in 1:n
        fix_variables_from_sample(m.variables, :assets_investment, Vector(samples[:, j]); m.capacity_lookup)
        JuMP.optimize!(m.model)
        mask[j] = JuMP.termination_status(m.model) == JuMP.OPTIMAL
        if !presolve_off && solver_sym == :Gurobi
            JuMP.set_optimizer_attribute(m.model, "Presolve", 0)
            presolve_off = true
        end
    end
    return mask
end

function main()
    config = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config.toml"))
    input_data_path = joinpath(@__DIR__, "..", "..", config["simulation"]["input_data"])
    n_scenarios = config["simulation"]["number_of_scenarios"]
    solver_sym = Symbol(first(config["simulation"]["solvers"]))

    Random.seed!(19990907)
    all_profiles_df =
        CSV.read(joinpath(@__DIR__, "..", "..", "create-scenarios", "profiles-wide-all-scenarios.csv"), DataFrame)
    selected = sort(unique(get_scenario_set(all_profiles_df, n_scenarios).scenario))
    println("Selected scenarios (the SD loop screens these): $selected")

    # Reproduce the SD loop's first Sobol-Gaussian sequence, then take N_SAMPLES.
    bounds = load_investment_bounds()
    cov = investment_covariance()
    Σ = shrink_covariance(cov.matrix; α=0.2)
    full = sobol_gaussian_samples_nonneg(512, bounds.mean, Σ; mode=:clip, ub=bounds.ub, seed=1)
    samples = full[:, 1:N_SAMPLES]
    println("Testing $(N_SAMPLES) samples (of 512) from sequence 1; centered on bounds.mean.")

    masks = Dict{Int,BitVector}()
    for s in selected
        @info "Building + screening scenario $s"
        m = build_scenario_model(s, all_profiles_df, input_data_path, solver_sym)
        masks[s] = feasible_mask(m, samples, solver_sym)
    end

    joint = reduce(.&, values(masks))

    println("\n" * "#"^78)
    println("# INFEASIBILITY QUANTIFICATION  (N=$(N_SAMPLES) Sobol-Gaussian samples)")
    println("#"^78)
    for s in selected
        f = count(masks[s]); pct = round(100 * f / N_SAMPLES; digits=1)
        println("  scenario $s : feasible $f/$(N_SAMPLES)  ($pct%)  -> infeasible $(round(100-pct;digits=1))%")
    end
    jf = count(joint); jpct = round(100 * jf / N_SAMPLES; digits=1)
    println("  JOINT (both): feasible $jf/$(N_SAMPLES)  ($jpct%)  -> infeasible $(round(100-jpct;digits=1))%")
    # Also report bounds.mean itself (sample 0).
    println("\n# Reference: is bounds.mean feasible per scenario?")
    for s in selected
        m = build_scenario_model(s, all_profiles_df, input_data_path, solver_sym)
        fix_variables_from_sample(m.variables, :assets_investment, Vector{Float64}(bounds.mean); m.capacity_lookup)
        JuMP.optimize!(m.model)
        println("  scenario $s : bounds.mean -> $(JuMP.termination_status(m.model))")
    end
    return nothing
end

main()
