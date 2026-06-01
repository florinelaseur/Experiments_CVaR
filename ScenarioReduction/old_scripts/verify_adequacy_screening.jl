# Verify the adequacy-cut pre-filter + feasibility mean-shift against ground-truth LP.
#
# Part A (controlled, scenarios [96,129] — the SD loop's selection for seed 19990907):
#   * build adequacy cuts, compute the feasibility centre μ* (the LP),
#   * for samples centred on bounds.mean vs μ*, compare the cheap cut verdict to the
#     actual single-scenario LP feasibility,
#   * assert SOUNDNESS (no cut-rejected sample is LP-feasible) and report the yield lift.
# Part B (tiny end-to-end): run the real stochastic_dominance loop on 8 samples to
#   confirm the wiring + CSV outputs.
#
# Usage: julia --project=. ScenarioReduction/verify_adequacy_screening.jl

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
include(joinpath(@__DIR__, "..", "src", "stochastic_dominance.jl"))

const N_A = 64   # samples per centre in Part A

# Single-scenario warm-startable LP model (is_seasonal off), built from original-id profiles.
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
    return (variables=variables, model=model, capacity_lookup=build_capacity_lookup(conn))
end

function lp_feasible(m, x, solver_sym, first_solve)
    fix_variables_from_sample(m.variables, :assets_investment, x; m.capacity_lookup)
    JuMP.optimize!(m.model)
    feas = JuMP.termination_status(m.model) == JuMP.OPTIMAL
    if first_solve[] && solver_sym == :Gurobi
        JuMP.set_optimizer_attribute(m.model, "Presolve", 0)
        first_solve[] = false
    end
    return feas
end

function part_a(all_profiles_df, input_data_path, solver_sym, selected, bounds, cov, params)
    println("\n" * "="^78)
    println("PART A — soundness + yield for scenarios $selected")
    cuts = [build_adequacy_cuts(all_profiles_df, s, params) for s in selected]
    for (s, c) in zip(selected, cuts)
        println("  scenario $s: $(length(c.b)) demanding-hour cuts")
    end

    optimizer, parameters = get_solver_parameters(solver_sym)
    res = feasibility_center(cuts, bounds.mean, bounds.ub; optimizer=optimizer, optimizer_parameters=parameters)
    println("  feasibility_center LP=$(res.status)")
    res.center === nothing && error("feasibility_center LP not optimal: $(res.status)")
    println("  bounds.mean = ", Dict(INVESTABLE_ASSETS .=> round.(bounds.mean; digits=1)))
    println("  μ*          = ", Dict(INVESTABLE_ASSETS .=> round.(res.center; digits=1)))
    println("  shift MW    = ", Dict(INVESTABLE_ASSETS .=> round.(res.shift; digits=1)))

    Σ = shrink_covariance(cov.matrix; α=0.2)
    models = Dict(s => build_scenario_model(s, all_profiles_df, input_data_path, solver_sym) for s in selected)

    for (label, center) in (("bounds.mean", Vector{Float64}(bounds.mean)), ("μ* (shifted)", Vector{Float64}(res.center)))
        samples = sobol_gaussian_samples_nonneg(512, center, Σ; mode=:clip, ub=bounds.ub, seed=1)[:, 1:N_A]
        cut_pass = falses(N_A); lp_joint = falses(N_A)
        first_solve = Ref(true)
        for j in 1:N_A
            x = Vector(samples[:, j])
            cut_pass[j] = adequacy_verdict(x, cuts).passed
            lp_joint[j] = all(lp_feasible(models[s], x, solver_sym, first_solve) for s in selected)
        end
        false_rejections = count(.!cut_pass .& lp_joint)   # cut said NO but LP said feasible
        println("\n  centre = $label  (N=$N_A)")
        println("    cut-pass rate      : $(count(cut_pass))/$N_A  ($(round(100*count(cut_pass)/N_A;digits=1))%)")
        println("    LP-joint-feasible  : $(count(lp_joint))/$N_A  ($(round(100*count(lp_joint)/N_A;digits=1))%)")
        println("    SOUNDNESS false-rejections (cut=NO but LP=feasible): $false_rejections  (must be 0)")
        if false_rejections != 0
            @error "ADEQUACY FILTER UNSOUND: $false_rejections samples rejected by cuts were LP-feasible!"
        end
    end
    return cuts
end

function part_b(all_profiles_df, input_data_path, n_scenarios, solver_sym, selected)
    println("\n" * "="^78)
    println("PART B — tiny end-to-end stochastic_dominance (num_samples=8, 1 sequence)")
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)
    sp = filter(row -> row.scenario in selected, all_profiles_df)
    mapping = Dict(old => new for (new, old) in enumerate(sort(unique(sp.scenario))))
    sp = copy(sp); sp[!, :scenario] = [mapping[s] for s in sp.scenario]
    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, sp, "profiles_wide")
    prob = 1.0 / n_scenarios
    vals = join(["($s, $prob)" for s in 1:n_scenarios], ", ")
    DuckDB.query(conn, "CREATE OR REPLACE TABLE stochastic_scenario AS SELECT * FROM (VALUES $vals) AS t(scenario, probability)")
    TC.transform_wide_to_long!(conn, "profiles_wide", "profiles"; exclude_columns=["scenario", "milestone_year", "timestep"])

    out = joinpath(@__DIR__, "..", "outputs")
    result = stochastic_dominance(
        conn; solver=solver_sym, num_samples=8, number_of_samples_sequences=1, output_dir=out,
        input_data_path=input_data_path,
    )
    println("  returned: rejected=$(result.rejected) optimal=$(result.optimal) infeasible=$(result.infeasible)")
    for f in ("adequacy_cuts.csv", "feasibility_center.csv", "screening_diagnostics.csv")
        p = joinpath(out, f)
        println("  output $(f): exists=$(isfile(p))")
    end
    return result
end

function main()
    config = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config.toml"))
    input_data_path = joinpath(@__DIR__, "..", "..", config["simulation"]["input_data"])
    n_scenarios = config["simulation"]["number_of_scenarios"]
    solver_sym = Symbol(first(config["simulation"]["solvers"]))

    Random.seed!(19990907)
    all_profiles_df = CSV.read(joinpath(@__DIR__, "..", "..", "create-scenarios", "profiles-wide-all-scenarios.csv"), DataFrame)
    selected = sort(unique(get_scenario_set(all_profiles_df, n_scenarios).scenario))
    println("Selected scenarios: $selected")

    bounds = load_investment_bounds()
    cov = investment_covariance()

    # Read adequacy params from a throwaway connection (peak_demand, hydro/ens caps).
    conn0 = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn0, input_data_path)
    params = read_adequacy_params(conn0)
    println("Adequacy params: peak_demand=$(params.peak_demand) hydro_cap=$(params.hydro_cap) ens_cap=$(params.ens_cap)")

    part_a(all_profiles_df, input_data_path, solver_sym, selected, bounds, cov, params)
    part_b(all_profiles_df, input_data_path, n_scenarios, solver_sym, selected)
    println("\nDONE.")
    return nothing
end

main()
