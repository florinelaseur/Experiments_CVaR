# Multi-scenario isolation test for the SD-loop infeasibility.
#
# The single-scenario decisive test came back OPTIMAL in every variant, so the
# bug is not visible with one scenario. The SD loop solves the SELECTED scenario
# subset (number_of_scenarios) simultaneously: under dummy_cluster! each scenario
# becomes its own representative period, so `is_seasonal=true` couples storage
# ACROSS scenarios (chains scenario 1's reservoir into scenario 2). That coupling
# is degenerate with one period and only bites with >=2.
#
# This script reproduces the SD-loop model build EXACTLY (same scenario selection,
# same dummy_cluster, same single shared assets_investment), then fixes a known-
# generous investment and solves with `is_seasonal` toggled:
#   Variant A (seasonal_off=true)  — what main.jl benchmark / generation do
#   Variant B (seasonal_off=false) — what stochastic_dominance.jl actually does
#
# Non-destructive: registers profiles_wide / stochastic_scenario in-memory instead
# of overwriting the input CSVs.
#
# Usage: julia --project=. ScenarioReduction/multiscenario_seasonal_test.jl

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

function build_multiscenario_connection(profiles_df, n_scenarios, input_data_path; seasonal_off::Bool)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)

    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, profiles_df, "profiles_wide")

    prob = 1.0 / n_scenarios
    values_sql = join(["($s, $prob)" for s in 1:n_scenarios], ", ")
    DuckDB.query(
        conn,
        "CREATE OR REPLACE TABLE stochastic_scenario AS " *
        "SELECT * FROM (VALUES $values_sql) AS t(scenario, probability)",
    )

    TC.transform_wide_to_long!(
        conn, "profiles_wide", "profiles";
        exclude_columns=["scenario", "milestone_year", "timestep"],
    )
    layout = TC.ProfilesTableLayout(; year=:milestone_year, cols_to_groupby=[:milestone_year, :scenario])
    TC.dummy_cluster!(conn; layout=layout)
    TEM.populate_with_defaults!(conn)
    if seasonal_off
        DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    end
    return conn
end

function solve_with_fixed_investment(conn, inv_mw::Vector{Float64}, solver_sym::Symbol)
    TEM.create_internal_tables!(conn)
    variables = TEM.compute_variables_indices(conn)
    constraints = TEM.compute_constraints_indices(conn)
    profiles = TEM.prepare_profiles_structure(conn)
    model, _ = TEM.create_model(conn, variables, constraints, profiles)

    optimizer, parameters = get_solver_parameters(solver_sym)
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in parameters]...)
    JuMP.set_silent(model)

    capacity_lookup = build_capacity_lookup(conn)
    fix_variables_from_sample(variables, :assets_investment, inv_mw; capacity_lookup)

    n_rep = try
        nrow(DataFrame(DuckDB.query(conn, "SELECT DISTINCT rep_period FROM profiles_rep_periods")))
    catch
        -1
    end
    status = solve_model(model)
    obj = status == JuMP.OPTIMAL ? JuMP.objective_value(model) : NaN
    return status, obj, n_rep
end

function run_case(label, profiles_df, n_scenarios, input_data_path, inv_mw, solver_sym; seasonal_off::Bool)
    tag = seasonal_off ? "A is_seasonal=false (benchmark env)" : "B is_seasonal default (SD-loop env)"
    println("\n" * "="^78)
    println(">>> $label | Variant $tag")
    conn = build_multiscenario_connection(profiles_df, n_scenarios, input_data_path; seasonal_off)
    status, obj, n_rep = solve_with_fixed_investment(conn, inv_mw, solver_sym)
    println(">>> $label | Variant $tag => status=$status  obj=$obj  (rep_periods=$n_rep)")
    return (case=label, variant=(seasonal_off ? "A" : "B"), status=string(status), objective=obj)
end

function main()
    config = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config.toml"))
    input_data_path = joinpath(@__DIR__, "..", "..", config["simulation"]["input_data"])
    n_scenarios = config["simulation"]["number_of_scenarios"]
    solver_sym = Symbol(first(config["simulation"]["solvers"]))

    # Reproduce the SD loop's scenario selection exactly (seed + get_scenario_set first).
    Random.seed!(19990907)
    all_profiles_df =
        CSV.read(joinpath(@__DIR__, "..", "..", "create-scenarios", "profiles-wide-all-scenarios.csv"), DataFrame)
    profiles_df = get_scenario_set(all_profiles_df, n_scenarios)
    selected_original = sort(unique(profiles_df.scenario))
    println("Selected original scenarios: $selected_original")

    mapping = Dict(old => new for (new, old) in enumerate(selected_original))
    profiles_df = copy(profiles_df)
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    # Investments to fix (MW), in INVESTABLE_ASSETS order.
    bounds = load_investment_bounds()
    mean_mw = Vector{Float64}(bounds.mean)

    per_scenario_df = CSV.read(default_per_scenario_investments_path(), DataFrame)
    sel_rows = filter(r -> r.scenario in selected_original, per_scenario_df)
    generous_mw = Float64[maximum(sel_rows[!, Symbol(a)]) for a in INVESTABLE_ASSETS]

    println("bounds.mean MW   = ", Dict(INVESTABLE_ASSETS .=> round.(mean_mw; digits=1)))
    println("generous max MW  = ", Dict(INVESTABLE_ASSETS .=> round.(generous_mw; digits=1)))

    summary = NamedTuple[]
    for (label, inv) in (("bounds.mean", mean_mw), ("max-of-selected-optima", generous_mw))
        push!(summary, run_case(label, profiles_df, n_scenarios, input_data_path, inv, solver_sym; seasonal_off=true))
        push!(summary, run_case(label, profiles_df, n_scenarios, input_data_path, inv, solver_sym; seasonal_off=false))
    end

    println("\n" * "#"^78)
    println("# MULTI-SCENARIO SEASONAL TEST SUMMARY (scenarios $selected_original)")
    println("#"^78)
    for s in summary
        println("  case=$(rpad(s.case,24))  variant=$(s.variant)  status=$(s.status)  obj=$(s.objective)")
    end
    return summary
end

main()
