# Decisive single-scenario feasibility test.
#
# Reproduces the EXACT environment that produced per_scenario_investments.csv
# (single original scenario, full-hourly dummy_cluster, single-scenario solve),
# then FIXES that scenario's own optimal investment vector back in and re-solves.
#
# A scenario's own deterministic optimum MUST be feasible when re-solved against
# that same scenario under the same model. We toggle `is_seasonal` to isolate the
# divergence between the generation env and the SD-loop env:
#   Variant A (seasonal_off=true)  — generation env  -> expect OPTIMAL
#   Variant B (seasonal_off=false) — SD-loop env     -> expect INFEASIBLE
#
# Usage:
#   julia --project=. ScenarioReduction/decisive_single_scenario_test.jl
# or in the REPL:
#   include(joinpath(@__DIR__, "ScenarioReduction", "decisive_single_scenario_test.jl")); main()

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()
# Match main.jl:10 — the project's asset.csv uses the OLD TEM schema
# (storage_method_energy as a string enum), which only this git rev accepts.
# The registered v0.21.0 expects a BOOLEAN and fails in populate_with_defaults!.
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

Random.seed!(19990907)

include(joinpath(@__DIR__, "..", "..", "utils", "functions.jl"))
include(joinpath(@__DIR__, "..", "..", "utils", "constants.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))

# Build a single-scenario connection identical to solve_one_scenario in
# get_assets_investment_bounds.jl, except `is_seasonal` is toggled by `seasonal_off`.
function build_single_scenario_connection(
    scenario_id,
    all_profiles_df,
    input_data_path;
    seasonal_off::Bool,
)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)

    scenario_profiles = filter(row -> row.scenario == scenario_id, all_profiles_df)
    nrow(scenario_profiles) > 0 || error("No profile rows for scenario $scenario_id")
    scenario_profiles[!, :scenario] .= 1
    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, scenario_profiles, "profiles_wide")

    DuckDB.query(
        conn,
        """
        CREATE OR REPLACE TABLE stochastic_scenario AS
        SELECT 1 AS scenario, 1.0 AS probability
        """,
    )

    TC.transform_wide_to_long!(
        conn,
        "profiles_wide",
        "profiles";
        exclude_columns=["scenario", "milestone_year", "timestep"],
    )

    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    TC.dummy_cluster!(conn; layout=layout)
    TEM.populate_with_defaults!(conn)

    # The ONLY toggled difference. Generation + benchmark always force this off;
    # the SD loop (stochastic_dominance.jl) does not.
    if seasonal_off
        DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    end

    return conn
end

# Fix the (MW) investment vector via the verified alignment path and solve.
function solve_with_fixed_investment(conn, row_mw::Vector{Float64}, solver_sym::Symbol)
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
    fix_variables_from_sample(
        variables,
        :assets_investment,
        row_mw;
        capacity_lookup,
    )

    status = solve_model(model)  # prints IIS conflict on infeasible
    obj = status == JuMP.OPTIMAL ? JuMP.objective_value(model) : NaN
    return status, obj
end

function row_mw_for_scenario(per_scenario_df, scenario_id)
    rows = filter(r -> r.scenario == scenario_id, per_scenario_df)
    nrow(rows) == 1 || error("Expected exactly one per-scenario row for scenario $scenario_id, got $(nrow(rows))")
    r = rows[1, :]
    return Float64[r[Symbol(a)] for a in INVESTABLE_ASSETS]
end

function run_variant(scenario_id, row_mw, all_profiles_df, input_data_path, solver_sym; seasonal_off::Bool)
    label = seasonal_off ? "A (is_seasonal=false, generation env)" : "B (is_seasonal default, SD-loop env)"
    println("\n" * "="^78)
    println(">>> Scenario $scenario_id — Variant $label")
    println("    fixed MW = ", Dict(INVESTABLE_ASSETS .=> round.(row_mw; digits=1)))
    conn = build_single_scenario_connection(
        scenario_id, all_profiles_df, input_data_path; seasonal_off,
    )
    status, obj = solve_with_fixed_investment(conn, row_mw, solver_sym)
    println(">>> Scenario $scenario_id — Variant $label  => status=$status  obj=$obj")
    return (scenario=scenario_id, variant=(seasonal_off ? "A" : "B"), status=string(status), objective=obj)
end

function main()
    config = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config.toml"))
    input_data_path = joinpath(@__DIR__, "..", "..", config["simulation"]["input_data"])
    solver_sym = Symbol(first(config["simulation"]["solvers"]))

    all_profiles_df =
        CSV.read(joinpath(@__DIR__, "..", "..", "create-scenarios", "profiles-wide-all-scenarios.csv"), DataFrame)
    per_scenario_df = CSV.read(default_per_scenario_investments_path(), DataFrame)

    # Scenario 1 (representative) and scenario 19 (lowest combined VRE in the CSV).
    test_scenarios = [1, 19]

    summary = NamedTuple[]
    for sid in test_scenarios
        row_mw = row_mw_for_scenario(per_scenario_df, sid)
        push!(summary, run_variant(sid, row_mw, all_profiles_df, input_data_path, solver_sym; seasonal_off=true))
        push!(summary, run_variant(sid, row_mw, all_profiles_df, input_data_path, solver_sym; seasonal_off=false))
    end

    println("\n" * "#"^78)
    println("# DECISIVE TEST SUMMARY")
    println("#"^78)
    for s in summary
        println("  scenario=$(s.scenario)  variant=$(s.variant)  status=$(s.status)  obj=$(s.objective)")
    end
    return summary
end

# Called unconditionally (matches get_assets_investment_bounds.jl). The usual
# `abspath(PROGRAM_FILE) == @__FILE__` guard would fail here because cd(@__DIR__)
# above changes the working directory before the comparison.
main()
