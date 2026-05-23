# Solve the deterministic energy model independently for each scenario
# (full hourly resolution, no temporal reduction) and collect the chosen
# investment per asset. Output:
#   outputs/per_scenario_investments.csv  — one row per scenario
#   outputs/investment_bounds.csv         — component-wise max/mean/std
#
# Purpose: empirically bound the investment variables for downstream
# stochastic-dominance scenario screening.

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate("..")
Pkg.instantiate()

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
using Statistics: Statistics
using Random

Random.seed!(19990907)

include(joinpath("..", "utils", "functions.jl"))
include(joinpath("..", "utils", "constants.jl"))

const INVESTABLE_ASSETS =
    ["ccgt", "ocgt", "solar", "wind", "wind_offshore", "electrolizer", "battery"]

function solve_one_scenario(
    scenario_id,
    all_profiles_df,
    input_data_path,
    lambda,
    alpha,
    optimizer,
    optimizer_parameters,
)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)

    scenario_profiles = filter(row -> row.scenario == scenario_id, all_profiles_df)
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
    DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")

    energy_problem = TEM.EnergyProblem(conn)
    TEM.create_model!(
        energy_problem;
        optimizer=optimizer,
        optimizer_parameters=optimizer_parameters,
        model_file_name="",
        enable_names=true,
        direct_model=false,
    )
    solve_time = @elapsed TEM.solve_model!(energy_problem)

    status = string(energy_problem.termination_status)

    if energy_problem.termination_status != JuMP.OPTIMAL
        return (
            status=status,
            objective=NaN,
            solve_time=solve_time,
            investments=nothing,
        )
    end

    TEM.save_solution!(energy_problem)

    inv_df = DataFrame(TIO.get_table(conn, "var_assets_investment"))
    asset_df = DataFrame(TIO.get_table(conn, "asset"))
    cap_lookup = Dict(string(a) => c for (a, c) in zip(asset_df.asset, asset_df.capacity))

    investments = Dict{String,Float64}()
    for asset in INVESTABLE_ASSETS
        rows = filter(r -> string(r.asset) == asset, inv_df)
        sol = isempty(rows) ? 0.0 : sum(rows.solution)
        investments[asset] = cap_lookup[asset] * sol
    end

    return (
        status=status,
        objective=energy_problem.objective_value,
        solve_time=solve_time,
        investments=investments,
    )
end

function main()
    config = TOML.parsefile(joinpath("..", "config.toml"))
    input_data_path = joinpath("..", config["simulation"]["input_data"])
    lambda = config["simulation"]["risk_aversion_weight_lambda"]
    alpha = config["simulation"]["risk_aversion_confidence_level"]
    solver_sym = Symbol(first(config["simulation"]["solvers"]))
    optimizer, optimizer_parameters = get_solver_parameters(solver_sym)

    profiles_path =
        joinpath("..", "create-scenarios", "profiles-wide-all-scenarios.csv")
    all_profiles_df = CSV.read(profiles_path, DataFrame)
    scenarios = sort(unique(all_profiles_df.scenario))
    @info "Solving deterministic model for $(length(scenarios)) scenarios using $solver_sym"

    per_scenario = DataFrame(;
        scenario=Int[],
        ccgt=Float64[],
        ocgt=Float64[],
        solar=Float64[],
        wind=Float64[],
        wind_offshore=Float64[],
        electrolizer=Float64[],
        battery=Float64[],
        objective_value=Float64[],
        termination_status=String[],
        solve_time=Float64[],
    )

    total_time = @elapsed begin
        for s in scenarios
            @info "Scenario $s"
            try
                r = solve_one_scenario(
                    s,
                    all_profiles_df,
                    input_data_path,
                    lambda,
                    alpha,
                    optimizer,
                    optimizer_parameters,
                )
                if r.investments === nothing
                    @warn "Scenario $s not optimal ($(r.status)); skipping"
                    continue
                end
                push!(
                    per_scenario,
                    (
                        s,
                        r.investments["ccgt"],
                        r.investments["ocgt"],
                        r.investments["solar"],
                        r.investments["wind"],
                        r.investments["wind_offshore"],
                        r.investments["electrolizer"],
                        r.investments["battery"],
                        r.objective,
                        r.status,
                        r.solve_time,
                    ),
                )
            catch e
                @warn "Scenario $s failed: $e"
            end
        end
    end
    @info "Solved $(nrow(per_scenario))/$(length(scenarios)) scenarios in $(round(total_time; digits=1)) s"

    output_folder = joinpath("..", "outputs")
    mkpath(output_folder)
    CSV.write(joinpath(output_folder, "per_scenario_investments.csv"), per_scenario)

    bounds = DataFrame(;
        asset=String[],
        max_investment=Float64[],
        mean_investment=Float64[],
        std_investment=Float64[],
    )
    if nrow(per_scenario) == 0
        @warn "No successful solves; investment_bounds.csv will be empty"
    else
        for asset in INVESTABLE_ASSETS
            col = per_scenario[!, Symbol(asset)]
            push!(
                bounds,
                (
                    asset,
                    maximum(col),
                    Statistics.mean(col),
                    length(col) > 1 ? Statistics.std(col) : 0.0,
                ),
            )
        end
    end
    CSV.write(joinpath(output_folder, "investment_bounds.csv"), bounds)
    @info "Investment bounds:" bounds
    return nothing
end

main()
