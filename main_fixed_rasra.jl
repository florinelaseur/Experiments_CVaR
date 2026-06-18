# Takes RASRA's investment decisions from var_assets_investment.csv and re-solves the full hourly benchmark on all N scenarios
# Must be run after rasra.jl has completed for the same (N, seed)

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using CSV: CSV
using DataFrames
using Random
using TOML: TOML
using JuMP: JuMP
using Plots

seed = parse(Int, get(ENV, "EXPERIMENT_SEED", "19990907"))
Random.seed!(seed)

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

config = TOML.parsefile("config.toml")
input_data_path = config["simulation"]["input_data"]
solvers = [Symbol(el) for el in config["simulation"]["solvers"]]
lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]

# Re-derive the same N scenarios with the same seed so profiles_df matches exactly what rasra.jl used as its full scenario starting point
profiles_path = joinpath(@__DIR__, "create-scenarios", "profiles-wide-all-scenarios.csv")
all_profiles_df = CSV.read(profiles_path, DataFrame)
profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
selected_scenarios = sort(unique(profiles_df.scenario))
mapping = Dict(old => new for (new, old) in enumerate(selected_scenarios))
profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

df_stochastic_full = DataFrame(;
    scenario = sort(unique(profiles_df.scenario)),
    probability = fill(1.0 / number_of_scenarios, number_of_scenarios),
)

# Fix RASRA's investment decisions into the benchmark model
function fix_investments_from_csv!(energy_problem, csv_path)
    if !isfile(csv_path)
        error("var_assets_investment.csv not found at $csv_path")
    end
    df = CSV.read(csv_path, DataFrame)

    lookup = Dict(
        (String(row.asset), row.milestone_year) => row.solution
        for row in eachrow(df)
    )

    container = energy_problem.variables[:assets_investment].container
    n_fixed = 0
    n_missing = 0

    for var in container
        raw = JuMP.name(var)
        m = match(r"\[(\d+),\s*(.+)\]$", raw)
        if isnothing(m)
            @warn "Could not parse variable name: $raw: skipping"
            continue
        end
        year = parse(Int, m[1])
        asset = m[2]
        key = (asset, year)
        if haskey(lookup, key)
            JuMP.fix(var, lookup[key]; force=true)
            n_fixed += 1
        else
            @warn "No CSV entry for $key: skipping"
            n_missing += 1
        end
    end

    @info "Fixed $n_fixed investment variables ($(n_missing) missing from CSV)"
end

results_df = DataFrame(;
    solver = Symbol[],
    num_scenarios = Int[],
    seed = Int[],
    time_to_build = Float64[],
    time_to_solve_fixed = Float64[],
    time_to_save = Float64[],
    objective_value_fixed = Float64[],
    termination_status_fixed = String[],
    num_constraints = Int[],
    num_variables = Int[],
    num_loss_of_load_e_demand = Int[],
    num_loss_of_load_h2_demand = Int[],
    water_borrowed = Float64[],
    value_at_risk_threshold_mu = Float64[],
)

function run_fixed_rasra()
    for solver in solvers
        optimizer, parameters = get_solver_parameters(solver)

        rasra_output_folder = joinpath(
            @__DIR__, "outputs",
            "N$(number_of_scenarios)_seed$(seed)",
            "RASRA", string(solver),
        )

        if !isdir(rasra_output_folder)
            @warn "RASRA output folder not found: $rasra_output_folder"
            continue
        end

        rasra_inv_path = joinpath(rasra_output_folder, "var_assets_investment.csv")

        # Set up full N-scenario hourly benchmark, identical to 0_HourlyBenchmark
        @info "[$solver] Setting up full hourly benchmark (N=$number_of_scenarios scenarios)"

        CSV.write(joinpath(input_data_path, "profiles-wide.csv"),
                  profiles_df; writeheader=true)
        CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"),
                  df_stochastic_full; writeheader=true)

        conn_full = DuckDB.DBInterface.connect(DuckDB.DB)
        TIO.read_csv_folder(conn_full, input_data_path)
        DuckDB.query(
            conn_full,
            """
            UPDATE model_parameters
            SET risk_aversion_weight_lambda          = $lambda,
                risk_aversion_confidence_level_alpha = $alpha;
            """,
        )

        TC.transform_wide_to_long!(
            conn_full, "profiles_wide", "profiles";
            exclude_columns = ["scenario", "milestone_year", "timestep"],
        )

        layout_full = TC.ProfilesTableLayout(;
            year = :milestone_year,
            cols_to_groupby = [:milestone_year, :scenario],
        )
        TC.dummy_cluster!(conn_full; layout = layout_full)
        TEM.populate_with_defaults!(conn_full)
        DuckDB.query(conn_full, "UPDATE asset SET is_seasonal = false")

        energy_problem_full = TEM.EnergyProblem(conn_full)

        t_build = @elapsed TEM.create_model!(
            energy_problem_full;
            optimizer = optimizer,
            optimizer_parameters = parameters,
            model_file_name = "",
            enable_names = true,
            direct_model = false,
        )

        # Fix RASRA's investment decisions from the saved CSV into the full model
        fix_investments_from_csv!(energy_problem_full, rasra_inv_path)

        @info "[$solver] Solving full benchmark with RASRA investments fixed"
        t_solve = @elapsed TEM.solve_model!(energy_problem_full)

        if energy_problem_full.termination_status == JuMP.INFEASIBLE
            @warn "[$solver] Fixed re-solve is INFEASIBLE — printing IIS"
            JuMP.compute_conflict!(energy_problem_full.model)
            iis_model, _ = JuMP.copy_conflict(energy_problem_full.model)
            print(iis_model)
        end

        output_folder = joinpath(
            @__DIR__, "outputs",
            "N$(number_of_scenarios)_seed$(seed)",
            "RASRA_fixed", string(solver),
        )
        mkpath(output_folder)

        t_save = @elapsed begin
            TEM.save_solution!(energy_problem_full)
            TEM.export_solution_to_csv_files(output_folder, energy_problem_full)
            df_cost_per_scenario = export_operational_cost_per_scenario(energy_problem_full, output_folder)
            plot_operational_cost_per_scenario(df_cost_per_scenario, output_folder)
        end

        var_flow_df = TIO.get_table(conn_full, "var_flow")
        flow_ens = filter(r -> r.from_asset == "ens" && r.to_asset == "e_demand", var_flow_df)
        flow_smr_ccs = filter(r -> r.from_asset == "smr_ccs" && r.to_asset == "h2_demand", var_flow_df)
        water_borrow = filter(r -> r.from_asset == "water_borrower" && r.to_asset == "hydro_reservoir", var_flow_df)

        n_lol_ens = count(r -> r.solution > 0.0, eachrow(flow_ens))
        n_lol_smr_ccs = count(r -> r.solution > 0.0, eachrow(flow_smr_ccs))
        amount_water = sum(water_borrow.solution)

        mu_df = TIO.get_table(conn_full, "var_value_at_risk_threshold_mu")
        mu_value = nrow(mu_df) == 0 ? NaN : only(mu_df.solution)

        push!(results_df, (
            solver = solver,
            num_scenarios = number_of_scenarios,
            seed = seed,
            time_to_build = t_build,
            time_to_solve_fixed = t_solve,
            time_to_save = t_save,
            objective_value_fixed = energy_problem_full.objective_value,
            termination_status_fixed = string(energy_problem_full.termination_status),
            num_constraints = JuMP.num_constraints(
                energy_problem_full.model; count_variable_in_set_constraints = false,
            ),
            num_variables = JuMP.num_variables(energy_problem_full.model),
            num_loss_of_load_e_demand = n_lol_ens,
            num_loss_of_load_h2_demand = n_lol_smr_ccs,
            water_borrowed = amount_water,
            value_at_risk_threshold_mu = mu_value,
        ))

        @info "[$solver] Done. Fixed objective value = $(energy_problem_full.objective_value)"
    end

    out_path = joinpath(
        @__DIR__, "outputs",
        "results_rasra_fixed_N$(number_of_scenarios)_seed$(seed).csv",
    )
    results_df |> CSV.write(out_path; writeheader=true)
    @info "Results written to $out_path"

    return results_df
end

run_fixed_rasra()