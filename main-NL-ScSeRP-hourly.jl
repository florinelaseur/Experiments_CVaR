# Copyright (c) 2025: Diego Tejada and contributors
#
# Use of this source code is governed by an Apache 2.0 license that can be found
# in the LICENSE.md file or at https://opensource.org/license/apache-2-0.

cd(@__DIR__)

ENV["GKSwstype"] = "100"
# using Pkg: Pkg
# Pkg.activate(".")
# Pkg.instantiate()

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using Distances: Distances
using CSV: CSV
using Statistics: Statistics
using JuMP: JuMP
using TOML: TOML
using Plots
using Random
using DataFrames

base_seed = 19990907

# Defaults to seed1 when main.jl is run directly
seed = parse(Int, get(ENV, "EXPERIMENT_SEED", "1"))

# Reproduce the seed-th draw from the fixed base seed
rng = MersenneTwister(base_seed)
random_seeds = rand(rng, 1:typemax(Int32), seed)
random_seed = random_seeds[end]

Random.seed!(random_seed)

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

distance_map = Dict(
    :Euclidean => Distances.Euclidean(),
    :SqEuclidean => Distances.SqEuclidean(),
    :CosineDist => Distances.CosineDist(),
    :Cityblock => Distances.Cityblock(),
    :Chebyshev => Distances.Chebyshev(),
)

# Read and transform user input files to Tulipa input files
config = TOML.parsefile("config.toml")
input_data_path = joinpath(
    homedir(),
    "Nextcloud",
    config["simulation"]["input_data_NL"],
)

input_data_path_CC = joinpath(
    homedir(),
    "Nextcloud",
    config["simulation"]["input_data_NL_CC"],
)
use_ratio = config["clustering"]["use_ratio"]
heuristic_distance = config["clustering"]["heuristic_distance"]
fix_level_storage = config["simulation"]["fix_level_storage"]
representative_periods = config["simulation"]["representative_periods"]
solvers = [Symbol(el) for el in config["simulation"]["solvers"]]
lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]
fix_benchmark = config["simulation"]["fix_benchmark"]

#for new scenarios

profiles_path = joinpath(homedir(), "Nextcloud", "ExperimentData", "profiles-wide-all-scenarios.csv")
all_profiles_df = CSV.read(profiles_path, DataFrame)
profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
selected_scenarios = sort(unique(profiles_df.scenario))
mapping = Dict(old => new for (new, old) in enumerate(selected_scenarios))
profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]
CSV.write(joinpath(input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

model_parameters_df = DataFrame(
    risk_aversion_weight_lambda=[lambda],
    risk_aversion_confidence_level_alpha=[alpha],
)
for data_path in (input_data_path, input_data_path_CC)
    mkpath(data_path)
    CSV.write(joinpath(data_path, "model-parameters.csv"), model_parameters_df; writeheader=true)
end

df_stochastic_scenario = DataFrame(;
    scenario=sort(unique(profiles_df.scenario)),
    probability=fill(1.0 / number_of_scenarios, number_of_scenarios),
)
CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

# to keep scenarios

# profiles_df = CSV.read(joinpath(input_data_path, "profiles-wide.csv"), DataFrame)
# df_stochastic_scenario = CSV.read(joinpath(input_data_path, "stochastic-scenario.csv"), DataFrame)

case_studies_info = CSV.read(
    "case-studies-info.csv",
    DataFrame;
    types=Dict(
        :base_name => String,
        :period_duration => Int,
        :method => Symbol,
        :distance => Symbol,
        :weight_type => Symbol,
        :niters => Int,
        :learning_rate => Float64,
        :stochastic_method => Symbol,
        :run_case => Bool,
    ),
)

enable_names = true
direct_model = false
results_df = DataFrame(;
    case_name=String[],
    rp=Int[],
    solver=Symbol[],
    time_to_cluster=Float64[],
    time_to_read=Float64[],
    time_to_create=Float64[],
    time_to_solve=Float64[],
    time_to_save=Float64[],
    objective_value=Float64[],
    termination_status=String[],
    value_at_risk_threshold_mu_red=Float64[],
    num_constraints=Int[],
    num_variables=Int[],
    time_to_resolve_benchmark=Float64[],
    objective_value_resolve_benchmark=Float64[],
    termination_status_resolve_benchmark=String[],
    num_loss_of_load_e_demand_benchmark=Int[],
    lole_e_demand_benchmark=Float64[],
    num_loss_of_load_h2_demand_benchmark=Int[],
    lole_h2_demand_benchmark=Float64[],
    water_borrowed_benchmark=Float64[],
    value_at_risk_threshold_mu_benchmark=Float64[],
    time_to_resolve_baseline=Float64[],
    objective_value_resolve_baseline=Float64[],
    termination_status_resolve_baseline=String[],
    num_loss_of_load_e_demand_baseline=Int[],
    lole_e_demand_baseline=Float64[],
    num_loss_of_load_h2_demand_baseline=Int[],
    lole_h2_demand_baseline=Float64[],
    water_borrowed_baseline=Float64[],
    value_at_risk_threshold_mu_baseline=Float64[],
    scenario_set=String[],
    seed=Int[],
    number_of_scenarios=Int[],
)


function main()
    connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection_benchmark, input_data_path)
    profiles_wide = TIO.get_table(connection_benchmark, "profiles_wide")
    n_scenarios = length(unique(profiles_wide.scenario))

    @info "Running the base case study (0_HourlyBaseline)"
    base_name = "0_HourlyBaseline"

    # set up the connection and read the data
    connection_baseline = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection_baseline, input_data_path)
    # update the CSV input data for Tulipa from the config file info
    DuckDB.query(
        connection_baseline,
        "
        UPDATE model_parameters -- tables are with underscore in DuckDB world
        SET
            risk_aversion_weight_lambda = $(lambda) ,
            risk_aversion_confidence_level_alpha = $(alpha);
        ",
    )
    # transform the profiles data from wide to long
    TC.transform_wide_to_long!(
        connection_baseline,
        "profiles_wide",
        "profiles";
        exclude_columns=["scenario", "milestone_year", "timestep"],
    )

    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    time_to_cluster = @elapsed TC.dummy_cluster!(connection_baseline; layout=layout)
    TEM.populate_with_defaults!(connection_baseline)
    DuckDB.query(connection_baseline, "UPDATE asset SET is_seasonal = false")

    time_to_read = @elapsed energy_problem_baseline = TEM.EnergyProblem(connection_baseline)

    baseline_investment_by_solver = Dict{Symbol,DataFrame}()
    baseline_n_lol_ens_by_solver = Dict{Symbol,Int}()
    baseline_n_lol_smr_ccs_by_solver = Dict{Symbol,Int}()
    baseline_objective_by_solver = Dict{Symbol,Float64}()

    for solver in solvers
        optimizer, parameters = get_solver_parameters(solver)

        @info "Creating the model for the base case study $base_name with $solver"
        time_to_create = @elapsed TEM.create_model!(
            energy_problem_baseline;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=enable_names,
            direct_model=direct_model,
        )

        baseline_output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", base_name, "N$(number_of_scenarios)_seed$(seed)", string(solver))
        mkpath(baseline_output_folder)

        if !(isfile(joinpath(baseline_output_folder, "var_assets_investment.csv")) &&
             isfile(joinpath(baseline_output_folder, "var_flow.csv")) &&
             isfile(joinpath(baseline_output_folder, "baseline_breakdown.csv")) &&
             isfile(joinpath(baseline_output_folder, "var_value_at_risk_threshold_mu.csv",)) &&
             isfile(joinpath(baseline_output_folder, "total_operational_cost_per_scenario.csv",)))

            @info "Solving the model and saving the solution for the base case study $base_name with $solver"
            time_to_solve = @elapsed TEM.solve_model!(energy_problem_baseline)
            #        mu_value =
            #            JuMP.value(energy_problem_baseline.variables[:value_at_risk_threshold_mu].container)
            time_to_save = @elapsed TEM.save_solution!(energy_problem_baseline)
            TEM.export_solution_to_csv_files(baseline_output_folder, energy_problem_baseline)

            baseline_investment_df = TIO.get_table(connection_baseline, "var_assets_investment")
            baseline_objective = energy_problem_baseline.objective_value
            termination_status = string(energy_problem_baseline.termination_status)
            baseline_df = DataFrame(time_to_solve=[time_to_solve], time_to_save=[time_to_save], objective_value=[baseline_objective], termination_status=[termination_status])
            CSV.write(joinpath(baseline_output_folder, "baseline_breakdown.csv"), baseline_df; writeheader=true)

            mu_value_df = TIO.get_table(connection_baseline, "var_value_at_risk_threshold_mu")
            var_flow_df = TIO.get_table(connection_baseline, "var_flow")

            df_cost_per_scenario = export_total_operational_cost_per_scenario(energy_problem_baseline, baseline_output_folder)
            plot_cost_per_scenario(df_cost_per_scenario, baseline_output_folder, mu_value_df)

        else
            baseline_df = CSV.read(joinpath(baseline_output_folder, "baseline_breakdown.csv"), DataFrame)
            time_to_solve = only(baseline_df.time_to_solve)
            time_to_save = only(baseline_df.time_to_save)
            baseline_investment_df = CSV.read(joinpath(baseline_output_folder, "var_assets_investment.csv"), DataFrame)
            baseline_objective = only(baseline_df.objective_value)
            termination_status = only(baseline_df.termination_status)

            mu_value_df = CSV.read(joinpath(baseline_output_folder, "var_value_at_risk_threshold_mu.csv"), DataFrame)
            var_flow_df = CSV.read(joinpath(baseline_output_folder, "var_flow.csv"), DataFrame)

            df_cost_per_scenario = CSV.read(joinpath(baseline_output_folder, "total_operational_cost_per_scenario.csv"), DataFrame)
            plot_cost_per_scenario(df_cost_per_scenario, baseline_output_folder, mu_value_df)
        end

        mu_value = if nrow(mu_value_df) == 0
            NaN
        else
            only(mu_value_df.solution)
        end

        flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
        flow_smr_ccs =
            filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
        water_borrowed = filter(
            row -> row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
            var_flow_df,
        )

        baseline_n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
        baseline_lole_e_demand = baseline_n_lol_ens / number_of_scenarios
        baseline_n_lol_smr_ccs = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
        baseline_lole_h2_demand = baseline_n_lol_smr_ccs / number_of_scenarios

        amount_water_borrowed_b = sum(water_borrowed.solution)

        baseline_investment_by_solver[solver] = copy(baseline_investment_df)
        baseline_n_lol_ens_by_solver[solver] = baseline_n_lol_ens
        baseline_n_lol_smr_ccs_by_solver[solver] = baseline_n_lol_smr_ccs
        baseline_objective_by_solver[solver] = baseline_objective

        new_results_row = (
            case_name=base_name,
            rp=1,
            solver=solver,
            time_to_cluster=0.0,
            time_to_read=time_to_read,
            time_to_create=time_to_create,
            time_to_solve=time_to_solve,
            time_to_save=time_to_save,
            objective_value=baseline_objective,
            termination_status=termination_status,
            value_at_risk_threshold_mu_red=0.0,
            num_constraints=JuMP.num_constraints(
                energy_problem_baseline.model;
                count_variable_in_set_constraints=false,
            ),
            num_variables=JuMP.num_variables(energy_problem_baseline.model),
            time_to_resolve_benchmark=0.0,
            objective_value_resolve_benchmark=0.0,
            termination_status_resolve_benchmark="",
            num_loss_of_load_e_demand_benchmark=0,
            lole_e_demand_benchmark=0.0,
            num_loss_of_load_h2_demand_benchmark=0,
            lole_h2_demand_benchmark=0.0,
            water_borrowed_benchmark=0.0,
            value_at_risk_threshold_mu_benchmark=0.0,
            time_to_resolve_baseline=0.0,
            objective_value_resolve_baseline=0.0,
            termination_status_resolve_baseline="",
            num_loss_of_load_e_demand_baseline=baseline_n_lol_ens,
            lole_e_demand_baseline=baseline_lole_e_demand,
            num_loss_of_load_h2_demand_baseline=baseline_n_lol_smr_ccs,
            lole_h2_demand_baseline=baseline_lole_h2_demand,
            water_borrowed_baseline=amount_water_borrowed_b,
            value_at_risk_threshold_mu_baseline=mu_value,
            scenario_set="full",
            seed=seed,
            number_of_scenarios=number_of_scenarios,
        )
        push!(results_df, new_results_row)
    end


    # optimize the energy system for each case study
    for row in eachrow(case_studies_info)
        base_name = row[:base_name]
        period_duration = row[:period_duration]
        method = row[:method]
        distance = distance_map[row[:distance]]
        weight_type = row[:weight_type]
        niters = row[:niters]
        learning_rate = row[:learning_rate]
        stochastic_method = row[:stochastic_method]
        run_case = row[:run_case]

        weight_fitting_kwargs = Dict(:learning_rate => learning_rate, :niters => niters)
        clustering_kwargs = Dict(:learning_rate => learning_rate, :niters => niters)

        if !run_case
            continue
        end

        for rp in representative_periods
            case_name = base_name * "_rp_" * "$rp" * "_benchmark"

            @info "Processing case study: $case_name"

            connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
            TIO.read_csv_folder(connection_benchmark, input_data_path)

            DuckDB.query(
                connection_benchmark,
                "
                UPDATE model_parameters -- tables are with underscore in DuckDB world
                SET
                    risk_aversion_weight_lambda = $(lambda) ,
                    risk_aversion_confidence_level_alpha = $(alpha);
                ",
            )
            # to use the ratio availability/demand
            if use_ratio == true # be careful: this works now that we have only one demand location, so we divide each availability and inflow by that only demand
                DuckDB.query(
                    connection_benchmark,
                    "
                    UPDATE profiles_wide
                    SET
                        solar = solar / demand,
                        wind_offshore = wind_offshore / demand,
                        wind_onshore = wind_onshore / demand,
                        hydro_inflow = hydro_inflow / demand;
                    ",
                )
            end

            # transform the profiles data from wide to long
            TC.transform_wide_to_long!(
                connection_benchmark,
                "profiles_wide",
                "profiles";
                exclude_columns=["scenario", "milestone_year", "timestep"],
            )

            if stochastic_method == :per_scenario
                layout = TC.ProfilesTableLayout(;
                    year=:milestone_year,
                    cols_to_groupby=[:milestone_year, :scenario],
                )
                time_to_cluster = @elapsed TC.cluster!(
                    connection_benchmark,
                    period_duration,
                    rp; #round(Int, rp / n_scenarios);
                    method=method,
                    distance=distance,
                    weight_type=weight_type,
                    layout=layout,
                    clustering_kwargs,
                    weight_fitting_kwargs,
                )

            elseif stochastic_method == :cross_scenario
                layout = TC.ProfilesTableLayout(;
                    year=:milestone_year,
                    cols_to_groupby=[:milestone_year],
                    cols_to_crossby=[:scenario],
                )
                time_to_cluster = @elapsed TC.cluster!(
                    connection_benchmark,
                    period_duration,
                    rp;
                    method=method,
                    distance=distance,
                    weight_type=weight_type,
                    layout=layout,
                    clustering_kwargs,
                    weight_fitting_kwargs,
                )
            else
                error("Unknown stochastic method: $stochastic_method")
            end
            if use_ratio == true
                DuckDB.query(
                    connection_benchmark,
                    "UPDATE profiles AS x
                        SET value =
                            CASE
                                WHEN x.profile_name = 'demand' THEN x.value
                                ELSE x.value * d.value
                            END
                        FROM profiles AS d
                        WHERE d.timestep   = x.timestep
                        AND d.milestone_year       = x.milestone_year
                        AND d.scenario   = x.scenario
                        AND d.profile_name = 'demand';
                            ",
                )
            end
            TEM.populate_with_defaults!(connection_benchmark)

            time_to_read = @elapsed energy_problem_benchmark = TEM.EnergyProblem(connection_benchmark)

            for solver in solvers
                optimizer, parameters = get_solver_parameters(solver)

                @info "Creating the model for the case study: $case_name"
                time_to_create = @elapsed TEM.create_model!(
                    energy_problem_benchmark;
                    optimizer=optimizer,
                    optimizer_parameters=parameters,
                    model_file_name="",
                    enable_names=enable_names,
                )

                output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, string(solver))
                mkpath(output_folder)

                @info "Solving the model and saving the solution for the case study: $case_name with $solver"
                time_to_solve = @elapsed TEM.solve_model!(energy_problem_benchmark)
                time_to_save = @elapsed TEM.save_solution!(energy_problem_benchmark)
                TEM.export_solution_to_csv_files(output_folder, energy_problem_benchmark)

                output_file = joinpath(output_folder, "rep_periods_mapping.csv")
                DuckDB.execute(connection_benchmark, "COPY rep_periods_mapping TO '$output_file' (HEADER, DELIMITER ',')")

                benchmark_investment_df = TIO.get_table(connection_benchmark, "var_assets_investment")

                mu_value_df = TIO.get_table(connection_benchmark, "var_value_at_risk_threshold_mu")
                mu_value_benchmark = if nrow(mu_value_df) == 0
                    NaN
                else
                    only(mu_value_df.solution)
                end
                if !isnan(mu_value_benchmark)
                    @info "mu_value of Benchmark (24 periods per scenario on full scenario set) is defined"
                    @show mu_value_benchmark
                end


                benchmark_cost_df = export_total_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
                plot_cost_per_scenario(benchmark_cost_df, output_folder, mu_value_df)

                sorted_cost_df = sort(benchmark_cost_df, :total_cost)

                var_index = ceil(Int, alpha * number_of_scenarios)
                var_empirical = sorted_cost_df.total_cost[var_index]

                var_flow_df = TIO.get_table(connection_benchmark, "var_flow")
                flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
                flow_smr_ccs =
                    filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
                water_borrowed = filter(
                    row ->
                        row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                    var_flow_df,
                )
                # count steps with loss of load
                bm_n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
                lole_e_demand = bm_n_lol_ens / number_of_scenarios
                bm_n_lol_smr_ccs = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
                lole_h2_demand = bm_n_lol_smr_ccs / number_of_scenarios

                # count how much water_borrowed
                amount_water_borrowed_b = sum(water_borrowed.solution)

                amount_water_borrowed_err = sum(water_borrowed.solution)
                if amount_water_borrowed_err > 0.0
                    error("Borrowed water has been used: $amount_water_borrowed_err")
                end

                baseline_investment_df = baseline_investment_by_solver[solver]
                baseline_n_lol_ens = baseline_n_lol_ens_by_solver[solver]
                baseline_n_lol_smr_ccs = baseline_n_lol_smr_ccs_by_solver[solver]
                baseline_objective = baseline_objective_by_solver[solver]

                new_results_row = (
                    case_name=case_name,
                    rp=rp,
                    solver=solver,
                    time_to_cluster=time_to_cluster,
                    time_to_read=time_to_read,
                    time_to_create=time_to_create,
                    time_to_solve=time_to_solve,
                    time_to_save=time_to_save,
                    objective_value=energy_problem_benchmark.objective_value,
                    termination_status=string(energy_problem_benchmark.termination_status),
                    value_at_risk_threshold_mu_red=0.0,
                    num_constraints=JuMP.num_constraints(
                        energy_problem_benchmark.model;
                        count_variable_in_set_constraints=false,
                    ),
                    num_variables=JuMP.num_variables(energy_problem_benchmark.model),
                    time_to_resolve_benchmark=0.0,
                    objective_value_resolve_benchmark=0.0,
                    termination_status_resolve_benchmark="",
                    num_loss_of_load_e_demand_benchmark=bm_n_lol_ens,
                    lole_e_demand_benchmark=lole_e_demand,
                    num_loss_of_load_h2_demand_benchmark=bm_n_lol_smr_ccs,
                    lole_h2_demand_benchmark=lole_h2_demand,
                    water_borrowed_benchmark=amount_water_borrowed_b,
                    value_at_risk_threshold_mu_benchmark=mu_value_benchmark,
                    time_to_resolve_baseline=0.0,
                    objective_value_resolve_baseline=0.0,
                    termination_status_resolve_baseline="",
                    num_loss_of_load_e_demand_baseline=0,
                    lole_e_demand_baseline=0.0,
                    num_loss_of_load_h2_demand_baseline=0,
                    lole_h2_demand_baseline=0.0,
                    water_borrowed_baseline=0.0,
                    value_at_risk_threshold_mu_baseline=0.0,
                    scenario_set="full",
                    seed=seed,
                    number_of_scenarios=number_of_scenarios,
                )
                push!(results_df, new_results_row)

                @info "Starting Scenario selection"

                #insert scenario selection and create and solve energy_problem_red
                output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, "scenario_selection", string(solver))
                mkpath(output_folder)

                CSV.write(
                    joinpath(output_folder, "total_operational_cost_per_scenario.csv"),
                    benchmark_cost_df;
                    writeheader=true,
                )

                tol = 1e-5

                df_tail_scenarios = copy(benchmark_cost_df)

                df_tail_scenarios[!, :solution] =
                    max.(0.0, df_tail_scenarios.total_cost .- var_empirical)

                df_tail_scenarios = filter(
                    row -> row.total_cost > var_empirical + tol,
                    df_tail_scenarios,
                )

                df_mu_scenario = filter(
                    row -> abs(row.total_cost - var_empirical) <= tol,
                    benchmark_cost_df,
                )

                df_tail_scenarios = df_tail_scenarios[:, [:id, :scenario, :probability, :total_cost]]
                df_mu_scenario = df_mu_scenario[:, [:id, :scenario, :probability, :total_cost]]

                plot_cost_per_scenario_inc_tail(
                    benchmark_cost_df,
                    df_tail_scenarios,
                    output_folder,
                    df_mu_scenario,
                    var_empirical,
                    case_name,
                )

                CSV.write(
                    joinpath(output_folder, "tail_scenarios.csv"),
                    df_tail_scenarios;
                    writeheader=true,
                )

                profiles_df = CSV.read(
                    joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-input-data", "NL-case-study", "profiles-wide.csv"),
                    DataFrame,
                )

                tail_scenarios_ids = unique(df_tail_scenarios.scenario)
                mu_scenario_id = unique(df_mu_scenario.scenario)

                explicit_ids = unique(vcat(
                    tail_scenarios_ids,
                    mu_scenario_id,
                ))

                expected_cost = sum(benchmark_cost_df.probability .* benchmark_cost_df.total_cost)
                expected_idx = argmin(abs.(benchmark_cost_df.total_cost .- expected_cost))
                average_case_row = benchmark_cost_df[expected_idx, :]

                df_expected_cost_scenario = DataFrame(
                    scenario=[average_case_row.scenario],
                    total_cost=[average_case_row.total_cost],
                )
                CSV.write(joinpath(@__DIR__, output_folder, "expected_cost_scenario.csv"), df_expected_cost_scenario; writeheader=true)

                plot_cost_per_scenario_inc_tail_inc_representative(
                    benchmark_cost_df,
                    df_tail_scenarios,
                    df_expected_cost_scenario,
                    output_folder,
                    df_mu_scenario,
                    var_empirical,
                    "$case_name",
                )

                n_tail = length(tail_scenarios_ids)
                # tail_probability = (1.0 - alpha) / n_tail
                tail_probability = 1 / number_of_scenarios

                mu_probability = 1 / number_of_scenarios

                expected_cost_scenario = only(df_expected_cost_scenario.scenario)

                if expected_cost_scenario in explicit_ids
                    selected_scenarios_ids = copy(explicit_ids)
                    probabilities = [
                        scenario == expected_cost_scenario ?
                        1.0 - (length(explicit_ids) - 1) / number_of_scenarios :
                        1.0 / number_of_scenarios
                        for scenario in selected_scenarios_ids
                    ]
                else
                    selected_scenarios_ids = vcat(
                        explicit_ids,
                        expected_cost_scenario,
                    )
                    probabilities = vcat(
                        fill(1 / number_of_scenarios, length(explicit_ids),),
                        1 - length(explicit_ids) / number_of_scenarios,
                    )
                end

                @assert isapprox(sum(probabilities), 1.0; atol=1e-10)

                df_stochastic_scenario_CC = DataFrame(
                    scenario=selected_scenarios_ids,
                    probability=probabilities,
                )

                stochpath = joinpath(input_data_path_CC, "N$(number_of_scenarios)_seed$(seed)", case_name)
                mkpath(stochpath)

                CSV.write(
                    joinpath(stochpath, "stochastic-scenario.csv"),
                    df_stochastic_scenario_CC;
                    writeheader=true,
                )

                CSV.write(
                    joinpath(input_data_path_CC, "stochastic-scenario.csv"),
                    df_stochastic_scenario_CC;
                    writeheader=true,
                )

                profiles_df_CC = filter(
                    row -> row.scenario in selected_scenarios_ids,
                    profiles_df,
                )

                CSV.write(
                    joinpath(stochpath, "profiles-wide.csv"),
                    profiles_df_CC;
                    writeheader=true,
                )

                CSV.write(
                    joinpath(input_data_path_CC, "profiles-wide.csv"),
                    profiles_df_CC;
                    writeheader=true,
                )

                case_name = "0_HourlyBaseline_reduced_scenario_set"

                @info "Processing case study: $case_name"

                connection = DuckDB.DBInterface.connect(DuckDB.DB)
                TIO.read_csv_folder(connection, input_data_path_CC)
                DuckDB.query(
                    connection,
                    "
                    UPDATE model_parameters -- tables are with underscore in DuckDB world
                    SET
                        risk_aversion_weight_lambda = $(lambda) ,
                        risk_aversion_confidence_level_alpha = $(alpha);
                    ",
                )
                # to use the ratio availability/demand
                if use_ratio == true # be careful: this works now that we have only one demand location, so we divide each availability and inflow by that only demand
                    DuckDB.query(
                        connection,
                        "
                        UPDATE profiles_wide
                        SET
                            solar = solar / demand,
                            wind_offshore = wind_offshore / demand,
                            wind_onshore = wind_onshore / demand,
                            hydro_inflow = hydro_inflow / demand;
                        ",
                    )
                end

                # transform the profiles data from wide to long
                TC.transform_wide_to_long!(
                    connection,
                    "profiles_wide",
                    "profiles";
                    exclude_columns=["scenario", "milestone_year", "timestep"],
                )

                layout = TC.ProfilesTableLayout(;
                    year=:milestone_year,
                    cols_to_groupby=[:milestone_year, :scenario],
                )

                time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)

                TEM.populate_with_defaults!(connection)
                DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")
                time_to_read = @elapsed energy_problem_red = TEM.EnergyProblem(connection)

                @info "Creating the model for the case study: $case_name"
                time_to_create = @elapsed TEM.create_model!(
                    energy_problem_red;
                    optimizer=optimizer,
                    optimizer_parameters=parameters,
                    model_file_name="",
                    enable_names=enable_names,
                )

                output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, string(solver))
                mkpath(output_folder)

                @info "Solving the model and saving the solution for the case study: $case_name with $solver"
                time_to_solve = @elapsed TEM.solve_model!(energy_problem_red)
                time_to_save = @elapsed TEM.save_solution!(energy_problem_red)
                TEM.export_solution_to_csv_files(output_folder, energy_problem_red)

                output_file = joinpath(output_folder, "rep_periods_mapping.csv")
                DuckDB.execute(connection, "COPY rep_periods_mapping TO '$output_file' (HEADER, DELIMITER ',')")


                CC_investment_df = TIO.get_table(connection, "var_assets_investment")
                investment_output_folder_benchmark = joinpath(
                    @__DIR__,
                    "outputs",
                    "N$(number_of_scenarios)_seed$(seed)",
                    case_name,
                    "investment_analysis",
                    "benchmark",
                )
                mkpath(investment_output_folder_benchmark)

                mu_value_df = TIO.get_table(connection, "var_value_at_risk_threshold_mu")
                mu_value_red = if nrow(mu_value_df) == 0
                    NaN
                else
                    only(mu_value_df.solution)
                end

                if !isnan(mu_value_red)
                    @info "mu_value of CC ($rp periods per scenario on reduced scenario set) is defined"
                    @show mu_value_red
                end

                df_cost_per_scenario = export_total_operational_cost_per_scenario(energy_problem_red, output_folder)
                plot_cost_per_scenario(df_cost_per_scenario, output_folder, mu_value_df)

                var_flow_df = TIO.get_table(connection, "var_flow")
                water_borrowed = filter(
                    row ->
                        row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                    var_flow_df,
                )
                amount_water_borrowed_err = sum(water_borrowed.solution)
                if amount_water_borrowed_err > 0.0
                    error("Borrowed water has been used: $amount_water_borrowed_err")
                end

                time_to_resolve_benchmark = 0.0
                objective_value_resolve_benchmark = 0.0
                termination_status_resolve_benchmark = ""
                n_lol_ens_resolve_benchmark = 0
                lole_e_demand_resolve_benchmark = 0.0
                n_lol_smr_ccs_resolve_benchmark = 0
                lole_h2_demand_resolve_benchmark = 0.0
                amount_water_borrowed_resolve_benchmark = 0.0
                mu_value_resolve_benchmark = 0.0

                outlier_found = false

                if fix_benchmark
                    @info "Fixing investment from reduced scenario set in the benchmark case study: $rp RP on full set with $solver"
                    fix_variables_from_solution!(
                        energy_problem_benchmark,
                        energy_problem_red,
                        :assets_investment,
                    )
                    fix_variables_from_solution!(
                        energy_problem_benchmark,
                        energy_problem_red,
                        :assets_investment_energy,
                    )

                    # to fix also level of the seasonal storage
                    if fix_level_storage
                        df_profiles = TIO.get_table(connection, "profiles")
                        scenarios = unique(df_profiles.scenario)
                        scenario_to_rep_period_map = Dict(i => val for (i, val) in enumerate(scenarios))
                        fix_storage_levels!(
                            energy_problem_benchmark,
                            energy_problem_red,
                            scenario_to_rep_period_map,
                            period_duration,
                            "hydro_reservoir",
                        )
                        fix_storage_levels!(
                            energy_problem_benchmark,
                            energy_problem_red,
                            scenario_to_rep_period_map,
                            period_duration,
                            "h2_storage",
                        )
                    end

                    @info "Resolving the benchmark case study: RP on full set with $solver"
                    time_to_resolve_benchmark = @elapsed TEM.solve_model!(energy_problem_benchmark)
                    termination_status_resolve_benchmark = string(energy_problem_benchmark.termination_status)
                    objective_value_resolve_benchmark = energy_problem_benchmark.objective_value

                    if energy_problem_benchmark.termination_status == JuMP.INFEASIBLE
                        JuMP.compute_conflict!(energy_problem_benchmark.model)
                        iis_model, reference_map = JuMP.copy_conflict(energy_problem_benchmark.model)
                        print(iis_model)
                    end

                    TEM.save_solution!(energy_problem_benchmark)
                    var_flow_df = TIO.get_table(connection_benchmark, "var_flow")
                    flow_ens = filter(
                        row -> row.from_asset == "ens" && row.to_asset == "e_demand",
                        var_flow_df,
                    )
                    flow_smr_ccs = filter(
                        row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand",
                        var_flow_df,
                    )
                    water_borrowed = filter(
                        row ->
                            row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                        var_flow_df,
                    )

                    n_lol_ens_resolve_benchmark = count(row -> row.solution > 0.0, eachrow(flow_ens))
                    lole_e_demand_resolve_benchmark = n_lol_ens_resolve_benchmark / number_of_scenarios
                    n_lol_smr_ccs_resolve_benchmark = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
                    lole_h2_demand_resolve_benchmark = n_lol_smr_ccs_resolve_benchmark / number_of_scenarios
                    amount_water_borrowed_resolve_benchmark = sum(water_borrowed.solution)

                    mu_value_df = TIO.get_table(connection_benchmark, "var_value_at_risk_threshold_mu")
                    mu_value_resolve_benchmark = if nrow(mu_value_df) == 0
                        NaN
                    else
                        only(mu_value_df.solution)
                    end

                    if !isnan(mu_value_resolve_benchmark)
                        @info "mu_value of Resolve Benchmark (24 periods per scenario on full scenario set) is defined"
                        @show mu_value_resolve_benchmark
                    end

                    plot_normalized_asset_investment_differences(
                        benchmark_investment_df,
                        CC_investment_df;
                        output_folder=investment_output_folder_benchmark,
                        case_name=case_name,
                        benchmark_num_loss_of_load_e_demand=bm_n_lol_ens,
                        benchmark_num_loss_of_load_h2_demand=bm_n_lol_smr_ccs,
                        approximation_num_loss_of_load_e_demand=
                        n_lol_ens_resolve_benchmark,
                        approximation_num_loss_of_load_h2_demand=
                        n_lol_smr_ccs_resolve_benchmark,
                    )

                    output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, "fixed_in_benchmark", string(solver))
                    mkpath(output_folder)
                    TEM.export_solution_to_csv_files(output_folder, energy_problem_benchmark)

                    output_file = joinpath(output_folder, "rep_periods_mapping.csv")
                    DuckDB.execute(connection_benchmark, "COPY rep_periods_mapping TO '$output_file' (HEADER, DELIMITER ',')")

                    resolve_cost_df = export_total_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
                    plot_cost_per_scenario(resolve_cost_df, output_folder, mu_value_df)

                    comparison = innerjoin(
                        benchmark_cost_df,
                        resolve_cost_df;
                        on=:scenario,
                        renamecols="_benchmark" => "_resolve",
                    )
                    @info "showing comparison"
                    @show comparison
                    outlier_df = filter(
                        row -> row.total_cost_resolve > 2.0 * row.total_cost_benchmark,
                        comparison,
                    )

                    outlier_ids = unique(outlier_df.scenario)

                    if !isempty(outlier_df)

                        new_results_row = (
                            case_name=case_name,
                            rp=rp,
                            solver=solver,
                            time_to_cluster=time_to_cluster,
                            time_to_read=time_to_read,
                            time_to_create=time_to_create,
                            time_to_solve=time_to_solve,
                            time_to_save=time_to_save,
                            objective_value=energy_problem_red.objective_value,
                            termination_status=string(energy_problem_red.termination_status),
                            value_at_risk_threshold_mu_red=mu_value_red,
                            num_constraints=JuMP.num_constraints(
                                energy_problem_red.model;
                                count_variable_in_set_constraints=false,
                            ),
                            num_variables=JuMP.num_variables(energy_problem_red.model),
                            time_to_resolve_benchmark=time_to_resolve_benchmark,
                            objective_value_resolve_benchmark=objective_value_resolve_benchmark,
                            termination_status_resolve_benchmark=termination_status_resolve_benchmark,
                            num_loss_of_load_e_demand_benchmark=n_lol_ens_resolve_benchmark,
                            lole_e_demand_benchmark=lole_e_demand_resolve_benchmark,
                            num_loss_of_load_h2_demand_benchmark=n_lol_smr_ccs_resolve_benchmark,
                            lole_h2_demand_benchmark=lole_h2_demand_resolve_benchmark,
                            water_borrowed_benchmark=amount_water_borrowed_resolve_benchmark,
                            value_at_risk_threshold_mu_benchmark=mu_value_resolve_benchmark,
                            time_to_resolve_baseline=0.0,
                            objective_value_resolve_baseline=0.0,
                            termination_status_resolve_baseline="",
                            num_loss_of_load_e_demand_baseline=0,
                            lole_e_demand_baseline=0.0,
                            num_loss_of_load_h2_demand_baseline=0,
                            lole_h2_demand_baseline=0.0,
                            water_borrowed_baseline=0.0,
                            value_at_risk_threshold_mu_baseline=0.0,
                            scenario_set="reduced",
                            seed=seed,
                            number_of_scenarios=number_of_scenarios,
                        )
                        push!(results_df, new_results_row)

                        case_name = base_name * "_rp_" * "$rp" * "_reduced_scenario_set_inc_outlier"
                        outlier_found = true

                        time_to_resolve_benchmark = 0.0
                        objective_value_resolve_benchmark = 0.0
                        termination_status_resolve_benchmark = ""
                        n_lol_ens_resolve_benchmark = 0
                        lole_e_demand_resolve_benchmark = 0.0
                        n_lol_smr_ccs_resolve_benchmark = 0
                        lole_h2_demand_resolve_benchmark = 0.0
                        amount_water_borrowed_resolve_benchmark = 0.0
                        mu_value_resolve_benchmark = 0.0

                        output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, "scenario_selection", string(solver))
                        mkpath(output_folder)

                        # extra =
                        #     filter(
                        #         row ->
                        #             row.scenario in outlier_ids &&
                        #                 !(row.scenario in df_tail_scenarios.scenario),
                        #         benchmark_cost_df,
                        #     )
                        # extra = select(extra, names(df_tail_scenarios))
                        # append!(df_tail_scenarios, extra)

                        # CSV.write(
                        #     joinpath(output_folder, "tail_scenarios.csv"),
                        #     df_tail_scenarios;
                        #     writeheader=true,
                        # )

                        profiles_df = CSV.read(
                            joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-input-data", "NL-case-study", "profiles-wide.csv"),
                            DataFrame,
                        )

                        tail_scenarios_ids = unique(df_tail_scenarios.scenario)

                        expected_cost = sum(benchmark_cost_df.probability .* benchmark_cost_df.total_cost)
                        expected_idx = argmin(abs.(benchmark_cost_df.total_cost .- expected_cost))
                        average_case_row = benchmark_cost_df[expected_idx, :]

                        df_expected_cost_scenario = DataFrame(
                            scenario=[average_case_row.scenario],
                            total_cost=[average_case_row.total_cost],
                        )
                        expected_cost_scenario = only(df_expected_cost_scenario.scenario)
                        CSV.write(joinpath(@__DIR__, output_folder, "expected_cost_scenario.csv"), df_expected_cost_scenario; writeheader=true)

                        # plot_cost_per_scenario_inc_tail_inc_representative(
                        #     benchmark_cost_df,
                        #     df_tail_scenarios,
                        #     df_expected_cost_scenario,
                        #     output_folder,
                        #     mu_value_benchmark,
                        #     "$case_name",
                        # )

                        new_outlier_ids = [
                            s for s in outlier_ids
                            if !(s in tail_scenarios_ids) &&
                            s != expected_cost_scenario &&
                            !(s in mu_scenario_id)
                        ]

                        new_outlier_df = filter(
                            row -> row.scenario in new_outlier_ids,
                            outlier_df,
                        )

                        new_outlier_df = select(
                            new_outlier_df,
                            :scenario,
                            :total_cost_benchmark => :total_cost,
                        )

                        n_new_outliers = length(new_outlier_ids)

                        explicit_ids = unique(vcat(
                            tail_scenarios_ids,
                            mu_scenario_id,
                            new_outlier_ids,
                        ))

                        plot_cost_per_scenario_inc_tail_inc_representative_inc_outliers(
                            benchmark_cost_df,
                            df_tail_scenarios,
                            df_expected_cost_scenario,
                            new_outlier_df,
                            output_folder,
                            df_mu_scenario,
                            var_empirical,
                            "$case_name",
                        )

                        n_tail = length(tail_scenarios_ids)
                        #tail_probability = (1.0 - alpha) / n_tail
                        tail_probability = 1 / number_of_scenarios

                        expected_cost_scenario = only(df_expected_cost_scenario.scenario)

                        if expected_cost_scenario in explicit_ids
                            selected_scenarios_ids = vcat(explicit_ids)
                            probabilities = [
                                scenario == expected_cost_scenario ?
                                1 - (length(explicit_ids) - 1) / number_of_scenarios :
                                1 / number_of_scenarios for scenario in selected_scenarios_ids
                            ]
                            #append!(probabilities, fill(1.0 / number_of_scenarios, length(explicit_ids) - 1),)
                        else
                            selected_scenarios_ids = vcat(
                                explicit_ids,
                                expected_cost_scenario,
                            )
                            probabilities = vcat(
                                fill(1 / number_of_scenarios, length(explicit_ids)),
                                1 - length(explicit_ids) / number_of_scenarios,
                            )
                        end

                        @assert isapprox(sum(probabilities), 1.0; atol=1e-10)

                        df_stochastic_scenario_CC = DataFrame(
                            scenario=selected_scenarios_ids,
                            probability=probabilities,
                        )

                        profiles_df_CC = filter(
                            row -> row.scenario in selected_scenarios_ids,
                            profiles_df,
                        )

                        stochpath_twice = joinpath(input_data_path_CC, "N$(number_of_scenarios)_seed$(seed)", case_name, "twice")
                        mkpath(stochpath_twice)

                        CSV.write(
                            joinpath(stochpath_twice, "profiles-wide.csv"),
                            profiles_df_CC;
                            writeheader=true,
                        )

                        CSV.write(
                            joinpath(input_data_path_CC, "twice", "profiles-wide.csv"),
                            profiles_df_CC;
                            writeheader=true,
                        )

                        CSV.write(
                            joinpath(stochpath_twice, "stochastic-scenario.csv"),
                            df_stochastic_scenario_CC;
                            writeheader=true,
                        )

                        CSV.write(
                            joinpath(input_data_path_CC, "twice", "stochastic-scenario.csv"),
                            df_stochastic_scenario_CC;
                            writeheader=true,
                        )

                        mkpath(joinpath(output_folder, "input_profiles_inc_outlier"))
                        CSV.write(
                            joinpath(output_folder, "input_profiles_inc_outlier", "profiles-wide.csv"),
                            profiles_df_CC;
                            writeheader=true,
                        )

                        @info "Processing case study: $case_name"

                        connection = DuckDB.DBInterface.connect(DuckDB.DB)
                        TIO.read_csv_folder(connection, joinpath(input_data_path_CC, "twice"))

                        DuckDB.query(
                            connection,
                            "
                            UPDATE model_parameters -- tables are with underscore in DuckDB world
                            SET
                                risk_aversion_weight_lambda = $(lambda) ,
                                risk_aversion_confidence_level_alpha = $(alpha);
                            ",
                        )
                        # to use the ratio availability/demand
                        if use_ratio == true # be careful: this works now that we have only one demand location, so we divide each availability and inflow by that only demand
                            DuckDB.query(
                                connection,
                                "
                                UPDATE profiles_wide
                                SET
                                    solar = solar / demand,
                                    wind_offshore = wind_offshore / demand,
                                    wind_onshore = wind_onshore / demand,
                                    hydro_inflow = hydro_inflow / demand;
                                ",
                            )
                        end

                        # transform the profiles data from wide to long
                        TC.transform_wide_to_long!(
                            connection,
                            "profiles_wide",
                            "profiles";
                            exclude_columns=["scenario", "milestone_year", "timestep"],
                        )

                        layout = TC.ProfilesTableLayout(;
                            year=:milestone_year,
                            cols_to_groupby=[:milestone_year, :scenario],
                        )
                        time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)

                        TEM.populate_with_defaults!(connection)
                        DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")
                        time_to_read = @elapsed energy_problem_red = TEM.EnergyProblem(connection)

                        @info "Creating the model for the case study: $case_name"
                        time_to_create = @elapsed TEM.create_model!(
                            energy_problem_red;
                            optimizer=optimizer,
                            optimizer_parameters=parameters,
                            model_file_name="",
                            enable_names=enable_names,
                        )

                        output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, string(solver))
                        mkpath(output_folder)

                        @info "Solving the model and saving the solution for the case study: $case_name with $solver"
                        time_to_solve = @elapsed TEM.solve_model!(energy_problem_red)
                        time_to_save = @elapsed TEM.save_solution!(energy_problem_red)
                        TEM.export_solution_to_csv_files(output_folder, energy_problem_red)

                        output_file = joinpath(output_folder, "rep_periods_mapping.csv")
                        DuckDB.execute(connection, "COPY rep_periods_mapping TO '$output_file' (HEADER, DELIMITER ',')")


                        CC_investment_df = TIO.get_table(connection, "var_assets_investment")
                        investment_output_folder_benchmark = joinpath(
                            @__DIR__,
                            "outputs",
                            "N$(number_of_scenarios)_seed$(seed)",
                            case_name,
                            "investment_analysis",
                            "benchmark",
                        )
                        mkpath(investment_output_folder_benchmark)

                        mu_value_df = TIO.get_table(connection, "var_value_at_risk_threshold_mu")
                        mu_value_red = if nrow(mu_value_df) == 0
                            NaN
                        else
                            only(mu_value_df.solution)
                        end

                        if !isnan(mu_value_red)
                            @info "mu_value of CC ($rp periods per scenario on reduced scenario set) is defined"
                            @show mu_value_red
                        end

                        df_cost_per_scenario = export_total_operational_cost_per_scenario(energy_problem_red, output_folder)
                        plot_cost_per_scenario(df_cost_per_scenario, output_folder, mu_value_df)

                        var_flow_df = TIO.get_table(connection, "var_flow")
                        water_borrowed = filter(
                            row ->
                                row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                            var_flow_df,
                        )
                        amount_water_borrowed_err = sum(water_borrowed.solution)
                        if amount_water_borrowed_err > 0.0
                            error("Borrowed water has been used: $amount_water_borrowed_err")
                        end
                    end
                end

                @info outlier_found ?
                      "Fixing variables in the baseline: reduced scenario set including outliers from resolving benchmark solved hourly with $solver" :
                      "Fixing variables in the baseline: reduced scenario set solved hourly with $solver"

                fix_variables_from_solution!(
                    energy_problem_baseline,
                    energy_problem_red,
                    :assets_investment,
                )
                fix_variables_from_solution!(
                    energy_problem_baseline,
                    energy_problem_red,
                    :assets_investment_energy,
                )

                # to fix also level of the seasonal storage
                if fix_level_storage
                    df_profiles = TIO.get_table(connection, "profiles")
                    scenarios = unique(df_profiles.scenario)
                    scenario_to_rep_period_map = Dict(i => val for (i, val) in enumerate(scenarios))
                    fix_storage_levels!(
                        energy_problem_baseline,
                        energy_problem_red,
                        scenario_to_rep_period_map,
                        period_duration,
                        "hydro_reservoir",
                    )
                    fix_storage_levels!(
                        energy_problem_baseline,
                        energy_problem_red,
                        scenario_to_rep_period_map,
                        period_duration,
                        "h2_storage",
                    )
                end

                @info "Resolving the hourly baseline: hourly on full set with $solver"
                time_to_resolve_baseline = @elapsed TEM.solve_model!(energy_problem_baseline)

                if energy_problem_baseline.termination_status == JuMP.INFEASIBLE
                    JuMP.compute_conflict!(energy_problem_baseline.model)
                    iis_model, reference_map = JuMP.copy_conflict(energy_problem_baseline.model)
                    print(iis_model)
                end

                TEM.save_solution!(energy_problem_baseline)
                var_flow_df = TIO.get_table(connection_baseline, "var_flow")
                flow_ens = filter(
                    row -> row.from_asset == "ens" && row.to_asset == "e_demand",
                    var_flow_df,
                )
                flow_smr_ccs = filter(
                    row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand",
                    var_flow_df,
                )
                water_borrowed = filter(
                    row ->
                        row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                    var_flow_df,
                )

                # count steps with loss of load
                n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
                lole_e_demand = n_lol_ens / number_of_scenarios
                n_lol_smr_ccs = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
                lole_h2_demand = n_lol_smr_ccs / number_of_scenarios

                investment_output_folder_baseline = joinpath(
                    @__DIR__,
                    "outputs",
                    "N$(number_of_scenarios)_seed$(seed)",
                    case_name,
                    "investment_analysis",
                    "baseline",
                )
                mkpath(investment_output_folder_baseline)

                plot_normalized_asset_investment_differences(
                    baseline_investment_df,
                    CC_investment_df;
                    output_folder=investment_output_folder_baseline,
                    case_name=case_name,
                    benchmark_num_loss_of_load_e_demand=baseline_n_lol_ens,
                    benchmark_num_loss_of_load_h2_demand=baseline_n_lol_smr_ccs,
                    approximation_num_loss_of_load_e_demand=n_lol_ens,
                    approximation_num_loss_of_load_h2_demand=n_lol_smr_ccs,
                )
                # count how much water_borrowed
                amount_water_borrowed_b = sum(water_borrowed.solution)

                # get mu solution
                mu_value_df = TIO.get_table(connection_baseline, "var_value_at_risk_threshold_mu")
                mu_value_baseline = if nrow(mu_value_df) == 0
                    NaN
                else
                    only(mu_value_df.solution)
                end

                if !isnan(mu_value_baseline)
                    @info "mu_value of Resolve Baseline (hourly on full scenario set) is defined"
                    @show mu_value_baseline
                end

                output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)", case_name, "fixed_in_baseline", string(solver))
                mkpath(output_folder)
                TEM.export_solution_to_csv_files(output_folder, energy_problem_baseline)

                output_file = joinpath(output_folder, "rep_periods_mapping.csv")
                DuckDB.execute(connection_baseline, "COPY rep_periods_mapping TO '$output_file' (HEADER, DELIMITER ',')")


                df_cost_per_scenario = export_total_operational_cost_per_scenario(energy_problem_baseline, output_folder)
                plot_cost_per_scenario(df_cost_per_scenario, output_folder, mu_value_df)

                new_results_row = (
                    case_name=case_name,
                    rp=rp,
                    solver=solver,
                    time_to_cluster=time_to_cluster,
                    time_to_read=time_to_read,
                    time_to_create=time_to_create,
                    time_to_solve=time_to_solve,
                    time_to_save=time_to_save,
                    objective_value=energy_problem_red.objective_value,
                    termination_status=string(energy_problem_red.termination_status),
                    value_at_risk_threshold_mu_red=mu_value_red,
                    num_constraints=JuMP.num_constraints(
                        energy_problem_red.model;
                        count_variable_in_set_constraints=false,
                    ),
                    num_variables=JuMP.num_variables(energy_problem_red.model),
                    time_to_resolve_benchmark=time_to_resolve_benchmark,
                    objective_value_resolve_benchmark=objective_value_resolve_benchmark,
                    termination_status_resolve_benchmark=termination_status_resolve_benchmark,
                    num_loss_of_load_e_demand_benchmark=n_lol_ens_resolve_benchmark,
                    lole_e_demand_benchmark=lole_e_demand_resolve_benchmark,
                    num_loss_of_load_h2_demand_benchmark=n_lol_smr_ccs_resolve_benchmark,
                    lole_h2_demand_benchmark=lole_h2_demand_resolve_benchmark,
                    water_borrowed_benchmark=amount_water_borrowed_resolve_benchmark,
                    value_at_risk_threshold_mu_benchmark=mu_value_resolve_benchmark,
                    time_to_resolve_baseline=time_to_resolve_baseline,
                    objective_value_resolve_baseline=energy_problem_baseline.objective_value,
                    termination_status_resolve_baseline=string(
                        energy_problem_baseline.termination_status,
                    ),
                    num_loss_of_load_e_demand_baseline=n_lol_ens,
                    lole_e_demand_baseline=lole_e_demand,
                    num_loss_of_load_h2_demand_baseline=n_lol_smr_ccs,
                    lole_h2_demand_baseline=lole_h2_demand,
                    water_borrowed_baseline=amount_water_borrowed_b,
                    value_at_risk_threshold_mu_baseline=mu_value_baseline,
                    scenario_set="reduced",
                    seed=seed,
                    number_of_scenarios=number_of_scenarios,
                )
                push!(results_df, new_results_row)
            end
        end
    end

    output_folder = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N$(number_of_scenarios)_seed$(seed)")
    results_df |> CSV.write(joinpath(output_folder, "results_ScSeRP_N$(number_of_scenarios)_seed$(seed).csv"); writeheader=true)

    return nothing
end

main()
