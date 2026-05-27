# Copyright (c) 2025: Diego Tejada and contributors
#
# Use of this source code is governed by an Apache 2.0 license that can be found
# in the LICENSE.md file or at https://opensource.org/license/apache-2-0.

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev="227a80f7907e2c7178edb0697874cfb6666ad644")

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
using JuMP: JuMP, Model, @variable, @constraint, @objective
using TOML: TOML
using Plots
using Random
using DataFrames

Random.seed!(19990907)

using DataFrames

# helper functions
@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")
include("utils/ipdsr_engine.jl")

distance_map = Dict(
    :Euclidean => Distances.Euclidean(),
    :SqEuclidean => Distances.SqEuclidean(),
    :CosineDist => Distances.CosineDist(),
    :Cityblock => Distances.Cityblock(),
    :Chebyshev => Distances.Chebyshev(),
)

# Read and transform user input files to Tulipa input files
config = TOML.parsefile("config.toml")
input_data_path = config["simulation"]["input_data"]
use_ratio = config["clustering"]["use_ratio"]
heuristic_distance = config["clustering"]["heuristic_distance"]
fix_level_storage = config["simulation"]["fix_level_storage"]
representative_periods = config["simulation"]["representative_periods"]
solvers = [Symbol(el) for el in config["simulation"]["solvers"]]
lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]
run_benchmark = config["simulation"]["run_benchmark"]

profiles_path = joinpath(@__DIR__, "create-scenarios", "profiles-wide-all-scenarios.csv")
all_profiles_df = CSV.read(profiles_path, DataFrame)
profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
selected_scenarios = sort(unique(profiles_df.scenario))
mapping = Dict(old => new for (new, old) in enumerate(selected_scenarios))
profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]
CSV.write(joinpath(input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

df_stochastic_scenario = DataFrame(;
    scenario=sort(unique(profiles_df.scenario)),
    probability=fill(1.0 / number_of_scenarios, number_of_scenarios),
)
CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

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
    base_name=String[],
    rp=Int[],
    solver=Symbol[],
    time_to_cluster=Float64[],
    time_to_read=Float64[],
    time_to_create=Float64[],
    time_to_solve=Float64[],
    time_to_save=Float64[],
    objective_value=Float64[],
    termination_status=String[],
    num_constraints=Int[],
    num_variables=Int[],
    time_to_resolve_benchmark=Float64[],
    objective_value_resolve_benchmark=Float64[],
    termination_status_resolve_benchmark=String[],
    num_loss_of_load_e_demand=Int[],
    num_loss_of_load_h2_demand=Int[],
    water_borrowed=Float64[],
    value_at_risk_threshold_mu=Float64[],
)

function main()
    # optimize for the base case study (0_HourlyBenchmark)
    # set up the connection and read the data
    connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection_benchmark, input_data_path)
    profiles_wide = TIO.get_table(connection_benchmark, "profiles_wide")
    n_scenarios = length(unique(profiles_wide.scenario))
    # To make number of rps comparable with per and cross scenario
    # we consider the case that n_rps is not divisible by the number of scenarios
    #representative_periods .= n_scenarios .* round.(Int, representative_periods ./ n_scenarios)

    if run_benchmark
        @info "Running the base case study (0_HourlyBenchmark)"
        base_name = "0_HourlyBenchmark"

        # set up the connection and read the data
        connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
        TIO.read_csv_folder(connection_benchmark, input_data_path)
        # update the CSV input data for Tulipa from the config file info
        DuckDB.query(
            connection_benchmark,
            "
            UPDATE model_parameters -- tables are with underscore in DuckDB world
            SET
                risk_aversion_weight_lambda = $(lambda) ,
                risk_aversion_confidence_level_alpha = $(alpha);
            ",
        )
        # --- NEW: Ensure Relatively Complete Recourse ---
        DuckDB.query(
            connection_benchmark,
            "UPDATE asset SET capacity = 1000000 WHERE asset IN ('ens', 'smr_ccs');"
        )
        # ------------------------------------------------

        # transform the profiles data from wide to long
        TC.transform_wide_to_long!(
            connection_benchmark,
            "profiles_wide",
            "profiles";
            exclude_columns=["scenario", "milestone_year", "timestep"],
        )

        layout = TC.ProfilesTableLayout(;
            year=:milestone_year,
            cols_to_groupby=[:milestone_year, :scenario],
        )
        time_to_cluster = @elapsed TC.dummy_cluster!(connection_benchmark; layout=layout)
        TEM.populate_with_defaults!(connection_benchmark)
        DuckDB.query(connection_benchmark, "UPDATE asset SET is_seasonal = false")

        time_to_read = @elapsed energy_problem_benchmark = TEM.EnergyProblem(connection_benchmark)

        for solver in solvers
            optimizer, parameters = get_solver_parameters(solver)

            @info "Creating the model for the base case study (0_HourlyBenchmark) with $solver"
            time_to_create = @elapsed TEM.create_model!(
                energy_problem_benchmark;
                optimizer=optimizer,
                optimizer_parameters=parameters,
                model_file_name="",
                enable_names=enable_names,
                direct_model=direct_model,
            )

            output_folder = joinpath(@__DIR__, "outputs", base_name, string(solver))
            mkpath(output_folder)

            @info "Solving the model and saving the solution for the base case study (0_HourlyBenchmark) with $solver"
            time_to_solve = @elapsed TEM.solve_model!(energy_problem_benchmark)
            #        mu_value =
            #            JuMP.value(energy_problem_benchmark.variables[:value_at_risk_threshold_mu].container)
            time_to_save = @elapsed TEM.save_solution!(energy_problem_benchmark)
            TEM.export_solution_to_csv_files(output_folder, energy_problem_benchmark)


            df_cost_per_scenario = export_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
            plot_operational_cost_per_scenario(df_cost_per_scenario, output_folder)


            mu_value_df = TIO.get_table(connection_benchmark, "var_value_at_risk_threshold_mu")
            mu_value = only(mu_value_df.solution)
            var_flow_df = TIO.get_table(connection_benchmark, "var_flow")
            flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
            flow_smr_ccs =
                filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
            water_borrowed = filter(
                row -> row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                var_flow_df,
            )

            # count steps with loss of load
            n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
            n_lol_smr_cca = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))

            # count how much water_borrowed
            amount_water_borrowed_b = sum(water_borrowed.solution)

            new_results_row = (
                base_name=base_name,
                rp=1,
                solver=solver,
                time_to_cluster=0.0,
                time_to_read=time_to_read,
                time_to_create=time_to_create,
                time_to_solve=time_to_solve,
                time_to_save=time_to_save,
                objective_value=energy_problem_benchmark.objective_value,
                termination_status=string(energy_problem_benchmark.termination_status),
                num_constraints=JuMP.num_constraints(
                    energy_problem_benchmark.model;
                    count_variable_in_set_constraints=false,
                ),
                num_variables=JuMP.num_variables(energy_problem_benchmark.model),
                time_to_resolve_benchmark=0.0,
                objective_value_resolve_benchmark=0.0,
                termination_status_resolve_benchmark="",
                num_loss_of_load_e_demand=n_lol_ens,
                num_loss_of_load_h2_demand=n_lol_smr_cca,
                water_borrowed=amount_water_borrowed_b,
                value_at_risk_threshold_mu=mu_value,
            )
            push!(results_df, new_results_row)
        end
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
            case_name = base_name * "_rp_" * "$rp"

            @info "Processing case study: $case_name"

            iter = 0
            scenario_file = joinpath(input_data_path, "stochastic-scenario.csv")
            last_selected_scenarios = Any[]
            last_weights = Float64[]

            while iter < IPDSR_MAX_ITER
                iter += 1
                @info "=== IPDSR Iteration $iter for rp=$rp ==="
                ipdsr_converged = false

                connection = DuckDB.DBInterface.connect(DuckDB.DB)
                TIO.read_csv_folder(connection, input_data_path)

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

                if stochastic_method == :per_scenario
                    layout = TC.ProfilesTableLayout(;
                        year=:milestone_year,
                        cols_to_groupby=[:milestone_year, :scenario],
                    )
                    time_to_cluster = @elapsed TC.cluster!(
                        connection,
                        period_duration,
                        rp; #round(Int, rp / n_scenarios);
                        method=method,
                        distance=distance,
                        weight_type=weight_type,
                        layout=layout,
                        clustering_kwargs,
                        weight_fitting_kwargs,
                    )
                    if use_ratio == true
                        DuckDB.query(
                            connection,
                            "UPDATE profiles_rep_periods AS x
                                SET value =
                                    CASE
                                        WHEN x.profile_name = 'demand' THEN x.value
                                        ELSE x.value * d.value
                                    END
                                FROM profiles_rep_periods AS d
                                WHERE d.timestep   = x.timestep
                                AND d.rep_period       = x.rep_period
                                AND d.milestone_year       = x.milestone_year
                                AND d.scenario   = x.scenario
                                AND d.profile_name = 'demand';
                                    ",
                        )
                    end

                elseif stochastic_method == :cross_scenario
                    layout = TC.ProfilesTableLayout(;
                        year=:milestone_year,
                        cols_to_groupby=[:milestone_year],
                        cols_to_crossby=[:scenario],
                    )
                    time_to_cluster = @elapsed TC.cluster!(
                        connection,
                        period_duration,
                        rp;
                        method=method,
                        distance=distance,
                        weight_type=weight_type,
                        layout=layout,
                        clustering_kwargs,
                        weight_fitting_kwargs,
                    )
                    if use_ratio == true
                        DuckDB.query(
                            connection,
                            "UPDATE profiles_rep_periods AS x
                                SET value =
                                    CASE
                                        WHEN x.profile_name = 'demand' THEN x.value
                                        ELSE x.value * d.value
                                    END
                                FROM profiles_rep_periods AS d
                                WHERE d.timestep   = x.timestep
                                AND d.rep_period       = x.rep_period
                                AND d.milestone_year       = x.milestone_year
                                AND d.profile_name = 'demand';
                                    ",
                        )
                    end
                else
                    error("Unknown stochastic method: $stochastic_method")
                end
                if use_ratio == true
                    DuckDB.query(
                        connection,
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
                TEM.populate_with_defaults!(connection)

                time_to_read = @elapsed energy_problem = TEM.EnergyProblem(connection)

                for solver in solvers
                    optimizer, parameters = get_solver_parameters(solver)

                    @info "Creating the model for the case study: $case_name"
                    time_to_create = @elapsed TEM.create_model!(
                        energy_problem;
                        optimizer=optimizer,
                        optimizer_parameters=parameters,
                        model_file_name="",
                        enable_names=enable_names,
                    )

                    output_folder = joinpath(@__DIR__, "outputs", case_name, string(solver))
                    mkpath(output_folder)

                    @info "Solving the model and saving the solution for the case study: $case_name with $solver"
                    time_to_solve = @elapsed TEM.solve_model!(energy_problem)
                    time_to_save = @elapsed TEM.save_solution!(energy_problem)
                    TEM.export_solution_to_csv_files(output_folder, energy_problem)
                    df_cost_per_scenario = export_operational_cost_per_scenario(energy_problem, output_folder)

                    plot_operational_cost_per_scenario(df_cost_per_scenario, output_folder)

                    var_flow_df = TIO.get_table(connection, "var_flow")
                    water_borrowed = filter(
                        row ->
                            row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
                        var_flow_df,
                    )
                    amount_water_borrowed_err = sum(water_borrowed.solution)
                    if amount_water_borrowed_err > 0.0
                        error("Borrowed water has been used: $amount_water_borrowed")
                    end
                    mu_value_df = TIO.get_table(connection, "var_value_at_risk_threshold_mu")
                    mu_value = only(mu_value_df.solution)

                    if run_benchmark
                        @info "Fixing variables in the benchmark case study: $case_name with $solver"
                        fix_variables_from_solution!(
                            energy_problem_benchmark,
                            energy_problem,
                            :assets_investment,
                        )
                        fix_variables_from_solution!(
                            energy_problem_benchmark,
                            energy_problem,
                            :assets_investment_energy,
                        )

                        # to fix also level of the seasonal storage
                        if fix_level_storage
                            df_profiles = TIO.get_table(connection, "profiles")
                            scenarios = unique(df_profiles.scenario)
                            scenario_to_rep_period_map = Dict(i => val for (i, val) in enumerate(scenarios))
                            fix_storage_levels!(
                                energy_problem_benchmark,
                                energy_problem,
                                scenario_to_rep_period_map,
                                period_duration,
                                "hydro_reservoir",
                            )
                            fix_storage_levels!(
                                energy_problem_benchmark,
                                energy_problem,
                                scenario_to_rep_period_map,
                                period_duration,
                                "h2_storage",
                            )
                        end

                        @info "Resolving the benchmark case study: $case_name with $solver"
                        time_to_resolve_benchmark = @elapsed TEM.solve_model!(energy_problem_benchmark)
                        
                        # --- 1. IPDSR CONVERGENCE CHECK (OPTIMALITY GAP) ---
                        LB = energy_problem.objective_value
                        UB = energy_problem_benchmark.objective_value
                        OG = abs(UB - LB) / abs(UB)
                        
                        @info "--- IPDSR Iteration $iter Status ---"
                        @info "LB (Reduced Cost): $(round(LB, digits=2)) | UB (True Cost): $(round(UB, digits=2))"
                        @info "Current Optimality Gap (OG): $(round(OG * 100, digits=3))%"
                        
                        # --- 2. GATHER METRICS FOR RESULTS ---
                        var_flow_df = TIO.get_table(connection_benchmark, "var_flow")
                        flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
                        flow_smr_ccs = filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
                        water_borrowed = filter(row -> row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir", var_flow_df)
                        
                        n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
                        n_lol_smr_cca = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
                        amount_water_borrowed = sum(water_borrowed.solution)
                        
                        mu_value_df = TIO.get_table(connection_benchmark, "var_value_at_risk_threshold_mu")
                        mu_value = only(mu_value_df.solution)
                        
                        new_results_row = (
                            base_name=base_name, rp=rp, solver=solver,
                            time_to_cluster=time_to_cluster, time_to_read=time_to_read,
                            time_to_create=time_to_create, time_to_solve=time_to_solve,
                            time_to_save=time_to_save, objective_value=LB,
                            termination_status=string(energy_problem.termination_status),
                            num_constraints=JuMP.num_constraints(energy_problem.model; count_variable_in_set_constraints=false),
                            num_variables=JuMP.num_variables(energy_problem.model),
                            time_to_resolve_benchmark=time_to_resolve_benchmark,
                            objective_value_resolve_benchmark=UB,
                            termination_status_resolve_benchmark=string(energy_problem_benchmark.termination_status),
                            num_loss_of_load_e_demand=n_lol_ens, num_loss_of_load_h2_demand=n_lol_smr_cca,
                            water_borrowed=amount_water_borrowed, value_at_risk_threshold_mu=mu_value,
                        )

                        # --- 3. CONVERGENCE OR MAX ITERATIONS HANDLING ---
                        if OG <= IPDSR_MIP_GAP || iter == IPDSR_MAX_ITER
                            if OG <= IPDSR_MIP_GAP
                                @info "✅ IPDSR SUCCESSFULLY CONVERGED at Iteration $iter!"
                            else
                                @warn "⚠️ IPDSR hit Max Iterations ($IPDSR_MAX_ITER) without perfect convergence."
                            end
                            
                            TEM.save_solution!(energy_problem_benchmark)
                            output_folder_fixed = joinpath(@__DIR__, "outputs", "fixed", case_name, string(solver))
                            mkpath(output_folder_fixed)
                            TEM.export_solution_to_csv_files(output_folder_fixed, energy_problem_benchmark)
                            
                            push!(results_df, new_results_row)
                            break # Break the solver loop
                        end

                        # --- 4. EXTRACT PROBLEM SPACE & UPDATE WEIGHTS (If not converged) ---
                        @info "--- Extracting Problem Space for IPDSR Feedback ---"
                        df_costs = export_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
                        sort!(df_costs, :scenario)
                        
                        F_costs = df_costs.operational_cost
                        gamma = fill(1.0 / number_of_scenarios, number_of_scenarios)
                        
                        N_prime = min(length(F_costs), max(20, rp + 5)) 
                        F_agg, gamma_agg, original_mapping = aggregate_objectives(F_costs, gamma, N_prime)
                        
                        target_scenarios = N_TARGET_SCENARIOS
                        @info "Solving IPDSR MIP to reduce to K = $target_scenarios scenarios..."
                        selected_agg_idx, new_weights = solve_ipdsr_mip(F_agg, gamma_agg, target_scenarios, lambda, alpha)

                        if isempty(selected_agg_idx)
                            @warn "IPDSR failed to find scenarios. Saving current best and aborting."
                            push!(results_df, new_results_row)
                            break
                        end
                        
                        selected_original_scenarios = [df_costs.scenario[original_mapping[idx][1]] for idx in selected_agg_idx]
                        
                        # --- 5. FIXED-POINT CONVERGENCE CHECK ---
                        if selected_original_scenarios == last_selected_scenarios && length(new_weights) == length(last_weights) && isapprox(new_weights, last_weights, atol=1e-4)
                            @info "✅ FIXED-POINT CONVERGENCE ACHIEVED! Scenarios and weights stabilized at Iteration $iter."
                            TEM.save_solution!(energy_problem_benchmark)
                            output_folder_fixed = joinpath(@__DIR__, "outputs", "fixed", case_name, string(solver))
                            mkpath(output_folder_fixed)
                            TEM.export_solution_to_csv_files(output_folder_fixed, energy_problem_benchmark)
                            
                            push!(results_df, new_results_row)
                            ipdsr_converged = true
                            break
                        end
                        
                        last_selected_scenarios = copy(selected_original_scenarios)
                        last_weights = copy(new_weights)
                        # ----------------------------------------
                        
                        # Overwrite the DuckDB input file
                        scen_df = CSV.read(scenario_file, DataFrame)
                        scen_df.probability .= 0.0 
                        
                        for (i, scen_name) in enumerate(selected_original_scenarios)
                            idx = findfirst(==(scen_name), scen_df.scenario) 
                            if !isnothing(idx)
                                scen_df.probability[idx] = new_weights[i]
                            end
                        end
                        CSV.write(scenario_file, scen_df)
                        @info "Weights updated in stochastic-scenario.csv. Proceeding to next iteration..."
                        
                    else # If run_benchmark is false (Standard non-IPDSR run)
                        new_results_row = (
                            base_name=base_name, rp=rp, solver=solver,
                            time_to_cluster=time_to_cluster, time_to_read=time_to_read,
                            time_to_create=time_to_create, time_to_solve=time_to_solve,
                            time_to_save=time_to_save, objective_value=energy_problem.objective_value,
                            termination_status=string(energy_problem.termination_status),
                            num_constraints=JuMP.num_constraints(energy_problem.model; count_variable_in_set_constraints=false),
                            num_variables=JuMP.num_variables(energy_problem.model),
                            time_to_resolve_benchmark=0.0, objective_value_resolve_benchmark=0.0,
                            termination_status_resolve_benchmark="", num_loss_of_load_e_demand=0.0,
                            num_loss_of_load_h2_demand=0.0, water_borrowed=0.0, value_at_risk_threshold_mu=mu_value,
                        )
                        push!(results_df, new_results_row)
                    end
                end # This ends the 'for solver in solvers' loop
                
                # Check if we should break the outer while loop
                if ipdsr_converged || iter == IPDSR_MAX_ITER
                    break
                end
                
            end # This ends the 'while iter < IPDSR_MAX_ITER' loop
        end
    end

    results_df |> CSV.write("outputs/results.csv"; writeheader=true)

    return nothing
end

main()
