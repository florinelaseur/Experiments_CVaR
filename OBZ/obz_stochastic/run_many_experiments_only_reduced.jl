
cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
import DuckDB
import Distances
import Xpress
import CSV
import Statistics
import JuMP
import TOML
using Revise
using DataFrames
using Random
using JuMP

# helper functions
@info "Including helper functions"
include("utils/functions.jl")

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
heuristic_distance = config["clustering"]["heuristic_distance"]
n_runs = config["simulation"]["n_runs"]
add_worst_every_hour = config["extreme_periods"]["add_worst_every_hour"]
add_best_every_hour = config["extreme_periods"]["add_best_every_hour"]
add_worst_sum = config["extreme_periods"]["add_worst_sum"]
if add_best_every_hour && !add_worst_every_hour
    error("You can add best only when also worst is added")
end
one_for_all_scenarios = config["extreme_periods"]["one_for_all_scenarios"]


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

solvers = [:Xpress] #[:HiGHS, :Gurobi]
representative_periods = [10, 20, 40, 60, 80, 100, 150, 225, 300]
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
    reduced_num_lol_e=Int[],
    reduced_num_lol_h2=Int[],
    reduced_lol_e=Float64[],
    reduced_lol_h2=Float64[],
    num_constraints=Int[],
    num_variables=Int[]
)

function main()
    # set up the connection and read the data
    connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection_benchmark, input_data_path)

    # To make number of rps comparable with per and cross scenario
    # we consider the case that n_rps is not divisible by the number of scenarios
    profiles = TIO.get_table(connection_benchmark, "profiles")
    n_scenarios = length(unique(profiles.scenario))
    representative_periods .= n_scenarios .* round.(Int, representative_periods ./ n_scenarios)

    DuckDB.close(connection_benchmark)

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

        weight_fitting_kwargs = Dict(
            :learning_rate => learning_rate,
            :niters => niters
        )
        if method ∉ [:k_means, :k_medoids]
            clustering_kwargs = Dict(
                :learning_rate => learning_rate,
                :niters => niters,
                :heuristic_distance => heuristic_distance,
            )
        else
            clustering_kwargs = Dict{Symbol,Any}()
        end

        if !run_case
            continue
        end

        for rp in representative_periods
            case_name = base_name * "_rp_" * "$rp"

            @info "Processing case study: $case_name"

            for n in 1:n_runs
                Random.seed!(n)

                connection = DuckDB.DBInterface.connect(DuckDB.DB)
                TIO.read_csv_folder(connection, input_data_path)
                # let us add extreme periods 
                if add_worst_every_hour
                    df_profiles = TIO.get_table(connection, "profiles")
                    profiles = string.(unique(df_profiles.profile_name))
                    # profiles = filter(p -> !occursin("demand", lowercase(p)), profiles)
                    # println(length(profiles))
                    create_init_rps_hourly(connection, period_duration, profiles) # BE CAREFUL: tested only for 1 year
                    if add_best_every_hour
                        create_init_rps_hourly_best(connection, period_duration, profiles)
                    end
                end

                if add_worst_sum
                    df_profiles = TIO.get_table(connection, "profiles")
                    profiles = string.(unique(df_profiles.profile_name))
                    # profiles = filter(p -> !occursin("demand", lowercase(p)), profiles)
                    # println(length(profiles))
                    if !one_for_all_scenarios
                        create_init_rps_daily(connection, period_duration, profiles) # BE CAREFUL: tested only for 1 year
                    else
                        create_init_rps_daily_ofas(connection, period_duration, profiles)
                    end
                end

                if stochastic_method == :per_scenario
                    if one_for_all_scenarios
                        error("We can select only one artificial period per scenario only if we are using cross")
                    end
                    layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year, :scenario])
                    if add_worst_every_hour || add_worst_sum
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        time_to_cluster = @elapsed TC.cluster!(
                            connection,
                            period_duration,
                            round(Int, rp / n_scenarios);
                            method=method,
                            distance=distance,
                            initial_representatives=init_rps_df,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    else
                        time_to_cluster = @elapsed TC.cluster!(
                            connection,
                            period_duration,
                            round(Int, rp / n_scenarios);
                            method=method,
                            distance=distance,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    end
                elseif stochastic_method == :cross_scenario # now here i just assume cross
                    layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year], cols_to_crossby=[:scenario])
                    if add_worst_every_hour || add_worst_sum
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        time_to_cluster = @elapsed TC.cluster!(
                            connection,
                            period_duration,
                            rp;
                            method=method,
                            distance=distance,
                            initial_representatives=init_rps_df,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    else
                        time_to_cluster = @elapsed TC.cluster!(
                            connection,
                            period_duration,
                            rp;
                            method=method,
                            distance=distance,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    end

                else
                    error("Unknown stochastic method: $stochastic_method")
                end
                println("Clustering time ", time_to_cluster)
                df_profiles_rp = TIO.get_table(connection, "profiles_rep_periods")
                df_rp_mapping = TIO.get_table(connection, "rep_periods_mapping")
                df_rep_periods_data = TIO.get_table(connection, "rep_periods_data")
                df_timeframe_data = TIO.get_table(connection, "timeframe_data")
                output_folder = joinpath(@__DIR__, "save_rps", case_name)
                mkpath(output_folder)
                CSV.write(joinpath(output_folder, "profiles_rep_periods"), df_profiles_rp)
                CSV.write(joinpath(output_folder, "rep_periods_mapping"), df_rp_mapping)
                CSV.write(joinpath(output_folder, "rep_periods_data"), df_rep_periods_data)
                CSV.write(joinpath(output_folder, "timeframe_data"), df_timeframe_data)

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
                    if energy_problem.termination_status == JuMP.INFEASIBLE
                        # JuMP.compute_conflict!(energy_problem.model)
                        # iis_model, reference_map = JuMP.copy_conflict(energy_problem.model)
                        # print(iis_model)
                        println("Infeasible - skipped")
                        break
                    end
                    println("Solving time ", time_to_solve)
                    time_to_save = @elapsed TEM.save_solution!(energy_problem)

                    TEM.export_solution_to_csv_files(output_folder, energy_problem)
                    var_flow_red_df = TIO.get_table(connection, "var_flow")
                    flow_ens_red = filter(row ->
                            occursin("ens", lowercase(row.from_asset)),
                            var_flow_red_df
                            )
                    flow_smr_ccs_red = filter(row ->
                            occursin("smr", lowercase(row.from_asset)),
                            var_flow_red_df
                            )
                    # count steps with loss of load
                    n_lol_ens_red = count(row -> row.solution > 1e-8, eachrow(flow_ens_red))
                    n_lol_smr_cca_red = count(row -> row.solution > 1e-8, eachrow(flow_smr_ccs_red))
                    lol_ens_red = sum(flow_ens_red.solution)
                    lol_smr_red = sum(flow_smr_ccs_red.solution)

                    new_results_row = (
                        base_name=base_name,
                        rp=rp,
                        solver=solver,
                        time_to_cluster=time_to_cluster,
                        time_to_read=time_to_read,
                        time_to_create=time_to_create,
                        time_to_solve=time_to_solve,
                        time_to_save=time_to_save,
                        objective_value=energy_problem.objective_value,
                        termination_status=string(energy_problem.termination_status),
                        reduced_num_lol_e=n_lol_ens_red,
                        reduced_num_lol_h2=n_lol_smr_cca_red,
                        reduced_lol_e=lol_ens_red,
                        reduced_lol_h2=lol_smr_red,
                        num_constraints=JuMP.num_constraints(
                            energy_problem.model;
                            count_variable_in_set_constraints=false,
                        ),
                        num_variables=JuMP.num_variables(energy_problem.model)
                    )
                    push!(results_df, new_results_row)
                    CSV.write(
                        "outputs/results_partial.csv",
                        results_df;
                        writeheader=true
                    )
                end
                                
                energy_problem = nothing

                DuckDB.close(connection)
                GC.gc()
            end
        end
    end

    results_df |> CSV.write("outputs/results.csv"; writeheader=true)

    return nothing
end

main()