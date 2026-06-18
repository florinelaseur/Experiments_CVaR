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
using Dates

seed = parse(Int, get(ENV, "EXPERIMENT_SEED", "19990907"))
Random.seed!(seed)

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
number_of_scenarios = parse(Int, get(ENV, "NUMBER_OF_SCENARIOS", string(config["simulation"]["number_of_scenarios"])))
target_scenarios = parse(Int, get(ENV, "TARGET_SCENARIOS", string(number_of_scenarios ÷ 2)))
output_base_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "outputs"))

if haskey(ENV, "PREFLIGHT") && ENV["PREFLIGHT"] == "1"
    @info "PREFLIGHT MODE: Overriding config for smoke test."
    representative_periods = [2]
    number_of_scenarios = 4
    target_scenarios = 2
end

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

all_profiles_df = nothing
profiles_df = nothing
GC.gc()

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

enable_names = false
direct_model = true
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
    # =====================================================================
    # STATIC BENCHMARK SETUP
    # =====================================================================
    benchmark_db_file = joinpath(output_base_dir, "benchmark.duckdb")
    rm(benchmark_db_file, force=true)
    connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB, benchmark_db_file)
    
    TIO.read_csv_folder(connection_benchmark, input_data_path)
    profiles_wide = TIO.get_table(connection_benchmark, "profiles_wide")
    
    all_scenarios = sort(unique(profiles_wide.scenario))
    n_scenarios = length(all_scenarios)
    time_to_cluster = 0.0

    solver = "Gurobi"
    optimizer, parameters = get_solver_parameters(Symbol(solver), seed)

    if run_benchmark
        @info "Building the STATIC N=$(n_scenarios) Benchmark Model in RAM ONCE..."
        base_name = "0_HourlyBenchmark"

        DuckDB.query(
            connection_benchmark,
            "UPDATE model_parameters SET risk_aversion_weight_lambda = $(lambda), risk_aversion_confidence_level_alpha = $(alpha);"
        )
        DuckDB.query(
            connection_benchmark,
            "UPDATE asset SET capacity = 1000000 WHERE asset IN ('ens', 'smr_ccs');"
        )

        TC.transform_wide_to_long!(connection_benchmark, "profiles_wide", "profiles"; exclude_columns=["scenario", "milestone_year", "timestep"])
        layout = TC.ProfilesTableLayout(; year=:milestone_year, cols_to_groupby=[:milestone_year, :scenario])
        time_to_cluster = @elapsed TC.dummy_cluster!(connection_benchmark; layout=layout)
        TEM.populate_with_defaults!(connection_benchmark)
        DuckDB.query(connection_benchmark, "UPDATE asset SET is_seasonal = false")

        energy_problem_benchmark = TEM.EnergyProblem(connection_benchmark)
        TEM.create_model!(
            energy_problem_benchmark;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=enable_names,
            direct_model=direct_model,
        )
        @info "✅ Static Benchmark Model built and retained in memory."
    end
    
    # =====================================================================
    # IPDSR STANDALONE EXPERIMENT
    # =====================================================================
    base_name = "IPDSR_Experiment"
    @info "Processing case study: $base_name"

    iter = 0
    last_selected_scenarios = shuffle(all_scenarios)[1:target_scenarios]
    last_weights = fill(1.0 / target_scenarios, target_scenarios)
    
    while iter < IPDSR_MAX_ITER
        iter += 1
        @info "=== IPDSR Iteration $iter ==="
        
        db_file = joinpath(output_base_dir, "temp_iter_$(iter).duckdb")
        connection = DuckDB.DBInterface.connect(DuckDB.DB, db_file)
        TIO.read_csv_folder(connection, input_data_path)

        scen_str = join(["'" * string(s) * "'" for s in last_selected_scenarios], ", ")
        DuckDB.query(connection, "DELETE FROM profiles_wide WHERE scenario NOT IN ($scen_str)")
        DuckDB.query(connection, "DELETE FROM stochastic_scenario WHERE scenario NOT IN ($scen_str)")

        for (i, scen) in enumerate(last_selected_scenarios)
            weight = last_weights[i]
            DuckDB.query(connection, "UPDATE stochastic_scenario SET probability = $weight WHERE scenario = '$scen'")
        end

        DuckDB.query(connection, "UPDATE model_parameters SET risk_aversion_weight_lambda = $(lambda), risk_aversion_confidence_level_alpha = $(alpha);")

        TC.transform_wide_to_long!(connection, "profiles_wide", "profiles"; exclude_columns=["scenario", "milestone_year", "timestep"])
        layout = TC.ProfilesTableLayout(; year=:milestone_year, cols_to_groupby=[:milestone_year, :scenario])
        time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)
        TEM.populate_with_defaults!(connection)
        DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")

        # --- BUILD EPHEMERAL REDUCED MODEL ---
        time_to_read = @elapsed energy_problem = TEM.EnergyProblem(connection)
        @info "Creating the REDUCED model (K=$target_scenarios)"
        time_to_create = @elapsed TEM.create_model!(
            energy_problem;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=enable_names,
            direct_model=direct_model,
        )
        
        output_folder = joinpath(output_base_dir, base_name, "Iter_$iter", string(solver))
        mkpath(output_folder)

        @info "Solving the REDUCED model"
        time_to_solve = @elapsed TEM.solve_model!(energy_problem)
        time_to_save = @elapsed TEM.save_solution!(energy_problem)
        
        LB = energy_problem.objective_value
        reduced_term_status = string(energy_problem.termination_status)
        reduced_num_cons = JuMP.num_constraints(energy_problem.model; count_variable_in_set_constraints=false)
        reduced_num_vars = JuMP.num_variables(energy_problem.model)
        
        val_assets_investment = JuMP.value.(energy_problem.variables[:assets_investment].container)
        val_assets_investment_energy = JuMP.value.(energy_problem.variables[:assets_investment_energy].container)

        # --- AGGRESSIVELY DESTROY THE REDUCED MODEL ---
        @info "Destroying Ephemeral Reduced Model to free RAM..."
        if hasproperty(energy_problem, :model) && !isnothing(energy_problem.model)
            empty!(energy_problem.model)
            finalize(energy_problem.model)
        end
        energy_problem = nothing
        GC.gc(true)
        if Sys.islinux()
            ccall(:malloc_trim, Cint, (Cint,), 0)
        end
        
        # --- APPLY TO STATIC BENCHMARK ---
        if run_benchmark
            @info "Evaluating validation decision against the Static Benchmark"

            var_to_fix_inv = energy_problem_benchmark.variables[:assets_investment].container
            for (var, val) in zip(var_to_fix_inv, val_assets_investment)
                JuMP.fix(var, val; force=true)
            end

            var_to_fix_ene = energy_problem_benchmark.variables[:assets_investment_energy].container
            for (var, val) in zip(var_to_fix_ene, val_assets_investment_energy)
                JuMP.fix(var, val; force=true)
            end
            
            time_to_resolve_benchmark = @elapsed TEM.solve_model!(energy_problem_benchmark)
            TEM.save_solution!(energy_problem_benchmark) # Required to populate DuckDB tables for the queries below!
            
            UB = energy_problem_benchmark.objective_value
            OG = abs(UB - LB) / abs(UB)
            
            @info "--- IPDSR Iteration $iter Status ---"
            @info "LB (Reduced Cost): $(round(LB, digits=2)) | UB (True Cost): $(round(UB, digits=2))"
            @info "Current Optimality Gap (OG): $(round(OG * 100, digits=3))%"
            @info "Current time: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.sss"))"
            
            ens_query = "SELECT COUNT(*) as count FROM var_flow WHERE from_asset = 'ens' AND to_asset = 'e_demand' AND solution > 0.0"
            n_lol_ens_df = DuckDB.query(connection_benchmark, ens_query) |> DataFrame
            n_lol_ens = nrow(n_lol_ens_df) > 0 ? n_lol_ens_df.count[1] : 0

            smr_query = "SELECT COUNT(*) as count FROM var_flow WHERE from_asset = 'smr_ccs' AND to_asset = 'h2_demand' AND solution > 0.0"
            n_lol_smr_df = DuckDB.query(connection_benchmark, smr_query) |> DataFrame
            n_lol_smr_cca = nrow(n_lol_smr_df) > 0 ? n_lol_smr_df.count[1] : 0

            water_query = "SELECT SUM(solution) as total FROM var_flow WHERE from_asset = 'water_borrower' AND to_asset = 'hydro_reservoir'"
            water_df = DuckDB.query(connection_benchmark, water_query) |> DataFrame
            amount_water_borrowed = (nrow(water_df) > 0 && !ismissing(water_df.total[1])) ? water_df.total[1] : 0.0
            
            mu_query = "SELECT solution FROM var_value_at_risk_threshold_mu LIMIT 1"
            mu_value_df = DuckDB.query(connection_benchmark, mu_query) |> DataFrame
            mu_value = (nrow(mu_value_df) > 0 && !ismissing(mu_value_df.solution[1])) ? Float64(mu_value_df.solution[1]) : 0.0

            new_results_row = (
                base_name=base_name, rp=1, solver=Symbol(solver),
                time_to_cluster=time_to_cluster, time_to_read=time_to_read,
                time_to_create=time_to_create, time_to_solve=time_to_solve,
                time_to_save=time_to_save, objective_value=LB,
                termination_status=reduced_term_status,
                num_constraints=reduced_num_cons,
                num_variables=reduced_num_vars,
                time_to_resolve_benchmark=time_to_resolve_benchmark,
                objective_value_resolve_benchmark=UB,
                termination_status_resolve_benchmark=string(energy_problem_benchmark.termination_status),
                num_loss_of_load_e_demand=n_lol_ens, num_loss_of_load_h2_demand=n_lol_smr_cca,
                water_borrowed=amount_water_borrowed, value_at_risk_threshold_mu=mu_value,
            )
            
            # Check Convergence
            if OG <= IPDSR_MIP_GAP || iter == IPDSR_MAX_ITER
                if OG <= IPDSR_MIP_GAP
                    @info "✅ IPDSR SUCCESSFULLY CONVERGED at Iteration $(iter)!"
                else
                    @warn "⚠️ IPDSR hit Max Iterations ($IPDSR_MAX_ITER) without perfect convergence."
                end
                
                output_folder_fixed = joinpath(output_base_dir, "fixed", base_name, string(solver))
                mkpath(output_folder_fixed)
                TEM.export_solution_to_csv_files(output_folder_fixed, energy_problem_benchmark)
                
                push!(results_df, new_results_row)
                break
            end

            # 5. Extract Problem Space & Solve MIP
            @info "--- Extracting Problem Space for IPDSR Feedback ---"
            df_costs = export_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
            sort!(df_costs, :scenario)
            
            F_costs = df_costs.operational_cost
            gamma = fill(1.0 / number_of_scenarios, number_of_scenarios)
            
            N_prime = min(length(F_costs), max(20, 10)) 
            F_agg, gamma_agg, original_mapping = aggregate_objectives(F_costs, gamma, N_prime)
            
            @info "Solving IPDSR MIP to reduce to K = $target_scenarios scenarios..."
            selected_agg_idx, new_weights = solve_ipdsr_mip(F_agg, gamma_agg, target_scenarios, lambda, alpha, seed)
            
            if isempty(selected_agg_idx)
                @warn "IPDSR failed to find scenarios. Saving current best and aborting."
                push!(results_df, new_results_row)
                break
            end

            selected_original_scenarios = [df_costs.scenario[original_mapping[idx][1]] for idx in selected_agg_idx]
            
            # FIXED-POINT CONVERGENCE CHECK
            if selected_original_scenarios == last_selected_scenarios && length(new_weights) == length(last_weights) && isapprox(new_weights, last_weights, atol=1e-4)
                @info "✅ FIXED-POINT CONVERGENCE ACHIEVED! Scenarios and weights stabilized at Iteration $iter."
                output_folder_fixed = joinpath(output_base_dir, "fixed", base_name, string(solver))
                mkpath(output_folder_fixed)
                TEM.export_solution_to_csv_files(output_folder_fixed, energy_problem_benchmark)
                
                push!(results_df, new_results_row)
                break
            end

            last_selected_scenarios = copy(selected_original_scenarios)
            last_weights = copy(new_weights)
            
        else
            # Handles the case if the user set run_benchmark = false in config
            new_results_row = (
                base_name=base_name, rp=1, solver=Symbol(solver),
                time_to_cluster=time_to_cluster, time_to_read=time_to_read,
                time_to_create=time_to_create, time_to_solve=time_to_solve,
                time_to_save=time_to_save, objective_value=LB,
                termination_status=reduced_term_status,
                num_constraints=reduced_num_cons,
                num_variables=reduced_num_vars,
                time_to_resolve_benchmark=0.0, objective_value_resolve_benchmark=0.0,
                termination_status_resolve_benchmark="", num_loss_of_load_e_demand=0,
                num_loss_of_load_h2_demand=0, water_borrowed=0.0, value_at_risk_threshold_mu=0.0,
            )
            push!(results_df, new_results_row)
            @info "Benchmark disabled. Breaking after one evaluation."
            break
        end
        
        # Explicitly close temporary loop DuckDB and force full GC
        DuckDB.close(connection)
        connection = nothing
        df_costs = nothing
        rm(db_file, force=true)
        
        GC.gc(true)
        if Sys.islinux()
            ccall(:malloc_trim, Cint, (Cint,), 0)
        end
    end 
    
    # Clean up benchmark connection at the very end
    if run_benchmark
        DuckDB.close(connection_benchmark)
        rm(benchmark_db_file, force=true)
    end

    results_df |> CSV.write(joinpath(output_base_dir, "results.csv"); writeheader=true)

    return nothing
end

main()