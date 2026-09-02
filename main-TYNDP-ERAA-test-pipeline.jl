# Copyright (c) 2025: Diego Tejada and contributors
#
# Use of this source code is governed by an Apache 2.0 license that can be found
# in the LICENSE.md file at https://opensource.org/license/apache-2-0.

cd(@__DIR__)

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

# ============================================================
# READ CONFIGURATION
# ============================================================

config = TOML.parsefile("config-test-pipeline.toml")

input_data_path = joinpath(
    homedir(),
    "Nextcloud",
    config["simulation"]["input_data_EU"],
)

input_data_path_CC = joinpath(
    homedir(),
    "Nextcloud",
    config["simulation"]["input_data_EU_CC"],
)

use_ratio = config["clustering"]["use_ratio"]
heuristic_distance = config["clustering"]["heuristic_distance"]
fix_level_storage = config["simulation"]["fix_level_storage"]
representative_periods = config["simulation"]["representative_periods"]

solvers = [
    Symbol(el)
    for el in config["simulation"]["solvers"]
]

lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]
fix_benchmark = config["simulation"]["fix_benchmark"]

# The old ratio formulation assumed one global demand profile.
# This is not valid anymore for the EU case with multiple demand locations.
if use_ratio
    error(
        "use_ratio=true is not supported for the EU case with multiple demand locations. " *
        "Set [clustering].use_ratio = false.",
    )
end


# ============================================================
# SELECT SCENARIOS
# ============================================================

profiles_path = joinpath(
    input_data_path,
    "profiles-wide-all-scenarios.csv",
)

all_profiles_df = CSV.read(
    profiles_path,
    DataFrame,
)

profiles_df = get_scenario_set(
    all_profiles_df,
    number_of_scenarios,
)

selected_scenarios = sort(
    unique(profiles_df.scenario),
)

mapping = Dict(
    old => new
    for (new, old) in enumerate(selected_scenarios)
)

profiles_df[!, :scenario] = [
    mapping[s]
    for s in profiles_df.scenario
]

CSV.write(
    joinpath(
        input_data_path,
        "profiles-wide.csv",
    ),
    profiles_df;
    writeheader=true,
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

model_parameters_df = DataFrame(
    risk_aversion_weight_lambda=[lambda],
    risk_aversion_confidence_level_alpha=[alpha],
)

for data_path in (
    input_data_path,
    input_data_path_CC,
)
    mkpath(data_path)

    CSV.write(
        joinpath(
            data_path,
            "model-parameters.csv",
        ),
        model_parameters_df;
        writeheader=true,
    )
end


# ============================================================
# STOCHASTIC SCENARIOS
# ============================================================

df_stochastic_scenario = DataFrame(
    scenario=sort(unique(profiles_df.scenario)),
    probability=fill(
        1.0 / number_of_scenarios,
        number_of_scenarios,
    ),
)

CSV.write(
    joinpath(
        input_data_path,
        "stochastic-scenario.csv",
    ),
    df_stochastic_scenario;
    writeheader=true,
)


# ============================================================
# CASE STUDY INFORMATION
# ============================================================

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


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = DataFrame(
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


# ============================================================
# MAIN
# ============================================================

function main()

    # --------------------------------------------------------
    # DETERMINE NUMBER OF SCENARIOS
    # --------------------------------------------------------

    connection_benchmark =
        DuckDB.DBInterface.connect(DuckDB.DB)

    TIO.read_csv_folder(
        connection_benchmark,
        input_data_path,
    )

    profiles_wide = TIO.get_table(
        connection_benchmark,
        "profiles_wide",
    )

    n_scenarios = length(
        unique(profiles_wide.scenario),
    )

    @assert n_scenarios == number_of_scenarios "Expected $number_of_scenarios scenarios, found $n_scenarios in profiles_wide"


    # ========================================================
    # HOURLY BASELINE
    # ========================================================

    @info "Running the base case study (0_HourlyPartitionBaseline)"

    base_name = "0_HourlyPartitionBaseline"

    connection_baseline =
        DuckDB.DBInterface.connect(DuckDB.DB)

    TIO.read_csv_folder(
        connection_baseline,
        input_data_path,
    )

    # Update risk parameters
    DuckDB.query(
        connection_baseline,
        """
        UPDATE model_parameters
        SET
            risk_aversion_weight_lambda = $(lambda),
            risk_aversion_confidence_level_alpha = $(alpha);
        """,
    )

    # Transform wide profiles to long format
    TC.transform_wide_to_long!(
        connection_baseline,
        "profiles_wide",
        "profiles";
        exclude_columns=[
            "scenario",
            "milestone_year",
            "timestep",
        ],
    )

    layout = TC.ProfilesTableLayout(
        year=:milestone_year,
        cols_to_groupby=[
            :milestone_year,
            :scenario,
        ],
    )

    time_to_cluster = @elapsed TC.dummy_cluster!(
        connection_baseline;
        layout=layout,
    )

    TEM.populate_with_defaults!(
        connection_baseline,
    )

    DuckDB.query(
        connection_baseline,
        "UPDATE asset SET is_seasonal = false",
    )

    time_to_read = @elapsed energy_problem_baseline =
        TEM.EnergyProblem(
            connection_baseline,
        )

    baseline_investment_by_solver =
        Dict{Symbol,DataFrame}()

    baseline_n_lol_ens_by_solver =
        Dict{Symbol,Int}()

    baseline_n_lol_smr_ccs_by_solver =
        Dict{Symbol,Int}()

    baseline_objective_by_solver =
        Dict{Symbol,Float64}()


    # --------------------------------------------------------
    # SOLVE HOURLY BASELINE
    # --------------------------------------------------------

    for solver in solvers

        optimizer, parameters =
            get_solver_parameters(solver)

        @info "Creating the model for the base case study $base_name with $solver"

        time_to_create = @elapsed TEM.create_model!(
            energy_problem_baseline;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=enable_names,
            direct_model=direct_model,
        )

        baseline_output_folder = joinpath(
            homedir(),
            "Nextcloud",
            "ExperimentData",
            "EU-output-data",
            base_name,
            "N$(number_of_scenarios)_seed$(seed)",
            string(solver),
        )

        mkpath(baseline_output_folder)

        if !(
            isfile(
                joinpath(
                    baseline_output_folder,
                    "var_assets_investment.csv",
                ),
            ) &&
            isfile(
                joinpath(
                    baseline_output_folder,
                    "var_flow.csv",
                ),
            ) &&
            isfile(
                joinpath(
                    baseline_output_folder,
                    "baseline_breakdown.csv",
                ),
            ) &&
            isfile(
                joinpath(
                    baseline_output_folder,
                    "var_value_at_risk_threshold_mu.csv",
                ),
            ) &&
            isfile(
                joinpath(
                    baseline_output_folder,
                    "total_operational_cost_per_scenario.csv",
                ),
            )
        )

            @info "Solving the model and saving the solution for the base case study $base_name with $solver"

            time_to_solve =
                @elapsed TEM.solve_model!(
                    energy_problem_baseline,
                )

            time_to_save =
                @elapsed TEM.save_solution!(
                    energy_problem_baseline,
                )

            TEM.export_solution_to_csv_files(
                baseline_output_folder,
                energy_problem_baseline,
            )

            baseline_investment_df =
                TIO.get_table(
                    connection_baseline,
                    "var_assets_investment",
                )

            baseline_objective =
                energy_problem_baseline.objective_value

            termination_status =
                string(
                    energy_problem_baseline.termination_status,
                )

            baseline_df = DataFrame(
                time_to_solve=[time_to_solve],
                time_to_save=[time_to_save],
                objective_value=[baseline_objective],
                termination_status=[termination_status],
            )

            CSV.write(
                joinpath(
                    baseline_output_folder,
                    "baseline_breakdown.csv",
                ),
                baseline_df;
                writeheader=true,
            )

            mu_value_df =
                TIO.get_table(
                    connection_baseline,
                    "var_value_at_risk_threshold_mu",
                )

            var_flow_df =
                TIO.get_table(
                    connection_baseline,
                    "var_flow",
                )

            df_cost_per_scenario =
                export_total_operational_cost_per_scenario(
                    energy_problem_baseline,
                    baseline_output_folder,
                )

            plot_cost_per_scenario(
                df_cost_per_scenario,
                baseline_output_folder,
                mu_value_df,
            )

        else

            baseline_df = CSV.read(
                joinpath(
                    baseline_output_folder,
                    "baseline_breakdown.csv",
                ),
                DataFrame,
            )

            time_to_solve =
                only(baseline_df.time_to_solve)

            time_to_save =
                only(baseline_df.time_to_save)

            baseline_investment_df =
                CSV.read(
                    joinpath(
                        baseline_output_folder,
                        "var_assets_investment.csv",
                    ),
                    DataFrame,
                )

            baseline_objective =
                only(baseline_df.objective_value)

            termination_status =
                only(baseline_df.termination_status)

            mu_value_df =
                CSV.read(
                    joinpath(
                        baseline_output_folder,
                        "var_value_at_risk_threshold_mu.csv",
                    ),
                    DataFrame,
                )

            var_flow_df =
                CSV.read(
                    joinpath(
                        baseline_output_folder,
                        "var_flow.csv",
                    ),
                    DataFrame,
                )

            df_cost_per_scenario =
                CSV.read(
                    joinpath(
                        baseline_output_folder,
                        "total_operational_cost_per_scenario.csv",
                    ),
                    DataFrame,
                )

            plot_cost_per_scenario(
                df_cost_per_scenario,
                baseline_output_folder,
                mu_value_df,
            )
        end


        # ----------------------------------------------------
        # BASELINE METRICS
        # ----------------------------------------------------

        mu_value =
            if nrow(mu_value_df) == 0
                NaN
            else
                only(mu_value_df.solution)
            end


        # Multiple electricity demand locations
        flow_ens = filter(
            row ->
                occursin(
                        "ens",
                        lowercase(string(row.from_asset)),
                    ) &&
                    occursin(
                        "demand",
                        lowercase(string(row.to_asset)),
                    ) &&
                    (
                        occursin(
                            "_e_",
                            lowercase(string(row.to_asset)),
                        ) ||
                        endswith(
                            lowercase(string(row.to_asset)),
                            "_e_demand",
                        ) ||
                        lowercase(string(row.to_asset)) ==
                        "e_demand"
                    ),
            var_flow_df,
        )


        # Multiple hydrogen demand locations
        flow_smr_ccs = filter(
            row ->
                occursin(
                        "smr_ccs",
                        lowercase(string(row.from_asset)),
                    ) &&
                    occursin(
                        "demand",
                        lowercase(string(row.to_asset)),
                    ) &&
                    (
                        occursin(
                            "h2",
                            lowercase(string(row.to_asset)),
                        ) ||
                        occursin(
                            "_h_",
                            lowercase(string(row.to_asset)),
                        ) ||
                        endswith(
                            lowercase(string(row.to_asset)),
                            "_h_demand",
                        ) ||
                        lowercase(string(row.to_asset)) ==
                        "h2_demand"
                    ),
            var_flow_df,
        )


        water_borrowed = filter(
            row ->
                occursin(
                    "water_borrower",
                    lowercase(string(row.from_asset)),
                ) &&
                    occursin(
                        "hydro_reservoir",
                        lowercase(string(row.to_asset)),
                    ),
            var_flow_df,
        )


        baseline_n_lol_ens =
            count(
                row -> row.solution > 0.0,
                eachrow(flow_ens),
            )

        baseline_lole_e_demand =
            baseline_n_lol_ens /
            number_of_scenarios


        baseline_n_lol_smr_ccs =
            count(
                row -> row.solution > 0.0,
                eachrow(flow_smr_ccs),
            )

        baseline_lole_h2_demand =
            baseline_n_lol_smr_ccs /
            number_of_scenarios


        amount_water_borrowed_b =
            sum(water_borrowed.solution)


        baseline_investment_by_solver[solver] =
            copy(baseline_investment_df)

        baseline_n_lol_ens_by_solver[solver] =
            baseline_n_lol_ens

        baseline_n_lol_smr_ccs_by_solver[solver] =
            baseline_n_lol_smr_ccs

        baseline_objective_by_solver[solver] =
            baseline_objective


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
            num_variables=JuMP.num_variables(
                energy_problem_baseline.model,
            ),
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
            num_loss_of_load_e_demand_baseline=
            baseline_n_lol_ens,
            lole_e_demand_baseline=
            baseline_lole_e_demand,
            num_loss_of_load_h2_demand_baseline=
            baseline_n_lol_smr_ccs,
            lole_h2_demand_baseline=
            baseline_lole_h2_demand,
            water_borrowed_baseline=
            amount_water_borrowed_b,
            value_at_risk_threshold_mu_baseline=
            mu_value,
            scenario_set="full",
            seed=seed,
            number_of_scenarios=
            number_of_scenarios,
        )

        push!(
            results_df,
            new_results_row,
        )
    end


    # ========================================================
    # REPRESENTATIVE-PERIOD CASES
    # ========================================================

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
            :niters => niters,
        )

        clustering_kwargs = Dict(
            :learning_rate => learning_rate,
            :niters => niters,
        )

        if !run_case
            continue
        end


        for rp in representative_periods

            case_name =
                base_name *
                "_rp_" *
                "$rp" *
                "_benchmark"

            @info "Processing case study: $case_name"


            # ------------------------------------------------
            # READ INPUT
            # ------------------------------------------------

            connection_benchmark =
                DuckDB.DBInterface.connect(
                    DuckDB.DB,
                )

            TIO.read_csv_folder(
                connection_benchmark,
                input_data_path,
            )


            DuckDB.query(
                connection_benchmark,
                """
                UPDATE model_parameters
                SET
                    risk_aversion_weight_lambda = $(lambda),
                    risk_aversion_confidence_level_alpha = $(alpha);
                """,
            )


            # ------------------------------------------------
            # TRANSFORM PROFILES
            # ------------------------------------------------

            TC.transform_wide_to_long!(
                connection_benchmark,
                "profiles_wide",
                "profiles";
                exclude_columns=[
                    "scenario",
                    "milestone_year",
                    "timestep",
                ],
            )


            # ------------------------------------------------
            # CLUSTER
            # ------------------------------------------------

            if stochastic_method == :per_scenario

                layout =
                    TC.ProfilesTableLayout(
                        year=:milestone_year,
                        cols_to_groupby=[
                            :milestone_year,
                            :scenario,
                        ],
                    )

                time_to_cluster =
                    @elapsed TC.cluster!(
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


            elseif stochastic_method == :cross_scenario

                layout =
                    TC.ProfilesTableLayout(
                        year=:milestone_year,
                        cols_to_groupby=[
                            :milestone_year,
                        ],
                        cols_to_crossby=[
                            :scenario,
                        ],
                    )

                time_to_cluster =
                    @elapsed TC.cluster!(
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
                error(
                    "Unknown stochastic method: $stochastic_method",
                )
            end


            # ------------------------------------------------
            # EXPAND PARTITIONS TO ALL REPRESENTATIVE PERIODS
            # ------------------------------------------------
            #
            # assets-rep-periods-partitions.csv and
            # flows-rep-periods-partitions.csv define the
            # partition assignment for rep_period = 1.
            #
            # Copy that assignment to rep_period = 1:rp
            # after clustering and before TEM processes the
            # representative-period structure.
            # ------------------------------------------------

            expand_rep_period_partitions!(
                connection_benchmark,
                rp,
            )


            # ------------------------------------------------
            # CREATE ENERGY PROBLEM
            # ------------------------------------------------

            TEM.populate_with_defaults!(
                connection_benchmark,
            )

            time_to_read =
                @elapsed energy_problem_benchmark =
                    TEM.EnergyProblem(
                        connection_benchmark,
                    )


            # =================================================
            # SOLVE RP MODEL
            # =================================================

            for solver in solvers

                optimizer, parameters =
                    get_solver_parameters(solver)

                @info "Creating the model for the case study: $case_name"

                time_to_create =
                    @elapsed TEM.create_model!(
                        energy_problem_benchmark;
                        optimizer=optimizer,
                        optimizer_parameters=parameters,
                        model_file_name="",
                        enable_names=enable_names,
                    )


                output_folder = joinpath(
                    homedir(),
                    "Nextcloud",
                    "ExperimentData",
                    "EU-output-data",
                    "N$(number_of_scenarios)_seed$(seed)",
                    case_name,
                    string(solver),
                )

                mkpath(output_folder)


                @info "Solving the model and saving the solution for the case study: $case_name with $solver"

                time_to_solve =
                    @elapsed TEM.solve_model!(
                        energy_problem_benchmark,
                    )

                time_to_save =
                    @elapsed TEM.save_solution!(
                        energy_problem_benchmark,
                    )

                TEM.export_solution_to_csv_files(
                    output_folder,
                    energy_problem_benchmark,
                )


                # ---------------------------------------------
                # EXPORT REPRESENTATIVE-PERIOD MAPPING
                # ---------------------------------------------

                rep_periods_mapping =
                    TIO.get_table(
                        connection_benchmark,
                        "rep_periods_mapping",
                    )

                CSV.write(
                    joinpath(
                        output_folder,
                        "rep_periods_mapping.csv",
                    ),
                    rep_periods_mapping;
                    writeheader=true,
                )


                # ---------------------------------------------
                # INVESTMENTS
                # ---------------------------------------------

                benchmark_investment_df =
                    TIO.get_table(
                        connection_benchmark,
                        "var_assets_investment",
                    )


                # ---------------------------------------------
                # VaR
                # ---------------------------------------------

                mu_value_df =
                    TIO.get_table(
                        connection_benchmark,
                        "var_value_at_risk_threshold_mu",
                    )

                mu_value_benchmark =
                    if nrow(mu_value_df) == 0
                        NaN
                    else
                        only(mu_value_df.solution)
                    end


                if !isnan(mu_value_benchmark)

                    @info "mu_value of benchmark is defined"

                    @show mu_value_benchmark
                end


                # ---------------------------------------------
                # COST PER SCENARIO
                # ---------------------------------------------

                benchmark_cost_df =
                    export_total_operational_cost_per_scenario(
                        energy_problem_benchmark,
                        output_folder,
                    )

                plot_cost_per_scenario(
                    benchmark_cost_df,
                    output_folder,
                    mu_value_df,
                )

                sorted_cost_df =
                    sort(
                        benchmark_cost_df,
                        :total_cost,
                    )

                var_index =
                    ceil(
                        Int,
                        alpha * n_scenarios,
                    )

                var_empirical =
                    sorted_cost_df.total_cost[
                        var_index
                    ]


                # ---------------------------------------------
                # FLOW METRICS
                # ---------------------------------------------

                var_flow_df =
                    TIO.get_table(
                        connection_benchmark,
                        "var_flow",
                    )


                flow_ens = filter(
                    row ->
                        occursin(
                                "ens",
                                lowercase(
                                    string(
                                        row.from_asset,
                                    ),
                                ),
                            ) &&
                            occursin(
                                "demand",
                                lowercase(
                                    string(
                                        row.to_asset,
                                    ),
                                ),
                            ) &&
                            (
                                occursin(
                                    "_e_",
                                    lowercase(
                                        string(
                                            row.to_asset,
                                        ),
                                    ),
                                ) ||
                                endswith(
                                    lowercase(
                                        string(
                                            row.to_asset,
                                        ),
                                    ),
                                    "_e_demand",
                                ) ||
                                lowercase(
                                    string(
                                        row.to_asset,
                                    ),
                                ) == "e_demand"
                            ),
                    var_flow_df,
                )


                flow_smr_ccs = filter(
                    row ->
                        occursin(
                                "smr_ccs",
                                lowercase(
                                    string(
                                        row.from_asset,
                                    ),
                                ),
                            ) &&
                            occursin(
                                "demand",
                                lowercase(
                                    string(
                                        row.to_asset,
                                    ),
                                ),
                            ) &&
                            (
                                occursin(
                                    "h2",
                                    lowercase(
                                        string(
                                            row.to_asset,
                                        ),
                                    ),
                                ) ||
                                occursin(
                                    "_h_",
                                    lowercase(
                                        string(
                                            row.to_asset,
                                        ),
                                    ),
                                ) ||
                                endswith(
                                    lowercase(
                                        string(
                                            row.to_asset,
                                        ),
                                    ),
                                    "_h_demand",
                                ) ||
                                lowercase(
                                    string(
                                        row.to_asset,
                                    ),
                                ) == "h2_demand"
                            ),
                    var_flow_df,
                )


                water_borrowed = filter(
                    row ->
                        occursin(
                            "water_borrower",
                            lowercase(
                                string(
                                    row.from_asset,
                                ),
                            ),
                        ) &&
                            occursin(
                                "hydro_reservoir",
                                lowercase(
                                    string(
                                        row.to_asset,
                                    ),
                                ),
                            ),
                    var_flow_df,
                )


                # ---------------------------------------------
                # LOSS OF LOAD
                # ---------------------------------------------

                bm_n_lol_ens =
                    count(
                        row ->
                            row.solution > 0.0,
                        eachrow(flow_ens),
                    )

                lole_e_demand =
                    bm_n_lol_ens /
                    number_of_scenarios


                bm_n_lol_smr_ccs =
                    count(
                        row ->
                            row.solution > 0.0,
                        eachrow(flow_smr_ccs),
                    )

                lole_h2_demand =
                    bm_n_lol_smr_ccs /
                    number_of_scenarios


                # ---------------------------------------------
                # WATER BORROWING
                # ---------------------------------------------

                amount_water_borrowed_b =
                    sum(
                        water_borrowed.solution,
                    )

                amount_water_borrowed_err =
                    sum(
                        water_borrowed.solution,
                    )

                if amount_water_borrowed_err > 0.0
                    error(
                        "Borrowed water has been used: $amount_water_borrowed_err",
                    )
                end


                # ---------------------------------------------
                # BASELINE RESULTS
                # ---------------------------------------------

                baseline_investment_df =
                    baseline_investment_by_solver[
                        solver
                    ]

                baseline_n_lol_ens =
                    baseline_n_lol_ens_by_solver[
                        solver
                    ]

                baseline_n_lol_smr_ccs =
                    baseline_n_lol_smr_ccs_by_solver[
                        solver
                    ]

                baseline_objective =
                    baseline_objective_by_solver[
                        solver
                    ]


                # ---------------------------------------------
                # STORE RESULTS
                # ---------------------------------------------

                new_results_row = (
                    case_name=case_name,
                    rp=rp,
                    solver=solver,
                    time_to_cluster=time_to_cluster,
                    time_to_read=time_to_read,
                    time_to_create=time_to_create,
                    time_to_solve=time_to_solve,
                    time_to_save=time_to_save,
                    objective_value=
                    energy_problem_benchmark.objective_value,
                    termination_status=string(
                        energy_problem_benchmark.termination_status,
                    ),
                    value_at_risk_threshold_mu_red=0.0,
                    num_constraints=
                    JuMP.num_constraints(
                        energy_problem_benchmark.model;
                        count_variable_in_set_constraints=false,
                    ),
                    num_variables=
                    JuMP.num_variables(
                        energy_problem_benchmark.model,
                    ),
                    time_to_resolve_benchmark=0.0,
                    objective_value_resolve_benchmark=0.0,
                    termination_status_resolve_benchmark="",
                    num_loss_of_load_e_demand_benchmark=
                    bm_n_lol_ens,
                    lole_e_demand_benchmark=
                    lole_e_demand,
                    num_loss_of_load_h2_demand_benchmark=
                    bm_n_lol_smr_ccs,
                    lole_h2_demand_benchmark=
                    lole_h2_demand,
                    water_borrowed_benchmark=
                    amount_water_borrowed_b,
                    value_at_risk_threshold_mu_benchmark=
                    mu_value_benchmark,
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
                    number_of_scenarios=
                    number_of_scenarios,
                )

                push!(
                    results_df,
                    new_results_row,
                )
            end
        end
    end


    # ========================================================
    # EXPORT RESULTS
    # ========================================================

    output_folder = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-output-data",
        "N$(number_of_scenarios)_seed$(seed)",
    )

    mkpath(output_folder)

    CSV.write(
        joinpath(
            output_folder,
            "results_ScSeRP_N$(number_of_scenarios)_seed$(seed).csv",
        ),
        results_df;
        writeheader=true,
    )

    return nothing
end


main()