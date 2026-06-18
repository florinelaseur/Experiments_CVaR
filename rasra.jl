# RASRA: Risk-Averse Scenario Reduction Algorithm (finetuned for mixed objectives)
#
# Based on Arpon et al. (2018) Algorithm 3, finetuned for mixed E[cost] + CVaR objectives from Nijhoff (2025)
# The original RASRA targets a pure CVaR objective, but our GEP problem minimizes (1-lambda) * E[cost] + lambda * CVaR_alpha[cost]
# Therefore we finetune by adjusting the reduced scenario probabilities to reflect both terms correctly
#
# Algorithm steps:
# 1. Backward reduction on all N scenarios to select subset J
# 2. Solve on J only to get the approximating solution x_hat
# 3. Approximate the total cost of all N scenarios at x_hat using dual variables from the J-solve
# 4. Identify tail scenarios (approximated cost >= VaR_alpha) and non-tail scenarios from J to represent the expected cost term
# 5. Assign finetuned probabilities to the reduced set (tail and non-tail scenarios) via Nijhoff dual representation
# 6. Solve the final problem on the reduced set with the adjusted probabilities

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
using Distances: Distances
using CSV: CSV
using Statistics: Statistics, quantile
using Plots
using JuMP: JuMP
using TOML: TOML
using DataFrames
using Random
using LinearAlgebra: dot

seed = parse(Int, get(ENV, "EXPERIMENT_SEED", "19990907"))
Random.seed!(seed)

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

config = TOML.parsefile("config.toml")
input_data_path = config["simulation"]["input_data"]
use_ratio = config["clustering"]["use_ratio"]
solvers = [Symbol(el) for el in config["simulation"]["solvers"]]
lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]

# Size of the initial subset J used to get x_hat (this needs to be much smaller than N for efficiency)
size_of_j = config["simulation"]["rasra_size_of_j"]

profiles_path = joinpath(@__DIR__, "create-scenarios", "profiles-wide-all-scenarios.csv")
all_profiles_df = CSV.read(profiles_path, DataFrame)
profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
selected_scenarios = sort(unique(profiles_df.scenario))
mapping = Dict(old => new for (new, old) in enumerate(selected_scenarios))
profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]
CSV.write(joinpath(input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

df_stochastic_scenario = DataFrame(;
    scenario = sort(unique(profiles_df.scenario)),
    probability = fill(1.0 / number_of_scenarios, number_of_scenarios),
)
CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

results_df = DataFrame(;
    base_name = String[],
    num_scenarios_initial = Int[],
    num_scenarios_j = Int[],
    num_scenarios_effective = Int[],
    num_scenarios_ineffective = Int[],
    num_scenarios_reduced = Int[],
    solver = Symbol[],
    time_step1_backward_reduction = Float64[],
    time_step2_solve_j = Float64[],
    time_step3_evaluate_costs = Float64[],
    time_step4_identify_effective = Float64[],
    time_step5_adjust_probabilities = Float64[],
    time_step6_solve_reduced = Float64[],
    time_to_save = Float64[],
    objective_value = Float64[],
    termination_status = String[],
    num_constraints = Int[],
    num_variables = Int[],
    value_at_risk_threshold_mu = Float64[],
    var_alpha_threshold = Float64[],
    num_loss_of_load_e_demand = Int[],
    num_loss_of_load_h2_demand = Int[],
    water_borrowed = Float64[],
)

# Helper function to set up a DuckDB connection with all input data
function setup_connection(input_data_path, lambda, alpha, use_ratio)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)

    DuckDB.query(
        conn,
        """
        UPDATE model_parameters
        SET
            risk_aversion_weight_lambda = $lambda,
            risk_aversion_confidence_level_alpha = $alpha;
        """,
    )

    if use_ratio
        DuckDB.query(
            conn,
            """
            UPDATE profiles_wide
            SET
                solar = solar / demand,
                wind_offshore = wind_offshore / demand,
                wind_onshore = wind_onshore / demand,
                hydro_inflow = hydro_inflow / demand;
            """,
        )
    end

    return conn
end

# Helper function to prepare the DuckDB connection for solving
# Applies a dummy clustering so each scenario maps to itself as its own representative period so the full temporal resolution is preserved
function prepare_connection_for_solve!(conn)
    layout = TC.ProfilesTableLayout(;
        year = :milestone_year,
        cols_to_groupby = [:milestone_year, :scenario],
    )
    TC.transform_wide_to_long!(
        conn, "profiles_wide", "profiles";
        exclude_columns = ["scenario", "milestone_year", "timestep"],
    )
    TC.dummy_cluster!(conn; layout=layout)
    TEM.populate_with_defaults!(conn)
    DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    return layout
end

# Helper function that performs the backward reduction algorithm to select a subset J of size_of_j
# It chooses scenarios that best represents the full distribution by minimizing the Kantorovich distance between the original and reduced set
function backward_reduction(profiles_df, size_of_j)
    scenario_ids = sort(unique(profiles_df.scenario))
    n = length(scenario_ids)

    profile_cols = setdiff(names(profiles_df), ["scenario", "milestone_year", "timestep"])
    scenario_vectors = Dict(
        s => vec(Matrix(profiles_df[profiles_df.scenario .== s, profile_cols]))
        for s in scenario_ids
    )
    probabilities = Dict(s => 1.0 / n for s in scenario_ids)

    # Compute pairwise distances
    dist_matrix = Dict(
        (i, j) => Distances.euclidean(scenario_vectors[i], scenario_vectors[j])
        for i in scenario_ids, j in scenario_ids if i != j
    )
    remaining = Set(scenario_ids)

    while length(remaining) > size_of_j
        removal_costs = Dict{Int, Float64}()
        nearest_neighbours = Dict{Int, Int}()

        for i in remaining
            others = setdiff(remaining, [i])
            nearest = argmin(j -> dist_matrix[(i, j)], others)
            nearest_neighbours[i] = nearest
            removal_costs[i] = probabilities[i] * dist_matrix[(i, nearest)]
        end

        # Remove the scenario with the lowest removal cost
        removed = argmin(i -> removal_costs[i], collect(remaining))
        probabilities[nearest_neighbours[removed]] += probabilities[removed]
        delete!(remaining, removed)
    end

    j_scenario_ids = sort(collect(remaining))
    final_probs = [probabilities[s] for s in j_scenario_ids]

    @info "Backward reduction selected J=$(length(j_scenario_ids)) scenarios: $j_scenario_ids"

    return j_scenario_ids, final_probs
end

# Helper function to select subset J and set up its DuckDB connection
function select_subset_j(profiles_df, size_of_j, input_data_path, lambda, alpha, use_ratio)
    # Run backward reduction to get the J scenario ids and their redistributed probs
    j_scenario_ids, j_probs = backward_reduction(profiles_df, size_of_j)

    j_profiles = filter(r -> r.scenario in j_scenario_ids, profiles_df)
    j_mapping = Dict(old => new for (new, old) in enumerate(j_scenario_ids))
    j_profiles[!, :scenario] = [j_mapping[s] for s in j_profiles.scenario]
    CSV.write(joinpath(input_data_path, "profiles-wide.csv"), j_profiles; writeheader=true)

    df_j_probs = DataFrame(;
        scenario = 1:size_of_j,
        probability = j_probs,
    )
    CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_j_probs; writeheader=true)

    conn_j = setup_connection(input_data_path, lambda, alpha, use_ratio)
    prepare_connection_for_solve!(conn_j)

    return conn_j, j_scenario_ids
end

# Helper function to approximate the second-stage cost G(x_hat, xi_i) for all N scenarios using dual variables from the J-scenario solve
# Based on Algorithm 2 of Arpon eq. (4.15)
# The approximated cost for scenario i is: approx_cost(i) = max_{j in J} dot(dual_vector[j], profile_vector[i])
function approximate_costs_via_duals(energy_problem_j, conn_j, profiles_df, j_scenario_ids, lambda)
    asset_to_col = Dict(
        "e_demand" => "demand",
        "solar" => "solar",
        "wind_offshore" => "wind_offshore",
        "wind" => "wind_onshore",
        "hydro_reservoir" => "hydro_inflow",
    )
    col_to_asset = Dict(v => k for (k, v) in asset_to_col)
    stochastic_assets = Set(keys(asset_to_col))

    profile_cols = setdiff(names(profiles_df), ["scenario", "milestone_year", "timestep"])

    # 1. Collect duals from all relevant constraint tables
    cons_table_names = [
        "cons_balance_consumer",
        "cons_capacity_outgoing_simple_method",
        "cons_balance_storage_rep_period",
        "cons_balance_conversion",
    ]
    all_duals_df = DataFrame(rep_period=Int[], asset=String[], time_block_start=Int[], dual=Float64[])

    for tbl in cons_table_names
        exists = only(DuckDB.query(conn_j,
            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = '$tbl'") |> DataFrame)[1]
        exists == 0 && continue

        dual_col_df = DuckDB.query(conn_j,
            "SELECT column_name FROM duckdb_columns()
             WHERE table_name = '$tbl' AND column_name LIKE 'dual%'") |> DataFrame
        isempty(dual_col_df) && (@warn "No dual column found in $tbl: skipping"; continue)
        dual_col = dual_col_df[1, :column_name]

        rows = DuckDB.query(conn_j,
            "SELECT asset, rep_period, time_block_start, $dual_col AS dual
             FROM $tbl
             WHERE asset IN ($(join(["'" * a * "'" for a in stochastic_assets], ", ")))
               AND $dual_col IS NOT NULL") |> DataFrame
        isempty(rows) || append!(all_duals_df, rows)
    end

    if nrow(all_duals_df) == 0
        error("""
        approximate_costs_via_duals: no balance-constraint duals found for any of $stochastic_assets.
        """)
    end

    # 2. Restrict to active_cols (profile columns that actually have duals)
    assets_with_duals = Set(unique(all_duals_df.asset))
    active_cols = filter(c -> haskey(col_to_asset, c) && col_to_asset[c] in assets_with_duals,
                         profile_cols)

    if isempty(active_cols)
        error("""
        approximate_costs_via_duals: duals found for $assets_with_duals but none
        map to a profile column. Check asset_to_col.
        Available profile columns: $profile_cols
        """)
    end

    @info "Dual approximation active profile columns: $active_cols"

    # 3. Build dual vectors for each j in J
    j_reindex = Dict(new => old for (new, old) in enumerate(j_scenario_ids))
    timesteps = sort(unique(all_duals_df.time_block_start))
    n_timesteps = length(timesteps)

    dual_vectors = Dict{Int, Vector{Float64}}()
    for j_new in sort(unique(all_duals_df.rep_period))
        j_orig = j_reindex[j_new]
        duals_j = filter(r -> r.rep_period == j_new, all_duals_df)

        dual_vec = Float64[]
        for col in active_cols
            asset = col_to_asset[col]
            duals_col = filter(r -> r.asset == asset, duals_j)
            sort!(duals_col, :time_block_start)
            if nrow(duals_col) != n_timesteps
                @warn "Expected $n_timesteps dual rows for $asset in scenario $j_orig, " *
                      "got $(nrow(duals_col)). Padding with zeros."
                padded = zeros(n_timesteps)
                padded[1:min(nrow(duals_col), n_timesteps)] .= duals_col.dual[1:min(nrow(duals_col), n_timesteps)]
                append!(dual_vec, padded)
            else
                append!(dual_vec, duals_col.dual)
            end
        end

        dual_vectors[j_orig] = dual_vec
    end

    # 4. For each scenario i in 1:N, approximate its second-stage cost as:
    # approx_cost(i) = max_{j in J} dot(dual_vector[j], profile_vector[i])
    # using active_cols only, with the same layout as the dual vectors above.
    all_scenario_ids = sort(unique(profiles_df.scenario))
    costs_df = DataFrame(scenario=Int[], operational_cost=Float64[])

    for i in all_scenario_ids
        rows_i = sort(profiles_df[profiles_df.scenario .== i, :], :timestep)
        profile_vec = vec(Matrix(rows_i[:, active_cols]))

        approx_cost = maximum(
            dot(dual_vectors[j], profile_vec) for j in j_scenario_ids
        )

        push!(costs_df, (scenario=i, operational_cost=approx_cost))
    end

    return costs_df
end

# Helper function for identifying effective scenarios (Algorithm 2 from Arpon)
# Effective scenarios are those with approximated cost >= VaR_alpha of those costs (these contribute to the tail risk)
function identify_effective_scenarios(costs_df, alpha)
    operational_costs = costs_df.operational_cost
    var_threshold = quantile(operational_costs, alpha)

    effective_mask = operational_costs .>= var_threshold
    effective_df = costs_df[effective_mask, :]

    @info "Raw cost VaR_$(alpha): $(round(var_threshold; digits=4))"
    @info "Effective scenarios: $(sum(effective_mask)) / $(length(operational_costs))"

    return effective_df, var_threshold
end

# Step 5: Assign finetuned probabilities based on tail membership.
# Tail scenarios (cost >= VaR_alpha) share total probability mass (1-alpha) equally and k non-tail representatives (k = size_of_j - num_effective) share total probability alpha equally
function compute_finetuned_probabilities(merged_df, all_costs_df, alpha, var_threshold)
    # Recompute tail membership from raw costs and the VaR threshold
    in_tail = merged_df.operational_cost .>= var_threshold

    n_tail = count(in_tail)
    n_nontail = count(.!in_tail)

    if n_tail == 0
        @warn "compute_finetuned_probabilities: no tail scenarios found"
    end
    if n_nontail == 0
        @warn "compute_finetuned_probabilities: no non-tail scenarios found"
    end

    tail_prob = n_tail > 0 ? (1.0 - alpha) / n_tail : 0.0
    nontail_prob = n_nontail > 0 ? alpha / n_nontail : 0.0

    finetuned_probs = ifelse.(in_tail, tail_prob, nontail_prob)

    result_df = copy(merged_df)
    result_df[!, :in_tail] = in_tail
    result_df[!, :finetuned_probability] = finetuned_probs

    return result_df
end

# Main RASRA function
function run_rasra()
    base_name = "RASRA"
    @info "RASRA: $number_of_scenarios initial scenarios, lambda=$lambda, alpha=$alpha"

    for solver in solvers
        optimizer, parameters = get_solver_parameters(solver)

        # Step 1: Backward reduction on all N scenarios to select subset J
        @info "Step 1 – Backward reduction: selecting J=$size_of_j from $number_of_scenarios scenarios"
        t1 = @elapsed begin
            conn_j, j_scenario_ids = select_subset_j(
                profiles_df, size_of_j, input_data_path, lambda, alpha, use_ratio,
            )
        end

        # Step 2: Solve only on J to get the approximating solution x_hat
        @info "Step 2 – Solving on J=$size_of_j scenarios to get x_hat"
        t2 = @elapsed begin
            energy_problem_j = TEM.EnergyProblem(conn_j)

            TEM.create_model!(
                energy_problem_j;
                optimizer = optimizer,
                optimizer_parameters = parameters,
                model_file_name = "",
                enable_names = true,
                direct_model = false,
            )
            TEM.solve_model!(energy_problem_j)
        end

        if string(energy_problem_j.termination_status) != "OPTIMAL"
            @warn "J-subset solve did not reach optimality: $(energy_problem_j.termination_status)"
        end

        # Step 3: Approximate the total cost for all N at x_hat using duals from J
        @info "Step 3 – Approximating the total cost of all $number_of_scenarios scenarios at x_hat using dual variables from the solve on subset J"
        t3 = @elapsed begin
            TEM.save_solution!(energy_problem_j; compute_duals = true)
            costs_df = approximate_costs_via_duals(
                energy_problem_j, conn_j, profiles_df, j_scenario_ids, lambda,
            )
        end

        # Step 4: Identify effective (tail) scenarios from all N, and select k (k = size_of_j - num_effective) representative non-tail scenarios from all N.
        # Representatives are chosen evenly spaced in the sorted non-tail because in this case they all have equal probabilities
        @info "Step 4 – Identifying effective scenarios and selecting representative non-tail from all N"
        t4 = @elapsed begin
            effective_df, var_threshold = identify_effective_scenarios(costs_df, alpha)
            num_effective = nrow(effective_df)

            all_nontail_df = filter(r -> r.scenario ∉ effective_df.scenario, costs_df)
            sorted_nontail = sort(all_nontail_df, :operational_cost)
            n_nontail_all = nrow(sorted_nontail)
            k = max(1, size_of_j - num_effective)

            # Pick k indices evenly spread across 1:n_nontail_all
            # Each representative stands for an equal share of the non-tail distribution.
            rep_indices = [round(Int, (i - 0.5) * n_nontail_all / k) + 1 for i in 1:k]
            rep_indices = clamp.(rep_indices, 1, n_nontail_all)
            representative_nontail_df = sorted_nontail[rep_indices, :]
            num_ineffective = nrow(representative_nontail_df)

            merged_df = vcat(effective_df, representative_nontail_df)
            num_reduced = nrow(merged_df)
        end

        # Step 5: Assign finetuned probabilities
        @info "Step 5 – Computing finetuned probabilities for $num_effective tail and $num_ineffective non-tail scenarios"
        t5 = @elapsed begin
            reduced_with_probs = compute_finetuned_probabilities(
                merged_df, costs_df, alpha, var_threshold,
            )

            # Build the reduced profile and probability tables for Tulipa
            reduced_scenario_ids = reduced_with_probs.scenario
            reduced_profiles = filter(r -> r.scenario in reduced_scenario_ids, profiles_df)

            reduced_sorted = sort(unique(reduced_profiles.scenario))
            reduced_mapping = Dict(old => new for (new, old) in enumerate(reduced_sorted))
            reduced_profiles[!, :scenario] = [reduced_mapping[s] for s in reduced_profiles.scenario]

            # Map finetuned probabilities to the new scenario indices
            old_to_prob = Dict(
                row.scenario => row.finetuned_probability for row in eachrow(reduced_with_probs)
            )
            new_probs = [old_to_prob[s] for s in sort(unique(reduced_with_probs.scenario))]

            df_reduced_scenario = DataFrame(;
                scenario = 1:num_reduced,
                probability = new_probs,
            )

            CSV.write(joinpath(input_data_path, "profiles-wide.csv"), reduced_profiles; writeheader=true)
            CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_reduced_scenario; writeheader=true)
        end

        # Step 6: Solve the final reduced problem with adjusted probabilities
        @info "Step 6 – Solving final problem on $num_reduced reduced scenarios"
        t6 = @elapsed begin
            conn_reduced = setup_connection(input_data_path, lambda, alpha, use_ratio)
            prepare_connection_for_solve!(conn_reduced)
            energy_problem_reduced = TEM.EnergyProblem(conn_reduced)

            TEM.create_model!(
                energy_problem_reduced;
                optimizer = optimizer,
                optimizer_parameters = parameters,
                model_file_name = "",
                enable_names = true,
                direct_model = false,
            )
            TEM.solve_model!(energy_problem_reduced)
        end

        output_folder = joinpath(@__DIR__, "outputs", "N$(number_of_scenarios)_seed$(seed)", base_name, string(solver))
        mkpath(output_folder)

        t_save = @elapsed begin
            TEM.save_solution!(energy_problem_reduced)
            TEM.export_solution_to_csv_files(output_folder, energy_problem_reduced)
            df_cost_per_scenario = export_operational_cost_per_scenario(energy_problem_reduced, output_folder)
            plot_operational_cost_per_scenario(df_cost_per_scenario, output_folder)
        end

        var_flow_df = TIO.get_table(conn_reduced, "var_flow")
        flow_ens = filter(r -> r.from_asset == "ens" && r.to_asset == "e_demand", var_flow_df)
        flow_smr_ccs = filter(r -> r.from_asset == "smr_ccs" && r.to_asset == "h2_demand", var_flow_df)
        water_borrowed = filter(r -> r.from_asset == "water_borrower" && r.to_asset == "hydro_reservoir", var_flow_df)

        n_lol_ens = count(r -> r.solution > 0.0, eachrow(flow_ens))
        n_lol_smr_ccs = count(r -> r.solution > 0.0, eachrow(flow_smr_ccs))
        amount_water = sum(water_borrowed.solution)

        mu_value_df = TIO.get_table(conn_reduced, "var_value_at_risk_threshold_mu")
        mu_value = nrow(mu_value_df) == 0 ? NaN : only(mu_value_df.solution)

        push!(results_df, (
            base_name = base_name,
            num_scenarios_initial = number_of_scenarios,
            num_scenarios_j = size_of_j,
            num_scenarios_effective = num_effective,
            num_scenarios_ineffective = num_ineffective,
            num_scenarios_reduced = num_reduced,
            solver = solver,
            time_step1_backward_reduction = t1,
            time_step2_solve_j = t2,
            time_step3_evaluate_costs = t3,
            time_step4_identify_effective = t4,
            time_step5_adjust_probabilities = t5,
            time_step6_solve_reduced = t6,
            time_to_save = t_save,
            objective_value = energy_problem_reduced.objective_value,
            termination_status = string(energy_problem_reduced.termination_status),
            num_constraints = JuMP.num_constraints(
                energy_problem_reduced.model; count_variable_in_set_constraints=false,
            ),
            num_variables = JuMP.num_variables(energy_problem_reduced.model),
            value_at_risk_threshold_mu = mu_value,
            var_alpha_threshold = var_threshold,
            num_loss_of_load_e_demand = n_lol_ens,
            num_loss_of_load_h2_demand = n_lol_smr_ccs,
            water_borrowed = amount_water,
        ))

        @info "RASRA done - solver: $solver | effective scenarios: $num_effective | reduced scenarios: $num_reduced | objective: $(energy_problem_reduced.objective_value)"
    end

    rasra_results_path = "outputs/results_rasra_N$(number_of_scenarios)_seed$(seed).csv"
    results_df |> CSV.write(rasra_results_path; writeheader=true)
    @info "Results written to $rasra_results_path"

    return results_df
end

results = Base.invokelatest(run_rasra)