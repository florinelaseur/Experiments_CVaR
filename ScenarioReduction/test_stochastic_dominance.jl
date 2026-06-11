#  Main script to test stochastic dominance scenario screening.
#
# REPL usage (from repo root — path is stable across re-includes):
#   include(joinpath(@__DIR__, "ScenarioReduction", "test_stochastic_dominance.jl"))
#   run!()
#
# Or run as a script:
#   julia --project=. ScenarioReduction/test_stochastic_dominance.jl
#
# Settings: ScenarioReduction/config.toml

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = joinpath(SCRIPT_DIR, "..")

using Pkg: Pkg
Pkg.activate(REPO_ROOT)
Pkg.instantiate()

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using Distances: Distances
using CSV: CSV
using JuMP: JuMP
using JSON: JSON
using Random
using DataFrames

include(joinpath(REPO_ROOT, "utils", "functions.jl"))
include(joinpath(REPO_ROOT, "utils", "constants.jl"))
include(joinpath(SCRIPT_DIR, "src", "utils.jl"))
include(joinpath(SCRIPT_DIR, "src", "stochastic_dominance.jl"))

# Covariance matrix derived from solving single scenario models to optimality.
const INV_COV = [
    3.78479695199275e7 2.749005459752808e6 -3.4577296873559463e6 2.7418219047064386e7 298344.2313214293 5.651907423141254 407842.3222137223
    2.749005459752808e6 1.4501887858885615e6 1.7561773638240807e6 9.315492633408496e6 -13717.616779138572 0.9189337918286634 -374638.69889195403
    -3.4577296873559463e6 1.7561773638240807e6 2.854556209827236e7 1.0685293361151338e7 -169352.5029630032 2.6711278243126007 635237.786307628
    2.7418219047064386e7 9.315492633408496e6 1.0685293361151338e7 1.7639204795704246e8 75078.97973361365 5.368008683366009 7.997575021518203e6
    298344.2313214293 -13717.616779138572 -169352.5029630032 75078.97973361365 151423.08336402554 0.04797874928315783 79097.28020948995
    5.651907423141254 0.9189337918286634 2.6711278243126007 5.368008683366009 0.04797874928315783 2.691847694694756e-5 -0.3741512568820247
    407842.3222137223 -374638.69889195403 635237.786307628 7.997575021518203e6 79097.28020948995 -0.3741512568820247 3.7310673190848944e6
]

# Prepare teh datat for the experiment. Mimics the main.jl script. 
function prepare_scenario_input!(cfg::ScenarioReductionConfig)
    all_profiles_df = CSV.read(cfg.profiles_wide_source, DataFrame)
    profiles_df = get_scenario_set(all_profiles_df, cfg.number_of_scenarios)
    # Original source ids picked for this seed, captured BEFORE renumbering to 1..N.
    source_ids = sort(unique(profiles_df.scenario))
    mapping = Dict(old => new for (new, old) in enumerate(source_ids))
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    CSV.write(joinpath(cfg.input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

    df_stochastic_scenario = DataFrame(;
        scenario=sort(unique(profiles_df.scenario)),
        probability=fill(1.0 / cfg.number_of_scenarios, cfg.number_of_scenarios),
    )
    CSV.write(joinpath(cfg.input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

    return source_ids
end

#connection and profiles preparation
function setup_connection(cfg::ScenarioReductionConfig)
    connection = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection, cfg.input_data_path)

    DuckDB.query(
        connection,
        """
        UPDATE model_parameters
        SET
            risk_aversion_weight_lambda = $(cfg.lambda),
            risk_aversion_confidence_level_alpha = $(cfg.alpha);
        """,
    )

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

    return connection
end

function prepare_stochastic_dominance_indices!(connection)
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    TC.dummy_cluster!(connection; layout=layout)
    TEM.populate_with_defaults!(connection)
    TEM.create_internal_tables!(connection)
    return TEM.compute_variables_indices(connection)
end

"""
verify_investment_fix_mapping!(connection; tol=1e-8)

Integration smoke test: fix sentinel MW values per asset and read back model-unit
fix values via the indices-aligned path. Returns the audit result.
"""
function verify_investment_fix_mapping!(connection, cfg::ScenarioReductionConfig; tol=cfg.verify_mapping_tol)
    variables = prepare_stochastic_dominance_indices!(connection)
    capacity_lookup = build_capacity_lookup(connection)
    audit = audit_investment_mapping(variables; capacity_lookup)

    #create the model
    constraints = TEM.compute_constraints_indices(connection)
    profiles = TEM.prepare_profiles_structure(connection)
    model, _ = TEM.create_model(connection, variables, constraints, profiles)
    JuMP.set_optimizer(model, HiGHS.Optimizer)
    JuMP.set_silent(model)

    sample_mw = Float64[1000, 2000, 3000, 4000, 5000, 6000, 7000]
    fix_variables_from_sample(
        variables,
        :assets_investment,
        sample_mw;
        capacity_lookup,
    )

    inv_df = DataFrame(variables[:assets_investment].indices)
    container = variables[:assets_investment].container
    sample_by_asset = sample_vector_by_asset(sample_mw)

    for (i, row) in enumerate(eachrow(inv_df))
        asset = string(row.asset)
        expected = sample_by_asset[asset] / capacity_lookup[asset]
        actual = JuMP.fix_value(container[i])
        @info "Fix verification" index=i asset=asset expected=expected actual=actual
        abs(actual - expected) > tol &&
            error("Fix mismatch at index $i ($asset): got $actual, expected $expected")
    end

    @info "Investment fix mapping verification passed" (
        permutation_ok=audit.permutation_ok,
        n_mismatches=length(audit.mismatches),
    )
    println("Value fixing validated (indices-based alignment + capacity lookup).")
    return audit
end

function run_filter!(connection, cfg::ScenarioReductionConfig)
    @info "Running stochastic-dominance filtering"
    bounds = load_investment_bounds()
    cov = investment_covariance()
    #cov = INV_COV
    return stochastic_dominance(connection, cfg; bounds, covariance=cov)
end

function run_solve!(connection, cfg::ScenarioReductionConfig)
    @info "Clustering with $(cfg.representative_periods) representative periods (per scenario)"
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    clustering_kwargs = Dict(
        :learning_rate => cfg.clustering_learning_rate,
        :niters => cfg.clustering_niters,
    )
    weight_fitting_kwargs = Dict(
        :learning_rate => cfg.weight_learning_rate,
        :niters => cfg.weight_niters,
    )

    # Swap TC.dummy_cluster!(connection; layout=layout) for a full-hourly smoke test.
    time_to_cluster = @elapsed TC.cluster!(
        connection,
        cfg.hours_per_period,
        cfg.representative_periods;
        method=cfg.clustering_method,
        distance=Distances.Euclidean(),
        weight_type=cfg.weight_type,
        layout=layout,
        clustering_kwargs,
        weight_fitting_kwargs,
    )

    TEM.populate_with_defaults!(connection)
    DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")

    results = DataFrame(;
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
        value_at_risk_threshold_mu=Float64[],
    )

    for solver in cfg.solvers
        optimizer, parameters = get_solver_parameters(solver)

        @info "Creating model (stochastic_dominance, rp=$(cfg.representative_periods)) with $solver"
        time_to_read = @elapsed energy_problem = TEM.EnergyProblem(connection)
        time_to_create = @elapsed TEM.create_model!(
            energy_problem;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=cfg.use_names,
        )

        output_folder = joinpath(cfg.output_dir, "stochastic_dominance", string(solver))
        mkpath(output_folder)

        @info "Solving model with $solver"
        time_to_solve = @elapsed TEM.solve_model!(energy_problem)
        time_to_save = @elapsed begin
            TEM.save_solution!(energy_problem)
            TEM.export_solution_to_csv_files(output_folder, energy_problem)
        end

        mu_value_df = TIO.get_table(connection, "var_value_at_risk_threshold_mu")
        mu_value = only(mu_value_df.solution)

        push!(
            results,
            (
                "stochastic_dominance",
                cfg.representative_periods,
                solver,
                time_to_cluster,
                time_to_read,
                time_to_create,
                time_to_solve,
                time_to_save,
                energy_problem.objective_value,
                string(energy_problem.termination_status),
                JuMP.num_constraints(energy_problem.model; count_variable_in_set_constraints=false),
                JuMP.num_variables(energy_problem.model),
                mu_value,
            ),
        )
    end

    return results
end

function run!()
    cfg = load_config(script_dir=SCRIPT_DIR, repo_root=REPO_ROOT)

    if cfg.tulipa_energy_model_rev !== nothing
        Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev=cfg.tulipa_energy_model_rev)
    end

    Random.seed!(cfg.random_seed)

    prepare_scenario_input!(cfg)
    connection = setup_connection(cfg)

    if cfg.run_verify_mapping
        verify_investment_fix_mapping!(connection, cfg)
    elseif cfg.run_filter
        @warn "Skipping value-fix validation (run.verify_mapping=false); starting screening"
    end

    if cfg.run_filter
        run_filter!(connection, cfg)
    end

    if cfg.run_solve
        results = run_solve!(connection, cfg)
        mkpath(cfg.output_dir)
        results_path = joinpath(cfg.output_dir, "results.csv")
        CSV.write(results_path, results; writeheader=true)
        @info "Results saved to $results_path"
    end

    return connection
end

# Repeatable, multi-seed experiment driver.
#
# For each run: pick a fresh random scenario subset (seed = base_seed + run-1),
# run stochastic-dominance screening (artifacts under runK/screening/), then
# optionally solve at full temporal resolution via TC.dummy_cluster!:
#   experiment.run_ground_truth=true  → full sampled set ("all") under full_resolution_all/
#   experiment.run_selected_scenarios=true → pointwise-undominated set under
#     full_resolution_undominated/ (skipped when undominated equals the full set).
# Screening always runs; set either flag false to skip that post-screening solve.
function run_experiment!()
    cfg = load_config(script_dir=SCRIPT_DIR, repo_root=REPO_ROOT)

    if cfg.tulipa_energy_model_rev !== nothing
        Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev=cfg.tulipa_energy_model_rev)
    end

    base = joinpath(cfg.experiment_base_dir, cfg.experiment_name)
    mkpath(base)
    @info "Starting experiment" name=cfg.experiment_name num_runs=cfg.experiment_num_runs base=base

    summary = DataFrame(;
        run=Int[],
        seed=Int[],
        label=String[],
        solver=Symbol[],
        num_scenarios=Int[],
        time_to_cluster=Float64[],
        time_to_read=Float64[],
        time_to_create=Float64[],
        time_to_solve=Float64[],
        time_to_save=Float64[],
        objective_value=Float64[],
        termination_status=String[],
        num_constraints=Int[],
        num_variables=Int[],
        num_loss_of_load_e_demand=Int[],
        num_loss_of_load_h2_demand=Int[],
        water_borrowed=Float64[],
        value_at_risk_threshold_mu=Float64[],
    )
    summary_path = joinpath(base, "summary.csv")

    # Record which source scenarios were chosen per run/seed, for reproduction.
    selection = DataFrame(;
        run=Int[],
        seed=Int[],
        num_scenarios=Int[],
        selected_source_scenarios=String[],
    )
    selection_path = joinpath(base, "scenario_selection.csv")

    for run_idx in 1:cfg.experiment_num_runs
        seed = cfg.experiment_base_seed + (run_idx - 1)
        run_dir = joinpath(base, "run$(run_idx)")
        screening_dir = joinpath(run_dir, "screening")
        mkpath(screening_dir)

        run_cfg = copy_config_with(
            cfg;
            output_dir=screening_dir,
            conflict_log_path=joinpath(screening_dir, "infeasibility_conflicts.jsonl"),
            number_of_scenarios=cfg.experiment_num_input_scenarios,
            random_seed=seed,
        )
        save_config_toml(run_cfg, joinpath(run_dir, "config_used.toml"))

        @info "=== Experiment run $run_idx/$(cfg.experiment_num_runs) (seed=$seed) ==="
        Random.seed!(seed)
        source_ids = prepare_scenario_input!(run_cfg)
        # Persist the selected source scenarios BEFORE any solve (does not affect the solve).
        push!(selection, (run_idx, seed, length(source_ids), format_selected_scenarios(source_ids)))
        CSV.write(selection_path, selection; writeheader=true)
        @info "Selected source scenarios" run=run_idx seed=seed scenarios=source_ids

        connection = setup_connection(run_cfg)

        if run_cfg.run_verify_mapping
            verify_investment_fix_mapping!(connection, run_cfg)
        else
            @warn "Skipping value-fix validation (run.verify_mapping=false)"
        end

        sd = run_filter!(connection, run_cfg)
        try
            DuckDB.DBInterface.close!(connection)
        catch err
            @debug "Could not close screening connection" err
        end

        undominated = sort(Int.(sd.scenario_dominance.undominated))
        all_scenarios = sort(Int.(sd.scenarios))
        @info "Screening result" run=run_idx scenarios=all_scenarios undominated=undominated

        if cfg.experiment_run_ground_truth
            res_all = solve_full_resolution!(
                sd.scenarios,
                run_cfg.input_data_path,
                joinpath(run_dir, "full_resolution_all");
                label="all",
                solvers=run_cfg.solvers,
                lambda=run_cfg.lambda,
                alpha=run_cfg.alpha,
                use_names=run_cfg.use_names,
            )
            for row in eachrow(res_all.results)
                push!(summary, (; run=run_idx, seed=seed, pairs(row)...))
            end
        else
            @info "Run $run_idx: skipping ground-truth solve (experiment.run_ground_truth=false)"
        end

        if !cfg.experiment_run_selected_scenarios
            @info "Run $run_idx: skipping selected-scenarios solve (experiment.run_selected_scenarios=false)"
        elseif isempty(undominated) || undominated == all_scenarios
            @info "Run $run_idx: undominated set equals full set (no reduction); skipping redundant undominated solve" undominated =
                undominated
        else
            @info "Undominated-set solve" run = run_idx undominated = undominated K =
                length(undominated) note =
                "downstream CSVs (tail_scenarios.csv) use LOCAL ids 1..K of these undominated scenarios"
            res_und = solve_full_resolution!(
                undominated,
                run_cfg.input_data_path,
                joinpath(run_dir, "full_resolution_undominated");
                label="undominated",
                solvers=run_cfg.solvers,
                lambda=run_cfg.lambda,
                alpha=run_cfg.alpha,
                use_names=run_cfg.use_names,
            )
            for row in eachrow(res_und.results)
                push!(summary, (; run=run_idx, seed=seed, pairs(row)...))
            end
        end

        CSV.write(summary_path, summary; writeheader=true)
        @info "Run $run_idx complete; summary updated at $summary_path"
    end

    @info "Experiment complete. Summary saved to $summary_path"
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    _cfg = load_config(script_dir=SCRIPT_DIR, repo_root=REPO_ROOT)
    if _cfg.experiment_enabled
        run_experiment!()
    else
        run!()
    end
end
