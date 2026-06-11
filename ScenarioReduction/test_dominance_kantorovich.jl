# Hybrid dominance + Kantorovich scenario-reduction experiment.
#
# Per run (seed = hybrid_experiment.base_seed + run - 1):
#   1. Sample N input scenarios (renumbered 1..N) and run dominance screening
#      (dominance_screening Phases A/B/C; artifacts under run<K>/screening/).
#   2. Pick k = dominance.dominance_pick_k scenarios by dominance peeling
#      (pick_n_scenarios with dominance.dominance_method).
#   3. Pick n = hybrid_experiment.kantorovich_pick_n scenarios by Kantorovich
#      forward selection on the profile features of the COMPLEMENT of the
#      dominance picks (the two sets are disjoint by construction).
#   4. Solve the reduced model on the k+n scenarios → run<K>/reduced_solve/.
#      Probabilities are uniform 1/(k+n) (prepare_scenario_subset!); Kantorovich's
#      redistributed probabilities are recorded in kantorovich_pick.csv as
#      diagnostics only.
#   5. Fix the reduced solve's investments (var_assets_investment.csv → MW) in the
#      full N-scenario model and re-solve → run<K>/full_fixed/. Out-of-sample
#      evaluation; can be INFEASIBLE — recorded in results.csv, not fatal.
#   6. Solve the full N-scenario benchmark (no fixed investments)
#      → run<K>/full_benchmark/. full_fixed vs full_benchmark gives the
#      out-of-sample optimality gap (optimality_gap.csv).
#
# Stages are resumable: a solve stage is skipped when <stage>/<solver>/results.csv
# exists; screening is skipped when screening/cost_matrix.csv exists. Runs are
# seed-deterministic, so resuming reproduces identical scenario selections.
#
# REPL usage (from repo root — path is stable across re-includes):
#   include(joinpath(@__DIR__, "ScenarioReduction", "test_dominance_kantorovich.jl"))
#   run!()                      # single run (run1 only)
#   run_hybrid_experiment!()    # hybrid_experiment.num_runs runs
#
# Or run as a script (requires hybrid_experiment.enabled=true):
#   julia --project=. ScenarioReduction/test_dominance_kantorovich.jl
#
# Settings: ScenarioReduction/config.toml ([dominance] + [hybrid_experiment])

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
include(joinpath(REPO_ROOT, "utils", "kantorovich_reduction.jl"))
include(joinpath(SCRIPT_DIR, "src", "utils.jl"))
include(joinpath(SCRIPT_DIR, "src", "dominance.jl"))
include(joinpath(SCRIPT_DIR, "src", "experiment_common.jl"))

ids_to_str(ids) = join(ids, " ")

# Map LOCAL scenario ids (1..N after renumbering) back to this run's SOURCE ids.
map_local_to_source(local_ids, source_ids_sorted) =
    Int[Int(source_ids_sorted[i]) for i in local_ids]

stage_complete(stage_dir, solvers) =
    all(isfile(joinpath(stage_dir, string(s), "results.csv")) for s in solvers)

function hybrid_summary_dataframe()
    return DataFrame(;
        run=Int[],
        seed=Int[],
        label=String[],
        dominance_method=String[],
        k=Int[],
        n=Int[],
        dominance_picked=String[],
        kantorovich_picked=String[],
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
end

function optimality_gap_dataframe()
    return DataFrame(;
        run=Int[],
        seed=Int[],
        solver=Symbol[],
        F_full_fixed=Float64[],
        F_full_benchmark=Float64[],
        og_percent=Float64[],
        fixed_status=String[],
        benchmark_status=String[],
        dominance_method=String[],
        k=Int[],
        n=Int[],
        reduced_source=String[],
    )
end

# Append the per-solver results.csv rows of one solve stage to the experiment
# summary. Reading back from disk gives a single code path for stages solved in
# this process and stages skipped as already solved (resume).
function append_stage_rows!(summary, stage_dir, run_idx, seed, meta, solvers)
    for s in solvers
        path = joinpath(stage_dir, string(s), "results.csv")
        if !isfile(path)
            @warn "No results.csv for stage; summary row skipped" stage = stage_dir solver = s
            continue
        end
        df = CSV.read(path, DataFrame)
        for r in eachrow(df)
            push!(
                summary,
                (
                    run=run_idx,
                    seed=seed,
                    label=String(r.label),
                    dominance_method=meta.dominance_method,
                    k=meta.k,
                    n=meta.n,
                    dominance_picked=meta.dominance_picked,
                    kantorovich_picked=meta.kantorovich_picked,
                    solver=Symbol(r.solver),
                    num_scenarios=Int(r.num_scenarios),
                    time_to_cluster=Float64(r.time_to_cluster),
                    time_to_read=Float64(r.time_to_read),
                    time_to_create=Float64(r.time_to_create),
                    time_to_solve=Float64(r.time_to_solve),
                    time_to_save=Float64(r.time_to_save),
                    objective_value=Float64(r.objective_value),
                    termination_status=String(r.termination_status),
                    num_constraints=Int(r.num_constraints),
                    num_variables=Int(r.num_variables),
                    num_loss_of_load_e_demand=Int(r.num_loss_of_load_e_demand),
                    num_loss_of_load_h2_demand=Int(r.num_loss_of_load_h2_demand),
                    water_borrowed=Float64(r.water_borrowed),
                    value_at_risk_threshold_mu=Float64(r.value_at_risk_threshold_mu),
                ),
            )
        end
    end
    return summary
end

function read_objective_status(path)
    isfile(path) || return (NaN, "MISSING")
    df = CSV.read(path, DataFrame)
    isempty(df) && return (NaN, "MISSING")
    obj = hasproperty(df, :objective_value) ? Float64(df.objective_value[1]) : NaN
    status = hasproperty(df, :termination_status) ? String(df.termination_status[1]) : "UNKNOWN"
    return (obj, status)
end

# Out-of-sample optimality gap per solver:
#   OG(%) = (F(z^reduced, full) − F(z^full, full)) / F(z^full, full) × 100
function append_og_rows!(og, run_idx, seed, run_dir, solvers, meta, reduced_source)
    for s in solvers
        F_fixed, fixed_status =
            read_objective_status(joinpath(run_dir, "full_fixed", string(s), "results.csv"))
        F_bench, bench_status =
            read_objective_status(joinpath(run_dir, "full_benchmark", string(s), "results.csv"))
        og_percent =
            (isnan(F_fixed) || isnan(F_bench)) ? NaN : (F_fixed - F_bench) / F_bench * 100
        push!(
            og,
            (
                run=run_idx,
                seed=seed,
                solver=Symbol(s),
                F_full_fixed=F_fixed,
                F_full_benchmark=F_bench,
                og_percent=og_percent,
                fixed_status=fixed_status,
                benchmark_status=bench_status,
                dominance_method=meta.dominance_method,
                k=meta.k,
                n=meta.n,
                reduced_source=ids_to_str(reduced_source),
            ),
        )
    end
    return og
end

function run_hybrid_run!(cfg, run_idx, base, summary, selection, selection_path, og)
    seed = cfg.hybrid_base_seed + (run_idx - 1)
    run_dir = joinpath(base, "run$(run_idx)")
    screening_dir = joinpath(run_dir, "screening")
    mkpath(screening_dir)

    run_cfg = copy_config_with(
        cfg;
        output_dir=screening_dir,
        conflict_log_path=joinpath(screening_dir, "infeasibility_conflicts.jsonl"),
        number_of_scenarios=cfg.hybrid_num_input_scenarios,
        random_seed=seed,
    )
    save_config_toml(run_cfg, joinpath(run_dir, "config_used.toml"))

    N = run_cfg.number_of_scenarios
    k_req = run_cfg.dominance_pick_k
    n_kant = run_cfg.hybrid_kantorovich_pick_n
    k_req >= 1 || error("dominance.dominance_pick_k must be >= 1, got $k_req")
    n_kant >= 0 || error("hybrid_experiment.kantorovich_pick_n must be >= 0, got $n_kant")
    k_req + n_kant <= N || error(
        "dominance_pick_k + kantorovich_pick_n = $(k_req + n_kant) exceeds num_input_scenarios = $N",
    )

    @info "=== Hybrid run $run_idx/$(cfg.hybrid_num_runs) (seed=$seed) ===" N = N method =
        run_cfg.dominance_method k = k_req n = n_kant

    # Step 0: sample + renumber the N input scenarios. Also restores the full
    # N-scenario profiles-wide.csv when resuming a partially solved run.
    Random.seed!(seed)
    source_ids = prepare_scenario_input!(run_cfg)
    push!(selection, (run_idx, seed, length(source_ids), format_selected_scenarios(source_ids)))
    CSV.write(selection_path, selection; writeheader=true)
    @info "Selected source scenarios" run = run_idx seed = seed scenarios = source_ids

    # Step 1: dominance screening (skipped when cost_matrix.csv already exists).
    cost_matrix_path = joinpath(screening_dir, "cost_matrix.csv")
    screening_status = "SCREENING_OK"
    screening_elapsed = NaN
    if isfile(cost_matrix_path)
        screening_status = "SCREENING_SKIPPED"
        @info "Run $run_idx: screening skipped (cost_matrix.csv exists)" path = cost_matrix_path
    else
        @info "Running dominance screening" run = run_idx method = run_cfg.dominance_method
        screening_elapsed = @elapsed begin
            connection = setup_connection(run_cfg)
            try
                dominance_screening(
                    connection,
                    run_cfg;
                    bounds=load_investment_bounds(),
                    covariance=investment_covariance(),
                )
            finally
                try
                    DuckDB.DBInterface.close!(connection)
                catch err
                    @debug "Could not close screening connection" err
                end
            end
        end
    end

    # Step 2: pick k scenarios by dominance peeling on the cost matrix.
    cost_df = CSV.read(cost_matrix_path, DataFrame)
    pick = pick_n_scenarios(cost_df; n=k_req, method=run_cfg.dominance_method)
    if length(pick.picked) < k_req
        @warn "Dominance peel exhausted before reaching k; using all picked scenarios" requested =
            k_req picked = length(pick.picked)
    end
    dom_local = sort(Int.(pick.picked[1:min(k_req, length(pick.picked))]))
    dom_source = map_local_to_source(dom_local, source_ids)
    @info "Dominance pick" run = run_idx method = run_cfg.dominance_method satisfied =
        pick.satisfied peel_rounds = length(pick.rounds) picked_local = dom_local picked_source =
        dom_source

    CSV.write(
        joinpath(run_dir, "dominance_pick.csv"),
        DataFrame(;
            run=run_idx,
            seed=seed,
            method=string(run_cfg.dominance_method),
            k_requested=k_req,
            satisfied=pick.satisfied,
            peel_rounds=length(pick.rounds),
            picked_order_local=ids_to_str(Int.(pick.picked)),
            selected_local=ids_to_str(dom_local),
            selected_source=ids_to_str(dom_source),
        ),
    )

    # Step 3: pick n scenarios by Kantorovich on the complement of the dominance
    # picks (LOCAL ids; profiles-wide.csv holds the full N from step 0).
    profiles_df = CSV.read(joinpath(run_cfg.input_data_path, "profiles-wide.csv"), DataFrame)
    kant_selected, kant_probs =
        select_kantorovich_scenarios(profiles_df, n_kant; exclude=dom_local)
    kant_local = sort(Int.(kant_selected))
    kant_source = map_local_to_source(kant_local, source_ids)
    @info "Kantorovich pick" run = run_idx n = n_kant picked_local = kant_local picked_source =
        kant_source

    CSV.write(
        joinpath(run_dir, "kantorovich_pick.csv"),
        DataFrame(;
            run=run_idx,
            seed=seed,
            n_requested=n_kant,
            picked_order_local=ids_to_str(Int.(kant_selected)),
            selected_local=ids_to_str(kant_local),
            selected_source=ids_to_str(kant_source),
            # Diagnostic only — the reduced solve uses uniform 1/(k+n) probabilities.
            kantorovich_probs=join(round.(kant_probs; digits=6), " "),
            excluded_local=ids_to_str(dom_local),
        ),
    )

    reduced_local = sort(vcat(dom_local, kant_local))
    @assert allunique(reduced_local) "dominance and Kantorovich picks overlap: $reduced_local"
    @assert length(reduced_local) == length(dom_local) + length(kant_local)
    reduced_source = map_local_to_source(reduced_local, source_ids)
    @info "Reduced scenario set" run = run_idx reduced_local = reduced_local reduced_source =
        reduced_source

    meta = (
        dominance_method=string(run_cfg.dominance_method),
        k=k_req,
        n=n_kant,
        dominance_picked=ids_to_str(dom_source),
        kantorovich_picked=ids_to_str(kant_source),
    )

    push!(
        summary,
        (;
            run=run_idx,
            seed=seed,
            label="screening",
            meta...,
            solver=run_cfg.sd_solver,
            num_scenarios=N,
            time_to_cluster=0.0,
            time_to_read=0.0,
            time_to_create=0.0,
            time_to_solve=screening_elapsed,
            time_to_save=0.0,
            objective_value=NaN,
            termination_status=screening_status,
            num_constraints=0,
            num_variables=0,
            num_loss_of_load_e_demand=0,
            num_loss_of_load_h2_demand=0,
            water_borrowed=NaN,
            value_at_risk_threshold_mu=NaN,
        ),
    )

    # Step 4: solve the reduced k+n model (uniform 1/(k+n) probabilities).
    reduced_dir = joinpath(run_dir, "reduced_solve")
    if stage_complete(reduced_dir, run_cfg.solvers)
        @info "Run $run_idx: reduced solve skipped (results.csv exists)" dir = reduced_dir
    else
        solve_full_resolution!(
            reduced_local,
            run_cfg.input_data_path,
            reduced_dir;
            label="dom_kant_reduced",
            solvers=run_cfg.solvers,
            lambda=run_cfg.lambda,
            alpha=run_cfg.alpha,
            use_names=run_cfg.use_names,
        )
    end
    append_stage_rows!(summary, reduced_dir, run_idx, seed, meta, run_cfg.solvers)

    # Step 5: fix the reduced investments in the full N-scenario model and re-solve.
    fixed_dir = joinpath(run_dir, "full_fixed")
    if cfg.hybrid_run_fixed_full
        # The reduced solve overwrote profiles-wide.csv with the k+n subset →
        # restore the full N-scenario input (seed-deterministic).
        Random.seed!(seed)
        prepare_scenario_input!(run_cfg)
        for s in run_cfg.solvers
            if isfile(joinpath(fixed_dir, string(s), "results.csv"))
                @info "Run $run_idx: full_fixed skipped for $s (results.csv exists)"
                continue
            end
            mw = read_investment_mw(joinpath(reduced_dir, string(s)))
            if mw === nothing
                @warn "Run $run_idx: no usable var_assets_investment.csv in reduced solve; skipping full_fixed" solver =
                    s
                continue
            end
            solve_full_resolution!(
                1:N,
                run_cfg.input_data_path,
                fixed_dir;
                label="full_fixed",
                solvers=[s],
                lambda=run_cfg.lambda,
                alpha=run_cfg.alpha,
                use_names=run_cfg.use_names,
                fix_investment_mw=mw,
            )
        end
        append_stage_rows!(summary, fixed_dir, run_idx, seed, meta, run_cfg.solvers)
    else
        @info "Run $run_idx: skipping full_fixed (hybrid_experiment.run_fixed_full=false)"
    end

    # Step 6: full N-scenario benchmark (no fixed investments).
    bench_dir = joinpath(run_dir, "full_benchmark")
    if cfg.hybrid_run_full_benchmark
        if stage_complete(bench_dir, run_cfg.solvers)
            @info "Run $run_idx: full benchmark skipped (results.csv exists)" dir = bench_dir
        else
            Random.seed!(seed)
            prepare_scenario_input!(run_cfg)
            solve_full_resolution!(
                1:N,
                run_cfg.input_data_path,
                bench_dir;
                label="full_benchmark",
                solvers=run_cfg.solvers,
                lambda=run_cfg.lambda,
                alpha=run_cfg.alpha,
                use_names=run_cfg.use_names,
            )
        end
        append_stage_rows!(summary, bench_dir, run_idx, seed, meta, run_cfg.solvers)
    else
        @info "Run $run_idx: skipping full benchmark (hybrid_experiment.run_full_benchmark=false)"
    end

    if cfg.hybrid_run_fixed_full || cfg.hybrid_run_full_benchmark
        append_og_rows!(og, run_idx, seed, run_dir, run_cfg.solvers, meta, reduced_source)
    end

    return nothing
end

function run_hybrid_experiment!(; num_runs::Union{Nothing,Int}=nothing)
    cfg = load_config(script_dir=SCRIPT_DIR, repo_root=REPO_ROOT)

    if cfg.tulipa_energy_model_rev !== nothing
        Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev=cfg.tulipa_energy_model_rev)
    end

    total_runs = something(num_runs, cfg.hybrid_num_runs)
    base = joinpath(cfg.hybrid_base_dir, cfg.hybrid_name)
    mkpath(base)
    @info "Starting hybrid dominance+Kantorovich experiment" name = cfg.hybrid_name num_runs =
        total_runs base = base method = cfg.dominance_method k = cfg.dominance_pick_k n =
        cfg.hybrid_kantorovich_pick_n

    summary = hybrid_summary_dataframe()
    summary_path = joinpath(base, "summary.csv")
    selection = DataFrame(;
        run=Int[],
        seed=Int[],
        num_scenarios=Int[],
        selected_source_scenarios=String[],
    )
    selection_path = joinpath(base, "scenario_selection.csv")
    og = optimality_gap_dataframe()
    og_path = joinpath(base, "optimality_gap.csv")

    for run_idx in 1:total_runs
        run_hybrid_run!(cfg, run_idx, base, summary, selection, selection_path, og)
        CSV.write(summary_path, summary; writeheader=true)
        CSV.write(og_path, og; writeheader=true)
        @info "Run $run_idx complete; summary updated at $summary_path"
    end

    @info "Hybrid experiment complete." summary = summary_path optimality_gap = og_path
    return summary
end

# Single run (run1 only), same artifacts/layout as the multi-run experiment.
run!() = run_hybrid_experiment!(num_runs=1)

if abspath(PROGRAM_FILE) == @__FILE__
    _cfg = load_config(script_dir=SCRIPT_DIR, repo_root=REPO_ROOT)
    if _cfg.hybrid_enabled
        run_hybrid_experiment!()
    else
        @info "hybrid_experiment.enabled=false in config.toml — nothing to do. Enable it, or include this file and call run!() / run_hybrid_experiment!()."
    end
end
