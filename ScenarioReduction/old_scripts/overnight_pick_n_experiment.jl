# ============================================================================
# overnight_pick_n_experiment.jl   (ADDITIVE — never edits existing code/data)
#
# pick_n_scenarios (n ∈ {2,4}) under pointwise / FSD / SSD dominance on a completed
# experiment's screening cost matrices, then full-resolution solves each picked set.
#
# pick_n_scenarios peels the most-expensive (undominated / cost-maximal) tier first
# and accumulates until >= n scenarios; we solve picked[1:n]. This targets the
# risk-relevant tail (same direction as the CVaR effective scenarios).
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/overnight_pick_n_experiment.jl <EXPERIMENT> [dry]
# e.g.
#   julia --project=. ScenarioReduction/old_scripts/overnight_pick_n_experiment.jl NightNight dry   # picks only, no solve
#   julia --project=. ScenarioReduction/old_scripts/overnight_pick_n_experiment.jl NightNight        # full solves
#
# Reuse: this includes overnight_distributional_experiment.jl, which itself
# Pkg.activate's the repo, pins the TEM rev, and includes the driver
# test_stochastic_dominance.jl (guarded, so nothing auto-runs). That single include
# provides the driver/module functions (load_config, prepare_scenario_input!,
# solve_full_resolution!, pick_n_scenarios, dominating_scenarios, fsd/ssd_*) AND the
# reusable helpers (reproduce_and_verify, read_run_solve_params, map_local_to_source,
# read_selection, discover_runs, read_baseline_obj, results_row_fields, ids_to_str).
#
# APPEND-ONLY: writes only NEW run<K>/full_resolution_<method>_pick<n>/ folders and the
# top-level pick_n_summary.csv / PICK_N_REPORT.md roll-ups. Skip-if-results.csv-exists
# makes it resumable; existing files are never overwritten (the regenerable
# base-input-data CSVs are the only exception, and are backed up first).
# ============================================================================

include(joinpath(@__DIR__, "overnight_distributional_experiment.jl"))

const PICK_METHODS = (:pointwise, :fsd, :ssd)
const PICK_NS = (2, 4)

"Sorted LOCAL scenario ids (1..N) parsed from a cost_matrix DataFrame's scenario_<id> columns."
function scenario_local_ids(cost_df)
    ids = Int[]
    for c in names(cost_df)
        m = match(r"^scenario_(\d+)$", String(c))
        m === nothing && continue
        push!(ids, parse(Int, m.captures[1]))
    end
    return sort(ids)
end

# One (run, method, n) cell: pick, then either solve into a new folder or just record.
function pick_and_solve(run_idx, run_dir, seed, run_cfg, cost_df, method, n, all_local,
                        source_ids, solve_params, cfg, baseline; do_solve::Bool, not_solving_reason::String="")
    pick = pick_n_scenarios(cost_df; n = n, method = method)
    to_solve = pick.picked[1:min(n, length(pick.picked))]
    picked_local = collect(Int.(to_solve))
    picked_source = map_local_to_source(picked_local, source_ids)
    peel_rounds = length(pick.rounds)
    method_name = string(method)
    pick.satisfied ||
        @warn "[pick] satisfied=false (pool < n) — using what was picked" run = run_idx method = method_name n = n picked = picked_local

    ran = false
    reason = not_solving_reason
    obj = missing
    status = missing
    tts = missing

    if !do_solve
        @info "[pick] (no solve)" run = run_idx method = method_name n = n picked_local = picked_local picked_source = picked_source satisfied = pick.satisfied peel_rounds = peel_rounds reason = reason
    else
        output_dir = joinpath(run_dir, "full_resolution_$(method_name)_pick$(n)")
        solver = first(cfg.solvers)
        results_csv = joinpath(output_dir, string(solver), "results.csv")
        if isfile(results_csv)
            try
                (obj, status, tts) = results_row_fields(CSV.read(results_csv, DataFrame))
            catch
            end
            reason = "already solved (results.csv exists)"
            @info "[pick] already solved → skip (resumable)" run = run_idx method = method_name n = n path = results_csv
        else
            # subset solve overwrites profiles-wide.csv → restore the full N-scenario input first.
            Random.seed!(seed)
            prepare_scenario_input!(run_cfg)
            @info "[pick] SOLVING" run = run_idx method = method_name n = n K = length(picked_local) picked_local = picked_local picked_source = picked_source satisfied = pick.satisfied peel_rounds = peel_rounds out = output_dir
            try
                solve_res = solve_full_resolution!(
                    picked_local, run_cfg.input_data_path, output_dir;
                    label = "$(method_name)_pick$(n)", solvers = cfg.solvers,
                    lambda = solve_params.lambda, alpha = solve_params.alpha, use_names = solve_params.use_names,
                )
                (obj, status, tts) = results_row_fields(solve_res.results)
                ran = true
                reason = ""
                @info "[pick] SOLVED" run = run_idx method = method_name n = n objective = obj status = status time_to_solve = tts
            catch err
                @error "[pick] solve FAILED" run = run_idx method = method_name n = n exception = (err, catch_backtrace())
                reason = "solve error: $err"
                status = "ERROR"
            end
        end
    end

    return (run = run_idx, seed = seed, method = method_name, n = n, n_total = length(all_local),
        n_picked = length(picked_local), satisfied = pick.satisfied, peel_rounds = peel_rounds,
        picked_local = ids_to_str(picked_local), picked_source = ids_to_str(picked_source),
        solve_ran = ran, skip_reason = ran ? "" : reason,
        objective_value = obj, termination_status = status, time_to_solve = tts,
        baseline_all_objective = baseline)
end

function write_pick_reports(exp_base, experiment_name, report)
    df = DataFrame(
        run = [r.run for r in report],
        seed = [r.seed for r in report],
        method = [r.method for r in report],
        n = [r.n for r in report],
        n_total = [r.n_total for r in report],
        n_picked = [r.n_picked for r in report],
        satisfied = [r.satisfied for r in report],
        peel_rounds = [r.peel_rounds for r in report],
        picked_local = [r.picked_local for r in report],
        picked_source = [r.picked_source for r in report],
        solve_ran = [r.solve_ran for r in report],
        skip_reason = [r.skip_reason for r in report],
        objective_value = [r.objective_value for r in report],
        termination_status = [r.termination_status for r in report],
        time_to_solve = [r.time_to_solve for r in report],
        baseline_all_objective = [r.baseline_all_objective for r in report],
    )
    sum_path = joinpath(exp_base, "pick_n_summary.csv")
    CSV.write(sum_path, df)
    @info "[pick] wrote $sum_path"

    io = IOBuffer()
    println(io, "# Pick-n tail-scenario solves (pointwise / FSD / SSD, n ∈ {2,4}) — $experiment_name")
    println(io)
    println(io, "Generated by `overnight_pick_n_experiment.jl` (additive; existing files untouched).")
    println(io, "`pick_n_scenarios` peels the most-expensive (undominated, cost-max) tier first; `picked[1:n]` is solved.")
    println(io, "`objective` is the reduced-set CVaR solve; `baseline_all_obj` = full-set `full_resolution_all`.")
    println(io)
    println(io, "| run | method | n | n_total | picked (local) | picked (source) | satisfied | rounds | solved | objective | status | t_solve s | baseline_all_obj |")
    println(io, "|----|--------|---|--------|----------------|-----------------|-----------|--------|--------|-----------|--------|-----------|------------------|")
    for r in report
        objs = r.objective_value === missing ? "" : string(round(r.objective_value; sigdigits = 9))
        bl = r.baseline_all_objective === missing ? "" : string(round(r.baseline_all_objective; sigdigits = 9))
        tts = r.time_to_solve === missing ? "" : string(round(r.time_to_solve; digits = 1))
        st = r.termination_status === missing ? "" : r.termination_status
        solved = r.solve_ran ? "yes" : (isempty(r.skip_reason) ? "no" : "no ($(r.skip_reason))")
        println(io, "| $(r.run) | $(r.method) | $(r.n) | $(r.n_total) | $(r.picked_local) | $(r.picked_source) | $(r.satisfied) | $(r.peel_rounds) | $solved | $objs | $st | $tts | $bl |")
    end
    md_path = joinpath(exp_base, "PICK_N_REPORT.md")
    write(md_path, String(take!(io)))
    @info "[pick] wrote $md_path"
end

function run_pick_n(experiment_name::String; dry::Bool = false)
    cfg = load_config(script_dir = SCRIPT_DIR, repo_root = REPO_ROOT)
    exp_base = joinpath(cfg.experiment_base_dir, experiment_name)
    @info "[pick] START" experiment = experiment_name base = exp_base dry = dry input_data = cfg.input_data_path master = cfg.profiles_wide_source methods = PICK_METHODS ns = PICK_NS

    isfile(cfg.profiles_wide_source) || error("Master profiles source not found: $(cfg.profiles_wide_source)")
    runs = discover_runs(exp_base)
    isempty(runs) && error("No runs with screening/cost_matrix.csv under $exp_base")
    @info "[pick] discovered runs" runs = [r[1] for r in runs]

    selection_path = joinpath(exp_base, "scenario_selection.csv")
    isfile(selection_path) || error("scenario_selection.csv not found: $selection_path")
    selection = read_selection(selection_path)

    if !dry
        try
            backup_dir = joinpath(REPO_ROOT, "ScenarioReduction", "outputs", "overnight_backup")
            mkpath(backup_dir)
            for f in ("profiles-wide.csv", "stochastic-scenario.csv")
                src = joinpath(cfg.input_data_path, f)
                dst = joinpath(backup_dir, f)
                if isfile(src) && !isfile(dst)
                    cp(src, dst)
                    @info "[pick] backed up input CSV" file = f dst = dst
                end
            end
        catch err
            @warn "[pick] input backup skipped" err
        end
    end

    report = NamedTuple[]

    for (run_idx, run_dir) in runs
        @info "==================== RUN $run_idx ===================="
        try
            cm_path = joinpath(run_dir, "screening", "cost_matrix.csv")
            isfile(cm_path) || (@warn "[pick] cost_matrix.csv missing → skip run" run = run_idx; continue)
            cost_df = CSV.read(cm_path, DataFrame)
            nrow(cost_df) > 0 || (@warn "[pick] cost_matrix.csv empty → skip run" run = run_idx; continue)
            all_local = scenario_local_ids(cost_df)
            isempty(all_local) && (@warn "[pick] no scenario_ columns → skip run" run = run_idx; continue)
            N = length(all_local)
            baseline = read_baseline_obj(run_dir)

            haskey(selection, run_idx) || (@warn "[pick] no scenario_selection row → skip run" run = run_idx; continue)
            seed = selection[run_idx].seed
            source_ids = selection[run_idx].source_ids
            if length(source_ids) != N
                @warn "[pick] selection size != cost-matrix scenario count → skip run" run = run_idx sel = length(source_ids) cm = N
                continue
            end

            do_solve = false
            not_reason = ""
            run_cfg = nothing
            solve_params = (lambda = cfg.lambda, alpha = cfg.alpha, use_names = cfg.use_names)
            if dry
                not_reason = "dry mode (no solve)"
            else
                solve_params = read_run_solve_params(run_dir, cfg)
                v = reproduce_and_verify(seed, source_ids, cfg)
                if v.ok
                    do_solve = true
                    run_cfg = v.run_cfg
                    @info "[pick] reproduction VERIFIED — input restored to full set" run = run_idx N = N lambda = solve_params.lambda alpha = solve_params.alpha
                else
                    not_reason = "reproduction not verified"
                    @error "[pick] reproduction verify FAILED — recording picks, no solves, preserving data" run = run_idx reason = v.reason
                end
            end

            for method in PICK_METHODS, n in PICK_NS
                push!(report, pick_and_solve(run_idx, run_dir, seed, run_cfg, cost_df, method, n,
                    all_local, source_ids, solve_params, cfg, baseline;
                    do_solve = do_solve, not_solving_reason = not_reason))
            end
        catch err
            @error "[pick] RUN FAILED — recording blocker, continuing" run = run_idx exception = (err, catch_backtrace())
            push!(report, (run = run_idx, seed = haskey(selection, run_idx) ? selection[run_idx].seed : -1,
                method = "-", n = -1, n_total = missing, n_picked = missing, satisfied = false, peel_rounds = missing,
                picked_local = "", picked_source = "", solve_ran = false, skip_reason = "RUN ERROR: $err",
                objective_value = missing, termination_status = missing, time_to_solve = missing,
                baseline_all_objective = missing))
        end
    end

    if dry
        @info "[pick] dry complete — picks logged; no solves, no roll-up."
        return report
    end

    write_pick_reports(exp_base, experiment_name, report)
    @info "[pick] DONE" experiment = experiment_name rows = length(report)
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: julia --project=. ScenarioReduction/old_scripts/overnight_pick_n_experiment.jl <EXPERIMENT> [dry]")
    _experiment = ARGS[1]
    _dry = length(ARGS) >= 2 && lowercase(ARGS[2]) == "dry"
    run_pick_n(_experiment; dry = _dry)
end
