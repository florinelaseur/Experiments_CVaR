# ============================================================================
# overnight_distributional_experiment.jl   (ADDITIVE — never edits existing code/data)
#
# FSD/SSD distributional dominance screening + undominated full-resolution
# re-solves for a COMPLETED experiment folder (e.g. NightNight, trialBoris).
#
#   Phase 1  for each run<K>/screening/cost_matrix.csv compute FSD and SSD and
#            write NEW fsd_dominance.csv / ssd_dominance.csv (skip if present).
#   Phase 2  full-resolution re-solve each method's UNDOMINATED set into NEW
#            run<K>/full_resolution_fsd|ssd/ folders. The run's profiles-wide.csv
#            is replayed from the master (seed + cardinality) and VERIFIED against
#            scenario_selection.csv before any solve. Skips when there is no
#            reduction, when already solved, or when reproduction can't be verified.
#   Phase 3  write NEW distributional_summary.csv + OVERNIGHT_REPORT.md roll-ups.
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/overnight_distributional_experiment.jl <EXPERIMENT> [phase1]
# e.g.
#   julia --project=. ScenarioReduction/old_scripts/overnight_distributional_experiment.jl NightNight phase1
#   julia --project=. ScenarioReduction/old_scripts/overnight_distributional_experiment.jl NightNight
#
# APPEND-ONLY guarantees:
#   * dominance CSVs and full_resolution_<method>/<solver>/results.csv are written
#     only when absent (skip-if-exists). Existing files are never overwritten.
#   * the only files this script regenerates are its own top-level roll-ups
#     (distributional_summary.csv, OVERNIGHT_REPORT.md) — distinct names from the
#     protected summary.csv / scenario_selection.csv.
# ============================================================================

using Pkg: Pkg

const ORCH_DIR = @__DIR__
const ORCH_REPO_ROOT = abspath(joinpath(ORCH_DIR, "..", ".."))
Pkg.activate(ORCH_REPO_ROOT)

# Pin TEM to the project's rev BEFORE the driver imports it (avoids registry
# schema-mismatch crash). Same rev/url as run_experiment! (test_*.jl:305).
const PINNED_TEM_REV = "227a80f7907e2c7178edb0697874cfb6666ad644"
@info "[overnight] pinning TulipaEnergyModel before loading driver" rev = PINNED_TEM_REV
Pkg.add(url = "https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev = PINNED_TEM_REV)

# Loads packages + all functions (prepare_scenario_input!, solve_full_resolution!,
# copy_config_with, load_config, fsd/ssd_dominating_scenarios, ...). The driver's
# `abspath(PROGRAM_FILE)==@__FILE__` guard is false when included, so nothing runs.
include(joinpath(ORCH_DIR, "..", "test_stochastic_dominance.jl"))

# ---------------------------------------------------------------------------
# Canonical dominance-matrix writer — verbatim from
# old_scripts/analyze_distributional_dominance.jl:38-49 so output is byte-compatible
# with that CLI. (We replicate rather than include, because that script's top-level
# cd / Pkg.activate / main(ARGS) make it include-unfriendly.)
# ---------------------------------------------------------------------------
function overnight_dominance_matrix_dataframe(scenarios, dominates, nan_scenarios)
    n = length(scenarios)
    nan_set = Set(Int.(nan_scenarios))
    df = DataFrame(;
        scenario = collect(Int.(scenarios)),
        has_nan = Int[Int(scenarios[i]) in nan_set ? 1 : 0 for i in 1:n],
    )
    for j in 1:n
        df[!, Symbol("scenario_$(scenarios[j])")] = Int[dominates[i, j] ? 1 : 0 for i in 1:n]
    end
    return df
end

const OVERNIGHT_METHODS = (
    (name = "fsd", fn = fsd_dominating_scenarios, file = "fsd_dominance.csv"),
    (name = "ssd", fn = ssd_dominating_scenarios, file = "ssd_dominance.csv"),
)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
ids_to_str(ids) = join(Int.(ids), " ")

"LOCAL id i (1..N) maps to the i-th smallest original source id."
function map_local_to_source(local_ids, source_ids_sorted)
    out = Int[]
    for l in local_ids
        (1 <= l <= length(source_ids_sorted)) ||
            error("local id $l out of range 1..$(length(source_ids_sorted))")
        push!(out, source_ids_sorted[Int(l)])
    end
    return out
end

function read_selection(selection_path)
    df = CSV.read(selection_path, DataFrame)
    sel = Dict{Int,NamedTuple{(:seed, :source_ids),Tuple{Int,Vector{Int}}}}()
    for row in eachrow(df)
        ids = parse.(Int, split(strip(string(row.selected_source_scenarios))))
        sel[Int(row.run)] = (seed = Int(row.seed), source_ids = sort(ids))
    end
    return sel
end

"Read the run's actual solve params from its config_used.toml (fall back to live cfg)."
function read_run_solve_params(run_dir, cfg)
    p = joinpath(run_dir, "config_used.toml")
    if isfile(p)
        try
            t = TOML.parsefile(p)
            sim = get(t, "simulation", Dict{String,Any}())
            slv = get(t, "solve", Dict{String,Any}())
            return (
                lambda = Float64(get(sim, "risk_aversion_weight_lambda", cfg.lambda)),
                alpha = Float64(get(sim, "risk_aversion_confidence_level", cfg.alpha)),
                use_names = Bool(get(slv, "use_names", cfg.use_names)),
            )
        catch err
            @warn "[overnight] could not parse config_used.toml; using live cfg params" path = p err
        end
    end
    return (lambda = cfg.lambda, alpha = cfg.alpha, use_names = cfg.use_names)
end

function discover_runs(exp_base)
    isdir(exp_base) || error("Experiment folder not found: $exp_base")
    runs = Tuple{Int,String}[]
    for name in readdir(exp_base)
        m = match(r"^run(\d+)$", name)
        m === nothing && continue
        rd = joinpath(exp_base, name)
        isfile(joinpath(rd, "screening", "cost_matrix.csv")) || continue
        push!(runs, (parse(Int, m.captures[1]), rd))
    end
    sort!(runs; by = x -> x[1])
    return runs
end

function read_baseline_obj(run_dir)
    p = joinpath(run_dir, "full_resolution_all", "Gurobi", "results.csv")
    isfile(p) || return missing
    try
        df = CSV.read(p, DataFrame)
        return nrow(df) >= 1 && hasproperty(df, :objective_value) ? Float64(df.objective_value[1]) : missing
    catch
        return missing
    end
end

"Pull (objective, status, time_to_solve) from a one-row-per-solver results DataFrame."
function results_row_fields(results_df)
    nrow(results_df) >= 1 || return (missing, missing, missing)
    r = results_df[1, :]
    obj = hasproperty(results_df, :objective_value) ? Float64(r.objective_value) : missing
    status = hasproperty(results_df, :termination_status) ? string(r.termination_status) : missing
    tts = hasproperty(results_df, :time_to_solve) ? Float64(r.time_to_solve) : missing
    return (obj, status, tts)
end

# ---------------------------------------------------------------------------
# Phase 1 — FSD/SSD dominance per run (no solver)
# ---------------------------------------------------------------------------
function phase1_for_run(run_idx, run_dir)
    screening = joinpath(run_dir, "screening")
    cm_path = joinpath(screening, "cost_matrix.csv")
    isfile(cm_path) || return (ok = false, reason = "cost_matrix.csv missing", results = nothing)
    local cost_df
    try
        cost_df = CSV.read(cm_path, DataFrame)
    catch err
        return (ok = false, reason = "cost_matrix.csv unreadable: $err", results = nothing)
    end
    nrow(cost_df) > 0 || return (ok = false, reason = "cost_matrix.csv empty", results = nothing)
    scen_cols = filter(c -> occursin(r"^scenario_\d+$", String(c)), names(cost_df))
    isempty(scen_cols) && return (ok = false, reason = "no scenario_ columns", results = nothing)

    results = Dict{String,Any}()
    for m in OVERNIGHT_METHODS
        out_path = joinpath(screening, m.file)
        local res
        try
            res = m.fn(cost_df)
        catch err
            return (ok = false, reason = "$(uppercase(m.name)) compute failed: $err", results = nothing)
        end
        und = sort(Int.(res.undominated))
        nanv = sort(Int.(res.nan_scenarios))
        if isfile(out_path)
            @info "[Phase1] exists → skip write (recomputed in-memory)" run = run_idx method = m.name path = basename(out_path) n_undominated = length(und) undominated = und pairs = length(res.pairs) nan = nanv
        else
            out_df = overnight_dominance_matrix_dataframe(res.scenarios, res.dominates, res.nan_scenarios)
            CSV.write(out_path, out_df)
            @info "[Phase1] WROTE dominance CSV" run = run_idx method = m.name path = out_path n_undominated = length(und) undominated = und pairs = length(res.pairs) nan = nanv
        end
        if !isempty(nanv)
            @warn "[Phase1] NaN-tagged scenarios present (always undominated)" run = run_idx method = m.name nan = nanv
        end
        results[m.name] = res
    end
    return (ok = true, reason = "", results = results)
end

# ---------------------------------------------------------------------------
# Phase 2 — reproduce input + verify, then solve undominated sets
# ---------------------------------------------------------------------------
function reproduce_and_verify(seed, source_ids, cfg)
    card = length(source_ids)
    run_cfg = copy_config_with(cfg; number_of_scenarios = card, random_seed = seed)
    Random.seed!(seed)
    reproduced = sort(Int.(prepare_scenario_input!(run_cfg)))
    expected = sort(Int.(source_ids))
    reproduced == expected ||
        return (ok = false, run_cfg = run_cfg,
            reason = "reproduction mismatch (got $(reproduced) expected $(expected))")
    return (ok = true, run_cfg = run_cfg, reason = "")
end

function maybe_solve_method(run_idx, run_dir, seed, run_cfg, method_name, res, all_local, solve_params, cfg)
    method_dir = joinpath(run_dir, "full_resolution_$(method_name)")
    solver = first(cfg.solvers)
    results_csv = joinpath(method_dir, string(solver), "results.csv")
    undominated = sort(Int.(res.undominated))

    if isfile(results_csv)
        obj, status, tts = (missing, missing, missing)
        try
            (obj, status, tts) = results_row_fields(CSV.read(results_csv, DataFrame))
        catch
        end
        @info "[Phase2] already solved → skip (resumable)" run = run_idx method = method_name path = results_csv
        return (ran = false, reason = "already solved (results.csv exists)", objective = obj, status = status, time_to_solve = tts)
    end
    if isempty(undominated)
        return (ran = false, reason = "undominated set empty", objective = missing, status = missing, time_to_solve = missing)
    end
    if undominated == all_local
        @info "[Phase2] no reduction (undominated == full set) → skip" run = run_idx method = method_name N = length(all_local)
        return (ran = false, reason = "no reduction (undominated == full set of $(length(all_local)))", objective = missing, status = missing, time_to_solve = missing)
    end
    if length(res.nan_scenarios) == length(all_local)
        return (ran = false, reason = "all scenarios NaN-tagged", objective = missing, status = missing, time_to_solve = missing)
    end

    # solve_full_resolution! -> prepare_scenario_subset! OVERWRITES profiles-wide.csv,
    # so restore the full N-scenario input immediately before each method's solve.
    Random.seed!(seed)
    prepare_scenario_input!(run_cfg)

    length(undominated) == 1 && @warn "[Phase2] single undominated scenario → TEM drops CVaR (mu=NaN, risk-neutral); expected" run = run_idx method = method_name
    @info "[Phase2] SOLVING undominated set" run = run_idx method = method_name K = length(undominated) undominated = undominated nan = sort(Int.(res.nan_scenarios)) lambda = solve_params.lambda alpha = solve_params.alpha use_names = solve_params.use_names out = method_dir

    local solve_res
    try
        solve_res = solve_full_resolution!(
            undominated, run_cfg.input_data_path, method_dir;
            label = "$(method_name)_undominated", solvers = cfg.solvers,
            lambda = solve_params.lambda, alpha = solve_params.alpha, use_names = solve_params.use_names,
        )
    catch err
        @error "[Phase2] solve FAILED" run = run_idx method = method_name exception = (err, catch_backtrace())
        return (ran = false, reason = "solve error: $err", objective = missing, status = "ERROR", time_to_solve = missing)
    end
    obj, status, tts = results_row_fields(solve_res.results)
    @info "[Phase2] SOLVED" run = run_idx method = method_name objective = obj status = status time_to_solve = tts
    return (ran = true, reason = "", objective = obj, status = status, time_to_solve = tts)
end

# ---------------------------------------------------------------------------
# Phase 3 — roll-up report (NEW files only; the only files we regenerate)
# ---------------------------------------------------------------------------
function write_reports(exp_base, experiment_name, report)
    report_df = DataFrame(
        run = [r.run for r in report],
        seed = [r.seed for r in report],
        method = [r.method for r in report],
        n_total = [r.n_total for r in report],
        n_undominated = [r.n_undominated for r in report],
        n_nan = [r.n_nan for r in report],
        n_pairs = [r.n_pairs for r in report],
        undominated_local = [r.undominated_local for r in report],
        undominated_source = [r.undominated_source for r in report],
        nan_local = [r.nan_local for r in report],
        solve_ran = [r.solve_ran for r in report],
        skip_reason = [r.skip_reason for r in report],
        objective_value = [r.objective_value for r in report],
        termination_status = [r.termination_status for r in report],
        time_to_solve = [r.time_to_solve for r in report],
        baseline_all_objective = [r.baseline_all_objective for r in report],
    )
    sum_path = joinpath(exp_base, "distributional_summary.csv")
    CSV.write(sum_path, report_df)
    @info "[Phase3] wrote $sum_path"

    io = IOBuffer()
    println(io, "# Distributional (FSD/SSD) dominance + undominated re-solves — $experiment_name")
    println(io)
    println(io, "Generated by `overnight_distributional_experiment.jl` (additive; existing files untouched).")
    println(io)
    println(io, "Convention: cost-maximization (aligned with pointwise scenario_dominance). FSD ⊆ SSD")
    println(io, "(every FSD pair is also SSD), so FSD yields the *largest* undominated set and SSD the")
    println(io, "smallest. `undominated` = expensive / tail scenarios. NaN-tagged scenarios are always")
    println(io, "undominated. A single-scenario undominated solve is risk-neutral (TEM drops CVaR, mu=NaN).")
    println(io)
    println(io, "| run | seed | method | n_total | n_undom | n_nan | pairs | undominated (local) | undominated (source) | solved | objective | status | t_solve s | baseline_all_obj |")
    println(io, "|----|------|--------|--------|--------|------|------|----------------------|----------------------|--------|-----------|--------|-----------|------------------|")
    for r in report
        objs = r.objective_value === missing ? "" : string(round(r.objective_value; sigdigits = 9))
        bl = r.baseline_all_objective === missing ? "" : string(round(r.baseline_all_objective; sigdigits = 9))
        tts = r.time_to_solve === missing ? "" : string(round(r.time_to_solve; digits = 1))
        st = r.termination_status === missing ? "" : r.termination_status
        solved = r.solve_ran ? "yes" : "no"
        reason = isempty(r.skip_reason) ? "" : " ($(r.skip_reason))"
        println(io, "| $(r.run) | $(r.seed) | $(r.method) | $(r.n_total) | $(r.n_undominated) | $(r.n_nan) | $(r.n_pairs) | $(r.undominated_local) | $(r.undominated_source) | $(solved)$(reason) | $objs | $st | $tts | $bl |")
    end
    println(io)
    println(io, "_objective vs baseline_all_obj: undominated tail scenarios typically yield objectives at or above the full-set baseline (not below)._")
    md_path = joinpath(exp_base, "OVERNIGHT_REPORT.md")
    write(md_path, String(take!(io)))
    @info "[Phase3] wrote $md_path"
end

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
function run_overnight(experiment_name::String; phase1_only::Bool = false)
    cfg = load_config(script_dir = SCRIPT_DIR, repo_root = REPO_ROOT)
    exp_base = joinpath(cfg.experiment_base_dir, experiment_name)
    @info "[overnight] START" experiment = experiment_name base = exp_base phase1_only = phase1_only input_data = cfg.input_data_path master = cfg.profiles_wide_source solvers = cfg.solvers

    isfile(cfg.profiles_wide_source) || error("Master profiles source not found: $(cfg.profiles_wide_source)")
    runs = discover_runs(exp_base)
    isempty(runs) && error("No runs with screening/cost_matrix.csv under $exp_base")
    @info "[overnight] discovered runs" runs = [r[1] for r in runs]

    selection_path = joinpath(exp_base, "scenario_selection.csv")
    isfile(selection_path) || error("scenario_selection.csv not found: $selection_path")
    selection = read_selection(selection_path)

    # nice-to-have one-time backup of the (regenerable) input CSVs
    try
        backup_dir = joinpath(REPO_ROOT, "ScenarioReduction", "outputs", "overnight_backup")
        mkpath(backup_dir)
        for f in ("profiles-wide.csv", "stochastic-scenario.csv")
            src = joinpath(cfg.input_data_path, f)
            dst = joinpath(backup_dir, f)
            if isfile(src) && !isfile(dst)
                cp(src, dst)
                @info "[overnight] backed up input CSV" file = f dst = dst
            end
        end
    catch err
        @warn "[overnight] input backup skipped" err
    end

    report = NamedTuple[]

    for (run_idx, run_dir) in runs
        @info "==================== RUN $run_idx ===================="
        try
            p1 = phase1_for_run(run_idx, run_dir)
            baseline = read_baseline_obj(run_dir)

            if !p1.ok
                @warn "[Phase1] BLOCKED — skipping run" run = run_idx reason = p1.reason
                push!(report, (run = run_idx, seed = haskey(selection, run_idx) ? selection[run_idx].seed : -1,
                    method = "-", n_total = missing, n_undominated = missing, n_nan = missing, n_pairs = missing,
                    undominated_local = "", undominated_source = "", nan_local = "",
                    solve_ran = false, skip_reason = "PHASE1 BLOCKED: $(p1.reason)",
                    objective_value = missing, termination_status = missing, time_to_solve = missing,
                    baseline_all_objective = baseline))
                continue
            end

            fsd_res = p1.results["fsd"]
            ssd_res = p1.results["ssd"]
            all_local = sort(Int.(fsd_res.scenarios))

            have_sel = haskey(selection, run_idx)
            seed = have_sel ? selection[run_idx].seed : -1
            source_ids = have_sel ? selection[run_idx].source_ids : Int[]
            sel_matches = have_sel && length(source_ids) == length(all_local)

            verify_ok = false
            run_cfg = nothing
            solve_params = (lambda = cfg.lambda, alpha = cfg.alpha, use_names = cfg.use_names)
            if !phase1_only
                if !have_sel
                    @warn "[Phase2] no scenario_selection row → cannot reproduce input; skipping solves" run = run_idx
                elseif !sel_matches
                    @warn "[Phase2] selection size != cost-matrix scenario count; skipping solves" run = run_idx sel = length(source_ids) cm = length(all_local)
                else
                    solve_params = read_run_solve_params(run_dir, cfg)
                    v = reproduce_and_verify(seed, source_ids, cfg)
                    verify_ok = v.ok
                    run_cfg = v.run_cfg
                    if verify_ok
                        @info "[Phase2] reproduction VERIFIED — input restored to full set" run = run_idx cardinality = length(source_ids) lambda = solve_params.lambda alpha = solve_params.alpha
                    else
                        @error "[Phase2] reproduction verify FAILED — skipping solves, preserving data" run = run_idx reason = v.reason
                    end
                end
            end

            for (mname, res) in (("fsd", fsd_res), ("ssd", ssd_res))
                undominated = sort(Int.(res.undominated))
                nanv = sort(Int.(res.nan_scenarios))
                und_src = sel_matches ? map_local_to_source(undominated, source_ids) : Int[]
                nan_src = sel_matches ? map_local_to_source(nanv, source_ids) : Int[]

                ran = false
                reason = ""
                obj = missing
                status = missing
                tts = missing
                if phase1_only
                    reason = "phase1 mode (no solve)"
                elseif !verify_ok
                    reason = "reproduction not verified"
                else
                    r = maybe_solve_method(run_idx, run_dir, seed, run_cfg, mname, res, all_local, solve_params, cfg)
                    ran = r.ran
                    reason = r.reason
                    obj = r.objective
                    status = r.status
                    tts = r.time_to_solve
                end

                push!(report, (run = run_idx, seed = seed, method = mname,
                    n_total = length(all_local), n_undominated = length(undominated),
                    n_nan = length(nanv), n_pairs = length(res.pairs),
                    undominated_local = ids_to_str(undominated), undominated_source = ids_to_str(und_src),
                    nan_local = ids_to_str(nanv),
                    solve_ran = ran, skip_reason = ran ? "" : reason,
                    objective_value = obj, termination_status = status, time_to_solve = tts,
                    baseline_all_objective = baseline))
                @info "[summary] run=$run_idx method=$mname" n_undominated = length(undominated) undominated_local = undominated undominated_source = und_src solved = ran reason = reason objective = obj
            end
        catch err
            @error "[overnight] RUN FAILED — recording blocker, continuing" run = run_idx exception = (err, catch_backtrace())
            push!(report, (run = run_idx, seed = haskey(selection, run_idx) ? selection[run_idx].seed : -1,
                method = "-", n_total = missing, n_undominated = missing, n_nan = missing, n_pairs = missing,
                undominated_local = "", undominated_source = "", nan_local = "",
                solve_ran = false, skip_reason = "RUN ERROR: $err",
                objective_value = missing, termination_status = missing, time_to_solve = missing,
                baseline_all_objective = missing))
        end
    end

    if phase1_only
        @info "[overnight] phase1-only complete — dominance CSVs written; no solves / no roll-up."
        return report
    end

    write_reports(exp_base, experiment_name, report)
    @info "[overnight] DONE" experiment = experiment_name rows = length(report)
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: julia --project=. ScenarioReduction/old_scripts/overnight_distributional_experiment.jl <EXPERIMENT> [phase1]")
    _experiment = ARGS[1]
    _phase1_only = length(ARGS) >= 2 && lowercase(ARGS[2]) == "phase1"
    run_overnight(_experiment; phase1_only = _phase1_only)
end
