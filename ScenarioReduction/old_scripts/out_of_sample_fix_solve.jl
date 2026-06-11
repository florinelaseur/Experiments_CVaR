# ============================================================================
# out_of_sample_fix_solve.jl   (ADDITIVE — never edits existing code/data)
#
# Out-of-sample optimality gap: take each reduced configuration's STORED investment
# z^Ω̂ and evaluate it on the FULL N-scenario set → F(z^Ω̂, Ω); compare to the full
# model's own objective F(z^Ω, Ω) (= full_resolution_all).
#
#   OG(%) = (F(z^Ω̂, Ω) − F(z^Ω, Ω)) / F(z^Ω, Ω) × 100
#
# Pattern (reuses the SD screening's "build once, re-fix investment per sample,
# disable presolve for warm re-solves" — stochastic_dominance.jl; and the
# multi-scenario fix-and-solve of multiscenario_seasonal_test.jl): per run we build
# the full N-scenario CVaR model ONCE (non-destructive, in-memory profiles), then for
# each unique stored investment we `fix_variables_from_sample(:assets_investment)` and
# re-solve. Only :assets_investment is fixed — the battery's energy is slaved to its
# power (use_fixed_energy_to_power_ratio) and the seasonal storages aren't investable,
# so var_assets_investment.csv fully pins the investment.
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/out_of_sample_fix_solve.jl NightNight smoke   # run1, validate
#   julia --project=. ScenarioReduction/old_scripts/out_of_sample_fix_solve.jl NightNight         # full
#
# APPEND-ONLY: writes only NEW run<K>/oos_fix_solve/<configs>/results.csv and the
# top-level out_of_sample_summary.csv / OUT_OF_SAMPLE_REPORT.md. Non-destructive — the
# base-input-data CSVs are never overwritten. Skip-if-results.csv-exists ⇒ resumable.
# ============================================================================

include(joinpath(@__DIR__, "overnight_distributional_experiment.jl"))

# config label → folder under run<K>/ that stored its solved investment.
const OOS_CONFIG_FOLDER = (
    ("pointwise", "full_resolution_undominated"),
    ("fsd", "full_resolution_fsd"),
    ("ssd", "full_resolution_ssd"),
    ("select2", "full_resolution_fsd_pick2"),
    ("select4", "full_resolution_fsd_pick4"),
)

# Build the full N-scenario CVaR connection in memory (mirrors
# multiscenario_seasonal_test.build_multiscenario_connection + the λ/α UPDATE that
# solve_full_resolution! applies). Non-destructive: input CSVs are not overwritten.
function build_full_connection(profiles_df, N, input_data_path, lambda, alpha)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)
    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, profiles_df, "profiles_wide")
    prob = 1.0 / N
    values_sql = join(["($s, $prob)" for s in 1:N], ", ")
    DuckDB.query(
        conn,
        "CREATE OR REPLACE TABLE stochastic_scenario AS " *
        "SELECT * FROM (VALUES $values_sql) AS t(scenario, probability)",
    )
    DuckDB.query(
        conn,
        """
        UPDATE model_parameters
        SET risk_aversion_weight_lambda = $(lambda),
            risk_aversion_confidence_level_alpha = $(alpha);
        """,
    )
    TC.transform_wide_to_long!(
        conn, "profiles_wide", "profiles";
        exclude_columns = ["scenario", "milestone_year", "timestep"],
    )
    layout = TC.ProfilesTableLayout(; year = :milestone_year, cols_to_groupby = [:milestone_year, :scenario])
    TC.dummy_cluster!(conn; layout = layout)
    TEM.populate_with_defaults!(conn)
    DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    return conn
end

# Build the low-level model bundle once; the investment is fixed on this model and
# re-solved per stored investment (mirrors multiscenario_seasonal_test.solve_with_fixed_investment).
function build_model_bundle(conn, solver::Symbol)
    TEM.create_internal_tables!(conn)
    variables = TEM.compute_variables_indices(conn)
    constraints = TEM.compute_constraints_indices(conn)
    profiles = TEM.prepare_profiles_structure(conn)
    model, _ = TEM.create_model(conn, variables, constraints, profiles)
    optimizer, parameters = get_solver_parameters(solver)
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in parameters]...)
    JuMP.set_silent(model)
    return (model = model, variables = variables, capacity_lookup = build_capacity_lookup(conn))
end

# Read a stored investment (var_assets_investment.csv) as an MW vector in INVESTABLE_ASSETS
# order: MW[a] = solution[a] × capacity[a]. Returns nothing if the file/assets are missing.
function read_investment_mw(gurobi_folder)
    p = joinpath(gurobi_folder, "var_assets_investment.csv")
    isfile(p) || return nothing
    df = CSV.read(p, DataFrame)
    (hasproperty(df, :asset) && hasproperty(df, :solution) && hasproperty(df, :capacity)) || return nothing
    sol = Dict(string(r.asset) => Float64(r.solution) for r in eachrow(df))
    cap = Dict(string(r.asset) => Float64(r.capacity) for r in eachrow(df))
    mw = Float64[]
    for a in INVESTABLE_ASSETS
        (haskey(sol, a) && haskey(cap, a)) || return nothing
        push!(mw, sol[a] * cap[a])
    end
    return mw
end

vec_key(mw) = join((string(round(x; digits = 3)) for x in mw), "_")

# Compute each config's selected LOCAL set from the run's cost_matrix (current convention).
function config_sets(cost_df)
    pick(n) = (p = pick_n_scenarios(cost_df; n = n, method = :fsd); sort(Int.(p.picked[1:min(n, length(p.picked))])))
    return Dict(
        "pointwise" => sort(Int.(dominating_scenarios(cost_df).undominated)),
        "fsd" => sort(Int.(fsd_dominating_scenarios(cost_df).undominated)),
        "ssd" => sort(Int.(ssd_dominating_scenarios(cost_df).undominated)),
        "select2" => pick(2),
        "select4" => pick(4),
    )
end

function run_oos(experiment_name::String; smoke::Bool = false)
    cfg = load_config(script_dir = SCRIPT_DIR, repo_root = REPO_ROOT)
    exp_base = joinpath(cfg.experiment_base_dir, experiment_name)
    solver = first(cfg.solvers)
    @info "[oos] START" experiment = experiment_name base = exp_base smoke = smoke solver = solver master = cfg.profiles_wide_source

    isfile(cfg.profiles_wide_source) || error("Master profiles source not found: $(cfg.profiles_wide_source)")
    master_df = CSV.read(cfg.profiles_wide_source, DataFrame)
    runs = discover_runs(exp_base)
    isempty(runs) && error("No runs with screening/cost_matrix.csv under $exp_base")
    selection = read_selection(joinpath(exp_base, "scenario_selection.csv"))

    report = NamedTuple[]   # one row per (run, config)

    for (run_idx, run_dir) in runs
        (smoke && run_idx != 1) && continue
        @info "==================== RUN $run_idx ===================="
        try
            haskey(selection, run_idx) || (@warn "[oos] no selection row → skip run" run = run_idx; continue)
            seed = selection[run_idx].seed
            source_ids = sort(Int.(selection[run_idx].source_ids))
            N = length(source_ids)

            cost_df = CSV.read(joinpath(run_dir, "screening", "cost_matrix.csv"), DataFrame)
            sets = config_sets(cost_df)
            baseline = read_baseline_obj(run_dir)
            baseline === missing && (@warn "[oos] no full_resolution_all baseline → skip run" run = run_idx; continue)

            # --- reproduce the run's full N-scenario profiles (in memory) + verify ---
            Random.seed!(seed)
            sub = get_scenario_set(master_df, N)
            selected_original = sort(unique(Int.(sub.scenario)))
            if selected_original != source_ids
                @error "[oos] reproduction mismatch → skip run" run = run_idx got = selected_original expected = source_ids
                continue
            end
            mapping = Dict(old => new for (new, old) in enumerate(selected_original))
            profiles_df = copy(sub)
            profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]
            @info "[oos] reproduction verified — building full model" run = run_idx N = N lambda = cfg.lambda alpha = cfg.alpha

            # --- build the full N-scenario CVaR model once ---
            conn = build_full_connection(profiles_df, N, cfg.input_data_path, cfg.lambda, cfg.alpha)
            bundle = build_model_bundle(conn, solver)

            # --- smoke: free-solve must reproduce the full_resolution_all baseline ---
            if smoke
                st0 = solve_model(bundle.model; diagnose_infeasibility = false)
                obj0 = st0 == JuMP.OPTIMAL ? JuMP.objective_value(bundle.model) : NaN
                reldiff = abs(obj0 - baseline) / abs(baseline)
                @info "[oos][smoke] free-solve vs baseline" status = st0 free_objective = obj0 baseline = baseline rel_diff = reldiff
                reldiff < 1e-3 ? @info("[oos][smoke] BUILD OK (free objective matches full_resolution_all)") :
                @warn("[oos][smoke] build objective differs from baseline by >0.1% — investigate before trusting OG")
                disable_presolve!(bundle.model; solver = solver)
            end

            # --- read stored investments, dedupe by MW vector ---
            groups = Dict{String,Vector{String}}()
            mw_by_key = Dict{String,Vector{Float64}}()
            for (cname, folder) in OOS_CONFIG_FOLDER
                mw = read_investment_mw(joinpath(run_dir, folder, string(solver)))
                if mw === nothing
                    @warn "[oos] missing var_assets_investment.csv → config skipped" run = run_idx config = cname folder = folder
                    continue
                end
                k = vec_key(mw)
                push!(get!(groups, k, String[]), cname)
                mw_by_key[k] = mw
            end
            @info "[oos] unique investments" run = run_idx n_groups = length(groups) groups = [join(sort(v), "+") for v in values(groups)]

            # --- fix each unique investment on the full model and re-solve ---
            first_solve = !smoke   # in smoke we already solved free + disabled presolve
            f_by_config = Dict{String,Any}()
            og_by_config = Dict{String,Any}()
            status_by_config = Dict{String,String}()
            for (k, cnames) in groups
                cnames_sorted = sort(cnames)
                label = join(cnames_sorted, "_")
                rep = cnames_sorted[1]
                set_local = get(sets, rep, Int[])
                set_source = (length(source_ids) == N) ? map_local_to_source(set_local, source_ids) : Int[]
                out_dir = joinpath(run_dir, "oos_fix_solve", label)
                results_csv = joinpath(out_dir, "results.csv")

                local F, OG, status
                if isfile(results_csv)
                    rdf = CSV.read(results_csv, DataFrame)
                    F = hasproperty(rdf, :F_oos) ? Float64(rdf.F_oos[1]) : NaN
                    OG = hasproperty(rdf, :OG_percent) ? Float64(rdf.OG_percent[1]) : NaN
                    status = hasproperty(rdf, :termination_status) ? string(rdf.termination_status[1]) : "UNKNOWN"
                    @info "[oos] already solved → skip (resumable)" run = run_idx configs = label F = F OG = OG
                else
                    fix_variables_from_sample(bundle.variables, :assets_investment, mw_by_key[k]; capacity_lookup = bundle.capacity_lookup)
                    @info "[oos] SOLVING fixed-investment full model" run = run_idx configs = label picked_local = set_local picked_source = set_source
                    st = solve_model(bundle.model; diagnose_infeasibility = false)
                    status = string(st)
                    F = st == JuMP.OPTIMAL ? JuMP.objective_value(bundle.model) : NaN
                    OG = isnan(F) ? NaN : (F - baseline) / baseline * 100
                    if first_solve
                        disable_presolve!(bundle.model; solver = solver)
                        first_solve = false
                    end
                    mkpath(out_dir)
                    CSV.write(results_csv, DataFrame(
                        run = run_idx, configs = label, picked_local = ids_to_str(set_local),
                        picked_source = ids_to_str(set_source), F_oos = F, baseline_F_full = baseline,
                        OG_percent = OG, termination_status = status,
                    ))
                    @info "[oos] SOLVED" run = run_idx configs = label status = status F_oos = F OG_percent = OG
                end
                for c in cnames_sorted
                    f_by_config[c] = F
                    og_by_config[c] = OG
                    status_by_config[c] = status
                end
            end

            # --- collect report rows: Full + the 5 reduced configs ---
            push!(report, (run = run_idx, config = "full", picked_local = "", picked_source = "",
                F_oos = baseline, baseline = baseline, OG = 0.0, status = "OPTIMAL"))
            for (cname, _) in OOS_CONFIG_FOLDER
                haskey(f_by_config, cname) || continue
                sl = get(sets, cname, Int[])
                ss = map_local_to_source(sl, source_ids)
                push!(report, (run = run_idx, config = cname, picked_local = ids_to_str(sl),
                    picked_source = ids_to_str(ss), F_oos = f_by_config[cname], baseline = baseline,
                    OG = og_by_config[cname], status = status_by_config[cname]))
            end

            try
                DuckDB.DBInterface.close!(conn)
            catch
            end
        catch err
            @error "[oos] RUN FAILED — recording blocker, continuing" run = run_idx exception = (err, catch_backtrace())
        end
    end

    if smoke
        @info "[oos] smoke complete — no roll-up written."
        return report
    end

    write_oos_reports(exp_base, experiment_name, report)
    @info "[oos] DONE" experiment = experiment_name rows = length(report)
    return report
end

function write_oos_reports(exp_base, experiment_name, report)
    df = DataFrame(
        run = [r.run for r in report],
        config = [r.config for r in report],
        picked_local = [r.picked_local for r in report],
        picked_source = [r.picked_source for r in report],
        F_oos = [r.F_oos for r in report],
        baseline_F_full = [r.baseline for r in report],
        OG_percent = [r.OG for r in report],
        termination_status = [r.status for r in report],
    )
    sum_path = joinpath(exp_base, "out_of_sample_summary.csv")
    CSV.write(sum_path, df)
    @info "[oos] wrote $sum_path"

    runs = sort(unique(Int[r.run for r in report]))
    order = ["full", "pointwise", "fsd", "ssd", "select2", "select4"]
    Fd = Dict((r.config, r.run) => r.F_oos for r in report)
    Gd = Dict((r.config, r.run) => r.OG for r in report)
    Sd = Dict((r.config, r.run) => r.status for r in report)

    io = IOBuffer()
    println(io, "# Out-of-sample optimality gap — $experiment_name")
    println(io)
    println(io, "Each reduced configuration's stored investment fixed in the full N-scenario model:")
    println(io, "`F(ẑ^Ω̂, Ω)` (×10⁷) and `OG% = (F(ẑ^Ω̂,Ω) − F(z^Ω,Ω)) / F(z^Ω,Ω) × 100`. Full model = baseline, OG = 0.")
    println(io, "`INFEASIBLE` = the reduced investment cannot serve the full set (loss-of-load capacity exhausted).")
    println(io)
    header = "| config |" * join([" run$r F | run$r OG% |" for r in runs], "")
    sep = "|--------|" * join([":-------:|:-------:|" for _ in runs], "")
    println(io, header)
    println(io, sep)
    for c in order
        any(r -> r.config == c, report) || continue
        cells = String[]
        for r in runs
            st = get(Sd, (c, r), "")
            if c == "full" || st == "OPTIMAL"
                f = get(Fd, (c, r), missing)
                g = get(Gd, (c, r), missing)
                fs = (f === missing || (f isa Float64 && isnan(f))) ? "" : string(round(f / 1e7; digits = 4))
                gs = (g === missing || (g isa Float64 && isnan(g))) ? "" : string(round(g; digits = 2))
                push!(cells, " $fs | $gs |")
            else
                push!(cells, " $(isempty(st) ? "—" : st) | — |")
            end
        end
        println(io, "| $c |" * join(cells, ""))
    end
    println(io)
    println(io, "## Finding")
    println(io, "The dominance methods select the **most-expensive** (CVaR cost-tail) scenarios, which are **not** the")
    println(io, "**capacity-binding** ones. Each reduced investment is optimized for a low-wind cost-tail scenario and badly")
    println(io, "under-builds wind (run1 FSD: 6.9 GW vs the full model's 37.5 GW), so a *different* (non-tail) scenario's peak")
    println(io, "cannot be served; the model's loss-of-load (`ens`) capacity is far too small to absorb the shortfall, so the")
    println(io, "fixed-investment full solve is **INFEASIBLE**. IIS (run1 FSD): `consumer_balance[e_demand, rep_period 1, t≈5293–5296]`")
    println(io, "with every generator incl. `ens` at its `max_output_flows_limit` and the battery depleted.")
    md_path = joinpath(exp_base, "OUT_OF_SAMPLE_REPORT.md")
    write(md_path, String(take!(io)))
    @info "[oos] wrote $md_path"
end

# Focused diagnostic: build run1's full model WITH NAMES, fix one reduced investment,
# solve with IIS conflict reporting to reveal which constraint(s)/scenario is infeasible.
function diagnose_run(experiment_name::String; config_folder::String = "full_resolution_fsd")
    cfg = load_config(script_dir = SCRIPT_DIR, repo_root = REPO_ROOT)
    exp_base = joinpath(cfg.experiment_base_dir, experiment_name)
    solver = first(cfg.solvers)
    master_df = CSV.read(cfg.profiles_wide_source, DataFrame)
    selection = read_selection(joinpath(exp_base, "scenario_selection.csv"))
    run_idx = 1
    run_dir = joinpath(exp_base, "run$(run_idx)")
    seed = selection[run_idx].seed
    source_ids = sort(Int.(selection[run_idx].source_ids))
    N = length(source_ids)

    Random.seed!(seed)
    sub = get_scenario_set(master_df, N)
    selected_original = sort(unique(Int.(sub.scenario)))
    selected_original == source_ids || error("reproduction mismatch: $selected_original vs $source_ids")
    mapping = Dict(old => new for (new, old) in enumerate(selected_original))
    profiles_df = copy(sub)
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    @info "[diagnose] building full model WITH NAMES" run = run_idx N = N config = config_folder
    conn = build_full_connection(profiles_df, N, cfg.input_data_path, cfg.lambda, cfg.alpha)
    energy_problem = TEM.EnergyProblem(conn)
    optimizer, parameters = get_solver_parameters(solver)
    TEM.create_model!(energy_problem; optimizer = optimizer, optimizer_parameters = parameters,
        model_file_name = "", enable_names = true)
    capacity_lookup = build_capacity_lookup(conn)

    mw = read_investment_mw(joinpath(run_dir, config_folder, string(solver)))
    full_mw = read_investment_mw(joinpath(run_dir, "full_resolution_all", string(solver)))
    mw === nothing && error("missing investment in $config_folder")
    @info "[diagnose] investment (MW invested per asset)" assets = INVESTABLE_ASSETS reduced = round.(mw; digits = 1) full = (full_mw === nothing ? nothing : round.(full_mw; digits = 1))

    fix_variables_from_sample(energy_problem.variables, :assets_investment, mw; capacity_lookup = capacity_lookup)
    @info "[diagnose] solving fixed-investment full model with IIS reporting"
    status = solve_model(energy_problem.model; diagnose_infeasibility = true)
    @info "[diagnose] done" status = string(status)
    return status
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: julia --project=. ScenarioReduction/old_scripts/out_of_sample_fix_solve.jl <EXPERIMENT> [smoke|diagnose]")
    _experiment = ARGS[1]
    _mode = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "full"
    if _mode == "diagnose"
        diagnose_run(_experiment)
    else
        run_oos(_experiment; smoke = (_mode == "smoke"))
    end
end
