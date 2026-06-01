function stochastic_dominance(
    connection;
    bounds::InvestmentBounds=load_investment_bounds(),
    covariance::InvestmentCovariance=investment_covariance(),
    sampling_mode::Symbol=:uniform,            # :uniform | :gaussian
    gaussian_shrinkage::Real=0.2,
    gaussian_nonneg_mode::Symbol=:clip,
    solver::Symbol=:Gurobi,
    use_adequacy_cuts::Bool=true,               # cheap pre-filter + feasibility mean-shift
    mean_shift::Bool=true,                       # centre samples on the feasibility frontier
    output_dir::String=joinpath(@__DIR__, "..", "outputs"),
    num_samples::Int=512,
    number_of_samples_sequences::Int=5,   # = number of scramble seeds
    input_data_path::AbstractString,      # base input folder; re-read per single-scenario model
    max_runtime_sec::Union{Nothing,Real}=nothing,      # wall-clock budget for the per-scenario eval
    solve_time_limit_sec::Union{Nothing,Real}=nothing, # per-LP solver time limit
)
    num_assets = length(INVESTABLE_ASSETS)
    for asset in INVESTABLE_ASSETS
        println("Asset values: $(get_asset_bounds(bounds, asset))")
    end
    println("Investment covariance: $(covariance.matrix)")

    lb = zeros(num_assets)
    ub = bounds.ub

    # === Cluster + internal tables (must precede index/cut construction) ===
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    TC.dummy_cluster!(connection; layout=layout)
    TEM.populate_with_defaults!(connection)
    TEM.create_internal_tables!(connection)

    variables   = TEM.compute_variables_indices(connection)
    constraints = TEM.compute_constraints_indices(connection)
    profiles    = TEM.prepare_profiles_structure(connection)

    capacity_lookup = build_capacity_lookup(connection)
    mapping_audit = audit_investment_mapping(
        variables;
        capacity_lookup,
    )
    if !mapping_audit.permutation_ok
        @warn "Naive sample-to-container zip would mis-assign investments; using indices-based alignment"
    end

    optimizer, parameters = get_solver_parameters(solver)

    # === Adequacy cuts (cheap necessary feasibility condition) + feasibility centre ===
    mkpath(output_dir)
    center = Vector{Float64}(bounds.mean)
    cuts_list = AdequacyCuts[]
    if use_adequacy_cuts
        profiles_wide_df = DataFrame(TIO.get_table(connection, "profiles_wide"))
        params = read_adequacy_params(connection)
        selected_scenarios = sort(unique(Int.(profiles_wide_df.scenario)))
        cuts_list = [build_adequacy_cuts(profiles_wide_df, s, params) for s in selected_scenarios]
        save_adequacy_cuts_csv(joinpath(output_dir, "adequacy_cuts.csv"), cuts_list)
        println("Adequacy cuts: scenarios=$selected_scenarios  demanding-hours per scenario=$( [length(c.b) for c in cuts_list] )  (peak_demand=$(params.peak_demand))")

        if mean_shift
            res = feasibility_center(
                cuts_list, bounds.mean, ub;
                optimizer=optimizer, optimizer_parameters=parameters,
            )
            save_feasibility_center_csv(
                joinpath(output_dir, "feasibility_center.csv"),
                INVESTABLE_ASSETS, bounds.mean, res.center, res.shift, res.status,
            )
            if res.center !== nothing
                center = Vector{Float64}(res.center)
                println("Feasibility centre μ* (LP=$(res.status)); shift MW=$(round.(res.shift; digits=1))")
            else
                @warn "feasibility_center LP not OPTIMAL ($(res.status)); no in-bounds portfolio meets the cuts — keeping bounds.mean as centre"
            end
        end
    end

    # === Samples ===
    # With adequacy cuts on, draw per seed until `num_samples` cut-passing samples are
    # found (reject-to-target), so the solver gets a full pool of probable candidates.
    # `:uniform` ignores `center` (the cuts steer); `:gaussian` draws around `center`
    # (μ* after the mean-shift, else bounds.mean). Without cuts, fall back to the
    # original clip/uniform behaviour.
    accept = (use_adequacy_cuts && !isempty(cuts_list)) ?
        (x -> adequacy_verdict(x, cuts_list).passed) : nothing
    Σ_shrunk = sampling_mode === :gaussian ? shrink_covariance(covariance.matrix; α=gaussian_shrinkage) : nothing
    sampling_stats = DataFrame(;
        sequence=Int[], seed=Int[], target=Int[], n_drawn=Int[], acceptance_rate=Float64[],
    )

    samples = if accept === nothing
        if sampling_mode === :uniform
            generate_scrambled_Sobol_samples(num_samples, num_assets, lb, ub, number_of_samples_sequences)
        elseif sampling_mode === :gaussian
            [sobol_gaussian_samples_nonneg(num_samples, center, Σ_shrunk;
                 mode=gaussian_nonneg_mode, ub=ub, seed=seq) for seq in 1:number_of_samples_sequences]
        else
            throw(ArgumentError("sampling_mode must be :uniform or :gaussian (got :$sampling_mode)"))
        end
    else
        out = Vector{Matrix{Float64}}(undef, number_of_samples_sequences)
        for seq in 1:number_of_samples_sequences
            res = if sampling_mode === :uniform
                scrambled_sobol_uniform_reject_to_target(num_samples, lb, ub; accept=accept, seed=seq)
            elseif sampling_mode === :gaussian
                sobol_gaussian_reject_to_target(num_samples, center, Σ_shrunk; accept=accept, ub=ub, seed=seq)
            else
                throw(ArgumentError("sampling_mode must be :uniform or :gaussian (got :$sampling_mode)"))
            end
            out[seq] = res.samples
            kept = size(res.samples, 2)
            rate = res.n_drawn > 0 ? kept / res.n_drawn : 0.0
            push!(sampling_stats, (seq, seq, num_samples, res.n_drawn, rate))
            println("  seed=$seq: kept $kept/$num_samples cut-passing from $(res.n_drawn) draws (acceptance $(round(100*rate; digits=1))%)")
        end
        CSV.write(joinpath(output_dir, "sampling_stats.csv"), sampling_stats)
        out
    end

    # === Phase B: per-scenario operational evaluation ===
    # SD screening needs the operational cost of each sample under EACH scenario
    # (the matrix C_i(z_k)), not one probability-weighted objective. So build ONE
    # single-scenario model at a time, solve every sample against it, record that
    # scenario's cost column, then free it — only one operational model is ever in
    # memory. No multi-scenario TEM.create_model is used for per-sample evaluation.
    #
    # Risk params are not threaded in: each model has a single scenario, for which
    # TEM drops the CVaR term entirely (n_scenarios <= 1), so the recorded objective
    # is just that scenario's total cost.
    selected_scenarios = sort(unique(Int.(DataFrame(TIO.get_table(connection, "profiles_wide")).scenario)))
    N = length(selected_scenarios)
    @info "Per-scenario evaluation" scenarios=selected_scenarios num_models=N

    if solve_time_limit_sec !== nothing
        if solver == :Gurobi
            parameters["TimeLimit"] = Float64(solve_time_limit_sec)
        elseif solver == :HiGHS
            parameters["time_limit"] = Float64(solve_time_limit_sec)
        end
    end

    # Stable row layout: one row per (sequence, sample), independent of scenario,
    # so every scenario's column lines up in the cost matrix.
    row_index  = [(seq, sid) for seq in 1:number_of_samples_sequences
                             for sid in 1:size(samples[seq], 2)]
    total_rows = length(row_index)
    cost = fill(NaN, total_rows, N)

    diagnostics = DataFrame(;
        sequence=Int[], sample_id=Int[], scenario=Int[],
        passed_cut=Bool[], lp_status=String[],
        objective=Float64[], solve_elapsed_sec=Float64[],
    )

    t_start = time()
    stop = false
    for (i, s) in enumerate(selected_scenarios)
        stop && break
        @info "Building single-scenario model" scenario=s index="$i/$N"
        time_to_build = @elapsed m = build_single_scenario_model(
            s, input_data_path, optimizer, parameters; solver=solver,
        )
        println("Built single-scenario model for scenario $s ($i/$N): build=$(time_to_build)s")

        presolve_disabled = false
        row = 0
        for sequence in 1:number_of_samples_sequences
            stop && break
            for (sample_id, sample) in enumerate(eachcol(samples[sequence]))
                row += 1
                x = Vector(sample)

                # Per-scenario adequacy safety net (reject-to-target already guarantees
                # every kept sample passes every scenario's cuts, so this should be a no-op).
                passed = (use_adequacy_cuts && !isempty(cuts_list)) ?
                         passes_adequacy(x, cuts_list[i]) : true
                if !passed
                    push!(diagnostics, (sequence, sample_id, s, false, "REJECTED_ADEQUACY", NaN, 0.0))
                    continue
                end

                fix_variables_from_sample(
                    m.variables, :assets_investment, x; capacity_lookup=m.capacity_lookup,
                )
                elapsed_time = @elapsed status = solve_model(m.model; diagnose_infeasibility=false)
                obj = status == JuMP.OPTIMAL ? JuMP.objective_value(m.model) : NaN
                cost[row, i] = obj
                push!(diagnostics, (sequence, sample_id, s, true, string(status), obj, elapsed_time))

                if !presolve_disabled
                    # After the first solve the basis is informative; presolve would
                    # discard it and prevent warm-starting the next sample on this model.
                    disable_presolve!(m.model; solver=solver)
                    presolve_disabled = true
                end

                if max_runtime_sec !== nothing && (time() - t_start) > max_runtime_sec
                    @warn "max_runtime_sec=$max_runtime_sec exceeded; stopping after scenario $s (seq $sequence, sample $sample_id)"
                    stop = true
                    break
                end
            end
        end

        # Free this scenario's connection + model before building the next one.
        try
            DuckDB.DBInterface.close!(m.connection)
        catch err
            @debug "Could not close connection for scenario $s" err
        end
        m = nothing
        GC.gc()
        println("Finished scenario $s: $(count(!isnan, view(cost, :, i)))/$(total_rows) samples solved")
    end

    CSV.write(joinpath(output_dir, "screening_diagnostics.csv"), diagnostics)

    # === Cost matrix: rows = (sequence, sample), columns = scenarios ===
    cost_matrix = DataFrame(;
        sample_id=[r[2] for r in row_index],
        sequence=[r[1] for r in row_index],
    )
    for (i, s) in enumerate(selected_scenarios)
        cost_matrix[!, Symbol("scenario_$(s)")] = cost[:, i]
    end
    CSV.write(joinpath(output_dir, "cost_matrix.csv"), cost_matrix)

    n_rejected   = count(!, diagnostics.passed_cut)
    n_optimal    = count(==("OPTIMAL"), diagnostics.lp_status)
    n_infeasible = nrow(diagnostics) - n_rejected - n_optimal
    println("Per-scenario screening summary: LP-optimal=$n_optimal  LP-non-optimal=$n_infeasible  cut-rejected=$n_rejected  (diagnostic rows=$(nrow(diagnostics)))")
    println("  Solver load = scenarios × samples × seeds = $N × $num_samples × $number_of_samples_sequences = $(N*num_samples*number_of_samples_sequences) single-scenario LP solves.")
    println("  cost_matrix.csv: $(nrow(cost_matrix)) samples × $N scenarios (NaN = infeasible / not evaluated).")

    return (
        scenarios=selected_scenarios,
        cost_matrix=cost_matrix,
        cost=cost,
        diagnostics=diagnostics,
        center=center,
        cuts=cuts_list,
        optimal=n_optimal,
        infeasible=n_infeasible,
        rejected=n_rejected,
    )
end

"""Set a per-`optimize!` time limit on `model` (seconds)."""
function set_solver_time_limit!(model; solver::Symbol, seconds::Real)
    seconds > 0 || error("solve time limit must be positive, got $seconds")
    if solver == :Gurobi
        JuMP.set_optimizer_attribute(model, "TimeLimit", Float64(seconds))
    elseif solver == :HiGHS
        JuMP.set_optimizer_attribute(model, "time_limit", Float64(seconds))
    else
        @warn "No time-limit mapping for solver :$solver; skipping"
    end
    return nothing
end

function configure_for_warmstart!(model; solver::Symbol=:Gurobi)
    # Use simplex, not interior-point — simplex maintains a basis reusable across re-solves.
    # Dual simplex: after fix() changes bounds, the current basis stays dual-feasible.
    if solver == :Gurobi
        JuMP.set_optimizer_attribute(model, "Method", 1)  # dual simplex
    elseif solver == :HiGHS
        JuMP.set_optimizer_attribute(model, "solver", "simplex")
        JuMP.set_optimizer_attribute(model, "simplex_strategy", 1)  # dual simplex
    end
    return nothing
end

function disable_presolve!(model; solver::Symbol=:Gurobi)
    if solver == :Gurobi
        JuMP.set_optimizer_attribute(model, "Presolve", 0)
    elseif solver == :HiGHS
        JuMP.set_optimizer_attribute(model, "presolve", "off")
    end
    return nothing
end
