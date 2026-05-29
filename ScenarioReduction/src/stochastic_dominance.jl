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

    # === Build model ONCE ===
    # copy_conflict (IIS) requires a non-direct JuMP model; TEM.create_model uses direct_model=false by default.
    time_to_create = @elapsed model, expressions = TEM.create_model(connection, variables, constraints, profiles)
    println("Time to create model (one-time): $(time_to_create)")

    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in parameters]...)
    JuMP.set_silent(model)

    configure_for_warmstart!(model; solver=solver)

    # === Screening loop: cheap adequacy pre-filter, then LP only on survivors ===
    diagnostics = DataFrame(;
        sequence=Int[], sample_id=Int[], passed_cut=Bool[],
        binding_scenario=Int[], lp_status=String[], objective=Float64[],
    )
    n_rejected = 0
    n_optimal = 0
    n_infeasible = 0
    presolve_disabled = false
    stop = false
    for sequence in 1:number_of_samples_sequences
        stop && break
        for (sample_id, sample) in enumerate(eachcol(samples[sequence]))
            x = Vector(sample)

            verdict = use_adequacy_cuts ? adequacy_verdict(x, cuts_list) :
                      (passed=true, binding_scenario=0)
            if !verdict.passed
                n_rejected += 1
                push!(diagnostics, (sequence, sample_id, false, verdict.binding_scenario, "REJECTED_ADEQUACY", NaN))
                continue
            end

            time_to_fix = @elapsed fix_variables_from_sample(
                variables,
                :assets_investment,
                x;
                capacity_lookup,
            )
            elapsed_time = @elapsed status = solve_model(model; diagnose_infeasibility=false)
            obj = status == JuMP.OPTIMAL ? JuMP.objective_value(model) : NaN
            status == JuMP.OPTIMAL ? (n_optimal += 1) : (n_infeasible += 1)
            push!(diagnostics, (sequence, sample_id, true, verdict.binding_scenario, string(status), obj))

            if status == JuMP.OPTIMAL
                println("seq=$sequence sample=$sample_id  fix=$(time_to_fix)  solve=$(elapsed_time)  obj=$(obj)  status=$status")
            else
                println("seq=$sequence sample=$sample_id  fix=$(time_to_fix)  solve=$(elapsed_time)  status=$status")
            end

            if !presolve_disabled
                # After the first solve the basis is informative; presolve would
                # discard it and prevent warm-starting subsequent re-solves.
                disable_presolve!(model; solver=solver)
                presolve_disabled = true
            end
        end
    end

    CSV.write(joinpath(output_dir, "screening_diagnostics.csv"), diagnostics)
    total = n_rejected + n_optimal + n_infeasible
    println("Screening summary: LP-optimal=$n_optimal  LP-infeasible=$n_infeasible  in-loop-rejected=$n_rejected  (total LP attempts=$total)")
    if accept !== nothing
        println("  Adequacy rejection happens inside the sampler (see sampling_stats.csv); the in-loop pre-filter is a safety net and should report 0.")
    end
    println("  Solver load = num_samples × seeds = $num_samples × $number_of_samples_sequences = $(num_samples*number_of_samples_sequences) LP solves; lower the kwargs for quick runs.")

    return (
        rejected=n_rejected,
        optimal=n_optimal,
        infeasible=n_infeasible,
        center=center,
        cuts=cuts_list,
        diagnostics=diagnostics,
    )
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
