function stochastic_dominance(
    connection;
    bounds::InvestmentBounds=load_investment_bounds(),
    covariance::InvestmentCovariance=investment_covariance(),
    sampling_mode::Symbol=:gaussian,            # :uniform | :gaussian
    gaussian_shrinkage::Real=0.2,
    gaussian_nonneg_mode::Symbol=:clip,
    solver::Symbol=:Gurobi,
)
    num_assets = length(INVESTABLE_ASSETS)
    num_samples = 512
    number_of_samples_sequences = 10
    for asset in INVESTABLE_ASSETS
        println("Asset values: $(get_asset_bounds(bounds, asset))")
    end
    println("Investment covariance: $(covariance.matrix)")

    lb = zeros(num_assets)
    ub = bounds.ub

    samples = if sampling_mode === :uniform
        generate_scrambled_Sobol_samples(num_samples, num_assets, lb, ub, number_of_samples_sequences)
    elseif sampling_mode === :gaussian
        Σ_shrunk = shrink_covariance(covariance.matrix; α=gaussian_shrinkage)
        [
            sobol_gaussian_samples_nonneg(
                num_samples,
                bounds.mean,
                Σ_shrunk;
                mode=gaussian_nonneg_mode,
                ub=ub,
                seed=seq,
            ) for seq in 1:number_of_samples_sequences
        ]
    else
        throw(ArgumentError("sampling_mode must be :uniform or :gaussian (got :$sampling_mode)"))
    end

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

    # === Build model ONCE ===
    # copy_conflict (IIS) requires a non-direct JuMP model; TEM.create_model uses direct_model=false by default.
    time_to_create = @elapsed model, expressions = TEM.create_model(connection, variables, constraints, profiles)
    println("Time to create model (one-time): $(time_to_create)")

    optimizer, parameters = get_solver_parameters(solver)
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in parameters]...)
    JuMP.set_silent(model)

    configure_for_warmstart!(model; solver=solver)

    presolve_disabled = false
    stop = false
    for sequence in 1:number_of_samples_sequences
        stop && break
        for sample in eachcol(samples[sequence])
            time_to_fix = @elapsed fix_variables_from_sample(
                variables,
                :assets_investment,
                Vector(sample);
                capacity_lookup,
            )
            elapsed_time = @elapsed status = solve_model(model)
            if status == JuMP.OPTIMAL
                println("fix=$(time_to_fix)  solve=$(elapsed_time)  obj=$(JuMP.objective_value(model))  status=$status")
            else
                println("fix=$(time_to_fix)  solve=$(elapsed_time)  status=$status")
            end
            if !presolve_disabled
                # After the first solve the basis is informative; presolve would
                # discard it and prevent warm-starting subsequent re-solves.
                disable_presolve!(model; solver=solver)
                presolve_disabled = true
            end
        end
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