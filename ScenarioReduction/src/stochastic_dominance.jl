function stochastic_dominance(
    connection;
    bounds::InvestmentBounds=load_investment_bounds(),
    covariance::InvestmentCovariance=investment_covariance(),
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

    samples = generate_scrambled_Sobol_samples(num_samples, num_assets, lb, ub, number_of_samples_sequences)

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
    
    # === Build model ONCE ===
    time_to_create = @elapsed model, expressions = TEM.create_model(connection, variables, constraints, profiles)
    println("Time to create model (one-time): $(time_to_create)")
    
    JuMP.set_silent(model)  # mute solver chatter; do once
    
    # (Optional) configure solver for warm-starting — see below
    configure_for_warmstart!(model)
    
    count = 0
    for sequence in 1:number_of_samples_sequences
        for sample in eachcol(samples[sequence])
            if count == 1
                JuMP.set_optimizer_attribute(model, "presolve", "off")
            end
            time_to_fix = @elapsed fix_variables_from_sample(variables, :assets_investment, Vector(sample))
            elapsed_time = @elapsed solve_model(model)
            println("fix=$(time_to_fix)  solve=$(elapsed_time)  obj=$(JuMP.objective_value(model))  status=$(JuMP.termination_status(model))")
            count += 1
        end
    end
end

function configure_for_warmstart!(model)
    # Use the simplex method, not interior-point.
    # Simplex maintains a "basis" (the set of active variables at the corner solution)
    # which can be reused across re-solves. Interior-point (IPM) cannot warm-start.
    JuMP.set_optimizer_attribute(model, "solver", "simplex")

    # Use the *dual* simplex specifically.
    # When you change variable bounds (which is what fix() does — it sets lb=ub=value),
    # the current basis remains feasible for the DUAL problem but not the primal.
    # Dual simplex restarts from there and converges in few iterations.
    # Primal simplex would have to repair primal infeasibility — slower in this case.
    # In HiGHS: simplex_strategy = 1 means dual simplex.
    JuMP.set_optimizer_attribute(model, "simplex_strategy", 1)
end