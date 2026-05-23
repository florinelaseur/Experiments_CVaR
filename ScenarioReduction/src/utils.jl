
function print_model_variables_before_clustering(connection)
    layout = TC.ProfilesTableLayout(;
    year=:milestone_year,
    cols_to_groupby=[:milestone_year, :scenario],
    )

    time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)
    TEM.populate_with_defaults!(connection)
    TEM.create_internal_tables!(connection)
    variables = TEM.compute_variables_indices(connection)
    

    # Inspect investment space dimension
    inv_df = DataFrame(variables[:assets_investment].indices)
    @info "Investment space dimension: $(nrow(inv_df))"
    for (i, row) in enumerate(eachrow(inv_df))
        @info "Investment $i: $row"
    end

    constraints = TEM.compute_constraints_indices(connection)
    for (name, cons) in constraints
        n = TEM.get_num_rows(connection, cons)
        @info "Constraint :$name — $n rows"
    end
    profiles   = TEM.prepare_profiles_structure(connection)          # profile setup
    model, expressions = TEM.create_model(connection, variables, constraints, profiles)
    solve_model(model)
    
end


function solve_model(model::JuMP.Model)

    JuMP.optimize!(model)

    # Check solution status
    if JuMP.termination_status(model) != JuMP.OPTIMAL
        @warn("Model status different from optimal")
        return nothing
    end

    return
end
