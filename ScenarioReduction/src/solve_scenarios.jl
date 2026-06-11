# Multi-scenario stochastic TEM solve for an explicit scenario list.
# Same core path as main.jl (:per_scenario cluster → EnergyProblem → solve).
# Only scenario selection differs: filter profiles-wide.csv instead of get_scenario_set.
# Needs TEM/TC/TIO/DuckDB/JuMP/Distances/get_solver_parameters from the includer's scope.

using DataFrames: DataFrame, nrow

"""Filter profiles-wide.csv to `scenario_ids`, renumber to 1:K, update stochastic-scenario.csv."""
function prepare_scenario_subset!(
    scenario_ids::AbstractVector{<:Integer},
    input_data_path::AbstractString,
)
    profiles_path = joinpath(input_data_path, "profiles-wide.csv")
    isfile(profiles_path) || error("profiles-wide.csv not found: $profiles_path")

    profiles_df = CSV.read(profiles_path, DataFrame)
    nrow(profiles_df) > 0 || error("profiles-wide.csv is empty: $profiles_path")

    original_ids = sort(unique(Int.(scenario_ids)))
    isempty(original_ids) && error("scenario_ids must be non-empty")

    available = Set(Int.(profiles_df.scenario))
    missing = [id for id in original_ids if id ∉ available]
    !isempty(missing) &&
        error("scenario id(s) not in profiles-wide.csv: $(missing)")

    id_set = Set(original_ids)
    profiles_df = profiles_df[profiles_df.scenario .∈ Ref(id_set), :]
    mapping = Dict(old => new for (new, old) in enumerate(original_ids))
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    K = length(original_ids)
    CSV.write(profiles_path, profiles_df; writeheader=true)
    CSV.write(
        joinpath(input_data_path, "stochastic-scenario.csv"),
        DataFrame(; scenario=collect(1:K), probability=fill(1.0 / K, K));
        writeheader=true,
    )

    return original_ids
end

"""
    solve_scenarios(scenario_ids, input_data_path; kwargs...)

Run the multi-scenario stochastic TEM solve on the given scenario indices (must exist in
`profiles-wide.csv` under `input_data_path`).

`clustering_mode`: `:dummy` (full hourly, default) or `:cluster` (TC.cluster! with
`representative_periods`).

Returns `(scenario_ids, objective_value, termination_status)`.
"""
function solve_scenarios(
    scenario_ids::AbstractVector{<:Integer},
    input_data_path::AbstractString;
    solver::Symbol=:Gurobi,
    lambda::Real=0.1,
    alpha::Real=0.95,
    clustering_mode::Symbol=:dummy,
    representative_periods::Integer=30,
    period_duration::Integer=24,
    use_ratio::Bool=false,
    use_names::Bool=false,
)
    clustering_mode in (:dummy, :cluster) ||
        error("clustering_mode must be :dummy or :cluster (got :$clustering_mode)")

    selected = prepare_scenario_subset!(scenario_ids, input_data_path)

    connection = DuckDB.DBInterface.connect(DuckDB.DB)
    try
        TIO.read_csv_folder(connection, input_data_path)

        DuckDB.query(
            connection,
            """
            UPDATE model_parameters
            SET
                risk_aversion_weight_lambda = $(lambda),
                risk_aversion_confidence_level_alpha = $(alpha);
            """,
        )

        if use_ratio
            DuckDB.query(
                connection,
                """
                UPDATE profiles_wide
                SET
                    solar = solar / demand,
                    wind_offshore = wind_offshore / demand,
                    wind_onshore = wind_onshore / demand,
                    hydro_inflow = hydro_inflow / demand;
                """,
            )
        end

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

        if clustering_mode == :dummy
            TC.dummy_cluster!(connection; layout=layout)
        else
            fit_kwargs = Dict(:learning_rate => 0.001, :niters => 2000)
            TC.cluster!(
                connection,
                period_duration,
                representative_periods;
                method=:convex_hull,
                distance=Distances.Euclidean(),
                weight_type=:dirac,
                layout=layout,
                clustering_kwargs=fit_kwargs,
                weight_fitting_kwargs=fit_kwargs,
            )
            if use_ratio
                DuckDB.query(
                    connection,
                    """
                    UPDATE profiles_rep_periods AS x
                        SET value =
                            CASE
                                WHEN x.profile_name = 'demand' THEN x.value
                                ELSE x.value * d.value
                            END
                        FROM profiles_rep_periods AS d
                        WHERE d.timestep = x.timestep
                        AND d.rep_period = x.rep_period
                        AND d.milestone_year = x.milestone_year
                        AND d.scenario = x.scenario
                        AND d.profile_name = 'demand';
                    """,
                )
            end
        end

        if use_ratio
            DuckDB.query(
                connection,
                """
                UPDATE profiles AS x
                    SET value =
                        CASE
                            WHEN x.profile_name = 'demand' THEN x.value
                            ELSE x.value * d.value
                        END
                    FROM profiles AS d
                    WHERE d.timestep = x.timestep
                    AND d.milestone_year = x.milestone_year
                    AND d.scenario = x.scenario
                    AND d.profile_name = 'demand';
                """,
            )
        end

        TEM.populate_with_defaults!(connection)
        DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")

        optimizer, parameters = get_solver_parameters(solver)
        energy_problem = TEM.EnergyProblem(connection)
        TEM.create_model!(
            energy_problem;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=use_names,
        )
        TEM.solve_model!(energy_problem)

        return (
            scenario_ids=selected,
            objective_value=energy_problem.objective_value,
            termination_status=string(energy_problem.termination_status),
        )
    finally
        try
            DuckDB.DBInterface.close!(connection)
        catch
        end
    end
end

"""Format selected source scenario ids as a sorted, space-separated string (for CSV)."""
format_selected_scenarios(ids) = join(sort(collect(Int.(ids))), " ")

"""Return true if a DuckDB table named `name` exists on `connection`."""
function table_exists(connection, name::AbstractString)
    df = DataFrame(DuckDB.query(connection, "SHOW TABLES"))
    col = hasproperty(df, :name) ? df.name : df[!, 1]
    return any(==(name), string.(col))
end

"""
    select_export_tables(table_names; exclude_vars=String[], keep_constraints_only=String[])

Pure selection of which solution tables to export, given the full list of
candidate `var_*` / `cons_*` / `obj_*` table names:

- `var_*`  : kept unless the name is in `exclude_vars`.
- `cons_*` : kept ONLY if the name is in `keep_constraints_only`.
- anything else (e.g. `obj_*`) : kept.

Returns the filtered list of table names (order preserved).
"""
function select_export_tables(
    table_names;
    exclude_vars::AbstractVector{<:AbstractString}=String[],
    keep_constraints_only::AbstractVector{<:AbstractString}=String[],
)
    keep = String[]
    for t in table_names
        s = string(t)
        if startswith(s, "cons_")
            (s in keep_constraints_only) && push!(keep, s)
        elseif startswith(s, "var_")
            (s in exclude_vars) || push!(keep, s)
        else
            push!(keep, s)
        end
    end
    return keep
end

"""
    export_selected_solution_tables(connection, output_folder;
        exclude_vars=["var_flow", "var_storage_level_rep_period"],
        keep_constraints_only=["cons_scenario_tail_excess"])

Lean replacement for `TEM.export_solution_to_csv_files`: writes only the wanted
solution tables straight from DuckDB (via `COPY`), so the large `var_flow` /
`var_storage_level_rep_period` tables and the unwanted `cons_*` tables are NEVER
written to disk (no write-then-delete). Assumes `TEM.save_solution!` has already
populated the solution tables on `connection`.

Returns the vector of table names that were exported.
"""
function export_selected_solution_tables(
    connection,
    output_folder;
    exclude_vars::AbstractVector{<:AbstractString}=["var_flow", "var_storage_level_rep_period"],
    keep_constraints_only::AbstractVector{<:AbstractString}=["cons_scenario_tail_excess"],
)
    mkpath(output_folder)
    table_names = String[
        row.table_name for row in DuckDB.query(
            connection,
            "FROM duckdb_tables() WHERE (table_name LIKE 'var_%' OR " *
            "table_name LIKE 'cons_%' OR table_name LIKE 'obj_%') AND estimated_size > 0",
        )
    ]
    selected = select_export_tables(
        table_names; exclude_vars=exclude_vars, keep_constraints_only=keep_constraints_only,
    )
    for table_name in selected
        output_file = joinpath(output_folder, "$(table_name).csv")
        DuckDB.execute(connection, "COPY $table_name TO '$output_file' (HEADER, DELIMITER ',')")
    end
    return selected
end

"""
    export_solution_stats(energy_problem, connection, output_folder; label, solver,
                          num_scenarios, time_to_cluster, time_to_read,
                          time_to_create, time_to_solve, time_to_save)

Extract the same solution statistics as `main.jl` (minus the resolve-benchmark
fields) from a solved `energy_problem`, write a one-row `results.csv` into
`output_folder`, and return the row as a `NamedTuple`.

`value_at_risk_threshold_mu` is read from `var_value_at_risk_threshold_mu` when
present; it is `NaN` when TEM drops the CVaR term (single scenario or `lambda<=0`).
"""
function export_solution_stats(
    energy_problem,
    connection,
    output_folder;
    label::AbstractString,
    solver::Symbol,
    num_scenarios::Integer,
    time_to_cluster::Real,
    time_to_read::Real,
    time_to_create::Real,
    time_to_solve::Real,
    time_to_save::Real,
)
    mu_value = NaN
    if table_exists(connection, "var_value_at_risk_threshold_mu")
        mu_df = DataFrame(TIO.get_table(connection, "var_value_at_risk_threshold_mu"))
        if nrow(mu_df) == 1
            mu_value = only(mu_df.solution)
        elseif nrow(mu_df) > 1
            mu_value = first(mu_df.solution)
        end
    end

    var_flow_df = DataFrame(TIO.get_table(connection, "var_flow"))
    flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
    flow_smr_ccs =
        filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
    water_borrowed = filter(
        row -> row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
        var_flow_df,
    )

    n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
    n_lol_smr_ccs = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))
    amount_water_borrowed = isempty(water_borrowed.solution) ? 0.0 : sum(water_borrowed.solution)

    row = (
        label=String(label),
        solver=solver,
        num_scenarios=Int(num_scenarios),
        time_to_cluster=Float64(time_to_cluster),
        time_to_read=Float64(time_to_read),
        time_to_create=Float64(time_to_create),
        time_to_solve=Float64(time_to_solve),
        time_to_save=Float64(time_to_save),
        objective_value=Float64(energy_problem.objective_value),
        termination_status=string(energy_problem.termination_status),
        num_constraints=JuMP.num_constraints(
            energy_problem.model; count_variable_in_set_constraints=false,
        ),
        num_variables=JuMP.num_variables(energy_problem.model),
        num_loss_of_load_e_demand=n_lol_ens,
        num_loss_of_load_h2_demand=n_lol_smr_ccs,
        water_borrowed=amount_water_borrowed,
        value_at_risk_threshold_mu=mu_value,
    )

    mkpath(output_folder)
    CSV.write(joinpath(output_folder, "results.csv"), DataFrame([row]); writeheader=true)
    return row
end

"""
    solve_full_resolution!(scenario_ids, input_data_path, output_dir; label,
                           solvers=[:Gurobi], lambda=0.1, alpha=0.95,
                           fix_investment_mw=nothing)

Solve the multi-scenario stochastic TEM model on `scenario_ids` at full temporal
resolution (`TC.dummy_cluster!`). For each solver, exports a lean set of solution
CSVs (via `export_selected_solution_tables`, skipping `var_flow`,
`var_storage_level_rep_period`, and all `cons_*` except `cons_scenario_tail_excess`),
`results.csv` (via `export_solution_stats`), `operational_cost_per_scenario.csv`,
and `tail_scenarios.csv` (via `export_cvar_tail_diagnostics`) into
`<output_dir>/<solver>/`.

`fix_investment_mw` (MW per asset, `INVESTABLE_ASSETS` order, e.g. from
`read_investment_mw`) fixes `assets_investment` after model creation — used for
out-of-sample evaluation of a stored investment. The battery's energy is slaved to
its power and seasonal storages aren't investable, so this fully pins the
investment. A fixed-investment solve can be INFEASIBLE: any non-OPTIMAL solve is
recorded in `results.csv` (objective `NaN`) without solution exports instead of
erroring.

Returns `(selected=..., results=DataFrame)` where `results` has one row per solver.
"""
function solve_full_resolution!(
    scenario_ids::AbstractVector{<:Integer},
    input_data_path::AbstractString,
    output_dir::AbstractString;
    label::AbstractString="full",
    solvers::AbstractVector{Symbol}=[:Gurobi],
    lambda::Real=0.1,
    alpha::Real=0.95,
    use_names::Bool=false,
    fix_investment_mw::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    selected = prepare_scenario_subset!(scenario_ids, input_data_path)

    results = DataFrame(;
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

    connection = DuckDB.DBInterface.connect(DuckDB.DB)
    try
        TIO.read_csv_folder(connection, input_data_path)
        DuckDB.query(
            connection,
            """
            UPDATE model_parameters
            SET
                risk_aversion_weight_lambda = $(lambda),
                risk_aversion_confidence_level_alpha = $(alpha);
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
        time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)
        TEM.populate_with_defaults!(connection)
        DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")

        capacity_lookup =
            fix_investment_mw === nothing ? nothing : build_capacity_lookup(connection)

        for solver in solvers
            optimizer, parameters = get_solver_parameters(solver)

            @info "Full-resolution solve" label=label solver=solver scenarios=selected
            time_to_read = @elapsed energy_problem = TEM.EnergyProblem(connection)
            time_to_create = @elapsed TEM.create_model!(
                energy_problem;
                optimizer=optimizer,
                optimizer_parameters=parameters,
                model_file_name="",
                enable_names=use_names,
            )

            if fix_investment_mw !== nothing
                @info "Fixing assets_investment (MW per asset)" label=label solver=solver mw=fix_investment_mw
                fix_variables_from_sample(
                    energy_problem.variables,
                    :assets_investment,
                    Float64.(fix_investment_mw);
                    capacity_lookup,
                )
            end

            output_folder = joinpath(output_dir, string(solver))
            mkpath(output_folder)

            time_to_solve = @elapsed TEM.solve_model!(energy_problem)
            if energy_problem.solved
                time_to_save = @elapsed begin
                    TEM.save_solution!(energy_problem)
                    # Lean export: skip the large var_flow / var_storage_level_rep_period
                    # tables and all cons_* except cons_scenario_tail_excess (never written).
                    export_selected_solution_tables(connection, output_folder)
                end

                row = export_solution_stats(
                    energy_problem,
                    connection,
                    output_folder;
                    label=label,
                    solver=solver,
                    num_scenarios=length(selected),
                    time_to_cluster=time_to_cluster,
                    time_to_read=time_to_read,
                    time_to_create=time_to_create,
                    time_to_solve=time_to_solve,
                    time_to_save=time_to_save,
                )
                push!(results, row)

                export_operational_cost_per_scenario(energy_problem, output_folder)
                export_cvar_tail_diagnostics(energy_problem, connection, output_folder)
            else
                # No solution to save/export (e.g. INFEASIBLE under fixed investments);
                # results.csv still records the outcome and acts as the resume marker.
                @warn "Solve did not reach OPTIMAL; recording status without solution exports" label =
                    label solver = solver termination_status =
                    string(energy_problem.termination_status)
                row = (
                    label=String(label),
                    solver=solver,
                    num_scenarios=Int(length(selected)),
                    time_to_cluster=Float64(time_to_cluster),
                    time_to_read=Float64(time_to_read),
                    time_to_create=Float64(time_to_create),
                    time_to_solve=Float64(time_to_solve),
                    time_to_save=0.0,
                    objective_value=NaN,
                    termination_status=string(energy_problem.termination_status),
                    num_constraints=JuMP.num_constraints(
                        energy_problem.model; count_variable_in_set_constraints=false,
                    ),
                    num_variables=JuMP.num_variables(energy_problem.model),
                    num_loss_of_load_e_demand=0,
                    num_loss_of_load_h2_demand=0,
                    water_borrowed=NaN,
                    value_at_risk_threshold_mu=NaN,
                )
                push!(results, row)
                CSV.write(joinpath(output_folder, "results.csv"), DataFrame([row]); writeheader=true)
            end
        end
    finally
        try
            DuckDB.DBInterface.close!(connection)
        catch
        end
    end

    return (selected=selected, results=results)
end
