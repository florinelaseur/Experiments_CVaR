
function get_solver_parameters(optimizer::Symbol)
    if optimizer == :HiGHS
        return HiGHS.Optimizer,
        Dict(
            "output_flag" => true,
            "solver" => "hipo",
            "parallel" => "on",
            "run_crossover" => "off"
        )
    elseif optimizer == :Gurobi
        return Gurobi.Optimizer, Dict("OutputFlag" => 1)
    elseif optimizer == :Xpress
        return Xpress.Optimizer, Dict("DEFAULTALG" => 4)
    else
        return HiGHS.Optimizer, Dict()
    end
end


function fix_variables_from_solution!(benchmark_model, reduced_model, var_symbol)

    var_to_fix = benchmark_model.variables[var_symbol].container
    val_to_fix = JuMP.value(reduced_model.variables[var_symbol].container)

    for (var, val) in zip(var_to_fix, val_to_fix)
        JuMP.fix(var, val; force=true)
    end
end

function create_init_rps_hourly(connection, period_duration, profiles)
    profiles_str = join(profiles, ", ")

    DuckDB.query(
        connection,
        """
        CREATE TABLE init_rps AS
        WITH melted AS (
            SELECT
                year,
                scenario,
                timestep,
                profile_name,
                value,
                split_part(profile_name, '_', 1) AS location
            FROM profiles_wide
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        with_hour AS (
            SELECT
                *,
                ((timestep - 1) % $period_duration) + 1 AS hour
            FROM melted
        ),

        -- isolate demand by location
        demand_by_loc AS (
            SELECT
                year,
                scenario,
                hour,
                location,
                value AS demand
            FROM with_hour
            WHERE profile_name LIKE '%_E_Demand'
        ),

        -- attach the correct local demand
        with_ratio AS (
            SELECT
                p.year,
                p.scenario,
                p.timestep,
                p.hour,
                p.location,
                p.profile_name,
                p.value,
                p.value / NULLIF(d.demand, 0) AS ratio
            FROM with_hour p
            LEFT JOIN demand_by_loc d
              ON p.year = d.year
             AND p.scenario = d.scenario
             AND p.hour = d.hour
             AND p.location = d.location
            WHERE p.profile_name NOT LIKE '%_E_Demand'
        ),

        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY year, scenario, hour, location, profile_name
                    ORDER BY ratio ASC
                ) AS rn
            FROM with_ratio
        ),

        min_profiles AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                1 AS period,
                profile_name,
                value
            FROM ranked
            WHERE rn = 1
        ),

        max_demand AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                1 AS period,
                profile_name,
                MAX(value) AS value
            FROM with_hour
            WHERE profile_name LIKE '%_E_Demand'
            GROUP BY year, scenario, hour, profile_name
        )

        SELECT * FROM min_profiles
        UNION ALL
        SELECT * FROM max_demand
        ORDER BY year, scenario, profile_name, timestep
        """
    )
end
function create_init_rps_hourly_best(connection, period_duration, profiles)
    profiles_str = join(profiles, ", ")

    DuckDB.query(
        connection,
        """
        CREATE TABLE init_rps AS
        WITH melted AS (
            SELECT
                year,
                scenario,
                timestep,
                profile_name,
                value,
                split_part(profile_name, '_', 1) AS location
            FROM profiles_wide
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        with_hour AS (
            SELECT
                *,
                ((timestep - 1) % $period_duration) + 1 AS hour
            FROM melted
        ),

        -- isolate demand by location
        demand_by_loc AS (
            SELECT
                year,
                scenario,
                hour,
                location,
                value AS demand
            FROM with_hour
            WHERE profile_name LIKE '%_E_Demand'
        ),

        -- attach the correct local demand
        with_ratio AS (
            SELECT
                p.year,
                p.scenario,
                p.timestep,
                p.hour,
                p.location,
                p.profile_name,
                p.value,
                p.value / NULLIF(d.demand, 0) AS ratio
            FROM with_hour p
            LEFT JOIN demand_by_loc d
              ON p.year = d.year
             AND p.scenario = d.scenario
             AND p.hour = d.hour
             AND p.location = d.location
            WHERE p.profile_name NOT LIKE '%_E_Demand'
        ),

        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY year, scenario, hour, location, profile_name
                    ORDER BY ratio DESC
                ) AS rn
            FROM with_ratio
        ),

        max_profiles AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                2 AS period,
                profile_name,
                value
            FROM ranked
            WHERE rn = 1
        ),

        min_demand AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                2 AS period,
                profile_name,
                MIN(value) AS value
            FROM with_hour
            WHERE profile_name LIKE '%_E_Demand'
            GROUP BY year, scenario, hour, profile_name
        )

        SELECT * FROM max_profiles
        UNION ALL
        SELECT * FROM min_demand
        ORDER BY year, scenario, profile_name, timestep
        """
    )
end

function create_init_rps_daily(connection, period_duration, profiles)
    profiles_str = join(profiles, ", ")

    DuckDB.query(
        connection,
        """
        CREATE TABLE init_rps AS
        WITH base AS (
            SELECT
                year,
                scenario,
                timestep,
                CAST(((timestep - 1) % $period_duration) + 1 AS INTEGER) AS hour,
                CAST(FLOOR((timestep - 1) / $period_duration) + 1 AS INTEGER) AS day,
                profile_name,
                value,
                split_part(profile_name, '_', 1) AS location
            FROM profiles_wide
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        demand AS (
            SELECT
                year,
                scenario,
                location,
                day,
                hour,
                profile_name,
                value AS demand
            FROM base
            WHERE profile_name LIKE '%_E_Demand'
        ),

        renewable_profiles AS (
            SELECT
                year,
                scenario,
                location,
                day,
                hour,
                profile_name,
                value
            FROM base
            WHERE profile_name NOT LIKE '%_E_Demand'
        ),

        renewable_day_ratio AS (
            SELECT
                r.year,
                r.scenario,
                r.location,
                r.day,
                SUM(r.value) AS renewable_sum,
                SUM(d.demand) AS demand_sum,
                SUM(r.value) / NULLIF(SUM(d.demand), 0) AS ratio
            FROM renewable_profiles r
            JOIN demand d
              ON r.year = d.year
             AND r.scenario = d.scenario
             AND r.location = d.location
             AND r.day = d.day
             AND r.hour = d.hour
            GROUP BY r.year, r.scenario, r.location, r.day
        ),

        worst_renewable_day AS (
            SELECT year, scenario, location, day
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY year, scenario, location
                           ORDER BY ratio ASC
                       ) AS rn
                FROM renewable_day_ratio
            )
            WHERE rn = 1
        ),

        peak_demand_day AS (
            SELECT year, scenario, location, day
            FROM (
                SELECT
                    year,
                    scenario,
                    location,
                    day,
                    SUM(demand) AS total_demand,
                    ROW_NUMBER() OVER (
                        PARTITION BY year, scenario, location
                        ORDER BY SUM(demand) DESC
                    ) AS rn
                FROM demand
                GROUP BY year, scenario, location, day
            )
            WHERE rn = 1
        ),

        renewable_output AS (
            SELECT
                r.year,
                r.scenario,
                r.hour AS timestep,
                1 AS period,
                r.profile_name,
                r.value
            FROM renewable_profiles r
            JOIN worst_renewable_day w
              ON r.year = w.year
             AND r.scenario = w.scenario
             AND r.location = w.location
             AND r.day = w.day
        ),

        demand_output AS (
            SELECT
                d.year,
                d.scenario,
                d.hour AS timestep,
                1 AS period,
                d.profile_name,
                d.demand AS value
            FROM demand d
            JOIN peak_demand_day p
              ON d.year = p.year
             AND d.scenario = p.scenario
             AND d.location = p.location
             AND d.day = p.day
        )

        SELECT * FROM renewable_output
        UNION ALL
        SELECT * FROM demand_output
        ORDER BY year, scenario, profile_name, timestep;
        """
    )
end

function plot_values_stocmethod_weight( #considering different options: stochastic_method, weight_type
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png"
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Number of representative_periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )

    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        weight_type = g.weight_type[1]
        mcol = get(COLOR_MAP_weight, weight_type) do
            error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        scatter!(
            p,
            xidx,
            g_sorted[!, column],
            markershape=mk,
            markersize=8,
            markercolor=mcol,
            label=""
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (weight types)
    for (label, color) in COLOR_MAP_weight
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                error("Unknown method: $label")
            end)
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end

function plot_values_stocmethod_method( # considering options: method, stochastic_method (possible to add weight type dirac)
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png",
    include_dirac=false,
    logscale=false
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)

    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
    #results_with_options = filter(row -> row.rp >= 45, results_with_options)

    results_with_options = filter(row -> row.method != "conical_hull", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )

    if logscale
        yaxis!(p, :log10)
    end

    if !include_dirac
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end
    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)


        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        method = g.method[1]
        mcolout = get(COLOR_MAP_method, method) do
            error("Unknown method: $method")
        end

        weight_type = g.weight_type[1]
        mcolin = get(FILLER_MAP, weight_type) do
            error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        if !include_dirac
            mcolout = :black
        end

        scatter!(
            p,
            xidx,
            g_sorted[!, column],
            markershape=mk,
            markersize=8,
            markercolor=mcolin,
            markerstrokecolor=mcolout,
            label=""
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (method types)
    for (label, color) in COLOR_MAP_method
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                error("Unknown method: $label")
            end
        )
    end
    if include_dirac
        # Legend for filler colors (weights type)
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=:white,
            label="dirac weights")
    end
    #ylims!(p, 0, 0.15)

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="stats_plot.png",
    logscale=false
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    #results_with_options = filter(row -> row.rp >= 60, results_with_options)


    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    stats_df = filter(row -> row.base_name != "0_HourlyBenchmark", stats_df)
    rp_vals = sort(unique(stats_df.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))


    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )
    if logscale
        yaxis!(p, :log10)
    end


    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end

        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end
        method = g.method[1]
        weight_type = g.weight_type[1]
        mcolin = get(COLOR_MAP_method_weight, method * "_" * weight_type) do
            error("Unknown method and weight: $(method * "_" * weight_type)")
        end

        mean_vals = g_sorted[!, col_mean]
        q25_vals = g_sorted[!, col_q25]
        q75_vals = g_sorted[!, col_q75]

        lower = mean_vals .- q25_vals
        upper = q75_vals .- mean_vals
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        plot!(
            p,
            xidx,
            mean_vals,
            ribbon=(lower, upper),
            label=name,
            lw=2,
            color=mcolin,
            fillalpha=0.20,
            markershape=mk,
            markersize=6,
            markercolor=mcolin
        )
    end
    # ylims!(p, 0.0, 0.15)

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles_panel(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::AbstractString;
    method::AbstractString,
    panel_title::AbstractString="$values by rp",
    plot_legend::Bool=false
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
    results_with_options = filter(row -> row.method == method, results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    # Value columns
    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown value: $values")
        end,
        title=panel_title,
        legend=plot_legend ? :topright : false,
        legendfont=font(10),
        xticks=(1:length(rp_vals), rp_labels),
        size=(200, 100),
        theme=:ggplot2,
        framestyle=:box,
        grid=:y,
        minorgrid=true,
        tick_direction=:out,
        guidefont=font(10),
        tickfont=font(9),
    )


    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end

        g_sorted = sort(g, :rp)

        stochastic_method = g_sorted.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        weight_type = g_sorted.weight_type[1]
        together = method * "_" * weight_type * "_" * stochastic_method
        mcolin = get(COLOR_MAP_method_weight_stmethod, together) do
            error("Unknown method and weight: $together")
        end

        mean_vals = g_sorted[!, col_mean]
        q25_vals = g_sorted[!, col_q25]
        q75_vals = g_sorted[!, col_q75]

        lower = mean_vals .- q25_vals
        upper = q75_vals .- mean_vals
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        plot!(
            p,
            xidx,
            mean_vals,
            ribbon=(lower, upper),
            label=get(LEGEND_MAP, together) do
                error("Unknown legend value: $together")
            end,
            lw=2,
            linealpha=0.9,
            color=mcolin,
            fillalpha=0.25,
            markershape=mk,
            markersize=7,
            markercolor=mcolin,
            markerstrokecolor=:black,
            markerstrokewidth=0.5,
        )

    end

    return p
end


function plot_values_quantiles_grid(
    stats_dfs::NamedTuple,
    case_studies_df::DataFrame,
    values::AbstractString;
    savepath::Union{Nothing,AbstractString}=nothing,
    size::Tuple{Int,Int}=(1500, 900),
    titles::Union{Nothing,NamedTuple}=nothing,
)

    keys_order = [:DISTANT, :HALFMIXED, :CLOSE, :MIXED]
    available = [k for k in keys(stats_dfs)]
    cols = [k for k in keys_order if k in available]
    if isempty(cols)
        cols = collect(keys(stats_dfs))
    end

    panels = Plots.Plot[]

    # k_means 
    for col in cols
        label = string(col)
        title_txt = isnothing(titles) ? label : get(titles, col, label)
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_means",
                panel_title=title_txt,
                plot_legend=string(col) == "MIXED" ? true : false
            )
        )
    end

    # k_medoids 
    for col in cols
        #label = string(col)
        #title_txt = isnothing(titles) ? "$label — k_medoids" : get(titles, col, "$label — k_medoids")
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_medoids",
                panel_title=" ",
                plot_legend=string(col) == "MIXED" ? true : false
            )
        )
    end



    grid = plot(
        panels...,
        layout=(2, length(cols)),
        size=size,
        link=:both, # link x and y across panels
        left_margin=5mm,
        right_margin=12mm,
        top_margin=5mm,
        bottom_margin=5mm
    )


    savefig(grid, savepath)

    return nothing
end


function plot_cost_columns(df::DataFrame, savepath::AbstractString)

    # Create label column: base_name_rp
    df.label = string.(df.base_name, "_", df.rp)


    # Keep only relevant columns
    labels = df.label
    investment = df.investment_cost
    operational = df.operational_cost + df.investment_cost - df.penalty_tot_loss_of_load
    penalty = df.investment_cost + df.operational_cost

    # Create stacked bar plot

    data_matrix = hcat(penalty, operational, investment)
    label_order = ["Penalty Loss" "Operational Cost" "Investment Cost"]

    # Create stacked bar plot
    b = bar(
        labels,
        data_matrix,
        label=label_order,
        legend=:topright,
        xlabel="Scenario",
        ylabel="Cost",
        title="Cost Breakdown per Setting",
        bar_position=:stack,
        xticks=(1:length(labels), labels),
        xrotation=45,
        size=(800, 600)
    )

    savefig(b, savepath)

end

function plot_cost_columns_precise(df::DataFrame, savepath::AbstractString)

    # Create label column: base_name_rp
    labels = string.(df.rp)


    # Keep only relevant columns
    investment_cost_storage = df.investment_cost_storage
    investment_cost_renewable = df.investment_cost_renewable + df.investment_cost_storage
    investment_cost_non_renewable = df.investment_cost_non_renewable + df.investment_cost_renewable + df.investment_cost_storage
    investment_cost_electrolyzer = df.investment_cost_electrolyzer + df.investment_cost_non_renewable + df.investment_cost_renewable + df.investment_cost_storage
    operational = df.operational_cost + df.investment_cost_electrolyzer + df.investment_cost_non_renewable + df.investment_cost_renewable + df.investment_cost_storage - df.penalty_tot_loss_of_load
    penalty = df.operational_cost + df.investment_cost_electrolyzer + df.investment_cost_non_renewable + df.investment_cost_renewable + df.investment_cost_storage

    # Create stacked bar plot

    data_matrix = hcat(penalty, operational, investment_cost_electrolyzer, investment_cost_non_renewable, investment_cost_renewable, investment_cost_storage)
    label_order = ["Penalty Loss" "Operational Cost" "Investment Cost Electrolyzer" "Investment Cost Non Renewable" "Investment Cost Renewable" "Investment Cost Storage"]

        
    # colors = [
    #         :red,          # penalty
    #         :steelblue,    # operational
    #         :orange,       # electrolyzer
    #         :gray,         # non-renewable
    #         :green,        # renewable
    #         :purple        # storage
    #     ]


    # Create stacked bar plot
    b = bar(
        labels,
        data_matrix,
        label=label_order,
        legend=:topright,
        # color = colors,
        xlabel="Number of representative periods",
        ylabel="Cost (kEUR)",
        title="Cost Breakdown",
        bar_position=:stack,
        xticks=(1:length(labels), labels),
        size=(900, 600)
    )

    savefig(b, savepath)

end

