
function get_solver_parameters(optimizer::Symbol)
    if optimizer == :HiGHS
        return HiGHS.Optimizer,
        Dict(
            "output_flag" => true,
            "solver" => "hipo",
            "parallel" => "on",
            "run_crossover" => "off",
        )
    elseif optimizer == :Gurobi
        return Gurobi.Optimizer, Dict("OutputFlag" => 1)
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

# function plot_mu_vs_rp(
#     results_df::DataFrame,
#     case_studies_df::DataFrame;
#     savepath="value_at_risk_threshold_mu.png",
# )
#     results_with_options =
#         outerjoin(case_studies_df, results_df; on="base_name", makeunique=true)

#     results_with_options =
#         filter(row -> !ismissing(row.value_at_risk_threshold_mu), results_with_options)

#     benchmark_df = filter(row -> row.base_name == "0_HourlyBenchmark", results_with_options)
#     nonbenchmark_df = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

#     rp_vals = sort(unique(nonbenchmark_df.rp))
#     rp_labels = string.(rp_vals)
#     rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

#     p = plot(;
#         xlabel="Number of representative_periods",
#         ylabel="Optimal value_at_risk_threshold_mu",
#         title="",
#         legend=:topright,
#         size=(800, 500),
#         xticks=(1:length(rp_vals), rp_labels),
#     )

#     for g in groupby(nonbenchmark_df, :base_name)
#         g_sorted = sort(g, :rp)

#         stochastic_method = g.stochastic_method[1]
#         mk = get(MARKER_MAP, stochastic_method, :circle)

#         weight_type = g.weight_type[1]
#         mcol = get(COLOR_MAP_weight, weight_type, :black)

#         xidx = [rp_index[rp] for rp in g_sorted.rp]

#         scatter!(
#             p,
#             xidx,
#             g_sorted.value_at_risk_threshold_mu;
#             markershape=mk,
#             markersize=8,
#             markercolor=mcol,
#             label="",
#         )

#         plot!(
#             p,
#             xidx,
#             g_sorted.value_at_risk_threshold_mu;
#             color=mcol,
#             linewidth=1.5,
#             label="",
#         )
#     end

#     # Benchmark as horizontal reference line
#     if nrow(benchmark_df) > 0
#         mu_benchmark = benchmark_df.value_at_risk_threshold_mu[1]

#         hline!(
#             p,
#             [mu_benchmark];
#             color=:black,
#             linestyle=:dash,
#             linewidth=2,
#             label="Hourly benchmark",
#         )
#     end

#     # Legend for shapes (stochastic methods)
#     for (label, marker) in MARKER_MAP
#         short_label = replace(string(label), "_scenario" => "-scenario")
#         scatter!(
#             p,
#             [NaN],
#             [NaN];
#             markershape=marker,
#             markersize=8,
#             markercolor=:gray30,
#             label=short_label,
#         )
#     end

#     # Legend for colors (weight types)
#     for (label, color) in COLOR_MAP_weight
#         scatter!(
#             p,
#             [NaN],
#             [NaN];
#             markershape=:rect,
#             markersize=8,
#             markercolor=color,
#             label=get(LEGEND_METHOD_MAP, label) do
#                 return error("Unknown method: $label")
#             end,
#         )
#     end

#     savefig(p, savepath)
#     @info "Plot saved in: $savepath"
# end

function plot_values_stocmethod_weight( #considering different options: stochastic_method, weight_type
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png",
)
    results_with_options =
        outerjoin(case_studies_df, results_df; on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(;
        xlabel="Number of representative_periods",
        ylabel=get(VALUE_MAP, values) do
            return error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels),
    )
    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            return error("Unknown stochastic_method: $stochastic_method")
        end

        weight_type = g.weight_type[1]
        mcol = get(COLOR_MAP_weight, weight_type) do
            return error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        scatter!(
            p,
            xidx,
            g_sorted[!, column];
            markershape=mk,
            markersize=8,
            markercolor=mcol,
            label="",
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(
            p,
            [NaN],
            [NaN];
            markershape=marker,
            markersize=8,
            markercolor=:gray30,
            label=short_label,
        )
    end

    # Legend for colors (weight types)
    for (label, color) in COLOR_MAP_weight
        scatter!(
            p,
            [NaN],
            [NaN];
            markershape=:rect,
            markersize=8,
            markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                return error("Unknown method: $label")
            end,
        )
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
    from_rp=0,
    chosen_method=nothing,
)
    results_with_options =
        outerjoin(case_studies_df, results_df; on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(;
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            return error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels),
    )
    if !include_dirac
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end

    if chosen_method !== nothing
        results_with_options = filter(row -> row.method == chosen_method, results_with_options)
    end
    results_with_options = filter(row -> row.rp >= from_rp, results_with_options)

    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            return error("Unknown stochastic_method: $stochastic_method")
        end

        method = g.method[1]
        mcolout = get(COLOR_MAP_method, method) do
            return error("Unknown method: $method")
        end

        weight_type = g.weight_type[1]
        mcolin = get(FILLER_MAP, weight_type) do
            return error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        if !include_dirac
            mcolout = :black
        end

        scatter!(
            p,
            xidx,
            g_sorted[!, column];
            markershape=mk,
            markersize=8,
            markercolor=mcolin,
            markerstrokecolor=mcolout,
            label="",
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(
            p,
            [NaN],
            [NaN];
            markershape=marker,
            markersize=8,
            markercolor=:gray30,
            label=short_label,
        )
    end

    # Legend for colors (method types)
    for (label, color) in COLOR_MAP_method
        scatter!(
            p,
            [NaN],
            [NaN];
            markershape=:rect,
            markersize=8,
            markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                return error("Unknown method: $label")
            end,
        )
    end
    if include_dirac
        # Legend for filler colors (weights type)
        scatter!(
            p,
            [NaN],
            [NaN];
            markershape=:rect,
            markersize=8,
            markercolor=:white,
            label="dirac weights",
        )
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end

function parse_rep_period_name(name::String) # the vars were created as storage_level_rep_period[$(row.asset),$(row.year),$(row.rep_period),$(row.time_block_start):$(row.time_block_end)]
    inside = name[findfirst('[', name)+1:end-1] # inside []
    parts = split(inside, ",")
    return (
        asset=parts[1],
        year=parse(Int, parts[2]),
        rep_period=parse(Int, parts[3]),
        time_block_start=parse(Int, split(parts[4], ":")[1]),
    )
end

function parse_over_clustered_name(name::String) # storage_level_inter_period[$(row.asset),$(row.year),$(row.scenario),$(row.period_block_start):$(row.period_block_end)]
    inside = name[findfirst('[', name)+1:end-1]
    parts = split(inside, ",")
    return (
        asset=parts[1],
        year=parse(Int, parts[2]),
        scenario=parse(Int, parts[3]),
        period_block_start=parse(Int, split(parts[4], ":")[1]),
    )
end

function fix_storage_levels!(
    benchmark_model,
    reduced_model,
    scenario_to_rep_period_map,
    period_duration,
    storage_asset,
)
    bench_vars = benchmark_model.variables[:storage_level_rep_period].container
    red_vars = reduced_model.variables[:storage_level_inter_period].container

    bench_pairs = [
        (bench_vars[i], parse_rep_period_name(JuMP.name(bench_vars[i]))) for
        i in eachindex(bench_vars)
    ]

    red_pairs = [
        (red_vars[i], parse_over_clustered_name(JuMP.name(red_vars[i]))) for
        i in eachindex(red_vars)
    ]

    bench_pairs = filter(p -> p[2].asset == storage_asset, bench_pairs)
    red_pairs = filter(p -> p[2].asset == storage_asset, red_pairs)

    val_to_fix = Dict()

    for (v, row) in red_pairs
        key = (row.asset, row.year, row.scenario, row.period_block_start)
        val_to_fix[key] = JuMP.value(v)
    end

    for (v, row) in bench_pairs

        # only at the end of each day
        if row.time_block_start % period_duration != 0
            continue
        end
        scenario = scenario_to_rep_period_map[row.rep_period]

        period = row.time_block_start ÷ period_duration

        key = (row.asset, row.year, scenario, period)

        if haskey(val_to_fix, key)
            JuMP.fix(v, val_to_fix[key]; force=true)
        else
            error("No reduced_model value found for key $key")
        end
    end

    return nothing
end

function plot_storage_behavior(
    results_df::DataFrame,
    case_studies_df::DataFrame,
    storage_levels_hourly::DataFrame,
    storage_asset::String,
    representative_periods::Vector{Int64},
    scenario::Int64;
    tables_path="outputs",
    savepath="storage.png",
)
    results_with_options =
        outerjoin(case_studies_df, results_df; on="base_name", makeunique=true)

    asset_to_filter = storage_asset
    hourly_filtered_asset = filter(row -> row.asset == asset_to_filter, storage_levels_hourly)
    hourly_filtered_asset = filter(row -> row.rep_period == scenario, hourly_filtered_asset)

    # grid for all rp subplots
    n = length(representative_periods)
    ncols = min(n, 3)
    nrows = ceil(Int, n / ncols)

    p = plot(; layout=grid(nrows, ncols), link=:x, size=(1500, 350 * nrows), legend=false)

    for (i, rp) in enumerate(representative_periods)
        # plotting the results for the hourly benchmark
        plot!(
            p,
            hourly_filtered_asset.time_block_end,
            hourly_filtered_asset.solution;
            subplot=i,
            label="hourly",
            color=:red,
            title="Storage level — $asset_to_filter (rp = $rp)",
            xlabel="Hour",
            ylabel="[GWh]",
            xlims=(1, 8760),
            legend=false,
            dpi=600,
        )

        # add storage levels for each base_name, but using the rp-specific folder
        for g in groupby(results_with_options, :base_name)
            name = g.base_name[1]
            if name == "0_HourlyBenchmark"
                continue
            end

            stochastic_method = g.stochastic_method[1]

            stochastic_method = g.stochastic_method[1]
            mk = get(LINE_MAP, stochastic_method) do
                return error("Unknown stochastic_method: $stochastic_method")
            end

            weight_type = g.weight_type[1]
            mcol = get(FILLER_MAP, weight_type) do
                return error("Unknown weight_type: $weight_type")
            end

            name_rp = string(name, "_rp_", rp)

            path = joinpath(
                tables_path,
                "fixed",
                name_rp,
                "Gurobi",
                "var_storage_level_rep_period.csv",
            )

            reduced_storage_levels = CSV.read(path, DataFrame)

            reduced_filtered_asset =
                filter(row -> row.asset == asset_to_filter, reduced_storage_levels)
            reduced_filtered_asset =
                filter(row -> row.rep_period == scenario, reduced_filtered_asset)

            plot!(
                p,
                reduced_filtered_asset.time_block_end,
                reduced_filtered_asset.solution;
                subplot=i,
                label="$stochastic_method selection",
                color=mcol,
                linestyle=mk,
            )
        end
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function get_scenario_set(input_df::DataFrame, cardinality::Int)
    scenarios = unique(input_df.scenario)
    @assert cardinality ≤ length(scenarios) "Requested more scenarios than available."
    selected = sort(shuffle(scenarios)[1:cardinality])
    return filter(row -> row.scenario in selected, input_df)
end

function export_base_cost(energy_problem, output_folder)
    base_cost = JuMP.AffExpr(0.0)

    for objective_name in (
        :assets_investment_cost,
        :assets_fixed_cost_compact_method,
        :assets_fixed_cost_simple_method,
        :storage_assets_energy_investment_cost,
        :storage_assets_energy_fixed_cost,
        :flows_investment_cost,
        :flows_fixed_cost,
    )
        if haskey(energy_problem.model, objective_name)
            JuMP.add_to_expression!(base_cost, energy_problem.model[objective_name])
        end
    end

    df = DataFrame(base_cost=[JuMP.value(base_cost)])

    CSV.write(joinpath(output_folder, "base_cost.csv"), df)

    return df
end

function export_operational_cost_per_scenario(energy_problem, output_folder)
    costs_per_scenario = energy_problem.expressions[:flows_operational_cost_per_scenario] +
                         energy_problem.expressions[:vintage_flows_operational_cost_per_scenario] +
                         energy_problem.expressions[:units_on_operational_cost_per_scenario]
    df = costs_per_scenario.indices |> DataFrame
    costs = JuMP.value.(costs_per_scenario.expressions[:cost])
    df[!, :operational_cost] = costs
    CSV.write(joinpath(output_folder, "operational_cost_per_scenario.csv"), df)
    return df
end

function export_total_cost_per_scenario(energy_problem, output_folder)
    expr = energy_problem.expressions[:scenario_tail_excess]
    df = expr.indices |> DataFrame
    total_costs = JuMP.value.(
        expr.expressions[:total_cost_per_scenario]
    )
    df[!, :total_cost] = total_costs
    CSV.write(
        joinpath(output_folder, "total_cost_per_scenario.csv"),
        df;
        writeheader=true,
    )
    return df
end

# function plot_operational_cost_per_scenario(input_df::DataFrame, output_folder)
#     p = plot(input_df.scenario, input_df.operational_cost; xlabel="Scenario", ylabel="Operational Cost", title="Operational Cost per Scenario", marker=:circle)
#     sorted_scenario_costs = input_df.operational_cost |> sort
#     h = histogram(sorted_scenario_costs; bins=100, normalize=true, label="Operational Cost Distribution")
#     savefig(p, joinpath(output_folder, "operational_cost_per_scenario.png"))
#     savefig(h, joinpath(output_folder, "operational_cost_distribution.png"))
#     @info "Plots saved in: $(joinpath(output_folder, "operational_cost_per_scenario.png")) and $(joinpath(output_folder, "operational_cost_distribution.png"))"
# end

function plot_cost_per_scenario(
    input_df::DataFrame,
    output_folder,
    mu_value_df::DataFrame,
)

    folder_parts = splitpath(output_folder)
    title_suffix = join(folder_parts[end-1:end], Base.Filesystem.path_separator)

    p = scatter(
        input_df.scenario,
        input_df.total_cost;
        xlabel="Scenario",
        ylabel="Total Cost",
        title="Total Cost per Scenario - $title_suffix",
        marker=:circle,
        color=:blue,
        label="Scenario cost",
    )

    if nrow(mu_value_df) > 0
        mu_value = only(mu_value_df.solution)

        hline!(
            p,
            [mu_value];
            linestyle=:dash,
            linewidth=2,
            label="VaR threshold μ",
        )
    end

    savefig(p, joinpath(output_folder, "total_cost_per_scenario.png"))

    @info "Plots saved in: $(joinpath(output_folder, "total_cost_per_scenario.png"))"

    return p
end

function plot_cost_per_scenario_inc_tail(
    total_cost_per_scenario_df::DataFrame,
    df_tail_scenarios::DataFrame,
    output_folder,
    mu_value_df::DataFrame,
    case_name,
)
    p = scatter(
        total_cost_per_scenario_df.scenario,
        total_cost_per_scenario_df.total_cost;
        xlabel="Scenario",
        ylabel="Total Cost",
        title="Tail scenarios of $case_name",
        marker=:circle,
        color=:blue,
        label="All scenarios",
    )

    scatter!(
        p,
        df_tail_scenarios.scenario,
        df_tail_scenarios.total_cost;
        marker=:circle,
        color=:red,
        label="Tail scenarios",
    )

    if nrow(mu_value_df) > 0
        mu_value = only(mu_value_df.solution)

        hline!(
            p,
            [mu_value];
            linestyle=:dash,
            linewidth=2,
            color=:black,
            label="VaR threshold μ",
        )
    end

    savefig(
        p,
        joinpath(output_folder, "total_cost_per_scenario.png"),
    )

    @info "Plots saved in: $(joinpath(output_folder, "total_cost_per_scenario.png"))"

    return p
end

function plot_cost_per_scenario_inc_tail_inc_representative(
    total_cost_per_scenario_df::DataFrame,
    df_tail_scenarios::DataFrame,
    df_representative_scenarios::DataFrame,
    output_folder,
    mu_value_df::DataFrame,
    case_name,
)
    p = scatter(
        total_cost_per_scenario_df.scenario,
        total_cost_per_scenario_df.total_cost;
        xlabel="Scenario",
        ylabel="Total Cost",
        title="Tail scenarios of $case_name",
        marker=:circle,
        color=:blue,
        label="All scenarios",
    )

    scatter!(
        p,
        df_tail_scenarios.scenario,
        df_tail_scenarios.total_cost;
        marker=:circle,
        color=:red,
        label="Tail scenarios",
    )

    scatter!(
        p,
        df_representative_scenarios.scenario,
        df_representative_scenarios.total_cost;
        marker=:circle,
        color=:green,
        label="Representative scenarios",
    )

    if nrow(mu_value_df) > 0
        mu_value = only(mu_value_df.solution)

        hline!(
            p,
            [mu_value];
            linestyle=:dash,
            linewidth=2,
            color=:black,
            label="VaR threshold μ",
        )
    end

    savefig(
        p,
        joinpath(output_folder, "total_cost_per_scenario.png"),
    )

    @info "Plots saved in: $(joinpath(output_folder, "total_cost_per_scenario.png"))"

    return p
end

function plot_normalized_asset_investment_differences(
    benchmark_df::DataFrame,
    approximation_df::DataFrame;
    output_folder,
    case_name,)
    assets = benchmark_df[!, :asset]

    benchmark_solution = benchmark_df[!, :solution]
    approximation_solution = approximation_df[!, :solution]

    inv_diff = zeros(length(benchmark_solution))

    for i in eachindex(benchmark_solution)
        if benchmark_solution[i] > 0
            inv_diff[i] =
                (approximation_solution[i] - benchmark_solution[i]) /
                benchmark_solution[i]
        else
            inv_diff[i] = approximation_solution[i] - benchmark_solution[i]
        end
    end

    inv_diff_df = DataFrame(
        asset=assets,
        diff=inv_diff,
    )

    p_investment = bar(
        inv_diff_df.asset,
        inv_diff_df.diff;
        xlabel="Asset",
        ylabel="Normalized Investment Difference",
        title="Normalized Investment Differences of $case_name Compared to Benchmark",
        titlefontsize=8,
    )
    savefig(
        p_investment,
        joinpath(output_folder, "normalized_investment_differences.png"),
    )

    @info "Plots saved in: $(joinpath(output_folder, "normalized_investment_differences.png"))"

    return p_investment
end

function zoom_limits(v; margin_fraction=0.08)
    vmin = minimum(v)
    vmax = maximum(v)

    if vmin == vmax
        margin = abs(vmin) > 0 ? margin_fraction * abs(vmin) : 1.0
    else
        margin = margin_fraction * (vmax - vmin)
    end

    return (vmin - margin, vmax + margin)
end

function create_case_label(base_name, rp)
    if occursin("HourlyBenchmark_CC", base_name)
        return "Hourly CC"

    elseif occursin("HourlyBenchmark", base_name)
        return "Hourly Benchmark"

    elseif rp > 1
        return "$(rp) RPs"

    else
        return base_name
    end
end

function plot_comparison(output_folder, number_of_scenarios)

    results = joinpath(output_folder, "results.csv")
    df_results = CSV.read(results, DataFrame)

    plots_folder = joinpath(output_folder, "plots")
    mkpath(plots_folder)

    df_plot = copy(df_results)

    # Dynamic labels
    df_plot[!, :case_label] = [
        create_case_label(row.base_name, row.rp)
        for row in eachrow(df_plot)
    ]

    runtime_case_labels = copy(df_plot.case_label)

    # Find important rows dynamically
    idx_rp_reference = findfirst(df_plot.rp .> 1 .&& .!occursin.("CC", df_plot.base_name))

    idx_cc = findfirst(occursin.("CC", df_plot.base_name))

    # More descriptive runtime label for CC
    if idx_cc !== nothing && idx_rp_reference !== nothing
        rp_reference = df_plot[idx_rp_reference, :rp]

        runtime_case_labels[idx_cc] = "CC ($(rp_reference) RPs + hourly reduced scenario set)"
    end

    # Runtime
    df_plot[!, :runtime_own] =
        df_plot.time_to_read .+
        df_plot.time_to_create .+
        df_plot.time_to_solve .+
        df_plot.time_to_save

    rp_rows = df_plot.rp .> 1

    df_plot[!, :runtime_total] = copy(df_plot.runtime_own)

    df_plot[rp_rows, :runtime_total] =
        df_plot[rp_rows, :runtime_own] .+
        df_plot[rp_rows, :time_to_resolve_benchmark]

    runtime_comparison = copy(df_plot.runtime_total)

    # CC runtime logic
    if idx_cc !== nothing && idx_rp_reference !== nothing

        runtime_reference_total =
            df_plot[idx_rp_reference, :runtime_total]

        runtime_cc_own =
            df_plot[idx_cc, :runtime_own]

        runtime_comparison[idx_cc] =
            runtime_reference_total + runtime_cc_own

        @show runtime_reference_total
        @show runtime_cc_own
        @show runtime_comparison[idx_cc]
    end

    p_runtime = bar(
        runtime_case_labels,
        runtime_comparison;
        xlabel="Case",
        ylabel="Runtime [s]",
        title="Runtime Comparison",
        label=false,
        xrotation=20,
        legend=false,
    )

    savefig(
        p_runtime,
        joinpath(plots_folder, "runtime_comparison.png"),
    )

    # System costs
    df_plot[!, :system_cost] = df_plot.objective_value

    df_plot[rp_rows, :system_cost] =
        df_plot[rp_rows, :objective_value_resolve_benchmark]

    p_costs = bar(
        df_plot.case_label,
        df_plot.system_cost;
        xlabel="Case",
        ylabel="System Cost",
        title="System Costs Comparison",
        label=false,
        xrotation=30,
        ylim=zoom_limits(df_plot.system_cost),
    )

    savefig(
        p_costs,
        joinpath(plots_folder, "system_costs_comparison.png"),
    )

    # LOLE
    df_plot[!, :lole_e_demand] =
        df_plot.num_loss_of_load_e_demand ./ number_of_scenarios

    df_plot[!, :lole_h2_demand] =
        df_plot.num_loss_of_load_h2_demand ./ number_of_scenarios

    x = collect(1:nrow(df_plot))
    w = 0.35

    p_lole = bar(
        x .- w / 2,
        df_plot.lole_e_demand;
        bar_width=w,
        xlabel="Case",
        ylabel="LOLE [expected events per scenario]",
        title="Loss of Load Expected Comparison",
        label="Electricity demand",
        xticks=(x, df_plot.case_label),
        xrotation=30,
        legend=:topright,
    )

    bar!(
        p_lole,
        x .+ w / 2,
        df_plot.lole_h2_demand;
        bar_width=w,
        label="Hydrogen demand",
    )

    savefig(
        p_lole,
        joinpath(plots_folder, "lole_comparison.png"),
    )

    # VaR
    p_var = bar(
        df_plot.case_label,
        df_plot.value_at_risk_threshold_mu;
        xlabel="Case",
        ylabel="VaR threshold μ",
        title="Value at Risk Threshold Comparison",
        label=false,
        xrotation=30,
        ylim=zoom_limits(df_plot.value_at_risk_threshold_mu),
    )

    savefig(
        p_var,
        joinpath(plots_folder, "var_threshold_comparison.png"),
    )

    @info "Comparison plots saved in: $plots_folder"

    return df_plot
end