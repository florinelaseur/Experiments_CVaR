cd(@__DIR__)
# using Pkg: Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using CSV: CSV
using TOML: TOML
using DataFrames
using Plots
using StatsPlots
using Statistics

gr()

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

config = TOML.parsefile("config.toml")
scenario_sizes = config["simulation"]["scenarios_starting_set_sizes"]
representative_periods = config["simulation"]["representative_periods"]

inputdir = joinpath(@__DIR__, "outputs")
outdir = joinpath(@__DIR__, "outputs", "plots")
mkpath(outdir)

col_cc = RGB(0.122, 0.471, 0.706)
col_zero = RGB(0.4, 0.4, 0.4)

function plot_investment_difference_boxplots(; evaluation::Symbol=:baseline)
    investment_files = String[]

    subfolder = if evaluation == :baseline
        "baseline"
    elseif evaluation == :benchmark
        "benchmark"
    else
        error("Unknown evaluation=$evaluation. Use :baseline or :benchmark.")
    end

    for n in scenario_sizes
        for seed in 1:config["simulation"]["seeds"]
            f = joinpath(
                inputdir,
                "N$(n)_seed$(seed)",
                "investment_analysis",
                subfolder,
                "normalized_investment_differences.csv",
            )
            isfile(f) && push!(investment_files, f)
        end
    end

    if isempty(investment_files)
        @warn "No investment difference files found for evaluation=$evaluation"
        return DataFrame()
    end

    raw_dfs = DataFrame[]

    for f in investment_files
        m = match(r"N(\d+)_seed(\d+)", f)
        isnothing(m) && continue

        n = parse(Int, m[1])
        seed = parse(Int, m[2])

        df = CSV.read(f, DataFrame)
        df[!, :number_of_scenarios] .= n
        df[!, :seed] .= seed
        df[!, :evaluation] .= string(evaluation)

        push!(raw_dfs, df)
    end

    investment_df = vcat(raw_dfs...; cols=:union)

    CSV.write(
        joinpath(outdir, "comparison_CC_per_investment_differences_$(evaluation).csv"),
        investment_df;
        writeheader=true,
    )

    assets = sort(unique(investment_df.asset))

    investment_plot_dir = joinpath(outdir, "investment_differences", string(evaluation))
    mkpath(investment_plot_dir)

    for asset in assets
        asset_df = filter(row -> row.asset == asset, investment_df)

        n_values = sort(unique(asset_df.number_of_scenarios))
        diff_vectors = [
            asset_df[asset_df.number_of_scenarios.==n, :diff]
            for n in n_values
        ]

        ylabel = asset in ["e_demand_lol", "h2_demand_lol"] ?
                 "Number of loss-of-load timesteps" :
                 evaluation == :baseline ?
                 "(CC - hourly baseline) / hourly baseline" :
                 "(CC - RP full) / RP full"

        title = evaluation == :baseline ?
                "Investment difference vs hourly baseline: $asset" :
                "Investment difference vs RP full benchmark: $asset"

        p = plot(;
            title=title,
            ylabel=ylabel,
            xlabel="Number of scenarios in starting set",
            size=(700, 450),
            grid=true,
            gridalpha=0.3,
            legend=false,
        )

        hline!(
            p,
            [0.0];
            color=col_zero,
            linestyle=:dash,
            linewidth=1.5,
        )

        for (i, diffs) in enumerate(diff_vectors)
            boxplot!(
                p,
                fill(i, length(diffs)),
                diffs;
                color=col_cc,
                fillalpha=0.45,
                linecolor=col_cc,
                outliers=true,
                markersize=4,
            )
        end

        plot!(p; xticks=(1:length(n_values), string.(n_values)))

        safe_asset = replace(string(asset), r"[^A-Za-z0-9_]+" => "_")

        savefig(
            p,
            joinpath(
                investment_plot_dir,
                "investment_difference_boxplot_$(safe_asset)_$(evaluation).png",
            ),
        )
    end

    return investment_df
end

function plot_boxplot(; evaluation::Symbol=:baseline)
    files = filter(f -> begin
            b = basename(f)
            startswith(b, "results_CC_per_N") && endswith(b, ".csv")
        end, readdir(inputdir; join=true))

    raw_dfs = DataFrame[]

    for f in sort(files)
        n, seed = parse_n_seed(basename(f))
        df = CSV.read(f, DataFrame)
        df[!, :number_of_scenarios] .= n
        df[!, :seed] .= seed
        push!(raw_dfs, df)
    end

    all_df = vcat(raw_dfs...; cols=:union)

    if evaluation == :baseline
        bench_df = select(
            filter(row ->
                    row.base_name == "0_HourlyBenchmark" &&
                        row.scenario_set == "full" &&
                        string(row.termination_status) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :objective_value => :objective_ref,
        )

        reduced_df = select(
            filter(row ->
                    row.scenario_set == "reduced" &&
                        string(row.termination_status_resolve_baseline) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :base_name,
            :rp,
            :objective_value_resolve_baseline => :objective_red,
        )

        title = "Relative gap: reduced solution evaluated on hourly baseline"
        ylabel = "(reduced fixed on baseline - hourly baseline) / hourly baseline"
        outfile = "comparison_CC_per_rel_gap_baseline.csv"
        plotfile = "1_rel_gap_boxplot_CC_per_baseline.png"

    elseif evaluation == :benchmark
        bench_df = select(
            filter(row ->
                    row.scenario_set == "full" &&
                        row.base_name != "0_HourlyBenchmark" &&
                        string(row.termination_status) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :rp,
            :objective_value => :objective_ref,
        )

        reduced_df = select(
            filter(row ->
                    row.scenario_set == "reduced" &&
                        string(row.termination_status_resolve_benchmark) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :base_name,
            :rp,
            :objective_value_resolve_benchmark => :objective_red,
        )

        title = "Relative gap: reduced solution evaluated on RP full benchmark"
        ylabel = "(reduced fixed on RP full - RP full) / RP full"
        outfile = "comparison_CC_per_rel_gap_benchmark.csv"
        plotfile = "1_rel_gap_boxplot_CC_per_benchmark.png"

    else
        error("Unknown evaluation=$evaluation. Use :baseline or :benchmark.")
    end

    join_cols = evaluation == :baseline ?
                [:number_of_scenarios, :seed, :solver] :
                [:number_of_scenarios, :seed, :solver, :rp]

    comparison_df = innerjoin(reduced_df, bench_df; on=join_cols)

    transform!(
        comparison_df,
        [:objective_red, :objective_ref] =>
            ByRow((red, ref) -> (red - ref) / ref) => :rel_gap,
    )

    sort!(comparison_df, [:number_of_scenarios, :seed, :rp])
    CSV.write(joinpath(outdir, outfile), comparison_df)

    n_values = sort(unique(comparison_df.number_of_scenarios))
    gap_vectors = [
        comparison_df[comparison_df.number_of_scenarios.==n, :rel_gap]
        for n in n_values
    ]

    p = plot(;
        title=title,
        ylabel=ylabel,
        xlabel="Number of scenarios in starting set",
        size=(700, 450),
        grid=true,
        gridalpha=0.3,
        legend=:topright,
    )

    hline!(
        p,
        [0.0];
        color=col_zero,
        linestyle=:dash,
        linewidth=1.5,
        label="Reference",
    )

    for (i, gaps) in enumerate(gap_vectors)
        boxplot!(
            p,
            fill(i, length(gaps)),
            gaps;
            color=col_cc,
            fillalpha=0.45,
            linecolor=col_cc,
            outliers=true,
            markersize=4,
            label=i == 1 ? "CC-per" : "",
        )
    end

    plot!(p; xticks=(1:length(n_values), string.(n_values)))
    savefig(p, joinpath(outdir, plotfile))

    return comparison_df
end

plot_boxplot(evaluation=:baseline)
plot_investment_difference_boxplots(evaluation=:baseline)

if config["simulation"]["fix_benchmark"]
    plot_boxplot(evaluation=:benchmark)
    plot_investment_difference_boxplots(evaluation=:benchmark)
end