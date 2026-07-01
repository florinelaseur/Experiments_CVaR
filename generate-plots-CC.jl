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

function parse_n_seed(filename::String)
    m = match(r"results_CC_per_N(\d+)_seed(\d+)\.csv$", filename)
    return parse(Int, m[1]), parse(Int, m[2])
end

function plot_boxplot()
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

    bench_df = select(
        filter(row ->
                row.scenario_set == "full" &&
                    string(row.termination_status) == "OPTIMAL",
            vcat(raw_dfs...),
        ),
        :number_of_scenarios,
        :seed,
        :solver,
        :objective_value => :objective_full,
    )

    reduced_df = select(
        filter(row ->
                row.scenario_set == "reduced" &&
                    string(row.termination_status) == "OPTIMAL",
            vcat(raw_dfs...),
        ),
        :number_of_scenarios,
        :seed,
        :solver,
        :objective_value_resolve_full => :objective_oos,
    )

    comparison_df = innerjoin(
        reduced_df,
        bench_df;
        on=[:number_of_scenarios, :seed, :solver],
    )

    transform!(
        comparison_df,
        [:objective_oos, :objective_full] =>
            ByRow((oos, full) -> (oos - full) / full * 100) => :oos_gap_pct,
    )

    CSV.write(joinpath(outdir, "comparison_CC_per_oos_gap.csv"), comparison_df)

    n_values = sort(unique(comparison_df.number_of_scenarios))
    gap_vectors = [comparison_df[comparison_df.number_of_scenarios.==n, :oos_gap_pct] for n in n_values]

    p = plot(;
        title="OOS Gap: reduced solution evaluated on scenario starting set",
        ylabel="OOS gap (%)",
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
        label="Full benchmark",
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

    plot!(
        p;
        xticks=(1:length(n_values), string.(n_values)),
    )

    savefig(p, joinpath(outdir, "1_oos_gap_boxplot_CC_per.png"))

    @info "Saved comparison CSV to $(joinpath(outdir, "comparison_CC_per_oos_gap.csv"))"
    @info "Saved plot to $(joinpath(outdir, "1_oos_gap_boxplot_CC_per.png"))"

    return comparison_df
end

function plot_investment_difference_boxplots()
    investment_files = String[]

    for n in scenario_sizes
        for seed in 1:config["simulation"]["seeds"]
            f = joinpath(
                inputdir,
                "N$(n)_seed$(seed)",
                "investment_analysis",
                "normalized_investment_differences.csv",
            )
            isfile(f) && push!(investment_files, f)
        end
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

        push!(raw_dfs, df)
    end

    investment_df = vcat(raw_dfs...; cols=:union)

    CSV.write(
        joinpath(outdir, "comparison_CC_per_investment_differences.csv"),
        investment_df;
        writeheader=true,
    )

    assets = sort(unique(investment_df.asset))

    investment_plot_dir = joinpath(outdir, "investment_differences")
    mkpath(investment_plot_dir)

    for asset in assets
        asset_df = filter(row -> row.asset == asset, investment_df)

        n_values = sort(unique(asset_df.number_of_scenarios))
        diff_vectors = [
            asset_df[asset_df.number_of_scenarios.==n, :diff]
            for n in n_values
        ]

        p = plot(;
            title="Normalized investment difference: $asset",
            ylabel="(CC - full) / full",
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

        plot!(
            p;
            xticks=(1:length(n_values), string.(n_values)),
        )

        safe_asset = replace(string(asset), r"[^A-Za-z0-9_]+" => "_")

        savefig(
            p,
            joinpath(
                investment_plot_dir,
                "investment_difference_boxplot_$(safe_asset).png",
            ),
        )
    end

    return investment_df
end

plot_boxplot()
plot_investment_difference_boxplots()