cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()

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
    isnothing(m) && return nothing
    return parse(Int, m[1]), parse(Int, m[2])
end

function plot_boxplot()
    files = filter(readdir(inputdir; join=true)) do f
        b = basename(f)
        startswith(b, "results_CC_per_N") && endswith(b, ".csv")
    end

    isempty(files) && error("No results_CC_per_N*_seed*.csv files found in $inputdir")

    frames = DataFrame[]

    for f in sort(files)
        ns = parse_n_seed(basename(f))

        if isnothing(ns)
            @warn "Could not parse N/seed from $(basename(f)); skipping"
            continue
        end

        n, seed = ns

        if !(n in scenario_sizes)
            @warn "Skipping $(basename(f)) because N=$n is not in config scenarios_starting_set_sizes"
            continue
        end

        df = CSV.read(f, DataFrame)
        df[!, :number_of_scenarios] .= n
        df[!, :seed] .= seed

        push!(frames, df)
    end

    isempty(frames) && error("No valid result files found")

    raw_df = vcat(frames...; cols=:union)

    required_cols = [
        :base_name,
        :rp,
        :solver,
        :objective_value,
        :objective_value_resolve_full,
        :termination_status,
        :termination_status_resolve_full,
        :scenario_set,
        :number_of_scenarios,
        :seed,
    ]

    for col in required_cols
        hasproperty(raw_df, col) || error("Missing required column: $col")
    end

    full_df = filter(row ->
            row.scenario_set == "full" &&
            row.termination_status == "OPTIMAL",
        raw_df,
    )

    reduced_df = filter(row ->
            row.scenario_set == "reduced" &&
            row.termination_status == "OPTIMAL" &&
            row.termination_status_resolve_full == "OPTIMAL" &&
            row.rp in representative_periods,
        raw_df,
    )

    bench_df = select(
        full_df,
        :number_of_scenarios,
        :seed,
        :solver,
        :objective_value => :objective_full,
    )

    oos_df = select(
        reduced_df,
        :number_of_scenarios,
        :seed,
        :solver,
        :rp,
        :base_name,
        :objective_value => :objective_reduced,
        :objective_value_resolve_full => :objective_oos,
        :value_at_risk_threshold_mu_full,
    )

    comparison_df = innerjoin(
        oos_df,
        bench_df;
        on=[:number_of_scenarios, :seed, :solver],
    )

    nrow(comparison_df) == 0 && error("Comparison join produced zero rows")

    transform!(
        comparison_df,
        [:objective_oos, :objective_full] =>
            ByRow((oos, full) -> (oos - full) / full * 100) => :oos_gap_pct,
    )

    CSV.write(joinpath(outdir, "comparison_CC_per_oos_gap.csv"), comparison_df)

    n_values = sort(unique(comparison_df.number_of_scenarios))
    gap_vectors = [comparison_df[comparison_df.number_of_scenarios .== n, :oos_gap_pct] for n in n_values]

    p = plot(;
        title="OOS Gap: reduced solution evaluated on full scenario set",
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

plot_boxplot()