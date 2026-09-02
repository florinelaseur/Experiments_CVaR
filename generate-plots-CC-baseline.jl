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

inputdir = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data")
outdir = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "plots")
mkpath(outdir)

col_cc = RGB(0.122, 0.471, 0.706)
col_zero = RGB(0.4, 0.4, 0.4)

function concatenate_dataframes(dfs::Vector{DataFrame})
    isempty(dfs) && return DataFrame()
    return vcat(dfs...; cols=:union)
end

function collect_result_files(prefix::String)
    files = String[]

    for (root, _, dirfiles) in walkdir(inputdir)
        for f in dirfiles
            if contains(f, prefix) && endswith(f, ".csv")
                push!(files, joinpath(root, f))
            end
        end
    end

    files = sort(files)
    @info "collect_result_files" prefix = prefix found = length(files)
    return files
end

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

    investment_df = concatenate_dataframes(raw_dfs)

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
    raw_dfs = DataFrame[]

    # keep track of which files we actually read
    loaded_files = String[]

    # deterministically load expected results files for each scenario size and seed
    for n in scenario_sizes
        for seed in 1:config["simulation"]["seeds"]
            f = joinpath(inputdir, "N$(n)_seed$(seed)", "results_ScSeRP_N$(n)_seed$(seed).csv")
            if isfile(f)
                df = CSV.read(f, DataFrame)
                # record source filename for debugging and later inspection
                df[!, :source_file] .= basename(f)
                df[!, :number_of_scenarios] .= n
                df[!, :seed] .= seed
                push!(raw_dfs, df)
                push!(loaded_files, f)
            else
                @debug "Missing expected results file" file = f
            end
        end
    end

    # show which files were read
    @info "Loaded result files" count = length(loaded_files)
    for f in sort(loaded_files)
        @info f
    end

    if isempty(raw_dfs)
        @warn "No result dataframes collected for evaluation=$evaluation"
        return DataFrame()
    end

    if isempty(raw_dfs)
        @warn "No dataframes collected for evaluation=$evaluation"
        return DataFrame()
    end

    all_df = vcat(raw_dfs...; cols=:union)
    @show names(all_df)

    # Normalize string columns (trim whitespace) so the filters match values from the CSVs
    for col in [:case_name,
        :scenario_set,
        :termination_status,
        :termination_status_resolve_baseline,
        :termination_status_resolve_benchmark,
        :solver]
        if col in names(all_df)
            all_df[!, col] = strip.(string.(all_df[!, col]))
        end
    end

    if evaluation == :baseline
        bench_df = select(
            filter(row ->
                    row.case_name == "0_HourlyBaseline" &&
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
                        row.case_name != "0_HourlyBaseline" &&
                        string(row.termination_status_resolve_baseline) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :case_name,
            :rp,
            :objective_value_resolve_baseline => :objective_red,
        )

        title = "Relative gap: reduced solution evaluated on hourly baseline"
        ylabel = "(reduced fixed in baseline - baseline) / baseline * 100%"
        outfile = "comparison_ScSeRP_rel_gap_baseline.csv"
        plotfile = "1_rel_gap_boxplot_ScSeRP_baseline.png"

    elseif evaluation == :benchmark
        bench_df = select(
            filter(row ->
                    row.scenario_set == "full" &&
                        row.case_name != "0_HourlyBaseline" &&
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
                        row.case_name != "0_HourlyBaseline" &&
                        string(row.termination_status_resolve_benchmark) == "OPTIMAL",
                all_df),
            :number_of_scenarios,
            :seed,
            :solver,
            :case_name,
            :rp,
            :objective_value_resolve_benchmark => :objective_red,
        )

        title = "Relative gap: reduced solution evaluated on RP full benchmark"
        ylabel = "(reduced fixed on RP full - RP full) / RP full"
        outfile = nothing
        plotfile = nothing

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
            ByRow((red, ref) -> 100 * (red - ref) / ref) => :rel_gap,
    )

    sort!(comparison_df, [:number_of_scenarios, :seed, :rp])
    if !isnothing(outfile)
        CSV.write(joinpath(outdir, outfile), comparison_df)
    end

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
            label=i == 1 ? "ScSeRP" : "",
        )
    end

    plot!(p; xticks=(1:length(n_values), string.(n_values)))
    if !isnothing(plotfile)
        savefig(p, joinpath(outdir, plotfile))
    end

    return comparison_df
end

function plot_regret_vs_rp()

    files = filter(f -> begin
            b = basename(f)
            startswith(b, "results_CC_per_N") && endswith(b, ".csv")
        end, readdir(inputdir; join=true))

    dfs = DataFrame[]

    for f in sort(files)
        n, seed = parse_n_seed(basename(f))
        df = CSV.read(f, DataFrame)
        df.number_of_scenarios .= n
        df.seed .= seed
        push!(dfs, df)
    end

    if isempty(dfs)
        @warn "No result files found for regret plotting"
        return DataFrame()
    end

    all_df = concatenate_dataframes(dfs)

    baseline_df = select(
        filter(row ->
                row.case_name == "0_HourlyBaseline" &&
                    row.scenario_set == "full",
            all_df),
        :number_of_scenarios,
        :seed,
        :solver,
        :objective_value => :objective_baseline,
    )

    reduced_df = select(
        filter(row ->
                row.scenario_set == "reduced" &&
                    string(row.termination_status_resolve_baseline) == "OPTIMAL",
            all_df),
        :number_of_scenarios,
        :seed,
        :solver,
        :rp,
        :objective_value_resolve_baseline,
    )

    regret_df = innerjoin(
        reduced_df,
        baseline_df;
        on=[:number_of_scenarios, :seed, :solver],
    )

    transform!(
        regret_df,
        [:objective_value_resolve_baseline, :objective_baseline] =>
            ByRow((x, y) -> (x - y) / y) => :regret,
    )

    CSV.write(
        joinpath(outdir, "comparison_CC_per_regret_vs_rp.csv"),
        regret_df;
        writeheader=true,
    )

    p = plot(
        xlabel="Representative periods",
        ylabel="Regret",
        title="Regret after fixing CC investments in hourly baseline",
        legend=:topright,
        size=(700, 450),
        grid=true,
        gridalpha=0.3,
    )

    for seed in sort(unique(regret_df.seed))
        df_seed = sort(
            filter(r -> r.seed == seed, regret_df),
            :rp,
        )

        plot!(
            p,
            df_seed.rp,
            df_seed.regret;
            marker=:circle,
            linewidth=2,
            label="Seed $seed",
        )
    end

    savefig(
        p,
        joinpath(outdir, "regret_vs_representative_periods.png"),
    )

    return regret_df
end

function plot_runtime_vs_rp()
    files = filter(
        f -> begin
            b = basename(f)
            startswith(b, "results_CC_per_N") && endswith(b, ".csv")
        end,
        readdir(inputdir; join=true),
    )

    if isempty(files)
        @warn "No results_CC_per_N*.csv files found in $inputdir"
        return DataFrame()
    end

    raw_dfs = DataFrame[]

    for f in sort(files)
        n, seed = parse_n_seed(basename(f))
        df = CSV.read(f, DataFrame)

        df[!, :number_of_scenarios] .= n
        df[!, :seed] .= seed

        push!(raw_dfs, df)
    end

    all_df = vcat(raw_dfs...; cols=:union)

    # Runtime of each solved model.
    all_df[!, :runtime_model] =
        all_df.time_to_cluster .+
        all_df.time_to_read .+
        all_df.time_to_create .+
        all_df.time_to_solve .+
        all_df.time_to_save

    # One hourly-baseline runtime per N, seed and solver.
    baseline_df = select(
        filter(
            row ->
                row.case_name == "0_HourlyBaseline" &&
                    row.scenario_set == "full" &&
                    string(row.termination_status) == "OPTIMAL",
            all_df,
        ),
        :number_of_scenarios,
        :seed,
        :solver,
        :runtime_model => :runtime_baseline,
        :time_to_read => :time_to_read_baseline,
        :time_to_create => :time_to_create_baseline,
    )

    # CC runtime plus the hourly resolve with CC investments fixed.
    cc_df = select(
        filter(
            row ->
                row.scenario_set == "reduced" &&
                    string(row.termination_status_resolve_baseline) == "OPTIMAL",
            all_df,
        ),
        :number_of_scenarios,
        :seed,
        :solver,
        :rp,
        :runtime_model,
        :time_to_resolve_baseline,
    )

    runtime_df = innerjoin(
        cc_df,
        baseline_df;
        on=[:number_of_scenarios, :seed, :solver],
    )

    transform!(
        runtime_df,
        [
            :runtime_model,
            :time_to_read_baseline,
            :time_to_create_baseline,
            :time_to_resolve_baseline,
        ] =>
            ByRow(
                (
                    runtime_cc,
                    time_read_baseline,
                    time_create_baseline,
                    time_resolve_baseline,
                ) ->
                    runtime_cc +
                    time_read_baseline +
                    time_create_baseline +
                    time_resolve_baseline,
            ) => :runtime_cc_resolve,
    )

    sort!(
        runtime_df,
        [:number_of_scenarios, :seed, :solver, :rp],
    )

    CSV.write(
        joinpath(outdir, "comparison_CC_per_runtime_vs_rp.csv"),
        runtime_df;
        writeheader=true,
    )

    for n in sort(unique(runtime_df.number_of_scenarios))
        df_n = filter(
            row -> row.number_of_scenarios == n,
            runtime_df,
        )

        p = plot(
            xlabel="Number of representative periods",
            ylabel="Runtime (seconds)",
            title="Runtime versus representative periods, N=$n",
            size=(750, 475),
            grid=true,
            gridalpha=0.3,
            legend=:topright,
        )

        for seed in sort(unique(df_n.seed))
            df_seed = sort(
                filter(row -> row.seed == seed, df_n),
                :rp,
            )

            plot!(
                p,
                df_seed.rp,
                df_seed.runtime_baseline;
                marker=:circle,
                linewidth=2,
                label="Hourly baseline, seed $seed",
            )

            plot!(
                p,
                df_seed.rp,
                df_seed.runtime_cc_resolve;
                marker=:circle,
                linewidth=2,
                label="CC + hourly resolve, seed $seed",
            )
        end

        savefig(
            p,
            joinpath(
                outdir,
                "runtime_vs_representative_periods_N$(n).png",
            ),
        )
    end

    return runtime_df
end

function plot_runtime_vs_N()
    n_values = [10, 15, 20]
    seed = 1

    runtime_baseline = Float64[]
    runtime_reduced = Float64[]

    runtime_cols = [
        :time_to_cluster,
        :time_to_read,
        :time_to_create,
        :time_to_solve,
        :time_to_save,
    ]

    for n in n_values
        f = joinpath(
            inputdir,
            "N$(n)_seed$(seed)",
            "results_ScSeRP_N$(n)_seed$(seed).csv",
        )

        if !isfile(f)
            error("Results file not found: $f")
        end

        all_df = CSV.read(f, DataFrame)

        if nrow(all_df) < 3
            error(
                "Expected at least 3 rows in $(basename(f)), " *
                "but found $(nrow(all_df)).",
            )
        end

        # Row 1: hourly baseline
        baseline_runtime = sum(
            Float64(all_df[1, col]) for col in runtime_cols
        )

        # Rows 2 + 3: complete reduced-model runtime
        reduced_runtime = sum(
            Float64(all_df[row, col])
            for row in 2:3
            for col in runtime_cols
        )

        push!(runtime_baseline, baseline_runtime)
        push!(runtime_reduced, reduced_runtime)
    end

    runtime_matrix = hcat(runtime_baseline, runtime_reduced)

    p = groupedbar(
        string.(n_values),
        runtime_matrix;
        bar_position=:dodge,
        label=["Hourly baseline" "Reduced model"],
        xlabel="Number of scenarios",
        ylabel="Runtime (seconds)",
        title="Runtime hourly baseline vs reduced model, seed 1",
        size=(700, 450),
        grid=true,
        gridalpha=0.3,
        legend=:topleft,
    )

    savefig(
        p,
        joinpath(outdir, "runtime_vs_N_seed1.png"),
    )

    return p
end

function plot_investment_difference_N20_seed1()
    baseline_file = joinpath(
        inputdir,
        "0_HourlyBaseline",
        "N20_seed1",
        "Gurobi",
        "var_assets_investment.csv",
    )

    reduced_file = joinpath(
        inputdir,
        "N20_seed1",
        "convex_convex_per_rp_36_reduced_scenario_set",
        "Gurobi",
        "var_assets_investment.csv",
    )

    isfile(baseline_file) || error("Baseline investment file not found: $baseline_file")
    isfile(reduced_file) || error("Reduced investment file not found: $reduced_file")

    baseline_df = CSV.read(baseline_file, DataFrame)
    reduced_df = CSV.read(reduced_file, DataFrame)

    @show names(baseline_df)
    @show names(reduced_df)

    # Adjust these two column names if your CSV uses different names.
    asset_col = :asset
    investment_col = :investment

    # Keep only the columns required for comparison.
    baseline_inv = select(
        baseline_df,
        asset_col,
        investment_col => :investment_baseline,
    )

    reduced_inv = select(
        reduced_df,
        asset_col,
        investment_col => :investment_reduced,
    )

    # outerjoin ensures assets occurring in only one solution are retained.
    comparison_df = outerjoin(
        baseline_inv,
        reduced_inv;
        on=asset_col,
    )

    # Missing means that the asset has no investment in that solution.
    comparison_df.investment_baseline =
        coalesce.(comparison_df.investment_baseline, 0.0)

    comparison_df.investment_reduced =
        coalesce.(comparison_df.investment_reduced, 0.0)

    comparison_df[!, :investment_difference] =
        comparison_df.investment_reduced .-
        comparison_df.investment_baseline

    # Only plot assets for which the investment actually differs.
    plot_df = filter(
        row -> abs(row.investment_difference) > 1e-8,
        comparison_df,
    )

    sort!(plot_df, :investment_difference)

    if isempty(plot_df)
        @warn "No investment differences found for N=20, seed=1"
        return comparison_df
    end

    p = bar(
        string.(plot_df[!, asset_col]),
        plot_df.investment_difference;
        xlabel="Asset",
        ylabel="Investment difference",
        title="Investment differences: reduced - hourly baseline, N=20 seed 1",
        legend=false,
        size=(1000, 550),
        grid=true,
        gridalpha=0.3,
        xrotation=45,
    )

    hline!(
        p,
        [0.0];
        color=col_zero,
        linestyle=:dash,
        linewidth=1.5,
    )

    savefig(
        p,
        joinpath(
            outdir,
            "investment_difference_N20_seed1.png",
        ),
    )

    CSV.write(
        joinpath(
            outdir,
            "investment_difference_N20_seed1.csv",
        ),
        comparison_df,
    )

    return comparison_df
end

plot_boxplot(evaluation=:baseline)
plot_runtime_vs_N()
plot_investment_difference_N20_seed1()
# plot_regret_vs_rp()
# plot_runtime_vs_rp()
