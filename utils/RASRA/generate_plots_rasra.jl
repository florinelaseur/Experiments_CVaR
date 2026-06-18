# Generate comparison plots for RASRA vs. the full benchmark
# Reads three CSV results from the methods and compares the Optimality Gap, and CVaR deviation, and runtime

using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using TOML: TOML

gr()

const PROJECT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const INPUTDIR = joinpath(PROJECT_DIR, "outputs")
const OUTDIR = joinpath(PROJECT_DIR, "outputs", "plots")

config = TOML.parsefile(joinpath(PROJECT_DIR, "config.toml"))
const LAMBDA = config["simulation"]["risk_aversion_weight_lambda"]
const ALPHA = config["simulation"]["risk_aversion_confidence_level"]

const BENCHMARK_BASE_NAME = "0_HourlyBenchmark"

mkpath(OUTDIR)

const COL_RASRA = RGB(0.122, 0.471, 0.706)
const COL_BENCHMARK = RGB(0.698, 0.094, 0.122)
const COL_FIXED = RGB(0.596, 0.306, 0.639)
const COL_ZERO = RGB(0.4,   0.4,   0.4)

const STEP_COLS = [
    RGB(0.122, 0.471, 0.706),
    RGB(0.200, 0.627, 0.173),
    RGB(1.000, 0.498, 0.055),
    RGB(0.890, 0.102, 0.110),
    RGB(0.596, 0.306, 0.639),
]
const STEP_LABELS = [
    "Step 1: backward reduction",
    "Step 2: solve J",
    "Step 3: evaluate costs",
    "Step 4+5: effective + probs",
    "Step 6: solve reduced",
]

# Loaders

"""Extract (N, seed) from a filename like results_rasra_fixed_N20_seed3.csv."""
function parse_n_seed(filename::String)
    m = match(r"_N(\d+)_seed(\d+)\.csv$", filename)
    isnothing(m) && return nothing
    return parse(Int, m[1]), parse(Int, m[2])
end

"""Load all CSVs in INPUTDIR whose basename starts with pattern_prefix,
appending :N and :seed columns parsed from the filename."""
function load_all(pattern_prefix::String)
    files = filter(readdir(INPUTDIR; join=true)) do f
        b = basename(f)
        startswith(b, pattern_prefix) && endswith(b, ".csv")
    end
    isempty(files) && error("No files in '$INPUTDIR' matching '$(pattern_prefix)*.csv'")

    frames = DataFrame[]
    for f in sort(files)
        ns = parse_n_seed(basename(f))
        if isnothing(ns)
            @warn "Could not parse N/seed from '$(basename(f))': skipping"
            continue
        end
        n, s = ns
        df = CSV.read(f, DataFrame)
        df[!, :N] .= n
        df[!, :seed] .= s
        push!(frames, df)
    end
    isempty(frames) && error("No valid files for prefix '$pattern_prefix'")
    return vcat(frames...; cols = :union)
end

# Load results
@info "Loading main.jl benchmark results"
main_raw = load_all("results_N")

@info "Loading RASRA results"
rasra_df = load_all("results_rasra_N")

@info "Loading fixed re-solve results"
rasra_fixed_df = load_all("results_rasra_fixed_N")

# Filter benchmark to the hourly benchmark rows only
if !hasproperty(main_raw, :base_name)
    error("main.jl CSVs have no 'base_name' column")
end
main_df = filter(r -> r.base_name == BENCHMARK_BASE_NAME, main_raw)
nrow(main_df) == 0 && error("No rows with base_name == \"$BENCHMARK_BASE_NAME\" in main results")

for df in (main_df, rasra_df, rasra_fixed_df)
    if hasproperty(df, :solver)
        df[!, :solver] = string.(df.solver)
    end
end

# Build comparison dataframe for OG and CVaR plots
fixed_cols = select(rasra_fixed_df,
    :N, :seed, :solver,
    :objective_value_fixed,
    :value_at_risk_threshold_mu => :mu_fixed,
)

bench_cols = select(main_df,
    :N, :seed, :solver,
    :objective_value => :objective_benchmark,
    :value_at_risk_threshold_mu => :mu_benchmark,
)

comparison_df = innerjoin(fixed_cols, bench_cols; on = [:N, :seed, :solver])

n_dropped = nrow(fixed_cols) + nrow(bench_cols) - 2 * nrow(comparison_df)
if n_dropped > 0
    @warn "$n_dropped rows lost in comparison join: check that (N, seed, solver) " *
          "pairs exist in both rasra_fixed and main results"
end
nrow(comparison_df) == 0 && error("Comparison join produced zero rows")

transform!(comparison_df,
    [:objective_value_fixed, :objective_benchmark] =>
        ByRow((fixed, bench) -> (fixed - bench) / bench * 100) => :og_pct,
    [:mu_fixed, :mu_benchmark] =>
        ByRow((mf, mb) -> isnan(mf) || isnan(mb) || mb == 0.0 ?
                          NaN : (mf - mb) / mb * 100) => :cvar_dev_pct,
)

N_values = sort(unique(comparison_df.N))
@info "N values: $N_values, seeds per N: $(nrow(filter(r -> r.N == N_values[1], comparison_df)))"

# Helpers for plots
function per_n_vectors(df, col)
    ns = sort(unique(df.N))
    return ns, [df[df.N .== n, col] for n in ns]
end

function add_boxplots!(p, n_vals, vecs; series_color, series_label)
    for (i, v) in enumerate(vecs)
        boxplot!(p, fill(i, length(v)), v;
            color = series_color,
            fillalpha = 0.45,
            linecolor = series_color,
            outliers = true,
            markersize = 4,
            label = i == 1 ? series_label : "",
        )
    end
    plot!(p;
        xticks = (1:length(n_vals), string.(n_vals)),
        xlabel = "N (number of scenarios)",
        legend = :topright,
    )
end

# Plot 1: Optimality Gap
@info "Plot 1: OG boxplot"
n_vals, og_vecs = per_n_vectors(comparison_df, :og_pct)

p1 = plot(;
    title = "Optimality Gap: RASRA vs. Full Benchmark",
    ylabel = "OG (%)",
    size = (700, 450),
    grid = true,
    gridalpha = 0.3,
)
hline!(p1, [0.0]; color = COL_ZERO, linestyle = :dash, linewidth = 1.5, label = "Zero reference")
add_boxplots!(p1, n_vals, og_vecs; series_color = COL_RASRA, series_label = "RASRA")
savefig(p1, joinpath(OUTDIR, "1_og_boxplot.png"))
@info "Saved 1_og_boxplot.png"

# Plot 2: CVaR Deviation
@info "Plot 2: CVaR deviation boxplot"
_, cvar_vecs = per_n_vectors(comparison_df, :cvar_dev_pct)

p2 = plot(;
    title = "CVaR Deviation: RASRA vs. Full Benchmark",
    ylabel = "CVaR deviation (%)",
    size = (700, 450),
    grid = true,
    gridalpha = 0.3,
)
hline!(p2, [0.0]; color = COL_ZERO, linestyle = :dash, linewidth = 1.5, label = "Zero reference")
add_boxplots!(p2, n_vals, cvar_vecs; series_color = COL_RASRA, series_label = "RASRA")
savefig(p2, joinpath(OUTDIR, "2_cvar_deviation_boxplot.png"))
@info "Saved 2_cvar_deviation_boxplot.png"

# Plot 3: Runtime breakdown
@info "Plot 3: Runtime breakdown"

rasra_timing = select(rasra_df,
    :N, :seed, :solver,
    :time_step1_backward_reduction,
    :time_step2_solve_j,
    :time_step3_evaluate_costs,
    :time_step4_identify_effective,
    :time_step5_adjust_probabilities,
    :time_step6_solve_reduced,
)
runtime_df = rasra_timing

n_groups = length(N_values)
bar_width = 0.35
x_rasra = collect(1:n_groups) .- bar_width / 2
x_main = collect(1:n_groups) .+ bar_width / 2

p3 = plot(;
    title = "Runtime for various N (mean across seeds)",
    ylabel = "Time (seconds)",
    xlabel = "N (number of scenarios)",
    size = (800, 500),
    legend = :topleft,
    grid = true,
    gridalpha = 0.3,
    xticks = (1:n_groups, string.(N_values)),
)

step_specs = [
    (:time_step1_backward_reduction, STEP_LABELS[1], STEP_COLS[1]),
    (:time_step2_solve_j, STEP_LABELS[2], STEP_COLS[2]),
    (:time_step3_evaluate_costs, STEP_LABELS[3], STEP_COLS[3]),
    (nothing, STEP_LABELS[4], STEP_COLS[4]),
    (:time_step6_solve_reduced, STEP_LABELS[5], STEP_COLS[5]),
]

mean_step_times = map(step_specs) do (col, _, _)
    map(N_values) do n
        sub = filter(r -> r.N == n, runtime_df)
        if isnothing(col)
            mean(sub.time_step4_identify_effective .+ sub.time_step5_adjust_probabilities)
        else
            mean(sub[!, col])
        end
    end
end

bottoms = zeros(n_groups)
for ((_, label, col_color), means) in zip(step_specs, mean_step_times)
    tops = bottoms .+ means
    for i in 1:n_groups
        xl, xr = x_rasra[i] - bar_width/2, x_rasra[i] + bar_width/2
        yb, yt = bottoms[i], tops[i]
        plot!(p3, Shape([xl, xr, xr, xl], [yb, yb, yt, yt]);
            fillcolor = col_color,
            linecolor = :white,
            linewidth = 0.5,
            fillalpha = 0.85,
            label = i == 1 ? label : "",
        )
    end
    bottoms .= tops
end

# Benchmark bar (full solve time only, no fixed overhead)
bench_times = map(N_values) do n
    mean(filter(r -> r.N == n, main_df).time_to_solve)
end

for i in 1:n_groups
    xl, xr = x_main[i] - bar_width/2, x_main[i] + bar_width/2
    plot!(p3, Shape([xl, xr, xr, xl], [0.0, 0.0, bench_times[i], bench_times[i]]);
        fillcolor = COL_BENCHMARK,
        linecolor = :white,
        linewidth = 0.5,
        fillalpha = 0.6,
        label = i == 1 ? "Full benchmark solve" : "",
    )
end

savefig(p3, joinpath(OUTDIR, "3_runtime_breakdown.png"))
@info "Saved 3_runtime_breakdown.png"

@info "Done. All plots saved to $OUTDIR"