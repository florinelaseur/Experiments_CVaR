# This script generates comparison plots between RASRA and the full benchmark
# Reads results CSVs from outputs/ produced by run_experiments_rasra.jl

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

# Filter main results to the full hourly benchmark only
const BENCHMARK_BASE_NAME = "0_HourlyBenchmark"

mkpath(OUTDIR)

const COL_RASRA = RGB(0.122, 0.471, 0.706)
const COL_BENCHMARK = RGB(0.698, 0.094, 0.122)
const COL_ZERO = RGB(0.4, 0.4, 0.4)

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

"""Extract (N, seed) from filenames."""
function parse_n_seed(filename::String)
    m = match(r"_N(\d+)_seed(\d+)\.csv$", filename)
    isnothing(m) && return nothing
    return parse(Int, m[1]), parse(Int, m[2])
end

"""Load all CSVs with the given prefix, append :N and :seed from the filename."""
function load_all(pattern_prefix::String)
    files = filter(readdir(INPUTDIR; join=true)) do f
        basename(f) |> b -> startswith(b, pattern_prefix) && endswith(b, ".csv")
    end
    isempty(files) && error("No files found in '$INPUTDIR' matching '$pattern_prefix*.csv'")

    frames = DataFrame[]
    for f in sort(files)
        ns = parse_n_seed(basename(f))
        if isnothing(ns)
            @warn "Could not parse N/seed from filename '$(basename(f))' (skipping)"
            continue
        end
        n, s = ns
        df = CSV.read(f, DataFrame)
        df[!, :N] .= n
        df[!, :seed] .= s
        push!(frames, df)
    end
    isempty(frames) && error("No valid files loaded for prefix '$pattern_prefix'")
    return vcat(frames...; cols=:union)
end

@info "Loading main.jl results"
main_raw = load_all("results_N")

@info "Loading RASRA results"
rasra_df = load_all("results_rasra_N")

if !hasproperty(main_raw, :base_name)
    error("main.jl CSVs have no 'base_name' column (cannot filter to benchmark rows)")
end

main_df = filter(r -> r.base_name == BENCHMARK_BASE_NAME, main_raw)

if nrow(main_df) == 0
    error("""No rows with base_name == "$BENCHMARK_BASE_NAME" found in main.jl results. Available base_name values: $(unique(main_raw.base_name))""")
end

joined = innerjoin(rasra_df, main_df; on = [:N, :seed], makeunique = true)

n_dropped = min(nrow(rasra_df), nrow(main_df)) - nrow(joined)
if n_dropped > 0
    @warn "$n_dropped rows dropped during join. Check that (N, seed) pairs exist in both main and RASRA results"
end
nrow(joined) == 0 && error("Join produced zero rows. No matching (N, seed) pairs between RASRA and main results")

N_values = sort(unique(joined.N))
@info "N values: $N_values, seeds per N: $(nrow(filter(r -> r.N == N_values[1], joined)))"

# RASRA columns come first in the join, so clashing main columns get a suffix of _1
rasra_obj_col = :objective_value
main_obj_col = hasproperty(joined, :objective_value_1) ? :objective_value_1 : :objective_value

# Use Tulipa's optimal mu for both sides
rasra_mu_col = :value_at_risk_threshold_mu
main_mu_col = hasproperty(joined, :value_at_risk_threshold_mu_1) ?
               :value_at_risk_threshold_mu_1 : :value_at_risk_threshold_mu

transform!(joined,
    [rasra_obj_col, main_obj_col] => ByRow((r, m) -> (r - m) / m * 100) => :og_pct,
    [rasra_mu_col, main_mu_col] => ByRow((r, m) -> (r - m) / m * 100) => :var_thresh_dev_pct,
)

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
n_vals, og_vecs = per_n_vectors(joined, :og_pct)

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
@info "Saved as 1_og_boxplot.png"

# Plot 2: VaR Threshold Deviation
@info "Plot 2: VaR threshold deviation boxplot"
_, var_dev_vecs = per_n_vectors(joined, :var_thresh_dev_pct)

p2 = plot(;
    title = "VaR Threshold Deviation: RASRA vs. Full Benchmark",
    ylabel = "VaR threshold deviation (%)",
    size = (700, 450),
    grid = true,
    gridalpha = 0.3,
)
hline!(p2, [0.0]; color = COL_ZERO, linestyle = :dash, linewidth = 1.5, label = "Zero reference")
add_boxplots!(p2, n_vals, var_dev_vecs; series_color = COL_RASRA, series_label = "RASRA")
savefig(p2, joinpath(OUTDIR, "2_var_threshold_boxplot.png"))
@info "Saved as 2_var_threshold_boxplot.png"

# Plot 3: Runtime breakdown
@info "Plot 3: Runtime breakdown"

n_groups = length(n_vals)
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
    xticks = (1:n_groups, string.(n_vals)),
)

step_specs = [
    (:time_step1_backward_reduction, STEP_LABELS[1], STEP_COLS[1]),
    (:time_step2_solve_j, STEP_LABELS[2], STEP_COLS[2]),
    (:time_step3_evaluate_costs, STEP_LABELS[3], STEP_COLS[3]),
    (nothing, STEP_LABELS[4], STEP_COLS[4]),
    (:time_step6_solve_reduced, STEP_LABELS[5], STEP_COLS[5]),
]

mean_step_times = map(step_specs) do (col, _, _)
    map(n_vals) do n
        sub = filter(r -> r.N == n, joined)
        isnothing(col) ?
            mean(sub.time_step4_identify_effective .+ sub.time_step5_adjust_probabilities) :
            mean(sub[!, col])
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

main_times = map(n_vals) do n
    mean(filter(r -> r.N == n, main_df).time_to_solve)
end

for i in 1:n_groups
    xl, xr = x_main[i] - bar_width/2, x_main[i] + bar_width/2
    plot!(p3, Shape([xl, xr, xr, xl], [0.0, 0.0, main_times[i], main_times[i]]);
        fillcolor = COL_BENCHMARK,
        linecolor = :white,
        linewidth = 0.5,
        fillalpha = 0.6,
        label = i == 1 ? "Full solve (main.jl)" : "",
    )
end

savefig(p3, joinpath(OUTDIR, "3_runtime_breakdown.png"))
@info "Saved as 3_runtime_breakdown.png"

@info "Done. All plots saved to $OUTDIR"