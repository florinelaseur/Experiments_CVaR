# Evaluation plots for the Kantorovich distance scenario reduction method (12-scenario set).
#
# Metrics follow the thesis evaluation criteria:
#   OG(%)    = (F(z^Ω_hat, Ω) - F(z^Ω, Ω)) / F(z^Ω, Ω) × 100
#   ΔCVaR(%) = (CVaR_α(z^Ω_hat, Ω) - CVaR_α(z^Ω, Ω)) / CVaR_α(z^Ω, Ω) × 100
#
# OG and out-of-sample ΔCVaR require a feasible benchmark re-solve; infeasible
# cases are flagged.  In-sample metrics (approx. VaR, approx. CVaR) are computed
# from the reduced model and are always available.
#
# Usage:  julia utils/generate_kantorovich_plots.jl
#         (run from the repo root)

using DataFrames
using CSV
using Plots

# ── Configuration (must match config.toml) ──────────────────────────────────
const ALPHA   = 0.7   # CVaR confidence level
const LAMBDA  = 0.3   # CVaR weight in objective

const RESULTS_CSV = "outputs/kantorovich_results_12.csv"
const BASE_DIR    = "outputs"
const PLOT_DIR    = "outputs/plots/kantorovich_12"
mkpath(PLOT_DIR)

# ── Helpers ──────────────────────────────────────────────────────────────────
function read_mu(folder)
    CSV.read(joinpath(folder, "var_value_at_risk_threshold_mu.csv"), DataFrame).solution[1]
end

function read_cvar(folder)
    mu = read_mu(folder)
    xi_df = CSV.read(joinpath(folder, "var_tail_excess_slack_xi.csv"), DataFrame)
    mu + (1 / (1 - ALPHA)) * sum(xi_df.probability .* xi_df.solution)
end

function read_cvar_from_breakdown(folder)
    df = CSV.read(joinpath(folder, "obj_breakdown.csv"), DataFrame)
    only(filter(r -> r.name == "conditional_value_at_risk_term", df)).value
end

# ── Load results ─────────────────────────────────────────────────────────────
results_df  = CSV.read(RESULTS_CSV, DataFrame)
bench_row   = only(filter(r -> r.base_name == "0_HourlyBenchmark", results_df))
kanto_df    = sort(filter(r -> r.base_name != "0_HourlyBenchmark", results_df), :rp)

# ── Benchmark metrics ─────────────────────────────────────────────────────────
bench_folder = joinpath(BASE_DIR, "0_HourlyBenchmark", "Gurobi")
bench_mu     = read_mu(bench_folder)
bench_cvar   = read_cvar(bench_folder)   # CVaR_α(z^Ω, Ω)
hourly_obj   = bench_row.objective_value  # F(z^Ω, Ω)

@info "Benchmark" obj=hourly_obj mu=bench_mu cvar=bench_cvar

# ── Per-RP metrics ────────────────────────────────────────────────────────────
rps = kanto_df.rp
n   = length(rps)

og_pct         = fill(NaN, n)   # OG (%) – out-of-sample, feasible only
dcvar_oos_pct  = fill(NaN, n)   # ΔCVaR (%) – out-of-sample, feasible only
dcvar_ins_pct  = fill(NaN, n)   # ΔCVaR (%) – in-sample (reduced model)
dvar_pct       = fill(NaN, n)   # ΔVaR (%) – in-sample (reduced model)
var_reduced    = fill(NaN, n)   # Approx. VaR from reduced model
var_oos        = fill(NaN, n)   # VaR from full re-solve (feasible only)
cvar_ins       = fill(NaN, n)   # Approx. CVaR from reduced model
cvar_oos       = fill(NaN, n)   # CVaR from full re-solve (feasible only)
is_feasible    = fill(false, n)
lol_e          = Int[row.num_loss_of_load_e_demand for row in eachrow(kanto_df)]
lol_h2         = Int[row.num_loss_of_load_h2_demand for row in eachrow(kanto_df)]
total_time     = [row.time_to_cluster + row.time_to_read + row.time_to_create +
                  row.time_to_solve   + row.time_to_save for row in eachrow(kanto_df)]
time_solve     = kanto_df.time_to_solve
time_cluster   = kanto_df.time_to_cluster

for (i, row) in enumerate(eachrow(kanto_df))
    rp = row.rp
    reduced_folder = joinpath(BASE_DIR, "kantorovich_rp_$rp", "Gurobi")

    # In-sample VaR and CVaR from the reduced model
    var_reduced[i]   = read_mu(reduced_folder)
    cvar_ins[i]      = read_cvar(reduced_folder)
    dvar_pct[i]      = (var_reduced[i]  - bench_mu)   / bench_mu   * 100
    dcvar_ins_pct[i] = (cvar_ins[i]     - bench_cvar) / bench_cvar * 100

    # Out-of-sample (full benchmark re-solve with fixed investments)
    feasible = !ismissing(row.termination_status_resolve_benchmark) &&
               row.termination_status_resolve_benchmark == "OPTIMAL"
    is_feasible[i] = feasible

    if feasible
        resolve_folder = joinpath(BASE_DIR, "fixed", "kantorovich_rp_$rp", "Gurobi")
        og_pct[i]        = (row.objective_value_resolve_benchmark - hourly_obj) / hourly_obj * 100
        cvar_oos[i]      = read_cvar_from_breakdown(resolve_folder)
        dcvar_oos_pct[i] = (cvar_oos[i] - bench_cvar) / bench_cvar * 100
        var_oos[i]       = read_mu(resolve_folder)

        @info "rp=$rp (OPTIMAL)" OG=og_pct[i] dCVaR=dcvar_oos_pct[i] dVaR_oos=(var_oos[i]-bench_mu)/bench_mu*100
    else
        @info "rp=$rp (INFEASIBLE) in-sample" dVaR=dvar_pct[i] dCVaR_ins=dcvar_ins_pct[i]
    end
end

# ── Plotting helpers ──────────────────────────────────────────────────────────
xi          = 1:n   # x positions
rp_labels   = string.(rps)
xtick_setup = (collect(xi), rp_labels)

BENCH_COL    = :black
FEASIBLE_COL = :steelblue
INFEAS_COL   = :tomato
INSAMPLE_COL = :darkorange

default(
    fontfamily = "Helvetica",
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.25,
    tickfontsize  = 10,
    guidefontsize = 11,
    legendfontsize = 9,
    dpi = 150,
)

infeas_idx  = findall(.!is_feasible)
feasible_idx = findall(is_feasible)

# ── Plot 1: Optimality Gap (OG %) ────────────────────────────────────────────
p1 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "OG (%)",
    title  = "Optimality Gap — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :topright,
)

# Infeasible markers at y=0 with annotation
if !isempty(infeas_idx)
    scatter!(p1, xi[infeas_idx], zeros(length(infeas_idx));
        markershape = :xcross, markersize = 12,
        markercolor = INFEAS_COL, markerstrokecolor = INFEAS_COL,
        markerstrokewidth = 2, label = "Infeasible re-solve")
end

# Feasible OG points
if !isempty(feasible_idx)
    scatter!(p1, xi[feasible_idx], og_pct[feasible_idx];
        markershape = :circle, markersize = 9,
        markercolor = FEASIBLE_COL, markerstrokecolor = FEASIBLE_COL,
        label = "OG — out-of-sample")
end

hline!(p1, [0.0];
    color = BENCH_COL, linestyle = :dash, linewidth = 1.5, label = "Benchmark (OG = 0)")

savefig(p1, joinpath(PLOT_DIR, "og_pct.png"))
@info "Saved og_pct.png"

# ── Plot 2: CVaR Deviation (ΔCVaR %) ────────────────────────────────────────
p2 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "ΔCVaR (%)",
    title  = "CVaR Deviation — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :bottomright,
)

# Connecting line through all in-sample points (no markers on the line itself)
plot!(p2, xi, dcvar_ins_pct;
    color = INSAMPLE_COL, linewidth = 1.5, linestyle = :dash, label = "")

# In-sample infeasible: hollow diamonds — reduced-model approx., but investments are infeasible
if !isempty(infeas_idx)
    scatter!(p2, xi[infeas_idx], dcvar_ins_pct[infeas_idx];
        markershape = :diamond, markersize = 8,
        markercolor = :white, markerstrokecolor = INSAMPLE_COL,
        markerstrokewidth = 2, label = "ΔCVaR in-sample (infeasible re-solve)")
end

# In-sample feasible: filled diamonds
if !isempty(feasible_idx)
    scatter!(p2, xi[feasible_idx], dcvar_ins_pct[feasible_idx];
        markershape = :diamond, markersize = 8,
        markercolor = INSAMPLE_COL, markerstrokecolor = INSAMPLE_COL,
        label = "ΔCVaR in-sample (feasible re-solve)")
end

# Out-of-sample (full benchmark re-solve, feasible only)
if !isempty(feasible_idx)
    scatter!(p2, xi[feasible_idx], dcvar_oos_pct[feasible_idx];
        markershape = :circle, markersize = 9,
        markercolor = FEASIBLE_COL, markerstrokecolor = FEASIBLE_COL,
        label = "ΔCVaR out-of-sample")
end

hline!(p2, [0.0];
    color = BENCH_COL, linestyle = :dash, linewidth = 1.5, label = "Benchmark (ΔCVaR = 0)")

savefig(p2, joinpath(PLOT_DIR, "dcvar_pct.png"))
@info "Saved dcvar_pct.png"

# ── Plot 3: VaR deviation (ΔVaR %) ──────────────────────────────────────────
p3 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "ΔVaR (%)",
    title  = "VaR Deviation — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :bottomright,
)

plot!(p3, xi, dvar_pct;
    color = INSAMPLE_COL, linewidth = 1.5, linestyle = :dash, label = "")

if !isempty(infeas_idx)
    scatter!(p3, xi[infeas_idx], dvar_pct[infeas_idx];
        markershape = :square, markersize = 8,
        markercolor = :white, markerstrokecolor = INSAMPLE_COL,
        markerstrokewidth = 2, label = "ΔVaR in-sample (infeasible re-solve)")
end

if !isempty(feasible_idx)
    scatter!(p3, xi[feasible_idx], dvar_pct[feasible_idx];
        markershape = :square, markersize = 8,
        markercolor = INSAMPLE_COL, markerstrokecolor = INSAMPLE_COL,
        label = "ΔVaR in-sample (feasible re-solve)")
    dvar_oos_pct = (var_oos[feasible_idx] .- bench_mu) ./ bench_mu .* 100
    scatter!(p3, xi[feasible_idx], dvar_oos_pct;
        markershape = :circle, markersize = 9,
        markercolor = FEASIBLE_COL, markerstrokecolor = FEASIBLE_COL,
        label = "ΔVaR out-of-sample")
end

hline!(p3, [0.0];
    color = BENCH_COL, linestyle = :dash, linewidth = 1.5, label = "Benchmark (ΔVaR = 0)")

savefig(p3, joinpath(PLOT_DIR, "dvar_pct.png"))
@info "Saved dvar_pct.png"

# ── Plot 4: Absolute VaR comparison ─────────────────────────────────────────
scale = 1e6   # display in MEUR
p4 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "VaR — μ (MEUR)",
    title  = "Value-at-Risk Comparison — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :bottomright,
)

hline!(p4, [bench_mu / scale];
    color = BENCH_COL, linestyle = :dash, linewidth = 2, label = "Benchmark VaR")

plot!(p4, xi, var_reduced ./ scale;
    color = INSAMPLE_COL, linewidth = 1.5, linestyle = :dash, label = "")

if !isempty(infeas_idx)
    scatter!(p4, xi[infeas_idx], var_reduced[infeas_idx] ./ scale;
        markershape = :square, markersize = 8,
        markercolor = :white, markerstrokecolor = INSAMPLE_COL,
        markerstrokewidth = 2, label = "Approx. VaR in-sample (infeasible re-solve)")
end

if !isempty(feasible_idx)
    scatter!(p4, xi[feasible_idx], var_reduced[feasible_idx] ./ scale;
        markershape = :square, markersize = 8,
        markercolor = INSAMPLE_COL, markerstrokecolor = INSAMPLE_COL,
        label = "Approx. VaR in-sample (feasible re-solve)")
    scatter!(p4, xi[feasible_idx], var_oos[feasible_idx] ./ scale;
        markershape = :circle, markersize = 9,
        markercolor = FEASIBLE_COL, markerstrokecolor = FEASIBLE_COL,
        label = "VaR out-of-sample")
end

savefig(p4, joinpath(PLOT_DIR, "var_absolute.png"))
@info "Saved var_absolute.png"

# ── Plot 5: Loss of Load ─────────────────────────────────────────────────────
p5 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "Loss-of-Load Steps",
    title  = "Expected Loss of Load — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :topleft,
)

bar!(p5, xi .- 0.2, lol_e;
    bar_width = 0.35, color = FEASIBLE_COL, label = "Electricity demand")
bar!(p5, xi .+ 0.2, lol_h2;
    bar_width = 0.35, color = INSAMPLE_COL, label = "H₂ demand")

savefig(p5, joinpath(PLOT_DIR, "loss_of_load.png"))
@info "Saved loss_of_load.png"

# ── Plot 6: Runtime ──────────────────────────────────────────────────────────
p6 = plot(;
    xlabel = "Representative Scenarios",
    ylabel = "Time (s)",
    title  = "Computational Time — Kantorovich (Ω = 12 scenarios)",
    xticks = xtick_setup,
    size   = (680, 380),
    legend = :topleft,
)

plot!(p6, xi, total_time;
    markershape = :circle, markersize = 7, color = FEASIBLE_COL,
    linewidth = 2, label = "Total time")
plot!(p6, xi, time_solve;
    markershape = :square, markersize = 6, color = INSAMPLE_COL,
    linewidth = 1.5, linestyle = :dash, label = "Solve time")
plot!(p6, xi, time_cluster;
    markershape = :diamond, markersize = 6, color = INFEAS_COL,
    linewidth = 1.5, linestyle = :dot, label = "Cluster time")

savefig(p6, joinpath(PLOT_DIR, "runtime.png"))
@info "Saved runtime.png"

# ── Combined overview panel ──────────────────────────────────────────────────
combined = plot(p1, p2, p3, p6;
    layout = (2, 2),
    size   = (1200, 760),
    plot_title = "Kantorovich Distance — Scenario Reduction Evaluation (Ω = 12)",
    margin = 5Plots.mm,
)

savefig(combined, joinpath(PLOT_DIR, "overview.png"))
@info "Saved overview.png"

@info "All plots written to $PLOT_DIR"
