using CSV, DataFrames, Plots

results = joinpath(@__DIR__, "outputs_proxy", "results.csv")
df_results = CSV.read(results, DataFrame)

plots_folder = joinpath(@__DIR__, "outputs_proxy", "plots")
mkpath(plots_folder)

case_labels = ["Hourly Benchmark", "30 RPs", "60 RPs", "90 RPs", "Hourly CC"]
runtime_case_labels = [
    "Hourly Benchmark",
    "30 RPs",
    "60 RPs",
    "90 RPs",
    "CC (30 RPs + hourly reduced scenario set)",
]

df_plot = copy(df_results)
df_plot[!, :case_label] = case_labels

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

idx_30 = findfirst(==("30 RPs"), df_plot.case_label)
idx_cc = findfirst(==("Hourly CC"), df_plot.case_label)

runtime_30_total = df_plot[idx_30, :runtime_total]
runtime_cc_own = df_plot[idx_cc, :runtime_own]

# System costs
df_plot[!, :system_cost] = df_plot.objective_value
df_plot[rp_rows, :system_cost] =
    df_plot[rp_rows, :objective_value_resolve_benchmark]

# LOLE
df_plot[!, :lole_e_demand] =
    df_plot.num_loss_of_load_e_demand ./ 20

df_plot[!, :lole_h2_demand] =
    df_plot.num_loss_of_load_h2_demand ./ 20

function zoom_limits(v; margin_fraction=0.08)
    vmin = minimum(v)
    vmax = maximum(v)
    margin = margin_fraction * (vmax - vmin)
    return (vmin - margin, vmax + margin)
end

# 1. Runtime comparison

runtime_comparison = copy(df_plot.runtime_total)

# CC runtime = full 30 RPs runtime + own hourly reduced scenario set runtime
runtime_comparison[idx_cc] = runtime_30_total + runtime_cc_own

@show runtime_30_total
@show runtime_cc_own
@show runtime_comparison[idx_cc]

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

savefig(p_runtime, joinpath(plots_folder, "runtime_comparison.png"))

# 2. System costs comparison
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

savefig(p_costs, joinpath(plots_folder, "system_costs_comparison.png"))

# 3. LOLE comparison

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

savefig(p_lole, joinpath(plots_folder, "lole_comparison.png"))

# 4. VaR comparison

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

savefig(p_var, joinpath(plots_folder, "var_threshold_comparison.png"))

@info "Comparison plots saved in: $plots_folder"