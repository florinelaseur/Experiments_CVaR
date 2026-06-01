output_folder = joinpath(@__DIR__, "outputs_copy")

results = joinpath(output_folder, "results.csv")
df_results = CSV.read(results, DataFrame)

plots_folder = joinpath(output_folder, "plots")
mkpath(plots_folder)

df_plot = copy(df_results)

df_plot = copy(df_results)

# RP model runtime, excluding clustering
df_plot[!, :runtime_rp] =
    df_plot.time_to_cluster .+
    df_plot.time_to_read .+
    df_plot.time_to_create .+
    df_plot.time_to_solve .+
    df_plot.time_to_save

comparison = DataFrame(
    case_label=String[],
    runtime=Float64[],
    system_cost=Float64[],
    var_mu=Float64[],
    lole_e_demand=Union{Missing,Float64}[],
    lole_h2_demand=Union{Missing,Float64}[],
)

for row in eachrow(df_plot)

    if row.scenario_set == "full"
        push!(comparison, (
            "$(row.rp) periods full set",
            row.runtime_rp,
            row.objective_value,
            row.value_at_risk_threshold_mu,
            row.num_loss_of_load_e_demand / number_of_scenarios,
            row.num_loss_of_load_h2_demand / number_of_scenarios,
        ))

    elseif row.scenario_set == "reduced"
        # RP reduced model
        push!(comparison, (
            "$(row.rp) RPs reduced set",
            row.runtime_rp,
            row.objective_value,
            row.value_at_risk_threshold_mu,
            missing,
            missing,
        ))

        # Hourly resolve on reduced set
        push!(comparison, (
            "$(row.rp) RPs + hourly reduced set",
            row.runtime_rp + row.time_to_resolve_hourly,
            row.objective_value_resolve_hourly,
            row.value_at_risk_threshold_mu_hourly,
            row.num_loss_of_load_e_demand / number_of_scenarios,
            row.num_loss_of_load_h2_demand / number_of_scenarios,
        ))
    end
end

p_runtime = bar(
    comparison.case_label,
    comparison.runtime;
    xlabel="Case",
    ylabel="Runtime [s]",
    title="Runtime Comparison",
    label=false,
    xrotation=30,
)

savefig(p_runtime, joinpath(plots_folder, "runtime_comparison.png"))

p_costs = bar(
    comparison.case_label,
    comparison.system_cost;
    xlabel="Case",
    ylabel="System Cost",
    title="System Costs Comparison",
    label=false,
    xrotation=30,
    ylim=zoom_limits(comparison.system_cost),
)

savefig(p_costs, joinpath(plots_folder, "system_costs_comparison.png"))

p_var = bar(
    comparison.case_label,
    comparison.var_mu;
    xlabel="Case",
    ylabel="VaR threshold μ",
    title="Value at Risk Threshold Comparison",
    label=false,
    xrotation=30,
    ylim=zoom_limits(comparison.var_mu),
)

savefig(p_var, joinpath(plots_folder, "var_threshold_comparison.png"))

df_lole = dropmissing(comparison, [:lole_e_demand, :lole_h2_demand])

x = collect(1:nrow(df_lole))
w = 0.35

p_lole = bar(
    x .- w / 2,
    df_lole.lole_e_demand;
    bar_width=w,
    xlabel="Case",
    ylabel="LOLE [expected events per scenario]",
    title="Loss of Load Expected Comparison",
    label="Electricity demand",
    xticks=(x, df_lole.case_label),
    xrotation=30,
    legend=:topright,
)

bar!(
    p_lole,
    x .+ w / 2,
    df_lole.lole_h2_demand;
    bar_width=w,
    label="Hydrogen demand",
)

savefig(p_lole, joinpath(plots_folder, "lole_comparison.png"))