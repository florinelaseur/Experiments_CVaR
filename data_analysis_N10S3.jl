using CSV: CSV
using TOML: TOML
using DataFrames
using Plots
using StatsPlots

#compare the flows for scenario 7:

var_flow_df = CSV.read(joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed3", "convex_conicalb_per_rp_24", "Gurobi", "var_flow.csv"), DataFrame)
var_flow_fixed_df = CSV.read(joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed3", "fixed", "convex_conicalb_per_rp_24CC", "Gurobi", "var_flow.csv"), DataFrame)

scenario_7_periods = 145:168

var_flow_s7 = filter(row -> row.rep_period in scenario_7_periods, var_flow_df)
var_flow_fixed_s7 = filter(row -> row.rep_period in scenario_7_periods, var_flow_fixed_df)

join_cols = [
    :from_asset,
    :to_asset,
    :milestone_year,
    :rep_period,
    :time_block_start,
    :time_block_end,
]

comparison_df = innerjoin(
    select(var_flow_s7, join_cols..., :solution => :solution_benchmark),
    select(var_flow_fixed_s7, join_cols..., :solution => :solution_fixed);
    on=join_cols,
)

comparison_df[!, :diff] =
    comparison_df.solution_fixed .- comparison_df.solution_benchmark

CSV.write(
    joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed6", "flow_difference_scenario_9.csv"),
    comparison_df;
    writeheader=true,
)

comparison_df_sorted = sort(comparison_df, :diff, rev=true)

CSV.write(
    joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed3", "flow_difference_scenario_9_sorted.csv"),
    comparison_df_sorted;
    writeheader=true,
)

first(comparison_df_sorted, 20)

smr_diff_df = filter(
    row ->
        row.from_asset == "smr_ccs" &&
        row.to_asset == "h2_demand" &&
        abs(row.diff) > 1e-9,
    comparison_df,
)

smr_diff_df #deze is leeg

obj_breakdown_df = CSV.read(joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed6", "convex_conicalb_per_rp_24", "Gurobi", "obj_breakdown.csv"), DataFrame)
obj_breakdown_fixed_df = CSV.read(joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed6", "fixed", "convex_conicalb_per_rp_24CC", "Gurobi", "obj_breakdown.csv"), DataFrame)


#check which flows show greatest differnces over whole scenario 9
groupby(comparison_df, [:from_asset, :to_asset])
flow_summary = combine(
    groupby(comparison_df, [:from_asset, :to_asset]),
    :solution_benchmark => sum => :benchmark,
    :solution_fixed => sum => :fixed,
    :diff => sum => :diff_sum,
    :diff => (x -> sum(abs.(x))) => :abs_diff_sum,
)
sort!(flow_summary, :abs_diff_sum, rev=true)

CSV.write(
    joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "N10_seed3", "aggregated_flow_differences_scenario7.csv"),
    flow_summary;
    writeheader=true,
)