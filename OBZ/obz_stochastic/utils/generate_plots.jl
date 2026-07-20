# Run this only after results given from main

using DataFrames
using CSV
using Plots

# helper functions
@info "Including constants and helper functions"
include("constants.jl")
include("functions.jl")

mkpath("outputs/plots")

script_dir = @__DIR__
output_path = joinpath(script_dir, "..", "outputs")

results_df = CSV.read(joinpath(output_path, "results.csv"), DataFrame)

hourly_row = results_df[results_df.base_name.=="0_HourlyBenchmark", :]
hourly_obj = only(hourly_row.objective_value)

# compute relative regret 
results_df.rel_regret = [
    row.base_name == "0_HourlyBenchmark" ? 0.0 :
    (row.objective_value_resolve_benchmark - hourly_obj) / hourly_obj
    for row in eachrow(results_df)
]
plot_path = joinpath(output_path, "plots")

case_studies_path = joinpath(script_dir, "..", "case-studies-info.csv")
case_studies_df = CSV.read(case_studies_path, DataFrame)

mkpath(plot_path)
@info "Plotting relative regret"
plot_values_stocmethod_method(results_df, case_studies_df, "rel_regret"; savepath=joinpath(plot_path, "relative_regret.png"))

@info "Plotting time to cluster"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_cluster"; savepath=joinpath(plot_path, "time_to_cluster.png"))

@info "Plotting time to solve"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_solve"; savepath=joinpath(plot_path, "time_to_solve.png"))

@info "Plotting time to create"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_create"; savepath=joinpath(plot_path, "time_to_create.png"))


# compute total time 
results_df.total_time = [
    row.time_to_cluster + row.time_to_read + row.time_to_create + row.time_to_solve + row.time_to_save
    for row in eachrow(results_df)
]

@info "Plotting total time"
plot_values_stocmethod_method(results_df, case_studies_df, "total_time"; savepath=joinpath(plot_path, "total_time.png"))

# compute number of steps with loss of load 
hourly_lol_e = only(hourly_row.num_loss_of_load_e_demand)
results_df.num_loss_of_load_e_demand = [
    row.num_loss_of_load_e_demand - hourly_lol_e
    for row in eachrow(results_df)
]
hourly_lol_h2 = only(hourly_row.num_loss_of_load_h2_demand)
results_df.num_loss_of_load_h2_demand = [
    row.num_loss_of_load_h2_demand - hourly_lol_h2
    for row in eachrow(results_df)
]

hourly_amount_lol_e = only(hourly_row.loss_of_load_e_demand)
results_df.loss_of_load_e_demand = [
    row.loss_of_load_e_demand - hourly_amount_lol_e
    for row in eachrow(results_df)
]
hourly_amount_lol_h2 = only(hourly_row.loss_of_load_h2_demand)
results_df.loss_of_load_h2_demand = [
    row.loss_of_load_h2_demand - hourly_amount_lol_h2
    for row in eachrow(results_df)
]
results_df.total_steps_loss_of_load = [
    row.num_loss_of_load_h2_demand + row.num_loss_of_load_e_demand
    for row in eachrow(results_df)
]

results_df.total_lol = [
    row.loss_of_load_e_demand + row.loss_of_load_h2_demand
    for row in eachrow(results_df)
]


@info "Plotting number of steps with lol"
plot_values_stocmethod_method(results_df, case_studies_df, "num_loss_of_load_e_demand"; savepath=joinpath(plot_path, "num_loss_of_load_e_demand"))
plot_values_stocmethod_method(results_df, case_studies_df, "num_loss_of_load_h2_demand"; savepath=joinpath(plot_path, "num_loss_of_load_h2_demand"))
plot_values_stocmethod_method(results_df, case_studies_df, "total_steps_loss_of_load"; savepath=joinpath(plot_path, "total_steps_lol"))
plot_values_stocmethod_method(results_df, case_studies_df, "total_lol"; savepath=joinpath(plot_path, "total_lol"))
