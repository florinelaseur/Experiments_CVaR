# Run this only after results given from main: this is for algorithms that are not deterministic such as k-medoids and k-means
using DataFrames
using CSV
using Plots
using Statistics

# helper functions
@info "Including constants and helper functions"
include("constants.jl")
include("functions.jl")

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

# compute total time 
results_df.total_time = [
    row.time_to_cluster + row.time_to_read + row.time_to_create + row.time_to_solve + row.time_to_save
    for row in eachrow(results_df)
]

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

results_df.total_steps_loss_of_load = [
    row.num_loss_of_load_e_demand + row.num_loss_of_load_h2_demand
    for row in eachrow(results_df)
]

lol_e = only(hourly_row.loss_of_load_e_demand)
results_df.loss_of_load_e_demand = [
    row.loss_of_load_e_demand - lol_e
    for row in eachrow(results_df)
]

lol_h2 = only(hourly_row.loss_of_load_h2_demand)
results_df.loss_of_load_h2_demand = [
    row.loss_of_load_h2_demand - lol_h2
    for row in eachrow(results_df)
]

results_df.total_lol = [
    row.loss_of_load_e_demand + row.loss_of_load_h2_demand
    for row in eachrow(results_df)
]



# compute statistics we need
results_df = combine(
    groupby(results_df, [:base_name, :rp]), :time_to_cluster => mean => :time_to_cluster_mean,
    :time_to_cluster => (x -> quantile(x, 0.25)) => :time_to_cluster_q25,
    :time_to_cluster => (x -> quantile(x, 0.75)) => :time_to_cluster_q75, :time_to_read => mean => :time_to_read_mean,
    :time_to_read => (x -> quantile(x, 0.25)) => :time_to_read_q25,
    :time_to_read => (x -> quantile(x, 0.75)) => :time_to_read_q75, :time_to_create => mean => :time_to_create_mean,
    :time_to_create => (x -> quantile(x, 0.25)) => :time_to_create_q25,
    :time_to_create => (x -> quantile(x, 0.75)) => :time_to_create_q75, :time_to_solve => mean => :time_to_solve_mean,
    :time_to_solve => (x -> quantile(x, 0.25)) => :time_to_solve_q25,
    :time_to_solve => (x -> quantile(x, 0.75)) => :time_to_solve_q75, :time_to_save => mean => :time_to_save_mean,
    :time_to_save => (x -> quantile(x, 0.25)) => :time_to_save_q25,
    :time_to_save => (x -> quantile(x, 0.75)) => :time_to_save_q75, :total_time => mean => :total_time_mean,
    :total_time => (x -> quantile(x, 0.25)) => :total_time_q25,
    :total_time => (x -> quantile(x, 0.75)) => :total_time_q75, :rel_regret => mean => :rel_regret_mean,
    :rel_regret => (x -> quantile(x, 0.25)) => :rel_regret_q25,
    :rel_regret => (x -> quantile(x, 0.75)) => :rel_regret_q75, 
    :num_loss_of_load_e_demand => mean => :num_loss_of_load_e_demand_mean,
    :num_loss_of_load_e_demand => (x -> quantile(x, 0.25)) => :num_loss_of_load_e_demand_q25,
    :num_loss_of_load_e_demand => (x -> quantile(x, 0.75)) => :num_loss_of_load_e_demand_q75,
    :num_loss_of_load_h2_demand => mean => :num_loss_of_load_h2_demand_mean,
    :num_loss_of_load_h2_demand => (x -> quantile(x, 0.25)) => :num_loss_of_load_h2_demand_q25,
    :num_loss_of_load_h2_demand => (x -> quantile(x, 0.75)) => :num_loss_of_load_h2_demand_q75,
    :total_steps_loss_of_load => mean => :total_steps_loss_of_load_mean,
    :total_steps_loss_of_load => (x -> quantile(x, 0.25)) => :total_steps_loss_of_load_q25,
    :total_steps_loss_of_load => (x -> quantile(x, 0.75)) => :total_steps_loss_of_load_q75,
    :loss_of_load_e_demand => mean => :loss_of_load_e_demand_mean,
    :loss_of_load_e_demand => (x -> quantile(x, 0.25)) => :loss_of_load_e_demand_q25,
    :loss_of_load_e_demand => (x -> quantile(x, 0.75)) => :loss_of_load_e_demand_q75,
    :loss_of_load_h2_demand => mean => :loss_of_load_h2_demand_mean,
    :loss_of_load_h2_demand => (x -> quantile(x, 0.25)) => :loss_of_load_h2_demand_q25,
    :loss_of_load_h2_demand => (x -> quantile(x, 0.75)) => :loss_of_load_h2_demand_q75,
    :total_lol => mean => :total_lol_mean,
    :total_lol => (x -> quantile(x, 0.25)) => :total_lol_q25,
    :total_lol => (x -> quantile(x, 0.75)) => :total_lol_q75,
)
results_df |> CSV.write(joinpath(output_path, "stats.csv"); writeheader=true)



plot_path = joinpath(output_path, "plots")

case_studies_path = joinpath(script_dir, "..", "case-studies-info.csv")
case_studies_df = CSV.read(case_studies_path, DataFrame)

mkpath(plot_path)
@info "Plotting relative regret"
plot_values_quantiles(results_df, case_studies_df, "rel_regret"; savepath=joinpath(plot_path, "relative_regret.png"))

@info "Plotting time to cluster"
plot_values_quantiles(results_df, case_studies_df, "time_to_cluster"; savepath=joinpath(plot_path, "time_to_cluster.png"))

@info "Plotting time to solve"
plot_values_quantiles(results_df, case_studies_df, "time_to_solve"; savepath=joinpath(plot_path, "time_to_solve.png"))

@info "Plotting time to create"
plot_values_quantiles(results_df, case_studies_df, "time_to_create"; savepath=joinpath(plot_path, "time_to_create.png"))

@info "Plotting total time"
plot_values_quantiles(results_df, case_studies_df, "total_time"; savepath=joinpath(plot_path, "total_time.png"))

@info "Plotting number of steps with lol e_demand"
plot_values_quantiles(results_df, case_studies_df, "num_loss_of_load_e_demand"; savepath=joinpath(plot_path, "num_loss_of_load_e_demand"))

@info "Plotting number of steps with lol h2_demand"
plot_values_quantiles(results_df, case_studies_df, "num_loss_of_load_h2_demand"; savepath=joinpath(plot_path, "num_loss_of_load_h2_demand"))

@info "Plotting number of steps with lol "
plot_values_quantiles(results_df, case_studies_df, "total_steps_loss_of_load"; savepath=joinpath(plot_path, "total_steps_loss_of_load"))

@info "Plotting lol e_demand"
plot_values_quantiles(results_df, case_studies_df, "loss_of_load_e_demand"; savepath=joinpath(plot_path, "loss_of_load_e_demand"))

@info "Plotting lol h2_demand"
plot_values_quantiles(results_df, case_studies_df, "loss_of_load_h2_demand"; savepath=joinpath(plot_path, "loss_of_load_h2_demand"))

@info "Plotting lol "
plot_values_quantiles(results_df, case_studies_df, "total_lol"; savepath=joinpath(plot_path, "total_lol"))
