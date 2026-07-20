
using DataFrames
using CSV
using Plots
using Statistics

# Import your constants and helper functions
include("constants.jl")
include("functions.jl")

script_dir = @__DIR__
output_path = joinpath(script_dir, "..", "outputs")
case_studies_path = joinpath(script_dir, "..", "case-studies-info.csv")
case_studies_df = CSV.read(case_studies_path, DataFrame)

stats_files = [
    joinpath(output_path, "stats1.csv"),
    joinpath(output_path, "stats2.csv"),
    joinpath(output_path, "stats3.csv")
]

titles = ["BLENDED WEIGHTS", "DIRAC WEIGHTS", "BLENDED WEIGHTS WITH β=0.2"]


metric = "rel_regret"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
mkpath(dirname(savepath))
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)

metric = "time_to_cluster"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)

metric = "time_to_solve"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)

metric = "time_to_create"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)

metric = "total_time"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)

metric = "num_loss_of_load_e_demand"
savepath = joinpath(output_path, "plots", "panel_$(metric).png")
plot_together(stats_files, titles, metric, case_studies_df; savepath=savepath)