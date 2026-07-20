# Run this only after results given from main: this is for algorithms that are not deterministic such as k-medoids and k-means

using DataFrames
using CSV
using Plots
using Statistics
using Measures

# helper functions
@info "Including constants and helper functions"
include("constants.jl")
include("functions.jl")

script_dir = @__DIR__
output_path = joinpath(script_dir, "..", "outputs")

stats_dist = CSV.read(joinpath(output_path, "stats_dist.csv"), DataFrame)
stats_cl = CSV.read(joinpath(output_path, "stats_cl.csv"), DataFrame)
stats_mix = CSV.read(joinpath(output_path, "stats_mix.csv"), DataFrame)
stats_hmix = CSV.read(joinpath(output_path, "stats_hmix.csv"), DataFrame)


plot_path = joinpath(output_path, "plots")

case_studies_path = joinpath(script_dir, "..", "case-studies-info.csv")
case_studies_df = CSV.read(case_studies_path, DataFrame)

mkpath(plot_path)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "rel_regret";
    savepath=joinpath(plot_path, "relative_regret.png"),
    size=(1600, 1000),
)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "num_loss_of_load_e_demand";
    savepath=joinpath(plot_path, "num_loss_of_load_e_demand.png"),
    size=(1600, 1000),
)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "time_to_cluster";
    savepath=joinpath(plot_path, "time_to_cluster.png"),
    size=(1600, 1000),
)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "time_to_create";
    savepath=joinpath(plot_path, "time_to_create.png"),
    size=(1600, 1000),
)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "time_to_solve";
    savepath=joinpath(plot_path, "time_to_solve.png"),
    size=(1600, 1000),
)
plot_values_quantiles_grid(
    (DISTANT=stats_dist, CLOSE=stats_cl, HALFMIXED=stats_hmix, MIXED=stats_mix),
    case_studies_df,
    "total_time";
    savepath=joinpath(plot_path, "total_time.png"),
    size=(1600, 1000),
)