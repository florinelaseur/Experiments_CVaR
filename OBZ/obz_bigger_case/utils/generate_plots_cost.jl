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
plot_path = joinpath(output_path, "plots_cost")
mkpath(plot_path)

results_df = CSV.read(joinpath(output_path, "results.csv"), DataFrame)

plot_cost_columns(results_df, joinpath(plot_path, "cost_columns.png"))

plot_cost_columns_precise(results_df, joinpath(plot_path, "cost_columns_precise.png"))