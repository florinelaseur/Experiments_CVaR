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

# compute statistics we need
results_df = combine(
    groupby(results_df, [:base_name, :rp]), :time_to_cluster => mean => :time_to_cluster_mean,
    :time_to_cluster => (x -> quantile(x, 0.25)) => :time_to_cluster_q25,
    :time_to_cluster => (x -> quantile(x, 0.75)) => :time_to_cluster_q75
)
results_df |> CSV.write(joinpath(output_path, "stats.csv"); writeheader=true)



