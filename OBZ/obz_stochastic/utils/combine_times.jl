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
results_onlyclustering_df = CSV.read(joinpath(output_path, "results_onlyclustering.csv"), DataFrame)

keys = [:base_name,:rp]


results_df.row_id = repeat([1], nrow(results_df))  # initialize

for g in groupby(results_df, keys)
    g.row_id = 1:nrow(g)
end

results_onlyclustering_df.row_id = repeat([1], nrow(results_onlyclustering_df))

for g in groupby(results_onlyclustering_df, keys)
    g.row_id = 1:nrow(g)
end



results_combined = leftjoin(
    results_df[:, Not(:time_to_cluster)], 
    results_onlyclustering_df[:, vcat(keys, :row_id, :time_to_cluster)],
    on = vcat(keys, :row_id)
)
select!(results_combined, Not(:row_id))

results_combined |> CSV.write(joinpath(output_path, "results_combined.csv"); writeheader=true)

