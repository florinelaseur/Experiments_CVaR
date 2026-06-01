# Read Phase-B cost_matrix.csv and compute pairwise scenario dominance.
#
# Scenario A dominates B iff cost(A,k) >= cost(B,k) for every sample k, with strict
# > for at least one k. NaN is treated as +Inf (worst outcome).
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/analyze_scenario_dominance.jl
# Optional custom paths:
#   julia --project=. ScenarioReduction/old_scripts/analyze_scenario_dominance.jl path/to/cost_matrix.csv path/to/out.csv

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using CSV: CSV
using DataFrames: DataFrame

include(joinpath(@__DIR__, "..", "src", "scenario_dominance.jl"))

const DEFAULT_COST_MATRIX = joinpath(@__DIR__, "..", "outputs", "cost_matrix.csv")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "..", "outputs", "scenario_dominance.csv")

function main(;
    cost_matrix_path::String=DEFAULT_COST_MATRIX,
    output_path::String=DEFAULT_OUTPUT,
)
    isfile(cost_matrix_path) || error("Cost matrix not found: $cost_matrix_path")
    cost_df = CSV.read(cost_matrix_path, DataFrame)
    nrow(cost_df) > 0 || error("Cost matrix is empty: $cost_matrix_path")

    result = dominating_scenarios(cost_df)
    n_scenarios = length(result.scenarios)
    n_samples = nrow(cost_df)

    save_scenario_dominance_csv(output_path, result.scenarios, result.dominates)

    println("Cost matrix: $cost_matrix_path")
    println("  samples=$n_samples  scenarios=$n_scenarios  ids=$(result.scenarios)")
    println("Dominance pairs (dominator → dominated): $(length(result.pairs))")
    for (a, b) in result.pairs
        println("  $a → $b")
    end
    println("Undominated scenarios ($(length(result.undominated))): $(result.undominated)")
    println("Wrote $(output_path)")

    return result
end

cost_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_COST_MATRIX
out_path = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
main(; cost_matrix_path=cost_path, output_path=out_path)
