# Distributional (FSD / SSD) scenario dominance from a Phase-B cost_matrix.csv.
#
# Each scenario column of the cost matrix is treated as an empirical cost
# distribution over the sampled investments. Dominance uses the cost-maximization
# convention (aligned with pointwise dominating_scenarios): D[i, j] = 1 means
# row scenario i stochastically dominates column scenario j (i has more mass on
# high costs; j is the cheaper scenario).
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/analyze_distributional_dominance.jl path/to/folder fsd
#   julia --project=. ScenarioReduction/old_scripts/analyze_distributional_dominance.jl path/to/folder ssd
#
# Reads `cost_matrix.csv` from the folder, computes the chosen method, and writes
# `fsd_dominance.csv` / `ssd_dominance.csv` into the same folder. If the output
# already exists it is loaded, summarized, and returned without recomputation.
#
# This script is the only place with file I/O and early-termination behavior; the
# core functions in src/scenario_dominance.jl remain pure.

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using CSV: CSV
using DataFrames: DataFrame, nrow, propertynames

include(joinpath(@__DIR__, "..", "src", "scenario_dominance.jl"))

const METHODS = Dict(
    "fsd" => (fn=fsd_dominating_scenarios, file="fsd_dominance.csv", name="FSD"),
    "ssd" => (fn=ssd_dominating_scenarios, file="ssd_dominance.csv", name="SSD"),
)

"""Build a matrix-style dominance DataFrame. Column `scenario` holds the row
(dominator) scenario id; `has_nan` flags scenarios that contained a NaN cost and
were excluded from ordering; each `scenario_<id>` column holds 0/1 where 1 means
the row scenario dominates that column scenario."""
function dominance_matrix_dataframe(scenarios, dominates, nan_scenarios)
    n = length(scenarios)
    nan_set = Set(Int.(nan_scenarios))
    df = DataFrame(;
        scenario=collect(Int.(scenarios)),
        has_nan=Int[Int(scenarios[i]) in nan_set ? 1 : 0 for i in 1:n],
    )
    for j in 1:n
        df[!, Symbol("scenario_$(scenarios[j])")] = Int[dominates[i, j] ? 1 : 0 for i in 1:n]
    end
    return df
end

"""Inverse of `dominance_matrix_dataframe`: parse a saved matrix CSV back into
`(scenarios, dominates, nan_scenarios)`."""
function parse_dominance_matrix(df::DataFrame)
    hasproperty(df, :scenario) ||
        error("Dominance matrix CSV must have a `scenario` column")
    scenario_ids = Int[]
    for col in propertynames(df)
        m = match(r"^scenario_(\d+)$", string(col))
        m === nothing && continue
        push!(scenario_ids, parse(Int, m.captures[1]))
    end
    sort!(scenario_ids)
    row_ids = Int.(df.scenario)
    row_ids == scenario_ids ||
        error("Row scenario ids $row_ids do not match column scenario ids $scenario_ids")
    n = length(scenario_ids)
    dominates = falses(n, n)
    for (j, sid) in enumerate(scenario_ids)
        col = df[!, Symbol("scenario_$sid")]
        for i in 1:n
            dominates[i, j] = col[i] != 0
        end
    end
    nan_scenarios = if hasproperty(df, :has_nan)
        sort(Int[row_ids[i] for i in 1:n if df.has_nan[i] != 0])
    else
        Int[]
    end
    return scenario_ids, dominates, nan_scenarios
end

function print_summary(method_name, scenarios, dominates, n_samples, nan_scenarios)
    n_scen = length(scenarios)
    pairs = [(scenarios[i], scenarios[j]) for i in 1:n_scen for j in 1:n_scen
             if i != j && dominates[i, j]]
    undominated = undominated_scenarios(collect(Int.(scenarios)), dominates)
    println("Method:               $method_name")
    println("Number of scenarios:  $n_scen")
    println("Number of samples:    $n_samples")
    println("Dominance pairs:      $(length(pairs))")
    for (a, b) in pairs
        println("  $a dominates $b")
    end
    println("Undominated scenarios ($(length(undominated))): $undominated")
    println("NaN-tagged scenarios (skipped, $(length(nan_scenarios))): $(sort(collect(Int.(nan_scenarios))))")
    return undominated
end

function main(folder::String, method::String)
    key = lowercase(method)
    if !haskey(METHODS, key)
        valid = join(sort(collect(keys(METHODS))), ", ")
        error("Unknown method '$method'. Choose one of: $valid")
    end
    spec = METHODS[key]

    isdir(folder) || error("Folder not found: $folder")
    cost_matrix_path = joinpath(folder, "cost_matrix.csv")
    output_path = joinpath(folder, spec.file)

    n_samples = isfile(cost_matrix_path) ? nrow(CSV.read(cost_matrix_path, DataFrame)) : "unknown"

    if isfile(output_path)
        println("Output already exists, loading without recomputing: $output_path")
        df = CSV.read(output_path, DataFrame)
        scenarios, dominates, nan_scenarios = parse_dominance_matrix(df)
        print_summary(spec.name, scenarios, dominates, n_samples, nan_scenarios)
        return (scenarios=scenarios, dominates=dominates, nan_scenarios=nan_scenarios)
    end

    isfile(cost_matrix_path) || error("Cost matrix not found: $cost_matrix_path")
    cost_df = CSV.read(cost_matrix_path, DataFrame)
    nrow(cost_df) > 0 || error("Cost matrix is empty: $cost_matrix_path")

    result = spec.fn(cost_df)
    out_df = dominance_matrix_dataframe(result.scenarios, result.dominates, result.nan_scenarios)
    CSV.write(output_path, out_df)

    println("Cost matrix: $cost_matrix_path")
    print_summary(spec.name, result.scenarios, result.dominates, nrow(cost_df), result.nan_scenarios)
    println("Wrote $output_path")
    return result
end

if length(ARGS) < 2
    error("Usage: julia --project=. analyze_distributional_dominance.jl path/to/folder fsd|ssd")
end
main(ARGS[1], ARGS[2])
