# Pairwise scenario dominance from the Phase-B cost matrix.
#
# Standalone — depends only on DataFrames (+ CSV for the writer, resolved from
# the includer's scope in utils.jl). No JuMP/TEM so unit tests can load it alone.

using DataFrames: DataFrame, nrow, propertynames

_compare_cost(x::Real) = isnan(x) ? Inf : Float64(x)

function _parse_scenario_column_names(cols)
    ids = Int[]
    for col in cols
        s = string(col)
        m = match(r"^scenario_(\d+)$", s)
        m === nothing && continue
        push!(ids, parse(Int, m.captures[1]))
    end
    sort(ids)
end

"""
    _extract_cost_matrix(cost::DataFrame) -> (Matrix{Float64}, Vector{Int})

Extract numeric cost block and scenario ids from a `cost_matrix`-style DataFrame
(columns `scenario_<id>`, ignoring `sample_id` / `sequence`).
"""
function _extract_cost_matrix(cost::DataFrame)
    scenario_ids = _parse_scenario_column_names(propertynames(cost))
    isempty(scenario_ids) &&
        error("No scenario_<id> columns found in cost DataFrame")
    C = Matrix{Float64}(undef, nrow(cost), length(scenario_ids))
    for (j, sid) in enumerate(scenario_ids)
        col = Symbol("scenario_$sid")
        hasproperty(cost, col) || error("Missing column $col in cost DataFrame")
        C[:, j] = Float64.(cost[!, col])
    end
    return C, scenario_ids
end

function _column_dominates(col_i::AbstractVector, col_j::AbstractVector)
    ci = _compare_cost.(col_i)
    cj = _compare_cost.(col_j)
    return all(ci .>= cj) && any(ci .> cj)
end

"""
    dominating_scenarios(cost; scenarios=nothing, num_scenarios=nothing, num_samples=nothing)

Pairwise scenario dominance on a cost matrix (rows = samples, columns = scenarios).

Scenario **A** dominates **B** iff for every sample `k` the compared cost satisfies
`cost(A,k) >= cost(B,k)`, with strict `>` for at least one sample. `NaN` is treated
as `+Inf` (worst outcome); two `NaN`s at the same sample compare equal.

Returns `(scenarios=..., dominates=..., pairs=..., undominated=...)` where
`dominates[i,j]` is true iff scenario `scenarios[i]` dominates `scenarios[j]`.
"""
function dominating_scenarios(
    cost::DataFrame;
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _extract_cost_matrix(cost)
    if scenarios !== nothing
        collect(scenarios) == scenario_ids ||
            error("scenarios kwarg $(collect(scenarios)) does not match columns $scenario_ids")
    end
    if num_scenarios !== nothing && length(scenario_ids) != num_scenarios
        error("num_scenarios=$num_scenarios but got $(length(scenario_ids)) scenario columns")
    end
    if num_samples !== nothing && size(C, 1) != num_samples
        error("num_samples=$num_samples but cost has $(size(C, 1)) rows")
    end
    return _dominating_scenarios_matrix(C, scenario_ids)
end

function dominating_scenarios(
    cost::AbstractMatrix{<:Real};
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C = Matrix{Float64}(cost)
    n_samples, n_scenarios = size(C)
    scenario_ids = if scenarios === nothing
        collect(1:n_scenarios)
    else
        sid = collect(Int.(scenarios))
        length(sid) == n_scenarios ||
            error("scenarios length $(length(sid)) != matrix columns $n_scenarios")
        sid
    end
    if num_scenarios !== nothing && n_scenarios != num_scenarios
        error("num_scenarios=$num_scenarios but matrix has $n_scenarios columns")
    end
    if num_samples !== nothing && n_samples != num_samples
        error("num_samples=$num_samples but matrix has $n_samples rows")
    end
    return _dominating_scenarios_matrix(C, scenario_ids)
end

function _dominated_scenario_set(scenarios, dominates::AbstractMatrix{Bool})
    n = length(scenarios)
    size(dominates) == (n, n) ||
        error("dominates size $(size(dominates)) != ($n, $n) for $(length(scenarios)) scenarios")
    return Set(
        Int(scenarios[j]) for j in 1:n if any(dominates[i, j] for i in 1:n if i != j)
    )
end

function _undominated_from_dominated_set(all_scenarios, dominated::Set{Int})
    return Int[s for s in all_scenarios if Int(s) ∉ dominated]
end

"""
    undominated_scenarios(pairs::DataFrame, all_scenarios)

Return scenario ids from `all_scenarios` that never appear in the `dominated`
column of `pairs` (the `scenario_dominance.csv` format).

`all_scenarios` must list every scenario under consideration (typically from
`cost_matrix` column names or `dominating_scenarios(...).scenarios`).
"""
function undominated_scenarios(
    pairs::DataFrame,
    all_scenarios::AbstractVector{<:Integer},
)
    hasproperty(pairs, :dominator) ||
        error("pairs DataFrame must have a :dominator column")
    hasproperty(pairs, :dominated) ||
        error("pairs DataFrame must have a :dominated column")
    dominated = isempty(pairs) ? Set{Int}() : Set(Int.(pairs.dominated))
    return sort(_undominated_from_dominated_set(all_scenarios, dominated))
end

"""
    undominated_scenarios(scenarios, dominates)

Return scenario ids from `scenarios` that are not dominated by any other scenario
according to the N×N `dominates` matrix (`dominates[i,j]` = scenario i dominates j).
"""
function undominated_scenarios(
    scenarios::AbstractVector{<:Integer},
    dominates::AbstractMatrix{Bool},
)
    dominated = _dominated_scenario_set(scenarios, dominates)
    return _undominated_from_dominated_set(scenarios, dominated)
end

function _dominating_scenarios_matrix(C::Matrix{Float64}, scenario_ids::Vector{Int})
    n = length(scenario_ids)
    dominates = falses(n, n)
    pairs = Tuple{Int, Int}[]
    for i in 1:n
        for j in 1:n
            i == j && continue
            if _column_dominates(view(C, :, i), view(C, :, j))
                dominates[i, j] = true
                push!(pairs, (scenario_ids[i], scenario_ids[j]))
            end
        end
    end
    undominated = undominated_scenarios(scenario_ids, dominates)
    return (
        scenarios=scenario_ids,
        dominates=dominates,
        pairs=pairs,
        undominated=undominated,
    )
end

"""Write dominating pairs to CSV (`dominator`, `dominated`). References `CSV` from includer scope."""
function save_scenario_dominance_csv(path::String, scenarios, dominates)
    df = DataFrame(dominator=Int[], dominated=Int[])
    n = length(scenarios)
    for i in 1:n
        for j in 1:n
            dominates[i, j] || continue
            push!(df, (scenarios[i], scenarios[j]))
        end
    end
    CSV.write(path, df)
    return df
end
