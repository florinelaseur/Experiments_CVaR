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

# =====================================================================
# Distributional (FSD / SSD) scenario dominance
# =====================================================================
#
# CONVENTION (cost-maximization). Aligned with the pointwise
# `dominating_scenarios` helper above. Each scenario column of the cost matrix
# is treated as an empirical distribution over the K sampled investments. For
# the FSD/SSD methods below:
#
#     dominates[i,j] == true  means  scenarios[i] stochastically dominates
#                                    scenarios[j].
#
# i.e. scenario i has more probability mass on high costs than j; scenario j is
# the cheaper / less risk-relevant one. Equivalently, i's empirical CDF is <= j's
# at every shared threshold, with strict < somewhere (lower CDF = more mass above
# each cost level).
#
# NaN handling: any scenario whose column contains a NaN (infeasible /
# unevaluated sample) is TAGGED and excluded from ordering entirely — it never
# dominates and is never dominated, is reported in `nan_scenarios`, and a
# warning is emitted. (Inf is still merely excluded from the finite domain and
# behaves as a worst-case +Inf.)

"""
Validate cost-matrix dims against optional `scenarios` / `num_scenarios` /
`num_samples` kwargs and return `(C::Matrix{Float64}, scenario_ids::Vector{Int})`.
Shared by the FSD/SSD entry points; mirrors `dominating_scenarios` validation.
"""
function _normalize_cost_input(
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
    return C, scenario_ids
end

function _normalize_cost_input(
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
    return C, scenario_ids
end

"""
    _finite_cost_domain(C) -> Vector{Float64}

Shared, sorted, deduplicated set of every finite cost value in the whole matrix.
Evaluating all CDFs at exactly these points means no interpolation is needed —
every CDF step aligns with a domain point.
"""
function _finite_cost_domain(C::AbstractMatrix{Float64})
    return sort(unique(filter(isfinite, vec(C))))
end

"""
    _tag_nan_scenarios(C, scenario_ids) -> Vector{Bool}

Per-scenario mask: `true` where a scenario's cost column contains any `NaN`.
A `NaN` marks an infeasible / unevaluated sample, so the scenario's empirical
distribution is incomplete and it is excluded from FSD/SSD ordering entirely
(it can neither dominate nor be dominated). Emits an `@warn` listing the tagged
scenarios. Only `NaN` triggers tagging; `Inf` keeps its existing treatment
(filtered from the finite domain, i.e. behaves as a worst-case +Inf).
"""
function _tag_nan_scenarios(C::AbstractMatrix{Float64}, scenario_ids::Vector{Int})
    mask = Bool[any(isnan, view(C, :, s)) for s in 1:size(C, 2)]
    if any(mask)
        tagged = Int[scenario_ids[s] for s in eachindex(scenario_ids) if mask[s]]
        @warn "NaN cost values encountered; scenarios excluded from FSD/SSD ordering" nan_scenarios = tagged
    end
    return mask
end

"""
    _empirical_cdf_matrix(C, domain) -> M×S Matrix{Float64}

`F[m, s] = count(C[k, s] <= domain[m]) / K`, the empirical CDF of scenario `s`
evaluated at every domain point. Each scenario's column is sorted once and the
count at each threshold is found with `searchsortedlast` (O(M + K) via the
sorted scan). Non-finite costs sort to the end and are never counted at or below
a finite threshold, so they correctly behave as +Inf.
"""
function _empirical_cdf_matrix(C::AbstractMatrix{Float64}, domain::Vector{Float64})
    K, S = size(C)
    M = length(domain)
    F = Matrix{Float64}(undef, M, S)
    invK = K > 0 ? 1.0 / K : 0.0
    for s in 1:S
        sorted = sort(C[:, s])
        for m in 1:M
            F[m, s] = searchsortedlast(sorted, domain[m]) * invK
        end
    end
    return F
end

"""
    _integrated_cdf_matrix(F, domain) -> M×S Matrix{Float64}

Running left-Riemann integral of each scenario's step CDF:

    G[1, s]  = 0
    G[m, s]  = G[m-1, s] + F[m-1, s] * (domain[m] - domain[m-1])

A non-decreasing, piecewise-linear function. O(M) per scenario.
"""
function _integrated_cdf_matrix(F::AbstractMatrix{Float64}, domain::Vector{Float64})
    M, S = size(F)
    G = zeros(Float64, M, S)
    for s in 1:S
        acc = 0.0
        for m in 2:M
            acc += F[m - 1, s] * (domain[m] - domain[m - 1])
            G[m, s] = acc
        end
    end
    return G
end

"""
    _curve_dominance(curves, scenario_ids, nan_mask) -> NamedTuple

All-pairs dominance from an M×S matrix of comparison curves (CDFs for FSD,
integrated CDFs for SSD) under **cost maximization**. Scenario `i` dominates `j`
iff `curves[m,i] <= curves[m,j]` for every `m`, with strict `<` at least once.
Short-circuits on the first violation.

Scenarios flagged in `nan_mask` are excluded from all comparisons: a tagged
scenario never dominates and is never dominated (every pair touching it is
skipped), so it always lands in `undominated` and is reported in
`nan_scenarios`.

Returns `(scenarios, dominates, pairs, undominated, nan_scenarios)`.
"""
function _curve_dominance(
    curves::AbstractMatrix{Float64},
    scenario_ids::Vector{Int},
    nan_mask::AbstractVector{Bool},
)
    M, S = size(curves)
    dominates = falses(S, S)
    pairs = Tuple{Int, Int}[]
    for i in 1:S
        nan_mask[i] && continue            # NaN scenario never dominates
        for j in 1:S
            (i == j || nan_mask[j]) && continue  # never dominated -> skip pair
            le_all = true
            strict = false
            for m in 1:M
                a = curves[m, i]
                b = curves[m, j]
                if a > b
                    le_all = false
                    break
                elseif a < b
                    strict = true
                end
            end
            if le_all && strict
                dominates[i, j] = true
                push!(pairs, (scenario_ids[i], scenario_ids[j]))
            end
        end
    end
    undominated = undominated_scenarios(scenario_ids, dominates)
    nan_scenarios = sort(Int[scenario_ids[k] for k in 1:S if nan_mask[k]])
    return (
        scenarios=scenario_ids,
        dominates=dominates,
        pairs=pairs,
        undominated=undominated,
        nan_scenarios=nan_scenarios,
    )
end

"""
    fsd_dominating_scenarios(cost; scenarios=nothing, num_scenarios=nothing, num_samples=nothing)

First-order stochastic dominance (FSD) over scenarios, treating each scenario
column of `cost` (rows = investment samples, columns = scenarios) as an
empirical cost distribution.

Scenario `i` **FSD-dominates** `j` iff its empirical CDF is `<=` that of `j` at
every shared domain point, with strict `<` somewhere. Lower CDF everywhere
means more probability mass on high costs — scenario `i` is more expensive /
risk-relevant and `j` is the cheaper scenario.

`cost` may be a `cost_matrix`-style `DataFrame` (`scenario_<id>` columns,
ignoring `sample_id` / `sequence`) or a numeric matrix. Pure: no file I/O.

Any scenario whose column contains a `NaN` (infeasible / unevaluated sample) is
tagged and excluded from ordering entirely: it never dominates and is never
dominated. Such scenarios are reported in `nan_scenarios` and (being undominated
by construction) also appear in `undominated`; a warning is emitted. This
supersedes the older "NaN as +Inf" convention, which now applies only to the
legacy pointwise `dominating_scenarios`.

Returns `(scenarios, dominates, pairs, undominated, nan_scenarios)` where
`dominates[i,j]` means `scenarios[i]` dominates `scenarios[j]`.
"""
function fsd_dominating_scenarios(
    cost::DataFrame;
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    nan_mask = _tag_nan_scenarios(C, scenario_ids)
    domain = _finite_cost_domain(C)
    F = _empirical_cdf_matrix(C, domain)
    return _curve_dominance(F, scenario_ids, nan_mask)
end

function fsd_dominating_scenarios(
    cost::AbstractMatrix{<:Real};
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    nan_mask = _tag_nan_scenarios(C, scenario_ids)
    domain = _finite_cost_domain(C)
    F = _empirical_cdf_matrix(C, domain)
    return _curve_dominance(F, scenario_ids, nan_mask)
end

"""
    ssd_dominating_scenarios(cost; scenarios=nothing, num_scenarios=nothing, num_samples=nothing)

Second-order stochastic dominance (SSD) over scenarios. SSD relaxes FSD by
allowing the CDFs to cross, as long as the accumulated CDF area (integral) of
the dominating scenario is never overcome.

Scenario `i` **SSD-dominates** `j` iff the integrated CDF of `i` is `<=` that of
`j` at every shared domain point, with strict `<` somewhere. Less accumulated
area on the cheap (left) side means more probability mass at high costs; `i` has
the heavier (more expensive) tail. Under cost maximization, SSD is the
second-order analogue of preferring higher-cost distributions.

Every FSD pair is also an SSD pair (FSD ⊂ SSD). Same arguments / return shape as
`fsd_dominating_scenarios`, including the `nan_scenarios` field: scenarios whose
column contains a `NaN` are tagged and excluded from ordering. Pure: no file I/O.
"""
function ssd_dominating_scenarios(
    cost::DataFrame;
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    nan_mask = _tag_nan_scenarios(C, scenario_ids)
    domain = _finite_cost_domain(C)
    F = _empirical_cdf_matrix(C, domain)
    G = _integrated_cdf_matrix(F, domain)
    return _curve_dominance(G, scenario_ids, nan_mask)
end

function ssd_dominating_scenarios(
    cost::AbstractMatrix{<:Real};
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    nan_mask = _tag_nan_scenarios(C, scenario_ids)
    domain = _finite_cost_domain(C)
    F = _empirical_cdf_matrix(C, domain)
    G = _integrated_cdf_matrix(F, domain)
    return _curve_dominance(G, scenario_ids, nan_mask)
end

"""
    dominator_scenarios(result)
    dominator_scenarios(pairs::DataFrame)

Return the sorted unique scenario ids that dominate at least one other scenario
(the "most expensive" set). Accepts either a `dominating_scenarios(...)` result
NamedTuple (uses `.pairs`) or a `scenario_dominance.csv`-style DataFrame with a
`:dominator` column. Returns an empty `Vector{Int}` when there are no dominance
pairs.
"""
function dominator_scenarios(pairs::DataFrame)
    hasproperty(pairs, :dominator) ||
        error("pairs DataFrame must have a :dominator column")
    isempty(pairs) && return Int[]
    return sort(unique(Int.(pairs.dominator)))
end

function dominator_scenarios(result)
    hasproperty(result, :pairs) ||
        error("dominator_scenarios expects a result with a `pairs` field or a DataFrame")
    isempty(result.pairs) && return Int[]
    return sort(unique(Int[Int(p[1]) for p in result.pairs]))
end

function _dominance_fn(method::Symbol)
    method == :pointwise && return dominating_scenarios
    method == :fsd && return fsd_dominating_scenarios
    method == :ssd && return ssd_dominating_scenarios
    error("unknown dominance method :$method (use :pointwise, :fsd, or :ssd)")
end

"""
    dominance_analysis(cost; method=:pointwise, scenarios=nothing, num_scenarios=nothing, num_samples=nothing)

Master dispatch over the dominance relations: runs `dominating_scenarios`
(`:pointwise`), `fsd_dominating_scenarios` (`:fsd`), or `ssd_dominating_scenarios`
(`:ssd`) on `cost` (a `cost_matrix.csv`-style DataFrame or a samples × scenarios
matrix) and normalizes the result so every method returns the same shape:

`(scenarios, dominates, pairs, undominated, nan_scenarios, method)`

`nan_scenarios` lists scenarios excluded from the FSD/SSD ordering because they
contain NaN costs; it is always empty for `:pointwise`, which treats NaN as `+Inf`
instead of excluding. Used by the screening Phase C (`dominance_screening`) and
available to experiment drivers alongside `pick_n_scenarios`.
"""
function dominance_analysis(
    cost;
    method::Symbol=:pointwise,
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    fn = _dominance_fn(method)
    res = fn(cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples)
    nan_scenarios = hasproperty(res, :nan_scenarios) ? res.nan_scenarios : Int[]
    return (
        scenarios=res.scenarios,
        dominates=res.dominates,
        pairs=res.pairs,
        undominated=res.undominated,
        nan_scenarios=nan_scenarios,
        method=method,
    )
end

function _subset_cost_matrix(C::Matrix{Float64}, all_ids::Vector{Int}, active::Vector{Int})
    idx = Int[findfirst(==(id), all_ids) for id in active]
    any(isnothing, idx) &&
        error("active scenario ids $active are not a subset of $all_ids")
    return C[:, idx]
end

function _subset_cost_dataframe(cost::DataFrame, keep_ids::Vector{Int})
    sorted = sort(collect(Int.(keep_ids)))
    meta = Symbol[]
    for col in propertynames(cost)
        s = string(col)
        match(r"^scenario_(\d+)$", s) === nothing && push!(meta, col)
    end
    cols = vcat(meta, [Symbol("scenario_$id") for id in sorted])
    return cost[:, cols]
end

function _pick_n_scenarios_impl(
    C::Matrix{Float64},
    scenario_ids::Vector{Int};
    n::Int,
    method::Symbol=:pointwise,
    num_samples=nothing,
)
    n < 1 && error("pick_n_scenarios: n must be >= 1, got $n")
    fn = _dominance_fn(method)
    active = copy(scenario_ids)
    picked = Int[]
    rounds = NamedTuple[]

    while !isempty(active) && length(picked) < n
        C_sub = _subset_cost_matrix(C, scenario_ids, active)
        if num_samples === nothing
            res = fn(C_sub; scenarios=active)
        else
            res = fn(C_sub; scenarios=active, num_samples=num_samples)
        end
        layer = sort(Int.(res.undominated))
        push!(rounds, (undominated=layer, n_active=length(active)))
        append!(picked, layer)
        length(picked) >= n && break
        layer_set = Set(layer)
        layer_set == Set(active) && break
        active = Int[id for id in active if id ∉ layer_set]
    end

    return (
        picked=picked,
        rounds=rounds,
        satisfied=length(picked) >= n,
        method=method,
    )
end

"""
    pick_n_scenarios(cost; n, method=:pointwise, scenarios=nothing, num_scenarios=nothing, num_samples=nothing)

Iteratively peel **undominated** (maximal / expensive-tail) scenarios until at least
`n` are collected or the pool is exhausted.

Each round runs the chosen dominance method on the remaining scenarios:

- `method=:pointwise` — `dominating_scenarios` (row-wise cost maximization)
- `method=:fsd` — first-order stochastic dominance (cost maximization)
- `method=:ssd` — second-order stochastic dominance (cost maximization)

Peel order is expensive tiers first: round 1 picks scenarios nobody dominates among
the full set; round 2 picks the top tier among what remains; and so on. The dominance
graph is acyclic (strict partial order), so peeling terminates in at most `S` rounds
for `S` scenarios.

Returns `(picked, rounds, satisfied, method)`:

- `picked` — accumulated scenario ids in peel order (may exceed `n` when the final
  layer adds multiple mutually undominated scenarios; use `picked[1:n]` if exactly
  `n` are needed)
- `rounds` — per-round `(undominated=..., n_active=...)` records
- `satisfied` — `true` iff `length(picked) >= n`
- `method` — the dominance method used

NaN scenarios follow each method's existing rules (FSD/SSD: tagged but still
undominated and eligible for peeling; pointwise: `NaN` treated as `+Inf`).
"""
function pick_n_scenarios(
    cost::DataFrame;
    n::Int,
    method::Symbol=:pointwise,
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    return _pick_n_scenarios_impl(C, scenario_ids; n=n, method=method, num_samples=num_samples)
end

function pick_n_scenarios(
    cost::AbstractMatrix{<:Real};
    n::Int,
    method::Symbol=:pointwise,
    scenarios=nothing,
    num_scenarios=nothing,
    num_samples=nothing,
)
    C, scenario_ids = _normalize_cost_input(
        cost; scenarios=scenarios, num_scenarios=num_scenarios, num_samples=num_samples,
    )
    return _pick_n_scenarios_impl(C, scenario_ids; n=n, method=method, num_samples=num_samples)
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
