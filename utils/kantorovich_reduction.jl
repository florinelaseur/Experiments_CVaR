# Kantorovich distance with forward selection for scenario reduction.
#
# Reference: Heitsch & Römisch (2003), "Scenario Reduction Algorithms in
# Stochastic Programming", Computational Optimization and Applications.

function minmax_normalize(mat::Matrix{Float64})
    result = copy(mat)
    for i in axes(mat, 1)
        lo, hi = extrema(view(mat, i, :))
        rng = hi - lo
        if rng > 0.0
            result[i, :] = (mat[i, :] .- lo) ./ rng
        else
            result[i, :] .= 0.0
        end
    end
    return result
end

# Returns (n_features × n_scenarios) normalized matrix and sorted scenario ids.
function build_scenario_matrix(profiles_df::DataFrame)
    profile_cols = [:solar, :wind_offshore, :wind_onshore, :demand, :hydro_inflow]
    scenarios = sort(unique(profiles_df.scenario))
    n_scenarios = length(scenarios)
    scen_index = Dict(s => i for (i, s) in enumerate(scenarios))

    ref = filter(row -> row.scenario == scenarios[1], profiles_df)
    sort!(ref, [:timestep])
    n_timesteps = nrow(ref)
    n_features = n_timesteps * length(profile_cols)

    mat = Matrix{Float64}(undef, n_features, n_scenarios)
    for s in scenarios
        rows = filter(row -> row.scenario == s, profiles_df)
        sort!(rows, [:timestep])
        col = scen_index[s]
        for (p_idx, pcol) in enumerate(profile_cols)
            offset = (p_idx - 1) * n_timesteps
            mat[(offset + 1):(offset + n_timesteps), col] = rows[!, pcol]
        end
    end

    return minmax_normalize(mat), scenarios
end

# Symmetric pairwise L2 distance matrix between columns of mat.
function compute_cost_matrix(mat::Matrix{Float64})
    n = size(mat, 2)
    C = Matrix{Float64}(undef, n, n)
    for j in 1:n
        for i in 1:j
            d = sqrt(sum(abs2, view(mat, :, i) .- view(mat, :, j)))
            C[i, j] = d
            C[j, i] = d
        end
        C[j, j] = 0.0
    end
    return C
end

# Greedy forward selection minimising Kantorovich distance objective.
# Returns (selected_1based_indices, redistributed_probabilities).
function kantorovich_forward_select(
    probs::Vector{Float64},
    C::Matrix{Float64},
    k::Int,
)
    n = length(probs)
    @assert 1 <= k <= n "k=$k must be in [1, $n]"
    @assert size(C) == (n, n)
    @assert abs(sum(probs) - 1.0) < 1e-9 "Probabilities must sum to 1, got $(sum(probs))"

    # Seed: scenario whose total weighted distance to all others is smallest.
    seed = argmin([sum(probs[j] * C[i, j] for j in 1:n) for i in 1:n])
    selected = [seed]

    # nearest_dist[i] = distance from scenario i to its closest selected scenario.
    nearest_dist = [C[i, seed] for i in 1:n]
    nearest_dist[seed] = 0.0

    while length(selected) < k
        selected_set = Set(selected)
        best_u = -1
        best_z = Inf

        for u in 1:n
            u in selected_set && continue
            z = 0.0
            for i in 1:n
                i in selected_set && continue
                i == u && continue
                z += probs[i] * min(nearest_dist[i], C[i, u])
            end
            if z < best_z
                best_z = z
                best_u = u
            end
        end

        push!(selected, best_u)
        for i in 1:n
            i in Set(selected) && continue
            nearest_dist[i] = min(nearest_dist[i], C[i, best_u])
        end
    end

    # Redistribute: each non-selected scenario transfers probability to nearest selected.
    sel_order = Dict(selected[j] => j for j in 1:k)
    new_probs = zeros(Float64, k)
    selected_set = Set(selected)
    for i in 1:n
        if i in selected_set
            new_probs[sel_order[i]] += probs[i]
        else
            nearest_j = selected[argmin([C[i, j] for j in selected])]
            new_probs[sel_order[nearest_j]] += probs[i]
        end
    end

    return selected, new_probs
end

# Pure Kantorovich forward selection on the scenarios of profiles_df, excluding
# `exclude` from the candidate pool first (selection AND probability redistribution
# happen within the remaining pool, uniformly weighted). Returns
# (selected scenario ids in selection order, redistributed probabilities).
# Does NOT touch any DuckDB connection — callers decide what to do with the picks.
function select_kantorovich_scenarios(
    profiles_df::DataFrame,
    k::Int;
    exclude::AbstractVector{<:Integer}=Int[],
)
    k == 0 && return (Int[], Float64[])
    exclude_set = Set(Int.(exclude))
    pool_df = filter(row -> row.scenario ∉ exclude_set, profiles_df)
    isempty(pool_df) &&
        error("select_kantorovich_scenarios: no scenarios left after excluding $(sort(collect(exclude_set)))")

    mat, scenarios = build_scenario_matrix(pool_df)
    n = length(scenarios)
    k <= n ||
        error("select_kantorovich_scenarios: k=$k exceeds the $(n) eligible scenarios")
    probs = fill(1.0 / n, n)

    C = compute_cost_matrix(mat)

    selected_idx, new_probs = kantorovich_forward_select(probs, C, k)
    return (scenarios[selected_idx], new_probs)
end

# Performs Kantorovich forward selection and mutates DuckDB in-place.
# Must be called BEFORE use_ratio UPDATE and TC.transform_wide_to_long!.
function apply_kantorovich_reduction!(connection, profiles_df::DataFrame, k::Int)
    k >= 1 || error("apply_kantorovich_reduction!: k must be >= 1, got $k")
    selected_ids, new_probs = select_kantorovich_scenarios(profiles_df, k)

    @info "Kantorovich: selected scenarios $selected_ids with probs $new_probs (sum=$(sum(new_probs)))"

    id_list = join(selected_ids, ", ")
    DuckDB.query(connection, "DELETE FROM profiles_wide WHERE scenario NOT IN ($id_list);")
    DuckDB.query(connection, "DELETE FROM stochastic_scenario WHERE scenario NOT IN ($id_list);")
    for (sid, p) in zip(selected_ids, new_probs)
        DuckDB.query(
            connection,
            "UPDATE stochastic_scenario SET probability = $p WHERE scenario = $sid;",
        )
    end

    return nothing
end
