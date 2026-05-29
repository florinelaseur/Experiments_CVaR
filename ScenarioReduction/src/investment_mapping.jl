using DataFrames: DataFrame, eachrow, nrow

const INVESTABLE_ASSETS = [
    "ccgt",
    "ocgt",
    "solar",
    "wind",
    "wind_offshore",
    "electrolizer",
    "battery",
]

"""
    sample_vector_by_asset(sample_mw, assets=INVESTABLE_ASSETS)

Map a sample vector (MW space, `INVESTABLE_ASSETS` order) to `asset => value`.
"""
function sample_vector_by_asset(
    sample_mw::AbstractVector{<:Real},
    assets=INVESTABLE_ASSETS,
)
    length(sample_mw) == length(assets) ||
        error(
            "Sample length $(length(sample_mw)) does not match assets length $(length(assets))",
        )
    return Dict(assets[i] => Float64(sample_mw[i]) for i in eachindex(assets))
end

"""
    audit_investment_mapping_order(inv_df; assets=INVESTABLE_ASSETS)

Compare TEM `assets_investment` indices row order to positional `assets` order.
Returns a NamedTuple reporting whether naive `zip(container, sample)` would align.
"""
function audit_investment_mapping_order(
    inv_df::DataFrame;
    assets=INVESTABLE_ASSETS,
)
    n_container = nrow(inv_df)
    n_sample = length(assets)
    dim_ok = n_container == n_sample
    mismatches = Tuple{Int,String,String}[]

    for i in 1:min(n_container, n_sample)
        container_asset = string(inv_df.asset[i])
        expected_asset = assets[i]
        if container_asset != expected_asset
            push!(mismatches, (i, expected_asset, container_asset))
        end
    end

    permutation_ok = dim_ok && isempty(mismatches)
    return (
        inv_df=inv_df,
        permutation_ok=permutation_ok,
        dim_ok=dim_ok,
        mismatches=mismatches,
        n_container=n_container,
        n_sample_assets=n_sample,
    )
end

"""
    align_investment_sample_to_indices(inv_df, sample_mw, capacity_lookup; assets=INVESTABLE_ASSETS)

Reorder a sample from `assets` order (MW) into TEM container/indices order (model units).
"""
function align_investment_sample_to_indices(
    inv_df::DataFrame,
    sample_mw::AbstractVector{<:Real},
    capacity_lookup::Dict{String,Float64};
    assets=INVESTABLE_ASSETS,
)
    sample_by_asset = sample_vector_by_asset(sample_mw, assets)
    n = nrow(inv_df)
    vals = Vector{Float64}(undef, n)

    for (i, row) in enumerate(eachrow(inv_df))
        asset = string(row.asset)
        haskey(sample_by_asset, asset) ||
            error("No sample for asset $asset at container index $i")
        haskey(capacity_lookup, asset) ||
            error("No capacity for asset $asset at container index $i")
        cap = capacity_lookup[asset]
        cap > 0 || error("Non-positive capacity for asset $asset: $cap")
        vals[i] = sample_by_asset[asset] / cap
    end

    return vals
end
