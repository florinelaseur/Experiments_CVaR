using CSV: CSV
using DataFrames: DataFrame
using Statistics: Statistics
using QuasiMonteCarlo: QuasiMonteCarlo

# Quasi-random Gaussian sampling (sobol_gaussian_samples,
# sobol_gaussian_samples_nonneg, shrink_covariance) lives in its own file so
# that the unit tests under ScenarioReduction/test/ can load it without
# pulling in JuMP/Tulipa/DuckDB.
include(joinpath(@__DIR__, "sampling.jl"))
include(joinpath(@__DIR__, "investment_mapping.jl"))
include(joinpath(@__DIR__, "adequacy_cuts.jl"))
include(joinpath(@__DIR__, "adequacy_center.jl"))  # JuMP-macro LP; needs JuMP in scope
include(joinpath(@__DIR__, "single_scenario.jl"))  # build_single_scenario_model; needs TEM/TC in scope

struct AssetInvestmentBounds
    max::Float64
    mean::Float64
    std::Float64
end

struct InvestmentBounds
    assets::Dict{String, AssetInvestmentBounds}
    path::String
    ub::Vector{Float64}
    mean::Vector{Float64}
end

function investment_ub_vector(assets::Dict{String, AssetInvestmentBounds})
    ub = Vector{Float64}(undef, length(INVESTABLE_ASSETS))
    for (i, asset) in pairs(INVESTABLE_ASSETS)
        haskey(assets, asset) || error("Missing investment bounds for asset: $asset")
        ub[i] = assets[asset].max
    end
    return ub
end

function investment_mean_vector(assets::Dict{String, AssetInvestmentBounds})
    mean = Vector{Float64}(undef, length(INVESTABLE_ASSETS))
    for (i, asset) in pairs(INVESTABLE_ASSETS)
        haskey(assets, asset) || error("Missing investment bounds for asset: $asset")
        mean[i] = assets[asset].mean
    end
    return mean
end

function default_investment_bounds_path()
    return joinpath(@__DIR__, "..", "investment_bounds.csv")
end

function load_investment_bounds(path::String=default_investment_bounds_path())
    isfile(path) || error("Investment bounds file not found: $path")
    df = CSV.read(path, DataFrame)
    nrow(df) > 0 || error("Investment bounds file is empty: $path")

    required = [:asset, :max_investment, :mean_investment, :std_investment]
    missing_cols = [col for col in required if !(col in propertynames(df))]
    !isempty(missing_cols) &&
        error("Missing columns in investment bounds file: $(join(missing_cols, ", "))")

    assets = Dict{String, AssetInvestmentBounds}()
    for row in eachrow(df)
        assets[row.asset] = AssetInvestmentBounds(
            row.max_investment,
            row.mean_investment,
            row.std_investment,
        )
    end

    for asset in INVESTABLE_ASSETS
        haskey(assets, asset) || @warn "Missing investment bounds for asset: $asset"
    end

    ub = investment_ub_vector(assets)
    mean = investment_mean_vector(assets)
    return InvestmentBounds(assets, path, ub, mean)
end

function get_asset_bounds(bounds::InvestmentBounds, asset::String)
    haskey(bounds.assets, asset) || error("No investment bounds for asset: $asset")
    return bounds.assets[asset]
end

max_investment(bounds::InvestmentBounds, asset::String) = get_asset_bounds(bounds, asset).max
mean_investment(bounds::InvestmentBounds, asset::String) = get_asset_bounds(bounds, asset).mean
std_investment(bounds::InvestmentBounds, asset::String) = get_asset_bounds(bounds, asset).std

struct InvestmentCovariance
    matrix::Matrix{Float64}
    assets::Vector{String}
    n_scenarios::Int
    path::String
end

function default_per_scenario_investments_path()
    return joinpath(@__DIR__, "..", "per_scenario_investments.csv")
end

function load_per_scenario_investments(path::String=default_per_scenario_investments_path())
    isfile(path) || error("Per-scenario investments file not found: $path")
    df = CSV.read(path, DataFrame)
    nrow(df) > 0 || error("Per-scenario investments file is empty: $path")

    if :termination_status in propertynames(df)
        df = filter(row -> row.termination_status == "OPTIMAL", df)
        nrow(df) > 0 || error("No OPTIMAL scenarios found in: $path")
    end

    asset_cols = Symbol.(INVESTABLE_ASSETS)
    missing_cols = [col for col in asset_cols if !(col in propertynames(df))]
    !isempty(missing_cols) &&
        error("Missing asset columns in per-scenario investments file: $(join(missing_cols, ", "))")

    return df[!, asset_cols]
end

function investment_covariance(path::String=default_per_scenario_investments_path())
    df = load_per_scenario_investments(path)
    n = nrow(df)
    n >= 2 || error("Need at least 2 scenarios to compute covariance; got $n")

    mat = Matrix(df)
    cov_matrix = Statistics.cov(mat)

    return InvestmentCovariance(cov_matrix, copy(INVESTABLE_ASSETS), n, path)
end

function asset_index(cov::InvestmentCovariance, asset::String)
    idx = findfirst(==(asset), cov.assets)
    idx === nothing && error("No index for asset: $asset")
    return idx
end

function print_model_variables_before_clustering(connection)    
    layout = TC.ProfilesTableLayout(;
    year=:milestone_year,
    cols_to_groupby=[:milestone_year, :scenario],
    )

    time_to_cluster = @elapsed TC.dummy_cluster!(connection; layout=layout)
    TEM.populate_with_defaults!(connection)
    TEM.create_internal_tables!(connection)
    variables = TEM.compute_variables_indices(connection)
    

    # Inspect investment space dimension
    inv_df = DataFrame(variables[:assets_investment].indices)
    @info "Investment space dimension: $(nrow(inv_df))"
    for (i, row) in enumerate(eachrow(inv_df))
        @info "Investment $i: $row"
    end

    constraints = TEM.compute_constraints_indices(connection)
    for (name, cons) in constraints
        n = TEM.get_num_rows(connection, cons)
        @info "Constraint :$name — $n rows"
    end
    profiles   = TEM.prepare_profiles_structure(connection)          # profile setup
    model, expressions = TEM.create_model(connection, variables, constraints, profiles)
    solve_model(model)
    
end


function solve_model(model::JuMP.Model; diagnose_infeasibility=true)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    if status == JuMP.OPTIMAL
        return status
    end
    @warn "Model status: $status"
    if diagnose_infeasibility && status in (JuMP.INFEASIBLE, JuMP.INFEASIBLE_OR_UNBOUNDED)
        print_infeasibility_conflict!(model)
    end
    return status
end

function generate_scrambled_Sobol_samples(
    n_samples::Int,
    dim::Int,
    lb::Vector{Float64},
    ub::Vector{Float64},
    number_of_samples_sequences::Int,
)
    return [
        QuasiMonteCarlo.sample(
            n_samples,
            lb,
            ub,
            QuasiMonteCarlo.SobolSample(;
                R=QuasiMonteCarlo.OwenScramble(base=2, pad=32, rng=Random.Xoshiro(seed)),
            ),
        ) for seed in 1:number_of_samples_sequences
    ]
end

function build_capacity_lookup(connection)
    asset_df = DataFrame(TIO.get_table(connection, "asset"))
    return Dict(string(a) => Float64(c) for (a, c) in zip(asset_df.asset, asset_df.capacity))
end

function audit_investment_mapping(
    variables;
    assets=INVESTABLE_ASSETS,
    capacity_lookup=nothing,
)
    inv_df = DataFrame(variables[:assets_investment].indices)
    result = audit_investment_mapping_order(inv_df; assets)

    @info "Investment mapping audit" (
        permutation_ok=result.permutation_ok,
        dim_ok=result.dim_ok,
        n_container=result.n_container,
        n_sample_assets=result.n_sample_assets,
    )

    if !result.permutation_ok
        for (i, expected, actual) in result.mismatches
            @warn "Positional mismatch at index $i" expected=expected actual=actual
        end
        !result.dim_ok &&
            @warn "Container dimension $(result.n_container) != sample dimension $(result.n_sample_assets)"
    end

    if capacity_lookup !== nothing
        for row in eachrow(inv_df)
            asset = string(row.asset)
            if haskey(capacity_lookup, asset)
                @debug "Capacity for $asset" capacity=capacity_lookup[asset]
            else
                @warn "Missing capacity for investment asset: $asset"
            end
        end
    end

    return result
end

function align_investment_sample_to_container(
    variables,
    sample_mw::AbstractVector{Float64};
    capacity_lookup::Dict{String,Float64},
    assets=INVESTABLE_ASSETS,
)
    inv_df = DataFrame(variables[:assets_investment].indices)
    return align_investment_sample_to_indices(
        inv_df,
        sample_mw,
        capacity_lookup;
        assets,
    )
end

function assert_variables_fixed!(vars, targets::AbstractVector{Float64}; tol=1e-8)
    length(vars) == length(targets) ||
        error(
            "Cannot fix $(length(targets)) values onto $(length(vars)) variables",
        )

    for (i, (var, target)) in enumerate(zip(vars, targets))
        JuMP.is_fixed(var) ||
            error("Variable at index $i is not fixed")
        abs(JuMP.fix_value(var) - target) > tol &&
            error(
                "Variable at index $i has fix_value=$(JuMP.fix_value(var)), expected $target",
            )
    end

    return nothing
end

function fix_variables_from_sample(
    variables,
    var_symbol,
    val_to_fix::AbstractVector{Float64};
    capacity_lookup=nothing,
    assets=INVESTABLE_ASSETS,
)
    if var_symbol == :assets_investment && capacity_lookup !== nothing
        val_to_fix = align_investment_sample_to_container(
            variables,
            val_to_fix;
            capacity_lookup,
            assets,
        )
    end

    var_to_fix = variables[var_symbol].container

    for (var, val) in zip(var_to_fix, val_to_fix)
        JuMP.fix(var, val; force=true)
    end

    assert_variables_fixed!(var_to_fix, val_to_fix)

    return var_to_fix
end