using TestItems: @testmodule

@testmodule SamplingSetup begin
    using LinearAlgebra
    using Statistics
    using Random
    using QuasiMonteCarlo

    include(joinpath(@__DIR__, "..", "src", "sampling.jl"))

    # Re-export the stdlib + dep names that test items rely on so they're
    # visible via `using ..SamplingSetup` inside each @testitem's module.
    export I, Diagonal, diag, cholesky, Symmetric
    export mean, cov

    export sobol_gaussian_samples,
           sobol_gaussian_samples_nonneg,
           shrink_covariance
end

@testmodule InvestmentMappingSetup begin
    using DataFrames: DataFrame, DataFrames

    include(joinpath(@__DIR__, "..", "src", "investment_mapping.jl"))

    export INVESTABLE_ASSETS,
           DataFrame,
           sample_vector_by_asset,
           audit_investment_mapping_order,
           align_investment_sample_to_indices
end
