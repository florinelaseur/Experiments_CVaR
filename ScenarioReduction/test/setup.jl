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
           shrink_covariance,
           sobol_gaussian_reject_to_target,
           scrambled_sobol_uniform_reject_to_target
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

@testmodule AdequacyCutsSetup begin
    using DataFrames: DataFrame, DataFrames, nrow

    # adequacy_cuts.jl needs INVESTABLE_ASSETS from investment_mapping.jl. It also
    # references JuMP/CSV/TIO in functions we do NOT call here (feasibility_center,
    # CSV writers, read_adequacy_params); those resolve lazily at call time, so
    # including the file without a solver is fine.
    include(joinpath(@__DIR__, "..", "src", "investment_mapping.jl"))
    include(joinpath(@__DIR__, "..", "src", "adequacy_cuts.jl"))

    export INVESTABLE_ASSETS, DataFrame, nrow,
           AdequacyParams, AdequacyCuts,
           build_adequacy_cuts, passes_adequacy, adequacy_verdict,
           feasibility_center_max_optima,
           _nondominated_indices, _cut_coeff_row
end
