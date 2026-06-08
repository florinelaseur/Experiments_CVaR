using TestItems: @testitem

@testitem "audit detects RIDM-like permutation mismatch" setup = [InvestmentMappingSetup] tags = [:mapping, :unit] begin
    # asset-milestone.csv investable row order for RIDM (single year)
    inv_df = DataFrame(;
        asset=["ccgt", "wind", "solar", "ocgt", "electrolizer", "wind_offshore", "battery"],
        milestone_year=fill(2030, 7),
    )

    result = audit_investment_mapping_order(inv_df)
    @test !result.permutation_ok
    @test result.dim_ok
    @test length(result.mismatches) > 0
    @test result.mismatches[1] == (2, "ocgt", "wind")
end

@testitem "audit passes when orders match" setup = [InvestmentMappingSetup] tags = [:mapping, :unit] begin
    inv_df = DataFrame(;
        asset=copy(INVESTABLE_ASSETS),
        milestone_year=fill(2030, length(INVESTABLE_ASSETS)),
    )

    result = audit_investment_mapping_order(inv_df)
    @test result.permutation_ok
    @test isempty(result.mismatches)
end

@testitem "align permuted indices with MW to model units" setup = [InvestmentMappingSetup] tags = [:mapping, :unit] begin
    inv_df = DataFrame(;
        asset=["ccgt", "wind", "solar", "ocgt", "electrolizer", "wind_offshore", "battery"],
        milestone_year=fill(2030, 7),
    )

    capacity_lookup = Dict(
        "ccgt" => 0.8,
        "ocgt" => 0.1,
        "solar" => 0.5,
        "wind" => 0.4,
        "wind_offshore" => 0.4,
        "electrolizer" => 0.1,
        "battery" => 0.05,
    )

    # Sentinel MW values keyed by INVESTABLE_ASSETS order
    sample_mw = Float64[1000, 2000, 3000, 4000, 5000, 6000, 7000]
    vals = align_investment_sample_to_indices(inv_df, sample_mw, capacity_lookup)

    @test length(vals) == 7
    @test vals[1] ≈ 1000 / 0.8      # ccgt at container index 1
    @test vals[2] ≈ 4000 / 0.4      # wind at container index 2 (sample dim 4)
    @test vals[4] ≈ 2000 / 0.1      # ocgt at container index 4 (sample dim 2)
    @test vals[6] ≈ 5000 / 0.4      # wind_offshore at index 6 (sample dim 5)
    @test vals[7] ≈ 7000 / 0.05     # battery at index 7
end

@testitem "sample_vector_by_asset length guard" setup = [InvestmentMappingSetup] tags = [:mapping, :unit] begin
    @test_throws ErrorException sample_vector_by_asset([1.0, 2.0])
end

@testitem "fix_variables_from_sample requires capacity_lookup for assets_investment" setup = [
    InvestmentFixSetup,
] tags = [:mapping, :unit] begin
    model = JuMP.Model()
    container = [JuMP.@variable(model, base_name = "inv_$i") for i in 1:length(INVESTABLE_ASSETS)]
    variables = Dict(
        :assets_investment => (
            indices = DataFrame(;
                asset = copy(INVESTABLE_ASSETS),
                milestone_year = fill(2030, length(INVESTABLE_ASSETS)),
            ),
            container = container,
        ),
    )
    @test_throws ErrorException fix_variables_from_sample(
        variables,
        :assets_investment,
        zeros(length(INVESTABLE_ASSETS)),
    )
end
