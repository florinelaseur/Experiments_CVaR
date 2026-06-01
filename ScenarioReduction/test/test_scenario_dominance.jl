using TestItems: @testitem

@testitem "column dominates when all >= and one strict >" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[
        5.0 3.0
        4.0 3.0
        6.0 6.0
    ]
    res = dominating_scenarios(C; scenarios=[10, 20])
    @test res.scenarios == [10, 20]
    @test res.dominates == [false true; false false]
    @test res.pairs == [(10, 20)]
    @test 10 in res.undominated
    @test !(20 in res.undominated)
end

@testitem "equal columns — no dominance" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 1.0; 2.0 2.0; 3.0 3.0]
    res = dominating_scenarios(C)
    @test !any(res.dominates)
    @test isempty(res.pairs)
    @test sort(res.undominated) == [1, 2]
end

@testitem "NaN treated as Inf for dominance" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col1: [1, Inf]; col2: [2, 3] — fails at k=1 (1>=2 false)
    C = Float64[1.0 2.0; NaN 3.0]
    res = dominating_scenarios(C)
    @test !res.dominates[1, 2]
    @test !res.dominates[2, 1]

    # col1 all >= col2 with strict > when NaN beats finite
    C2 = Float64[NaN 3.0; NaN 3.0]
    res2 = dominating_scenarios(C2)
    @test res2.dominates[1, 2]
    @test !res2.dominates[2, 1]
end

@testitem "both NaN same row — no strict inequality from that row" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[NaN NaN; 2.0 1.0]
    res = dominating_scenarios(C)
    @test res.dominates[1, 2]   # Inf>=Inf on row1 (no strict), row2: 2>1
    @test !res.dominates[2, 1]
end

@testitem "DataFrame scenario_* column parsing" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    df = DataFrame(;
        sample_id=[1, 2],
        sequence=[1, 1],
        scenario_5=[10.0, 12.0],
        scenario_3=[8.0, 9.0],
    )
    res = dominating_scenarios(df)
    @test res.scenarios == [3, 5]
    @test res.dominates[2, 1]   # scenario 5 dominates 3
    @test !res.dominates[1, 2]
end

@testitem "num_scenarios and num_samples validation" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = ones(3, 2)
    @test_throws ErrorException dominating_scenarios(C; num_scenarios=3)
    @test_throws ErrorException dominating_scenarios(C; num_samples=4)
end
