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

@testitem "dominator_scenarios from result NamedTuple" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # scenario 10 (col 1) dominates 20 (col 2): col1 >= col2 everywhere, strict once.
    C = Float64[5.0 3.0; 4.0 3.0; 6.0 6.0]
    res = dominating_scenarios(C; scenarios=[10, 20])
    @test dominator_scenarios(res) == [10]
end

@testitem "dominator_scenarios empty when no dominance" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 1.0; 2.0 2.0]
    res = dominating_scenarios(C)
    @test dominator_scenarios(res) == Int[]
end

@testitem "dominator_scenarios from pairs DataFrame, sorted unique" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    pairs = DataFrame(; dominator=[7, 3, 7], dominated=[2, 5, 4])
    @test dominator_scenarios(pairs) == [3, 7]
    empty_pairs = DataFrame(; dominator=Int[], dominated=Int[])
    @test dominator_scenarios(empty_pairs) == Int[]
    bad = DataFrame(; foo=[1])
    @test_throws ErrorException dominator_scenarios(bad)
end

# --- Distributional FSD / SSD dominance ----------------------------------
# Convention (cost-maximization, aligned with pointwise dominating_scenarios):
# dominates[i,j] == true means scenario i has more mass on high costs than j.

@testitem "FSD clear dominance: uniformly more expensive scenario" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col1 = [1,2,3] (cheap), col2 = [4,5,6] (expensive): col2 above col1 everywhere
    C = Float64[1.0 4.0; 2.0 5.0; 3.0 6.0]
    fsd = fsd_dominating_scenarios(C; scenarios=[1, 2])
    ssd = ssd_dominating_scenarios(C; scenarios=[1, 2])

    @test fsd.dominates[2, 1]
    @test !fsd.dominates[1, 2]
    @test fsd.pairs == [(2, 1)]
    @test fsd.undominated == [2]

    # Clear FSD must also be detected by SSD.
    @test ssd.dominates[2, 1]
    @test !ssd.dominates[1, 2]
end

@testitem "SSD detects crossing CDFs that FSD misses" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col1 = [1,4], col2 = [2,3]. CDFs cross so FSD finds nothing. Under cost
    # maximization, col2's integrated CDF stays <= col1's everywhere, so SSD
    # reports col2 (more expensive / risk-relevant) dominates col1.
    C = Float64[1.0 2.0; 4.0 3.0]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)

    @test !any(fsd.dominates)
    @test isempty(fsd.pairs)

    @test ssd.dominates[2, 1]
    @test !ssd.dominates[1, 2]
    @test ssd.pairs == [(2, 1)]
    @test ssd.undominated == [2]
end

@testitem "identical columns: no FSD or SSD dominance" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 1.0; 2.0 2.0; 3.0 3.0]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)
    @test !any(fsd.dominates)
    @test !any(ssd.dominates)
    @test isempty(fsd.pairs)
    @test isempty(ssd.pairs)
    @test sort(fsd.undominated) == [1, 2]
    @test sort(ssd.undominated) == [1, 2]
end

@testitem "FSD/SSD parse cost_matrix DataFrame columns, ignore metadata" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # scenario_5 cheap, scenario_3 expensive; columns extracted sorted as [3, 5].
    df = DataFrame(;
        sample_id=[1, 2, 3],
        sequence=[1, 1, 1],
        scenario_5=[1.0, 2.0, 3.0],
        scenario_3=[4.0, 5.0, 6.0],
    )
    fsd = fsd_dominating_scenarios(df)
    @test fsd.scenarios == [3, 5]
    @test fsd.dominates[1, 2]      # scenario 3 (expensive) dominates scenario 5
    @test !fsd.dominates[2, 1]
    @test fsd.pairs == [(3, 5)]
    @test fsd.undominated == [3]
end

@testitem "FSD ⊂ SSD: every FSD pair is also an SSD pair" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # Mix of clearly ordered scenarios (1<2<3) plus a crossing pair (col4).
    C = Float64[
        1.0 4.0 7.0 2.0
        2.0 5.0 8.0 8.0
        3.0 6.0 9.0 3.0
    ]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)
    # Wherever FSD reports dominance, SSD must too.
    @test all(.!fsd.dominates .| ssd.dominates)
    @test any(fsd.dominates)
end

# --- Distributional FSD / SSD edge cases ---------------------------------

@testitem "FSD/SSD tag and skip scenarios containing NaN" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col2 has a NaN -> scenario 2's distribution is incomplete, so it is tagged
    # and excluded from ordering entirely (never dominates, never dominated).
    C = Float64[1.0 3.0; 2.0 NaN]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)
    @test fsd.nan_scenarios == [2]
    @test ssd.nan_scenarios == [2]
    @test !any(fsd.dominates)
    @test !any(ssd.dominates)
    @test isempty(fsd.pairs)
    @test isempty(ssd.pairs)
    # Tagged scenario is never dominated, so it stays in undominated.
    @test sort(fsd.undominated) == [1, 2]
    @test sort(ssd.undominated) == [1, 2]
end

@testitem "FSD/SSD keep finite ordering while skipping a NaN scenario" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # scenarios 10,20 finite and clearly ordered (10 cheaper, 20 expensive); 30 has a NaN.
    C = Float64[1.0 4.0 1.0; 2.0 5.0 NaN; 3.0 6.0 3.0]
    fsd = fsd_dominating_scenarios(C; scenarios=[10, 20, 30])
    ssd = ssd_dominating_scenarios(C; scenarios=[10, 20, 30])

    @test fsd.nan_scenarios == [30]
    @test ssd.nan_scenarios == [30]

    # Finite pair is still ordered: 20 dominates 10.
    @test fsd.dominates[2, 1]
    @test !fsd.dominates[1, 2]
    @test (20, 10) in fsd.pairs

    # No pair touches the NaN scenario (index 3 = scenario 30).
    @test !any(fsd.dominates[3, :]) && !any(fsd.dominates[:, 3])
    @test !any(ssd.dominates[3, :]) && !any(ssd.dominates[:, 3])
    @test all(p -> 30 ∉ p, fsd.pairs)
    @test all(p -> 30 ∉ p, ssd.pairs)

    # Skipped scenario is undominated by construction.
    @test 30 in fsd.undominated
    @test 30 in ssd.undominated
end

@testitem "FSD/SSD warn when NaN scenarios are encountered" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 3.0; 2.0 NaN]
    @test_logs (:warn,) match_mode = :any fsd_dominating_scenarios(C)
    @test_logs (:warn,) match_mode = :any ssd_dominating_scenarios(C)
end

@testitem "FSD/SSD are distributional: invariant to row order" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # Same multiset of costs in each column, just permuted across samples.
    # Unlike the row-wise `dominating_scenarios`, FSD/SSD compare distributions,
    # so identical empirical CDFs => no dominance either way.
    C = Float64[1.0 3.0; 2.0 1.0; 3.0 2.0]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)
    @test !any(fsd.dominates)
    @test !any(ssd.dominates)
    @test sort(fsd.undominated) == [1, 2]
    @test sort(ssd.undominated) == [1, 2]
    # No NaN anywhere -> empty nan_scenarios.
    @test isempty(fsd.nan_scenarios)
    @test isempty(ssd.nan_scenarios)
end

@testitem "FSD/SSD single scenario: no pairs, itself undominated" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = reshape(Float64[1.0, 2.0, 3.0], 3, 1)
    fsd = fsd_dominating_scenarios(C; scenarios=[42])
    ssd = ssd_dominating_scenarios(C; scenarios=[42])
    @test fsd.scenarios == [42]
    @test isempty(fsd.pairs)
    @test isempty(ssd.pairs)
    @test fsd.undominated == [42]
    @test ssd.undominated == [42]
    @test size(fsd.dominates) == (1, 1)
    @test !fsd.dominates[1, 1]
end

@testitem "FSD transitive chain: full ordered pair set and single undominated" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col1 < col2 < col3 elementwise across the distribution (10 < 20 < 30).
    C = Float64[1.0 4.0 7.0; 2.0 5.0 8.0; 3.0 6.0 9.0]
    fsd = fsd_dominating_scenarios(C; scenarios=[10, 20, 30])
    @test Set(fsd.pairs) == Set([(20, 10), (30, 10), (30, 20)])
    @test fsd.undominated == [30]
    # Most expensive dominates both others; cheapest is dominated by everyone.
    @test fsd.dominates[2, 1] && fsd.dominates[3, 1] && fsd.dominates[3, 2]
    @test !any(fsd.dominates[:, 3])
end

@testitem "SSD reports no dominance when integrated CDFs cross" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    # col1 = [1,5], col2 = [2,3]: neither CDF nor integrated CDF dominates the other.
    C = Float64[1.0 2.0; 5.0 3.0]
    fsd = fsd_dominating_scenarios(C)
    ssd = ssd_dominating_scenarios(C)
    @test !any(fsd.dominates)
    @test !any(ssd.dominates)
    @test isempty(ssd.pairs)
    @test sort(ssd.undominated) == [1, 2]
end

@testitem "FSD/SSD kwarg validation errors" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = ones(3, 2)
    @test_throws ErrorException fsd_dominating_scenarios(C; num_scenarios=3)
    @test_throws ErrorException fsd_dominating_scenarios(C; num_samples=4)
    @test_throws ErrorException fsd_dominating_scenarios(C; scenarios=[1, 2, 3])
    @test_throws ErrorException ssd_dominating_scenarios(C; num_scenarios=3)
    @test_throws ErrorException ssd_dominating_scenarios(C; num_samples=4)

    df = DataFrame(; sample_id=[1, 2], scenario_3=[1.0, 2.0], scenario_5=[3.0, 4.0])
    @test_throws ErrorException fsd_dominating_scenarios(df; scenarios=[5, 3])
    @test_throws ErrorException fsd_dominating_scenarios(df; num_scenarios=3)
    @test_throws ErrorException ssd_dominating_scenarios(df; num_samples=5)
end

# --- pick_n_scenarios: iterative dominance peeling -----------------------

@testitem "pick_n_scenarios chain peel pointwise and FSD" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 4.0 7.0; 2.0 5.0 8.0; 3.0 6.0 9.0]
    for method in (:pointwise, :fsd)
        res = pick_n_scenarios(C; n=2, method=method, scenarios=[10, 20, 30])
        @test res.picked == [30, 20]
        @test res.satisfied
        @test res.method == method
        @test length(res.rounds) == 2
        @test res.rounds[1].undominated == [30]
        @test res.rounds[2].undominated == [20]
        one_shot = method == :pointwise ? dominating_scenarios(C; scenarios=[10, 20, 30]) :
            fsd_dominating_scenarios(C; scenarios=[10, 20, 30])
        @test res.rounds[1].undominated == one_shot.undominated
    end
end

@testitem "pick_n_scenarios exhausts pool before reaching n" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 4.0 7.0; 2.0 5.0 8.0; 3.0 6.0 9.0]
    res = pick_n_scenarios(C; n=10, method=:pointwise, scenarios=[10, 20, 30])
    @test res.picked == [30, 20, 10]
    @test !res.satisfied
    @test length(res.rounds) == 3
end

@testitem "pick_n_scenarios single round when all mutually undominated" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 2.0; 5.0 3.0]
    res = pick_n_scenarios(C; n=1, method=:pointwise)
    @test sort(res.picked) == [1, 2]
    @test res.satisfied
    @test length(res.rounds) == 1
    @test sort(res.rounds[1].undominated) == [1, 2]
end

@testitem "pick_n_scenarios FSD n=1 clear dominance" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 4.0; 2.0 5.0; 3.0 6.0]
    res = pick_n_scenarios(C; n=1, method=:fsd, scenarios=[1, 2])
    @test res.picked == [2]
    @test res.satisfied
    @test fsd_dominating_scenarios(C; scenarios=[1, 2]).undominated == [2]
end

@testitem "pick_n_scenarios SSD crossing peel" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 2.0; 4.0 3.0]
    res = pick_n_scenarios(C; n=1, method=:ssd)
    @test res.picked == [2]
    @test res.satisfied
    @test ssd_dominating_scenarios(C).undominated == [2]
end

@testitem "pick_n_scenarios DataFrame input" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    df = DataFrame(;
        sample_id=[1, 2, 3],
        sequence=[1, 1, 1],
        scenario_5=[1.0, 2.0, 3.0],
        scenario_3=[4.0, 5.0, 6.0],
    )
    res = pick_n_scenarios(df; n=1, method=:pointwise)
    @test res.picked == [3]
    @test res.satisfied
end

@testitem "pick_n_scenarios validation errors" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    C = Float64[1.0 2.0; 3.0 4.0]
    @test_throws ErrorException pick_n_scenarios(C; n=0)
    @test_throws ErrorException pick_n_scenarios(C; n=1, method=:bogus)
end

# --- undominated_scenarios from a pairs DataFrame ------------------------

@testitem "undominated_scenarios from pairs DataFrame" setup = [ScenarioDominanceSetup] tags = [:dominance, :unit] begin
    pairs = DataFrame(; dominator=[10, 10], dominated=[20, 30])
    @test undominated_scenarios(pairs, [10, 20, 30]) == [10]

    empty_pairs = DataFrame(; dominator=Int[], dominated=Int[])
    @test undominated_scenarios(empty_pairs, [1, 2, 3]) == [1, 2, 3]

    bad = DataFrame(; dominator=[1])
    @test_throws ErrorException undominated_scenarios(bad, [1, 2])
end
