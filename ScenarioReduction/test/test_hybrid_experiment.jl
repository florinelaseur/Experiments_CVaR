using TestItems: @testitem

# Deterministic, solver-free coverage for the hybrid dominance + Kantorovich
# driver's reusable pieces (test_dominance_kantorovich.jl):
#   - select_kantorovich_scenarios (disjoint-complement selection)
#   - read_investment_mw (var_assets_investment.csv → MW vector)
#   - the pure helpers in src/hybrid_helpers.jl (resume detection, results
#     readback, optimality-gap arithmetic, investment cross-check)

# --- select_kantorovich_scenarios -----------------------------------------

@testitem "select_kantorovich_scenarios excludes the dominance picks and stays in the pool" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    pdf = _toy_profiles(5, 3)
    sel, probs = select_kantorovich_scenarios(pdf, 2; exclude=[1, 4])
    @test length(sel) == 2
    @test isempty(intersect(sel, [1, 4]))          # disjoint from the excluded set
    @test all(in([2, 3, 5]), sel)                  # only eligible scenarios chosen
    @test allunique(sel)
    @test length(probs) == 2
    @test isapprox(sum(probs), 1.0; atol=1e-9)     # probabilities renormalized over the pool
end

@testitem "select_kantorovich_scenarios n=0 returns empties without touching the data" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    empty_pdf = DataFrame(;
        milestone_year=Int[], scenario=Int[], timestep=Int[],
        solar=Float64[], wind_offshore=Float64[], wind_onshore=Float64[],
        demand=Float64[], hydro_inflow=Float64[],
    )
    sel, probs = select_kantorovich_scenarios(empty_pdf, 0)
    @test isempty(sel)
    @test isempty(probs)
end

@testitem "select_kantorovich_scenarios errors when k exceeds the eligible pool" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    pdf = _toy_profiles(3, 2)
    @test_throws ErrorException select_kantorovich_scenarios(pdf, 5)                 # k > scenarios
    @test_throws ErrorException select_kantorovich_scenarios(pdf, 1; exclude=[1, 2, 3])  # pool empty after exclude
end

# --- read_investment_mw ----------------------------------------------------

@testitem "read_investment_mw maps solution×capacity into INVESTABLE_ASSETS order" setup = [InvestmentFixSetup] tags = [:hybrid, :unit] begin
    dir = mktempdir()
    n = length(INVESTABLE_ASSETS)
    # Encode each asset's identity in its solution (= its INVESTABLE_ASSETS index),
    # capacity = 10, and write the rows in REVERSED order to prove read_investment_mw
    # reorders by INVESTABLE_ASSETS rather than trusting CSV row order.
    perm = reverse(1:n)
    df = DataFrame(;
        asset=INVESTABLE_ASSETS[perm],
        solution=Float64.(perm),
        capacity=fill(10.0, n),
    )
    CSV.write(joinpath(dir, "var_assets_investment.csv"), df)
    mw = read_investment_mw(dir)
    @test mw == Float64.(1:n) .* 10.0
end

@testitem "read_investment_mw returns nothing on a missing file or missing column" setup = [InvestmentFixSetup] tags = [:hybrid, :unit] begin
    @test read_investment_mw(mktempdir()) === nothing   # no var_assets_investment.csv

    dir = mktempdir()
    # capacity column absent → cannot compute MW.
    CSV.write(
        joinpath(dir, "var_assets_investment.csv"),
        DataFrame(; asset=INVESTABLE_ASSETS, solution=ones(length(INVESTABLE_ASSETS))),
    )
    @test read_investment_mw(dir) === nothing
end

# --- investment cross-check (full_fixed vs reduced) ------------------------

@testitem "investment_mw_matches flags equality, tolerance, and mismatch" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    a = [100.0, 20000.0, 37500.0]
    @test investment_mw_matches(a, copy(a)).match
    @test investment_mw_matches(a, a .+ 1e-9).match            # within default tol
    mismatch = investment_mw_matches(a, [100.0, 20050.0, 37500.0])
    @test !mismatch.match
    @test isapprox(mismatch.max_abs_diff, 50.0; atol=1e-9)
    @test !investment_mw_matches(a, [1.0, 2.0]).match          # length mismatch
    @test investment_mw_matches(Float64[], Float64[]).match    # trivially equal
end

# --- optimality_gap_percent ------------------------------------------------

@testitem "optimality_gap_percent computes the signed gap and is NaN-safe" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    @test optimality_gap_percent(110.0, 100.0) ≈ 10.0          # fixed worse than benchmark
    @test optimality_gap_percent(90.0, 100.0) ≈ -10.0
    @test optimality_gap_percent(100.0, 100.0) == 0.0
    @test isnan(optimality_gap_percent(NaN, 100.0))            # full_fixed INFEASIBLE
    @test isnan(optimality_gap_percent(110.0, NaN))            # benchmark skipped
    @test isnan(optimality_gap_percent(110.0, 0.0))            # guard against /0
end

# --- stage_complete (resume detection) -------------------------------------

@testitem "stage_complete is true only when every solver wrote results.csv" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    base = mktempdir()
    solvers = [:Gurobi, :HiGHS]
    @test !stage_complete(base, solvers)                       # nothing written yet
    for s in solvers
        d = joinpath(base, string(s))
        mkpath(d)
        write(joinpath(d, "results.csv"), "label\nx\n")
    end
    @test stage_complete(base, solvers)                        # both present
    rm(joinpath(base, "HiGHS", "results.csv"))
    @test !stage_complete(base, solvers)                       # one missing → incomplete
end

# --- read_objective_status (results.csv readback) --------------------------

@testitem "read_objective_status reads objective/status and is NaN-safe on missing" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    obj_missing, status_missing = read_objective_status(joinpath(mktempdir(), "nope.csv"))
    @test isnan(obj_missing)
    @test status_missing == "MISSING"

    dir = mktempdir()
    p = joinpath(dir, "results.csv")
    CSV.write(p, DataFrame(; objective_value=[2.21e7], termination_status=["OPTIMAL"]))
    obj, status = read_objective_status(p)
    @test obj ≈ 2.21e7
    @test status == "OPTIMAL"
end

# --- map_local_to_source / ids_to_str --------------------------------------

@testitem "map_local_to_source and ids_to_str round-trip ids" setup = [HybridSetup] tags = [:hybrid, :unit] begin
    source_ids = [60, 123, 130, 132]      # sorted source ids for a run
    @test map_local_to_source([2, 4], source_ids) == [123, 132]
    @test map_local_to_source(Int[], source_ids) == Int[]
    @test ids_to_str([123, 132]) == "123 132"
    @test ids_to_str(Int[]) == ""
end
