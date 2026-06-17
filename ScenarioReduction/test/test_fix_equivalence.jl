using TestItems: @testitem

# These tests pin the numeric contract BETWEEN the two variable-fixing entry points:
#
#   fix_variables_from_solution!(benchmark, reduced, sym)   (utils/functions.jl)
#       copies JuMP.value(reduced container) onto the benchmark container by
#       positional zip — model units, container order, no conversion.
#
#   fix_variables_from_sample(variables, sym, sample; capacity_lookup)  (src/utils.jl)
#       for :assets_investment treats `sample` as MW in INVESTABLE_ASSETS order,
#       then reorders to container order and divides by capacity (MW -> model units);
#       for every other symbol it fixes `sample` directly in container order.
#
# So they agree iff the sample fed to the second function is the solved model-unit
# values converted to MW in INVESTABLE_ASSETS order:
#   :assets_investment        -> sample[a] = value(container_of a) * capacity[a]
#   :assets_investment_energy -> sample    = value.(container) (identity copy)
# RIDM permutes container order vs INVESTABLE_ASSETS, so the reorder is load-bearing.
#
# Each @testitem runs in its own isolated module, so the capacities / asset order are
# redefined inline per test. Non-unit, distinct capacities are used on purpose so the
# MW <-> model-unit conversion and the reorder actually matter (a no-op conversion
# would mask ordering/scaling bugs).

@testitem "investment_mw_from_solution returns MW in INVESTABLE_ASSETS order" setup = [
    FixEquivalenceSetup,
] tags = [:fix_equivalence, :unit] begin
    caps = Dict(
        "ccgt" => 0.8, "ocgt" => 0.1, "solar" => 0.5, "wind" => 0.4,
        "wind_offshore" => 0.4, "electrolizer" => 0.1, "battery" => 0.05,
    )
    asset_order = ["ccgt", "wind", "solar", "ocgt", "electrolizer", "wind_offshore", "battery"]
    model_units = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]   # solved value per container position

    m = build_solved_mock(asset_order, model_units)
    mw = investment_mw_from_solution(m.reduced.variables, caps)

    @test length(mw) == length(INVESTABLE_ASSETS)
    pos = Dict(a => i for (i, a) in enumerate(asset_order))   # asset -> container position
    for (k, a) in enumerate(INVESTABLE_ASSETS)
        @test mw[k] ≈ model_units[pos[a]] * caps[a]
    end
end

@testitem "fix paths agree for :assets_investment with RIDM-permuted indices" setup = [
    FixEquivalenceSetup,
] tags = [:fix_equivalence, :unit] begin
    caps = Dict(
        "ccgt" => 0.8, "ocgt" => 0.1, "solar" => 0.5, "wind" => 0.4,
        "wind_offshore" => 0.4, "electrolizer" => 0.1, "battery" => 0.05,
    )
    asset_order = ["ccgt", "wind", "solar", "ocgt", "electrolizer", "wind_offshore", "battery"]
    model_units = Float64[12.5, 4.0, 33.0, 9.0, 1.5, 7.0, 88.0]

    m = build_solved_mock(asset_order, model_units)
    r = assert_fix_paths_equivalent!(
        m.benchmark, m.reduced, :assets_investment; capacity_lookup=caps,
    )

    @test r.equivalent
    @test r.max_abs_diff ≤ 1e-8
    # Solution path copies solved model units verbatim, in container order.
    @test r.via_solution ≈ model_units
    @test r.via_sample ≈ model_units
end

@testitem "fix paths agree for :assets_investment in INVESTABLE_ASSETS order" setup = [
    FixEquivalenceSetup,
] tags = [:fix_equivalence, :unit] begin
    caps = Dict(
        "ccgt" => 0.8, "ocgt" => 0.1, "solar" => 0.5, "wind" => 0.4,
        "wind_offshore" => 0.4, "electrolizer" => 0.1, "battery" => 0.05,
    )
    asset_order = copy(INVESTABLE_ASSETS)
    model_units = Float64[2.0, 11.0, 0.5, 7.5, 3.0, 40.0, 100.0]

    m = build_solved_mock(asset_order, model_units)
    r = assert_fix_paths_equivalent!(
        m.benchmark, m.reduced, :assets_investment; capacity_lookup=caps,
    )

    @test r.equivalent
    @test r.max_abs_diff ≤ 1e-8
    @test r.via_solution ≈ model_units
end

@testitem "fix paths agree for :assets_investment_energy (identity copy)" setup = [
    FixEquivalenceSetup,
] tags = [:fix_equivalence, :unit] begin
    # Energy investments are fixed directly in container order with no capacity
    # conversion, so asset names/order are irrelevant to the contract here.
    asset_order = ["battery", "electrolizer", "phs"]
    model_units = Float64[10.0, 25.5, 3.0]

    m = build_solved_mock(asset_order, model_units; var_symbol=:assets_investment_energy)
    r = assert_fix_paths_equivalent!(m.benchmark, m.reduced, :assets_investment_energy)

    @test r.equivalent
    @test r.max_abs_diff ≤ 1e-8
    @test r.via_solution ≈ model_units
    @test r.via_sample ≈ model_units
end

@testitem "raw model-unit sample is NOT equivalent for :assets_investment" setup = [
    FixEquivalenceSetup,
] tags = [:fix_equivalence, :unit] begin
    # Negative control: prove the MW conversion + reorder are required. Feeding the
    # raw solved model-unit values (container order) to fix_variables_from_sample as
    # if they were MW in INVESTABLE_ASSETS order divides them by capacity (and
    # mis-keys order), so the fixed values must differ from the solution path.
    caps = Dict(
        "ccgt" => 0.8, "ocgt" => 0.1, "solar" => 0.5, "wind" => 0.4,
        "wind_offshore" => 0.4, "electrolizer" => 0.1, "battery" => 0.05,
    )
    asset_order = ["ccgt", "wind", "solar", "ocgt", "electrolizer", "wind_offshore", "battery"]
    model_units = Float64[12.5, 4.0, 33.0, 9.0, 1.5, 7.0, 88.0]

    m = build_solved_mock(asset_order, model_units)
    container = m.benchmark.variables[:assets_investment].container

    fix_variables_from_solution!(m.benchmark, m.reduced, :assets_investment)
    via_solution = Float64[JuMP.fix_value(v) for v in container]
    for v in container
        JuMP.unfix(v)
    end

    raw = collect(JuMP.value.(m.reduced.variables[:assets_investment].container))
    fix_variables_from_sample(m.benchmark.variables, :assets_investment, raw; capacity_lookup=caps)
    via_sample_raw = Float64[JuMP.fix_value(v) for v in container]

    @test !all(isapprox.(via_solution, via_sample_raw; atol=1e-8))
    @test maximum(abs.(via_solution .- via_sample_raw)) > 1e-8
end
