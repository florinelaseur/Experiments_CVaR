using TestItems: @testitem

@testitem "build_tail_diagnostics marks tail by xi > tol" setup = [CvarDiagnosticsSetup] tags = [:cvar, :unit] begin
    scenarios = [1, 2, 3]
    operational_cost = [100.0, 200.0, 300.0]
    xi = [0.0, 50.0, 0.0]
    mu = 250.0

    df = build_tail_diagnostics(scenarios, operational_cost, xi, mu)
    @test df.scenario == [1, 2, 3]
    @test df.operational_cost == operational_cost
    @test df.var_tail_excess_slack_xi == xi
    @test all(df.value_at_risk_threshold_mu .== 250.0)
    @test df.in_tail == [false, true, false]
end

@testitem "build_tail_diagnostics treats NaN xi as not in tail" setup = [CvarDiagnosticsSetup] tags = [:cvar, :unit] begin
    # No CVaR term: xi unavailable (NaN), mu NaN. Nothing should be in tail.
    scenarios = [1, 2]
    df = build_tail_diagnostics(scenarios, [10.0, 20.0], [NaN, NaN], NaN)
    @test df.in_tail == [false, false]
    @test all(isnan, df.value_at_risk_threshold_mu)
end

@testitem "build_tail_diagnostics tol boundary" setup = [CvarDiagnosticsSetup] tags = [:cvar, :unit] begin
    scenarios = [1, 2, 3]
    # exactly at tol => not in tail; just above => in tail
    df = build_tail_diagnostics(scenarios, [0.0, 0.0, 0.0], [1e-6, 2e-6, 0.0], 0.0; tol=1e-6)
    @test df.in_tail == [false, true, false]
end

@testitem "build_tail_diagnostics length mismatch errors" setup = [CvarDiagnosticsSetup] tags = [:cvar, :unit] begin
    @test_throws ErrorException build_tail_diagnostics([1, 2], [1.0], [0.0, 0.0], 1.0)
    @test_throws ErrorException build_tail_diagnostics([1, 2], [1.0, 2.0], [0.0], 1.0)
end
