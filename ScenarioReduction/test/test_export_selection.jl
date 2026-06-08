using TestItems: @testitem

@testitem "select_export_tables drops big vars and non-tail cons" setup = [
    ExportSelectionSetup,
] tags = [:export, :unit] begin
    tables = [
        "var_assets_investment",
        "var_flow",
        "var_storage_level_rep_period",
        "var_tail_excess_slack_xi",
        "var_value_at_risk_threshold_mu",
        "cons_balance_consumer",
        "cons_balance_conversion",
        "cons_scenario_tail_excess",
        "obj_breakdown",
    ]

    kept = select_export_tables(
        tables;
        exclude_vars=["var_flow", "var_storage_level_rep_period"],
        keep_constraints_only=["cons_scenario_tail_excess"],
    )

    @test "var_flow" ∉ kept
    @test "var_storage_level_rep_period" ∉ kept
    @test "var_assets_investment" ∈ kept
    @test "var_tail_excess_slack_xi" ∈ kept
    @test "var_value_at_risk_threshold_mu" ∈ kept
    # Only the tail-excess constraint survives among cons_*.
    @test "cons_scenario_tail_excess" ∈ kept
    @test "cons_balance_consumer" ∉ kept
    @test "cons_balance_conversion" ∉ kept
    # obj_* and other prefixes are always kept.
    @test "obj_breakdown" ∈ kept
    # Order is preserved.
    @test kept == [
        "var_assets_investment",
        "var_tail_excess_slack_xi",
        "var_value_at_risk_threshold_mu",
        "cons_scenario_tail_excess",
        "obj_breakdown",
    ]
end

@testitem "select_export_tables defaults keep all vars, drop all cons" setup = [
    ExportSelectionSetup,
] tags = [:export, :unit] begin
    tables = ["var_a", "var_b", "cons_x", "obj_y"]
    # Default kwargs: no excluded vars, no kept constraints.
    kept = select_export_tables(tables)
    @test kept == ["var_a", "var_b", "obj_y"]
end

@testitem "select_export_tables keeps multiple constraints when allowed" setup = [
    ExportSelectionSetup,
] tags = [:export, :unit] begin
    tables = ["cons_scenario_tail_excess", "cons_other", "cons_keep_me"]
    kept = select_export_tables(
        tables; keep_constraints_only=["cons_scenario_tail_excess", "cons_keep_me"],
    )
    @test kept == ["cons_scenario_tail_excess", "cons_keep_me"]
end
