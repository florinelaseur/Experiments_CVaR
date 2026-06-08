# CVaR / tail diagnostics for a solved full-resolution stochastic model.
#
# Tail membership uses the CVaR tail-excess slack ξ_s, which TEM defines from the
# per-scenario TOTAL cost: ξ_s >= total_cost_s - μ, ξ_s >= 0. So `ξ_s > 0` is
# exactly `total_cost_s > μ` (scenario s is in the α-tail). We also export each
# scenario's operational cost. Because total_cost_s = base_cost + operational_s
# (+ other operational terms) and `base_cost` is scenario-independent (first-stage
# investment/fixed costs), ordering scenarios by operational cost differs from the
# CVaR total-cost ordering only by that common base cost. When the CVaR term is
# dropped (single scenario or λ<=0), μ and ξ are unavailable: μ is NaN and no
# scenario is marked in_tail.
#
# The pure `build_tail_diagnostics` depends only on DataFrames (unit-testable).
# `export_cvar_tail_diagnostics` resolves JuMP/TIO/CSV and `table_exists` lazily
# from the includer's scope (as the other ScenarioReduction/src files do).
#
# Scenario IDs in `tail_scenarios.csv` are the LOCAL ids `1..K` of the SOLVED
# subset: `prepare_scenario_subset!` renumbers the selected scenarios to `1..K`
# before the solve. They are NOT the original screening ids, so e.g. an
# `undominated` solve of screening scenarios {2,3,5} produces tail rows with
# scenarios {1,2,3}. Map back via the per-solve `selected` vector if needed.

using DataFrames: DataFrame, combine, groupby, nrow

"""
    build_tail_diagnostics(scenarios, operational_cost, xi, mu; tol=1e-6) -> DataFrame

Assemble per-scenario tail diagnostics from position-aligned vectors. Returns a
DataFrame with columns `scenario`, `operational_cost`, `var_tail_excess_slack_xi`,
`value_at_risk_threshold_mu`, `in_tail`. A scenario is `in_tail` iff its ξ is a
finite value greater than `tol` (equivalently `total_cost > μ`). `NaN` entries in
`xi` (no CVaR term) yield `in_tail = false`.
"""
function build_tail_diagnostics(
    scenarios::AbstractVector,
    operational_cost::AbstractVector,
    xi::AbstractVector,
    mu::Real;
    tol::Real=1e-6,
)
    n = length(scenarios)
    (length(operational_cost) == n && length(xi) == n) ||
        error(
            "build_tail_diagnostics: length mismatch (scenarios=$n, " *
            "operational_cost=$(length(operational_cost)), xi=$(length(xi)))",
        )

    in_tail = Bool[(!isnan(Float64(x)) && Float64(x) > tol) for x in xi]
    return DataFrame(
        scenario=collect(Int.(scenarios)),
        operational_cost=collect(Float64.(operational_cost)),
        var_tail_excess_slack_xi=Float64[Float64(x) for x in xi],
        value_at_risk_threshold_mu=fill(Float64(mu), n),
        in_tail=in_tail,
    )
end

"""
    export_cvar_tail_diagnostics(energy_problem, connection, output_folder) -> DataFrame

Pull μ (`var_value_at_risk_threshold_mu`), per-scenario ξ (`var_tail_excess_slack_xi`),
and per-scenario operational cost (`flows_operational_cost_per_scenario`) from a
solved model, build the tail diagnostics, write `tail_scenarios.csv` into
`output_folder`, and return the DataFrame.

The `scenario` column holds LOCAL subset ids `1..K` (the renumbered ids the model
was solved with), not the original screening ids.
"""
function export_cvar_tail_diagnostics(energy_problem, connection, output_folder)
    costs_expr = energy_problem.expressions[:flows_operational_cost_per_scenario]
    op_df = DataFrame(costs_expr.indices)
    op_df[!, :operational_cost] = JuMP.value.(costs_expr.expressions[:cost])
    op_by_scenario = combine(
        groupby(op_df, :scenario), :operational_cost => sum => :operational_cost,
    )

    scenarios = sort(unique(Int.(op_by_scenario.scenario)))
    op_lookup = Dict(
        Int(r.scenario) => Float64(r.operational_cost) for r in eachrow(op_by_scenario)
    )

    xi_lookup = Dict{Int,Float64}()
    if table_exists(connection, "var_tail_excess_slack_xi")
        xi_df = DataFrame(TIO.get_table(connection, "var_tail_excess_slack_xi"))
        for r in eachrow(xi_df)
            xi_lookup[Int(r.scenario)] = Float64(r.solution)
        end
    end

    mu = NaN
    if table_exists(connection, "var_value_at_risk_threshold_mu")
        mu_df = DataFrame(TIO.get_table(connection, "var_value_at_risk_threshold_mu"))
        if nrow(mu_df) == 1
            mu = only(mu_df.solution)
        elseif nrow(mu_df) > 1
            mu = first(mu_df.solution)
        end
    end

    operational_cost = [get(op_lookup, s, NaN) for s in scenarios]
    xi = [get(xi_lookup, s, NaN) for s in scenarios]

    df = build_tail_diagnostics(scenarios, operational_cost, xi, mu)
    mkpath(output_folder)
    CSV.write(joinpath(output_folder, "tail_scenarios.csv"), df)
    return df
end
