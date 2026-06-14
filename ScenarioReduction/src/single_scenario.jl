# Build one single-scenario operational model for SD per-scenario evaluation.
#
# Phase B of the stochastic-dominance screen evaluates each investment sample
# against EVERY selected scenario separately (cost matrix C_i(z_k)). Rather than
# one multi-scenario model, we build one single-scenario model at a time, solve
# all samples against it, then free it — so only one model lives in memory.
#
# Like src/utils.jl, this file references TEM/TC/TIO/DuckDB/JuMP from the
# includer's scope (they resolve lazily at call time), and `configure_for_warmstart!`
# from dominance.jl (defined before this is ever called). So do NOT add
# `import`/`using` for those here.

using DataFrames: DataFrame, nrow

"""
    build_single_scenario_model(scenario_id, input_data_path,
                                optimizer, optimizer_parameters; solver=:Gurobi)

Build a fresh, full-hourly operational model restricted to a single scenario.

Reads the input folder, keeps only `scenario_id`'s profile rows (renumbered to 1),
sets a degenerate `stochastic_scenario` (one scenario, probability 1), runs the
low-level TEM build pipeline, attaches the optimizer (silent, warm-start configured),
and returns everything needed to fix investments and solve.

Risk parameters are intentionally NOT set: TEM drops the entire CVaR term when
`n_scenarios <= 1` (see TEM `objectives/conditional_value_at_risk_term.jl`), so
`risk_aversion_*` are dead values here. With one scenario at probability 1.0 the
objective is just that scenario's total cost — exactly the `C_i(z_k)` the cost matrix
should hold.

Returns a NamedTuple `(connection, model, variables, capacity_lookup, scenario)`.
Close `connection` (and drop the model) when done to reclaim memory before building
the next scenario's model.
"""
function build_single_scenario_model(
    scenario_id::Integer,
    input_data_path::AbstractString,
    optimizer,
    optimizer_parameters;
    solver::Symbol=:Gurobi,
)
    conn = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(conn, input_data_path)

    # Keep only this scenario's profiles, renumber to 1 (single-scenario model).
    # The folder's profiles-wide.csv already uses the renumbered 1..N ids that the
    # adequacy cuts were built from, so no original-id mapping is needed.
    profiles_wide = DataFrame(TIO.get_table(conn, "profiles_wide"))
    scenario_profiles = filter(row -> row.scenario == scenario_id, profiles_wide)
    nrow(scenario_profiles) > 0 || error("No profile rows for scenario $scenario_id")
    scenario_profiles[!, :scenario] .= 1
    DuckDB.query(conn, "DROP TABLE IF EXISTS profiles_wide")
    DuckDB.register_table(conn, scenario_profiles, "profiles_wide")

    DuckDB.query(
        conn,
        """
        CREATE OR REPLACE TABLE stochastic_scenario AS
        SELECT 1 AS scenario, 1.0 AS probability
        """,
    )

    TC.transform_wide_to_long!(
        conn,
        "profiles_wide",
        "profiles";
        exclude_columns=["scenario", "milestone_year", "timestep"],
    )
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    TC.dummy_cluster!(conn; layout=layout)
    TEM.populate_with_defaults!(conn)
    DuckDB.query(conn, "UPDATE asset SET is_seasonal = false")
    TEM.create_internal_tables!(conn)

    variables   = TEM.compute_variables_indices(conn)
    constraints = TEM.compute_constraints_indices(conn)
    profiles    = TEM.prepare_profiles_structure(conn)
    model, _ = TEM.create_model(conn, variables, constraints, profiles)

    JuMP.set_optimizer(model, optimizer)
    JuMP.set_optimizer_attributes(model, [pair for pair in optimizer_parameters]...)
    JuMP.set_silent(model)
    configure_for_warmstart!(model; solver=solver)

    capacity_lookup = build_capacity_lookup(conn)
    return (
        connection=conn,
        model=model,
        variables=variables,
        capacity_lookup=capacity_lookup,
        scenario=Int(scenario_id),
    )
end

# ─────────────────────────────────────────────────────────────────────
# LP warm-start theory — why dual simplex + no presolve
# ─────────────────────────────────────────────────────────────────────
#
# The operational subproblem (investments fixed) is a standard LP:
#
#   PRIMAL                          DUAL
#   min  cᵀx                       max  bᵀy
#   s.t. Ax = b     (constraints)  s.t. Aᵀy + s = c   (reduced costs)
#        l ≤ x ≤ u  (bounds)            s ≥ 0 for x_i at lower bound
#                                       s ≤ 0 for x_i at upper bound
#                                       s free for basic x_i
#
# A "basis" B is a partition of variables into basic / non-basic.
# A basis is:
#   • primal feasible  ⟺  Ax = b  and  l ≤ x ≤ u   (constraints + bounds met)
#   • dual feasible    ⟺  reduced costs s = c − Aᵀy  have correct signs
#                         (i.e., no non-basic variable can improve the objective)
#   • optimal          ⟺  both
#
# When we call JuMP.fix(assets_investment[i], new_val), we change the
# BOUNDS l_i, u_i (and effectively the RHS b via the fixed columns).
#
# Effect on the current basis:
#   • Dual feasibility:  s = c − Aᵀy  does NOT depend on bounds/RHS.
#     → The basis stays DUAL FEASIBLE after bound changes.
#   • Primal feasibility: Ax = b and l ≤ x ≤ u may be violated.
#     → The basis is generally PRIMAL INFEASIBLE.
#
# PRIMAL simplex starts from a primal-feasible basis and pivots to
# restore optimality (dual feasibility). After bound changes the basis
# is dual-feasible but primal-infeasible → wrong starting point.
#
# DUAL simplex starts from a dual-feasible basis and pivots to restore
# primal feasibility. After bound changes the basis is exactly this
# → ideal starting point. Only the few violated bound constraints need
# to be repaired — typically very few pivots.
#
# PRESOLVE transforms the model into a smaller equivalent form. This
# destroys the basis mapping (variables are eliminated / substituted),
# so the warm-start basis cannot be reused. We disable it after the
# first solve to preserve the basis across re-solves.
#
# Net effect: first solve is a cold start (presolve ON, any method).
# Every subsequent re-solve is a warm-started dual simplex — only a
# handful of pivots to repair the 7 changed investment bounds, vs.
# tens of thousands of variables in the full model.
# ─────────────────────────────────────────────────────────────────────



function configure_for_warmstart!(model; solver::Symbol=:Gurobi)
    # Use simplex, not interior-point — simplex maintains a basis reusable across re-solves.
    # Dual simplex: after fix() changes bounds, the current basis stays dual-feasible.
    if solver == :Gurobi
        JuMP.set_optimizer_attribute(model, "Method", 1)  # dual simplex
    elseif solver == :HiGHS
        JuMP.set_optimizer_attribute(model, "solver", "simplex")
        JuMP.set_optimizer_attribute(model, "simplex_strategy", 1)  # dual simplex
    end
    return nothing
end