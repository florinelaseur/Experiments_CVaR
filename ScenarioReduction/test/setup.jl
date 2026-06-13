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

@testmodule InvestmentFixSetup begin
    using JuMP: JuMP
    using CSV: CSV
    using DataFrames: DataFrame, DataFrames

    include(joinpath(@__DIR__, "..", "src", "investment_mapping.jl"))
    include(joinpath(@__DIR__, "..", "src", "utils.jl"))

    export JuMP,
           CSV,
           INVESTABLE_ASSETS,
           DataFrame,
           fix_variables_from_sample,
           read_investment_mw
end

# Solver-free helpers for the hybrid dominance + Kantorovich driver:
# select_kantorovich_scenarios (utils/kantorovich_reduction.jl) + the pure
# id/resume/gap helpers (src/hybrid_helpers.jl). Neither file uses DuckDB/TEM at
# top level, so this loads without a solver.
@testmodule HybridSetup begin
    using DataFrames: DataFrame, DataFrames, nrow
    using CSV: CSV

    include(joinpath(@__DIR__, "..", "..", "utils", "kantorovich_reduction.jl"))
    include(joinpath(@__DIR__, "..", "src", "hybrid_helpers.jl"))

    # Wide profiles_df fixture: one row per (scenario, timestep) with the 5
    # profile columns build_scenario_matrix expects, distinct per scenario.
    function _toy_profiles(n_scenarios, n_timesteps)
        rows = NamedTuple[]
        for s in 1:n_scenarios, t in 1:n_timesteps
            push!(rows, (
                milestone_year=2030,
                scenario=s,
                timestep=t,
                solar=1.0 * s + 0.1t,
                wind_offshore=2.0 * s - 0.2t,
                wind_onshore=0.5 * s + 0.3t,
                demand=10.0 * s - t,
                hydro_inflow=1.0 * s + 0.1t,
            ))
        end
        return DataFrame(rows)
    end

    export DataFrame, DataFrames, nrow, CSV,
           _toy_profiles,
           select_kantorovich_scenarios,
           build_scenario_matrix,
           compute_cost_matrix,
           kantorovich_forward_select,
           ids_to_str,
           map_local_to_source,
           stage_complete,
           read_objective_status,
           optimality_gap_percent,
           investment_mw_matches
end

@testmodule ScenarioDominanceSetup begin
    using DataFrames: DataFrame, DataFrames

    include(joinpath(@__DIR__, "..", "src", "scenario_dominance.jl"))

    export DataFrame,
           dominating_scenarios,
           fsd_dominating_scenarios,
           ssd_dominating_scenarios,
           dominance_analysis,
           dominator_scenarios,
           save_scenario_dominance_csv,
           undominated_scenarios,
           pick_n_scenarios,
           _compare_cost,
           _column_dominates,
           _finite_cost_domain,
           _empirical_cdf_matrix,
           _integrated_cdf_matrix
end

@testmodule CvarDiagnosticsSetup begin
    using DataFrames: DataFrame, DataFrames

    # cvar_diagnostics.jl's pure `build_tail_diagnostics` needs only DataFrames.
    # `export_cvar_tail_diagnostics` references JuMP/TIO/CSV/table_exists, which are
    # NOT called here, so they resolve lazily and the include is solver-free.
    include(joinpath(@__DIR__, "..", "src", "cvar_diagnostics.jl"))

    export DataFrame, build_tail_diagnostics
end

@testmodule ExportSelectionSetup begin
    using DataFrames: DataFrame, DataFrames

    # solve_scenarios.jl only needs DataFrames at include time; TEM/TC/TIO/DuckDB/CSV
    # are referenced lazily inside functions we do NOT call here. The pure
    # `select_export_tables` uses only string operations.
    include(joinpath(@__DIR__, "..", "src", "solve_scenarios.jl"))

    export select_export_tables, format_selected_scenarios
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

@testmodule ConflictLogSetup begin
    using JuMP: JuMP
    using HiGHS: HiGHS
    using JSON: JSON

    include(joinpath(@__DIR__, "..", "utils", "infeasibility_conflict.jl"))
    include(joinpath(@__DIR__, "..", "src", "investment_mapping.jl"))
    include(joinpath(@__DIR__, "..", "src", "conflict_log.jl"))

    export JuMP,
           HiGHS,
           INVESTABLE_ASSETS,
           collect_infeasibility_conflict,
           append_infeasibility_conflict_record!,
           sample_vector_by_asset,
           JSON
end
