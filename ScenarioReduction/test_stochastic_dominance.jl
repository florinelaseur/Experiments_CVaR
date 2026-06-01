# Minimal runner for stochastic-dominance scenario screening.
#
# REPL usage (from repo root — path is stable across re-includes):
#   include(joinpath(@__DIR__, "ScenarioReduction", "test_stochastic_dominance.jl"))
#   run!()
#
# Or run as a script:
#   julia --project=. ScenarioReduction/test_stochastic_dominance.jl

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = joinpath(SCRIPT_DIR, "..")
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "outputs")
const REPRESENTATIVE_PERIODS = 30
const RUN_FILTER = true
const RUN_SOLVE = false
const RUN_VERIFY_MAPPING = false

using Pkg: Pkg
Pkg.activate(REPO_ROOT)
Pkg.instantiate()
# The project's asset.csv uses the OLD TEM schema (storage_method_energy as a string
# enum); registry v0.21.0 expects a BOOLEAN and crashes in populate_with_defaults!.
# Pin the same git rev main.jl uses so this runner works without running main.jl first.
Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev="227a80f7907e2c7178edb0697874cfb6666ad644")

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using Distances: Distances
using CSV: CSV
using JuMP: JuMP
using TOML: TOML
using Random
using DataFrames

Random.seed!(19990907)

include(joinpath(REPO_ROOT, "utils", "functions.jl"))
include(joinpath(REPO_ROOT, "utils", "constants.jl"))
include(joinpath(SCRIPT_DIR, "src", "utils.jl"))
include(joinpath(SCRIPT_DIR, "src", "stochastic_dominance.jl"))

const CONFIG = TOML.parsefile(joinpath(REPO_ROOT, "config.toml"))
const INPUT_DATA_PATH = joinpath(REPO_ROOT, CONFIG["simulation"]["input_data"])
const SOLVERS = [Symbol(s) for s in CONFIG["simulation"]["solvers"]]
const LAMBDA = CONFIG["simulation"]["risk_aversion_weight_lambda"]
const ALPHA = CONFIG["simulation"]["risk_aversion_confidence_level"]
const NUMBER_OF_SCENARIOS = CONFIG["simulation"]["number_of_scenarios"]

const INV_COV = [
    3.78479695199275e7 2.749005459752808e6 -3.4577296873559463e6 2.7418219047064386e7 298344.2313214293 5.651907423141254 407842.3222137223
    2.749005459752808e6 1.4501887858885615e6 1.7561773638240807e6 9.315492633408496e6 -13717.616779138572 0.9189337918286634 -374638.69889195403
    -3.4577296873559463e6 1.7561773638240807e6 2.854556209827236e7 1.0685293361151338e7 -169352.5029630032 2.6711278243126007 635237.786307628
    2.7418219047064386e7 9.315492633408496e6 1.0685293361151338e7 1.7639204795704246e8 75078.97973361365 5.368008683366009 7.997575021518203e6
    298344.2313214293 -13717.616779138572 -169352.5029630032 75078.97973361365 151423.08336402554 0.04797874928315783 79097.28020948995
    5.651907423141254 0.9189337918286634 2.6711278243126007 5.368008683366009 0.04797874928315783 2.691847694694756e-5 -0.3741512568820247
    407842.3222137223 -374638.69889195403 635237.786307628 7.997575021518203e6 79097.28020948995 -0.3741512568820247 3.7310673190848944e6
]

function prepare_scenario_input!()
    profiles_path = joinpath(REPO_ROOT, "create-scenarios", "profiles-wide-all-scenarios.csv")
    all_profiles_df = CSV.read(profiles_path, DataFrame)
    profiles_df = get_scenario_set(all_profiles_df, NUMBER_OF_SCENARIOS)
    mapping = Dict(old => new for (new, old) in enumerate(sort(unique(profiles_df.scenario))))
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    CSV.write(joinpath(INPUT_DATA_PATH, "profiles-wide.csv"), profiles_df; writeheader=true)

    df_stochastic_scenario = DataFrame(;
        scenario=sort(unique(profiles_df.scenario)),
        probability=fill(1.0 / NUMBER_OF_SCENARIOS, NUMBER_OF_SCENARIOS),
    )
    CSV.write(joinpath(INPUT_DATA_PATH, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

    return nothing
end

function setup_connection()
    connection = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection, INPUT_DATA_PATH)

    DuckDB.query(
        connection,
        """
        UPDATE model_parameters
        SET
            risk_aversion_weight_lambda = $(LAMBDA),
            risk_aversion_confidence_level_alpha = $(ALPHA);
        """,
    )

    TC.transform_wide_to_long!(
        connection,
        "profiles_wide",
        "profiles";
        exclude_columns=["scenario", "milestone_year", "timestep"],
    )
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )

    return connection
end

function prepare_stochastic_dominance_indices!(connection)
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    TC.dummy_cluster!(connection; layout=layout)
    TEM.populate_with_defaults!(connection)
    TEM.create_internal_tables!(connection)
    return TEM.compute_variables_indices(connection)
end

"""
    verify_investment_fix_mapping!(connection; tol=1e-8)

Integration smoke test: fix sentinel MW values per asset and read back model-unit
fix values via the indices-aligned path. Returns the audit result.
"""
function verify_investment_fix_mapping!(connection; tol=1e-8)
    variables = prepare_stochastic_dominance_indices!(connection)
    capacity_lookup = build_capacity_lookup(connection)
    audit = audit_investment_mapping(variables; capacity_lookup)

    constraints = TEM.compute_constraints_indices(connection)
    profiles = TEM.prepare_profiles_structure(connection)
    model, _ = TEM.create_model(connection, variables, constraints, profiles)
    JuMP.set_optimizer(model, HiGHS.Optimizer)
    JuMP.set_silent(model)

    sample_mw = Float64[1000, 2000, 3000, 4000, 5000, 6000, 7000]
    fix_variables_from_sample(
        variables,
        :assets_investment,
        sample_mw;
        capacity_lookup,
    )

    inv_df = DataFrame(variables[:assets_investment].indices)
    container = variables[:assets_investment].container
    sample_by_asset = sample_vector_by_asset(sample_mw)

    for (i, row) in enumerate(eachrow(inv_df))
        asset = string(row.asset)
        expected = sample_by_asset[asset] / capacity_lookup[asset]
        actual = JuMP.fix_value(container[i])
        @info "Fix verification" index=i asset=asset expected=expected actual=actual
        abs(actual - expected) > tol &&
            error("Fix mismatch at index $i ($asset): got $actual, expected $expected")
    end

    @info "Investment fix mapping verification passed" (
        permutation_ok=audit.permutation_ok,
        n_mismatches=length(audit.mismatches),
    )
    return audit
end

function run_filter!(connection)
    @info "Running stochastic-dominance filtering"
    bounds = load_investment_bounds()
    cov = investment_covariance()
    #cov = INV_COV
    stochastic_dominance(connection; bounds, covariance=cov, input_data_path=INPUT_DATA_PATH)
    return bounds, cov
end

function run_solve!(connection)
    @info "Clustering with $(REPRESENTATIVE_PERIODS) representative periods (per scenario)"
    layout = TC.ProfilesTableLayout(;
        year=:milestone_year,
        cols_to_groupby=[:milestone_year, :scenario],
    )
    clustering_kwargs = Dict(:learning_rate => 0.001, :niters => 2000)
    weight_fitting_kwargs = Dict(:learning_rate => 0.001, :niters => 2000)

    # Swap TC.dummy_cluster!(connection; layout=layout) for a full-hourly smoke test.
    time_to_cluster = @elapsed TC.cluster!(
        connection,
        24,
        REPRESENTATIVE_PERIODS;
        method=:convex_hull,
        distance=Distances.Euclidean(),
        weight_type=:dirac,
        layout=layout,
        clustering_kwargs,
        weight_fitting_kwargs,
    )

    TEM.populate_with_defaults!(connection)
    DuckDB.query(connection, "UPDATE asset SET is_seasonal = false")

    results = DataFrame(;
        base_name=String[],
        rp=Int[],
        solver=Symbol[],
        time_to_cluster=Float64[],
        time_to_read=Float64[],
        time_to_create=Float64[],
        time_to_solve=Float64[],
        time_to_save=Float64[],
        objective_value=Float64[],
        termination_status=String[],
        num_constraints=Int[],
        num_variables=Int[],
        value_at_risk_threshold_mu=Float64[],
    )

    for solver in SOLVERS
        optimizer, parameters = get_solver_parameters(solver)

        @info "Creating model (stochastic_dominance, rp=$(REPRESENTATIVE_PERIODS)) with $solver"
        time_to_read = @elapsed energy_problem = TEM.EnergyProblem(connection)
        time_to_create = @elapsed TEM.create_model!(
            energy_problem;
            optimizer=optimizer,
            optimizer_parameters=parameters,
            model_file_name="",
            enable_names=true,
        )

        output_folder = joinpath(OUTPUT_DIR, "stochastic_dominance", string(solver))
        mkpath(output_folder)

        @info "Solving model with $solver"
        time_to_solve = @elapsed TEM.solve_model!(energy_problem)
        time_to_save = @elapsed begin
            TEM.save_solution!(energy_problem)
            TEM.export_solution_to_csv_files(output_folder, energy_problem)
        end

        mu_value_df = TIO.get_table(connection, "var_value_at_risk_threshold_mu")
        mu_value = only(mu_value_df.solution)

        push!(
            results,
            (
                "stochastic_dominance",
                REPRESENTATIVE_PERIODS,
                solver,
                time_to_cluster,
                time_to_read,
                time_to_create,
                time_to_solve,
                time_to_save,
                energy_problem.objective_value,
                string(energy_problem.termination_status),
                JuMP.num_constraints(energy_problem.model; count_variable_in_set_constraints=false),
                JuMP.num_variables(energy_problem.model),
                mu_value,
            ),
        )
    end

    return results
end

function run!()
    prepare_scenario_input!()
    connection = setup_connection()

    if RUN_VERIFY_MAPPING
        verify_investment_fix_mapping!(connection)
    end

    if RUN_FILTER
        run_filter!(connection)
    end

    if RUN_SOLVE
        results = run_solve!(connection)
        mkpath(OUTPUT_DIR)
        CSV.write(joinpath(OUTPUT_DIR, "results.csv"), results; writeheader=true)
        @info "Results saved to $(joinpath(OUTPUT_DIR, "results.csv"))"
    end

    return connection
end

if abspath(PROGRAM_FILE) == @__FILE__
    run!()
end
