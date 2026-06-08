# Quick audit + fix verification for investment mapping (no full SD loop).
const SCRIPT_DIR = @__DIR__
const REPO_ROOT = joinpath(SCRIPT_DIR, "..")

using Pkg: Pkg
Pkg.activate(REPO_ROOT)
Pkg.instantiate()

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using JuMP: JuMP
using TOML: TOML
using CSV: CSV
using DataFrames
using Random

Random.seed!(19990907)

include(joinpath(REPO_ROOT, "utils", "functions.jl"))
include(joinpath(REPO_ROOT, "utils", "constants.jl"))
include(joinpath(SCRIPT_DIR, "src", "utils.jl"))

const CONFIG = TOML.parsefile(joinpath(REPO_ROOT, "config.toml"))
const INPUT_DATA_PATH = joinpath(REPO_ROOT, CONFIG["simulation"]["input_data"])
const LAMBDA = CONFIG["simulation"]["risk_aversion_weight_lambda"]
const ALPHA = CONFIG["simulation"]["risk_aversion_confidence_level"]
const NUMBER_OF_SCENARIOS = CONFIG["simulation"]["number_of_scenarios"]

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

    return audit
end

prepare_scenario_input!()
connection = setup_connection()
audit = verify_investment_fix_mapping!(connection)

println()
println("=== Investment mapping audit summary ===")
println("permutation_ok (naive zip safe?): ", audit.permutation_ok)
println("dim_ok: ", audit.dim_ok)
println("n_container: ", audit.n_container)
println("mismatches:")
for m in audit.mismatches
    println("  index $(m[1]): expected=$(m[2]) container=$(m[3])")
end
println("Aligned fix verification: PASSED")
