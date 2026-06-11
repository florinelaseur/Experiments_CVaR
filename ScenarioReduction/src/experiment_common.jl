# Shared experiment-driver helpers (input preparation + screening connection),
# used by test_stochastic_dominance.jl and test_dominance_kantorovich.jl.
#
# Needs from the includer's scope: ScenarioReductionConfig (src/config.jl, loaded
# via src/dominance.jl — include this file AFTER src/dominance.jl), get_scenario_set
# (utils/functions.jl), and CSV/DataFrames/DuckDB/TIO/TC.

# Prepare the data for the experiment. Mimics the main.jl script.
function prepare_scenario_input!(cfg::ScenarioReductionConfig)
    all_profiles_df = CSV.read(cfg.profiles_wide_source, DataFrame)
    profiles_df = get_scenario_set(all_profiles_df, cfg.number_of_scenarios)
    # Original source ids picked for this seed, captured BEFORE renumbering to 1..N.
    source_ids = sort(unique(profiles_df.scenario))
    mapping = Dict(old => new for (new, old) in enumerate(source_ids))
    profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]

    CSV.write(joinpath(cfg.input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

    df_stochastic_scenario = DataFrame(;
        scenario=sort(unique(profiles_df.scenario)),
        probability=fill(1.0 / cfg.number_of_scenarios, cfg.number_of_scenarios),
    )
    CSV.write(joinpath(cfg.input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

    return source_ids
end

#connection and profiles preparation
function setup_connection(cfg::ScenarioReductionConfig)
    connection = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection, cfg.input_data_path)

    DuckDB.query(
        connection,
        """
        UPDATE model_parameters
        SET
            risk_aversion_weight_lambda = $(cfg.lambda),
            risk_aversion_confidence_level_alpha = $(cfg.alpha);
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
