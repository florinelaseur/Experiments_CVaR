using DataFrames
using CSV

cd(@__DIR__)


function prepare_demand_data()

    # ============================================================
    # PATHS
    # ============================================================

    # ERAA demand time series
    demand_data_folder = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
        "ERAA",
        "DemandData",
        "DemandTimeseries",
    )

    # TYNDP model input data
    # input_data_folder_TYNDP = joinpath(
    #     homedir(),
    #     "Nextcloud",
    #     "ExperimentData",
    #     "EU-input-data",
    #     "TYNDP",
    #     "Outputs",
    #     "tulipa_input_north_sea_2026_2035_de",
    # )

    input_data_folder_TYNDP = joinpath(
        homedir(),
        "Server",
        "tulipa_input_north_sea_2026_2035_de_bnl"
    )

    profiles_file = joinpath(
        input_data_folder_TYNDP,
        "assets-profiles.csv",
    )

    df_profiles = CSV.read(
        profiles_file,
        DataFrame,
    )


    # ============================================================
    # SETTINGS
    # ============================================================

    n_timesteps = 8760
    n_scenarios = 5

    # Use all 36 ERAA weather scenarios to determine one common
    # normalization factor for each market node.
    scenario_cols = [
        "WS" * lpad(string(j), 2, '0')
        for j in 1:36
    ]

    profiles_list_full = DataFrame[]


    # ============================================================
    # FIND REQUIRED ELECTRICITY-DEMAND PROFILES
    # ============================================================

    # assets-profiles.csv determines which demand profiles the
    # TYNDP model actually requires.
    demand_profiles = filter(
        row ->
            row.profile_type == "demand" &&
                occursin("_E_Demand", String(row.asset)),
        df_profiles,
    )

    if isempty(demand_profiles)
        error(
            """
            No electricity-demand profiles were found in:
            $profiles_file
            """
        )
    end


    # ============================================================
    # CREATE DEMAND PROFILE FOR EACH TYNDP MARKET NODE
    # ============================================================

    for row in eachrow(demand_profiles)

        asset = String(row.asset)
        profile_name = String(row.profile_name)

        # Example:
        #
        # DKE1_E_Demand_2035 -> DKE1
        # NL00_E_Demand_2035 -> NL00
        zone = first(
            split(
                asset,
                "_E_Demand",
            )
        )


        # --------------------------------------------------------
        # READ ERAA DEMAND
        # --------------------------------------------------------

        file = joinpath(
            demand_data_folder,
            "$(zone)_Demand_total_2035_National Trends.csv",
        )

        if !isfile(file)
            error(
                """
                ERAA demand file does not exist for
                TYNDP asset $asset:

                $file
                """
            )
        end


        demand_data = CSV.read(
            file,
            DataFrame,
        )


        # --------------------------------------------------------
        # CHECK ERAA SCENARIOS
        # --------------------------------------------------------

        missing_scenario_cols = [
            ws
            for ws in scenario_cols
            if Symbol(ws) ∉ propertynames(demand_data)
        ]

        if !isempty(missing_scenario_cols)
            error(
                """
                Missing ERAA scenario columns in:
                $(basename(file))

                Missing:
                $(join(missing_scenario_cols, ", "))
                """
            )
        end


        n_rows = nrow(demand_data)

        @assert n_rows == n_timesteps """
        Expected $n_timesteps demand timesteps for $zone,
        but found $n_rows in:

        $file
        """


        # ========================================================
        # NORMALIZE DEMAND
        # ========================================================

        # One common normalization factor for this market node,
        # calculated over:
        #
        #   8760 hours × 36 ERAA weather scenarios
        #
        # This preserves the relative demand levels between the
        # weather scenarios.

        max_val = maximum(
            maximum(
                demand_data[!, Symbol(ws)]
            )
            for ws in scenario_cols
        )

        if max_val <= 0
            error(
                "Maximum demand is zero or negative for $zone"
            )
        end


        for ws in scenario_cols
            demand_data[!, Symbol(ws)] ./=
                max_val
        end


        # ========================================================
        # CREATE THE 5 STOCHASTIC SCENARIOS
        # ========================================================

        for (s, ws) in enumerate(
            scenario_cols[1:n_scenarios]
        )

            push!(
                profiles_list_full,
                DataFrame(
                    milestone_year=fill(
                        2035,
                        n_timesteps,
                    ),
                    timestep=1:n_timesteps,
                    scenario=fill(
                        s,
                        n_timesteps,
                    ),
                    profile_name=fill(
                        profile_name,
                        n_timesteps,
                    ),
                    value=demand_data[
                        !,
                        Symbol(ws),
                    ],
                ),
            )
        end
    end


    # ============================================================
    # COMBINE
    # ============================================================

    if isempty(profiles_list_full)
        error(
            "No demand profiles were generated."
        )
    end


    all_profiles_df = vcat(
        profiles_list_full...
    )

    sort!(
        all_profiles_df,
        [
            :milestone_year,
            :scenario,
            :profile_name,
            :timestep,
        ],
    )


    # ============================================================
    # SANITY CHECK
    # ============================================================

    generated_profiles = unique(
        all_profiles_df.profile_name
    )

    println()
    println(
        "Generated $(length(generated_profiles)) ERAA demand profiles:"
    )

    for profile in sort(generated_profiles)
        println("  ", profile)
    end


    # ============================================================
    # WRITE OUTPUT
    # ============================================================

    # output_file = joinpath(
    #     homedir(),
    #     "Nextcloud",
    #     "ExperimentData",
    #     "EU-input-data",
    #     "profiles-demand.csv",
    # )

    output_file = joinpath(
        homedir(),
        "Server",
        "profiles-demand.csv",
    )

    mkpath(
        dirname(output_file)
    )

    CSV.write(
        output_file,
        all_profiles_df;
        writeheader=true,
    )


    println()
    println(
        "Demand profiles written to:"
    )
    println(
        output_file
    )
end


prepare_demand_data()