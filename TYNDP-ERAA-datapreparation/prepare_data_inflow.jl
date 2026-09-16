using DataFrames
using CSV

cd(@__DIR__)


function prepare_inflow_data()

    # ============================================================
    # PATHS
    # ============================================================

    # ERAA hydro inflow data
    input_data_folder = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
        "ERAA",
        "AvailabilityData",
    )

    hydro_inflow_folder = joinpath(
        input_data_folder,
        "HydroInflows_250704",
    )

    # TYNDP model input data
    input_data_folder_TYNDP = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
        "TYNDP",
        "Outputs",
        "tulipa_input_north_sea_2026_2035_de",
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
    # normalization factor for each hydro asset.
    scenario_cols = [
        "WS" * lpad(string(j), 2, '0')
        for j in 1:36
    ]

    profiles_list_full = DataFrame[]


    # ============================================================
    # HELPER: READ ERAA INFLOW FILE
    # ============================================================

    function read_inflow(
        zone::AbstractString,
        suffix::AbstractString,
        asset::AbstractString,
    )

        file = joinpath(
            hydro_inflow_folder,
            "$(zone)_Hydro_Inflows_$(suffix)_2035.csv",
        )

        if !isfile(file)
            error(
                """
                ERAA hydro inflow file does not exist
                for TYNDP asset $asset:

                $file
                """
            )
        end

        data = CSV.read(
            file,
            DataFrame,
        )

        missing_scenario_cols = [
            ws
            for ws in scenario_cols
            if Symbol(ws) ∉ propertynames(data)
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

        return data
    end


    # ============================================================
    # HELPER: NORMALIZE ERAA INFLOW
    # ============================================================

    function normalize_inflow!(
        data::DataFrame,
        asset::AbstractString,
    )

        max_val = maximum(
            maximum(
                data[!, Symbol(ws)]
            )
            for ws in scenario_cols
        )

        if max_val <= 0
            error(
                "Maximum inflow is zero or negative for $asset"
            )
        end


        for ws in scenario_cols
            data[!, Symbol(ws)] ./=
                max_val
        end
    end


    # ============================================================
    # HELPER: CONVERT TO HOURLY PROFILE AND STORE
    # ============================================================

    function add_hourly_profile!(
        data::DataFrame,
        profile_name::AbstractString,
        asset::AbstractString,
        hours_per_input_step::Int,
    )

        hourly_data = DataFrame()


        for ws in scenario_cols

            vals = repeat(
                data[!, Symbol(ws)],
                inner=hours_per_input_step,
            )

            if length(vals) < n_timesteps
                error(
                    """
                    ERAA inflow data for $asset produces only
                    $(length(vals)) hourly timesteps.

                    Expected at least $n_timesteps.
                    """
                )
            end

            hourly_data[!, Symbol(ws)] =
                vals[1:n_timesteps]
        end


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
                    value=hourly_data[
                        !,
                        Symbol(ws),
                    ],
                ),
            )
        end
    end


    # ============================================================
    # RUN-OF-RIVER / HRR
    # ============================================================

    run_of_river_profiles = filter(
        row ->
            occursin(
                "_Hydro_Run_of_River",
                String(row.asset),
            ),
        df_profiles,
    )


    for row in eachrow(run_of_river_profiles)

        asset = String(row.asset)
        profile_name = String(row.profile_name)

        zone = first(
            split(
                asset,
                "_Hydro_Run_of_River",
            )
        )


        inflow_data = read_inflow(
            zone,
            "HRR",
            asset,
        )

        normalize_inflow!(
            inflow_data,
            asset,
        )


        # ERAA HRR input is daily.
        add_hourly_profile!(
            inflow_data,
            profile_name,
            asset,
            24,
        )
    end


    # ============================================================
    # RESERVOIR / HRI
    # ============================================================

    reservoir_profiles = filter(
        row ->
            occursin(
                "_Hydro_Reservoir",
                String(row.asset),
            ),
        df_profiles,
    )


    for row in eachrow(reservoir_profiles)

        asset = String(row.asset)
        profile_name = String(row.profile_name)

        zone = first(
            split(
                asset,
                "_Hydro_Reservoir",
            )
        )


        inflow_data = read_inflow(
            zone,
            "HRI",
            asset,
        )

        normalize_inflow!(
            inflow_data,
            asset,
        )


        # ERAA HRI input is weekly.
        add_hourly_profile!(
            inflow_data,
            profile_name,
            asset,
            24 * 7,
        )
    end


    # ============================================================
    # OPEN PUMPED STORAGE / HOL
    # ============================================================

    pump_storage_open_profiles = filter(
        row ->
            occursin(
                "_Hydro_Pump_Storage_Open",
                String(row.asset),
            ),
        df_profiles,
    )


    for row in eachrow(pump_storage_open_profiles)

        asset = String(row.asset)
        profile_name = String(row.profile_name)

        zone = first(
            split(
                asset,
                "_Hydro_Pump_Storage_Open",
            )
        )


        inflow_data = read_inflow(
            zone,
            "HOL",
            asset,
        )

        normalize_inflow!(
            inflow_data,
            asset,
        )


        # ERAA HOL input is weekly.
        add_hourly_profile!(
            inflow_data,
            profile_name,
            asset,
            24 * 7,
        )
    end


    # ============================================================
    # CHECK FOR UNSUPPORTED TYNDP HYDRO INFLOW TYPES
    # ============================================================

    # TYNDP contains Hydro_Pondage assets with inflow profiles,
    # but the old ERAA preparation script does not define which
    # ERAA inflow source (HRR/HRI/HOL) should be used for them.
    #
    # Do not silently invent this mapping.

    # ============================================================
    # PONDAGE
    # ============================================================

    pondage_profiles = filter(
        row ->
            occursin(
                "_Hydro_Pondage",
                String(row.asset),
            ),
        df_profiles,
    )

    for row in eachrow(pondage_profiles)

        asset = String(row.asset)
        profile_name = String(row.profile_name)

        zone = first(
            split(
                asset,
                "_Hydro_Pondage",
            )
        )

        # Preferred ERAA inflow source for pondage:
        #
        # 1. HPI: dedicated pondage inflow
        # 2. HRI: reservoir inflow
        # 3. HOL: open-loop pumped-storage inflow
        # 4. HRR: run-of-river inflow
        #
        # Prefer 2035 data where available; otherwise fall back to 2025.

        inflow_priority = [
            ("HPI", 2035),
            ("HPI", 2025),
            ("HRI", 2035),
            ("HRI", 2025),
            ("HOL", 2035),
            ("HOL", 2025),
            ("HRR", 2035),
            ("HRR", 2025),
        ]

        selected_file = nothing
        selected_type = nothing
        selected_year = nothing

        for (inflow_type, year) in inflow_priority

            candidate_file = joinpath(
                hydro_inflow_folder,
                "$(zone)_Hydro_Inflows_$(inflow_type)_$(year).csv",
            )

            if isfile(candidate_file)
                selected_file = candidate_file
                selected_type = inflow_type
                selected_year = year
                break
            end
        end

        if selected_file === nothing
            error(
                """
                No ERAA inflow profile was found for pondage asset:
                $asset

                Tried HPI, HRI, HOL, and HRR for 2035 and 2025
                in:
                $hydro_inflow_folder
                """
            )
        end

        println(
            "Pondage $asset: using ERAA $selected_type $selected_year inflow."
        )

        inflow_data = CSV.read(
            selected_file,
            DataFrame,
        )

        missing_scenario_cols = [
            ws
            for ws in scenario_cols
            if Symbol(ws) ∉ propertynames(inflow_data)
        ]

        if !isempty(missing_scenario_cols)
            error(
                """
                Missing ERAA scenario columns in:
                $(basename(selected_file))

                Missing:
                $(join(missing_scenario_cols, ", "))
                """
            )
        end

        normalize_inflow!(
            inflow_data,
            asset,
        )

        # Pondage inflow series are treated as weekly.
        add_hourly_profile!(
            inflow_data,
            profile_name,
            asset,
            24 * 7,
        )
    end


    # ============================================================
    # COMBINE
    # ============================================================

    if isempty(profiles_list_full)
        error(
            "No inflow profiles were generated."
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
        "Generated $(length(generated_profiles)) ERAA hydro profiles:"
    )

    for profile in sort(generated_profiles)
        println("  ", profile)
    end


    # ============================================================
    # WRITE OUTPUT
    # ============================================================

    output_file = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
        "profiles-inflow.csv",
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
        "Inflow profiles written to:"
    )
    println(
        output_file
    )
end


prepare_inflow_data()