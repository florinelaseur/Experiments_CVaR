using DataFrames
using CSV


function prepare_availability_data()

    # ------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------

    # ERAA availability data
    input_data_folder = joinpath(
        homedir(),
        "Nextcloud",
        "ExperimentData",
        "EU-input-data",
        "ERAA",
        "AvailabilityData",
    )

    capacity_factors_folder = joinpath(
        input_data_folder,
        "CapacityFactors_250716",
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


    # ------------------------------------------------------------
    # SETTINGS
    # ------------------------------------------------------------

    n_timesteps = 8760
    n_scenarios = 5

    scenario_cols = [
        "WS" * lpad(string(j), 2, '0')
        for j in 1:n_scenarios
    ]

    profiles_list_full = DataFrame[]

    # Only generate availability profiles for assets that are
    # actually referenced in the TYNDP model input.
    profile_assets = unique(
        String.(df_profiles.asset)
    )


    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    """
    Add one availability profile to profiles_list_full
    for all ERAA weather scenarios.
    """
    function add_profile!(
        asset_name::String,
        data::DataFrame,
    )

        @assert nrow(data) == n_timesteps """
        Expected $n_timesteps timesteps for $asset_name,
        but found $(nrow(data)).
        """

        for (s, ws) in enumerate(scenario_cols)

            @assert Symbol(ws) in propertynames(data) """
            Scenario column $ws not found for $asset_name.
            """

            push!(
                profiles_list_full,
                DataFrame(
                    year=fill(2035, n_timesteps),
                    timestep=1:n_timesteps,
                    scenario=fill(s, n_timesteps),
                    profile_name=fill(asset_name, n_timesteps),
                    value=data[!, Symbol(ws)],
                ),
            )
        end
    end


    """
    Read one ERAA capacity-factor file.
    """
    function read_capacity_factor(
        file::String,
        asset_name::String,
    )

        if !isfile(file)
            error(
                """
                Availability file not found for $asset_name:
                $file
                """
            )
        end

        data = CSV.read(
            file,
            DataFrame;
            header=11,
        )

        @assert nrow(data) == n_timesteps """
        Expected $n_timesteps timesteps for $asset_name,
        but found $(nrow(data)) in:
        $file
        """

        for ws in scenario_cols
            @assert Symbol(ws) in propertynames(data) """
            Scenario column $ws not found for $asset_name in:
            $file
            """
        end

        return data
    end


    """
    Aggregate ERAA solar capacity-factor profiles within one market node.

    No installed-capacity data are used.

    The TYNDP assets-profiles.csv determines which model availability
    profiles are required. ERAA determines the corresponding
    weather-dependent capacity-factor time series.

    :photovoltaic combines the available ERAA profiles:
        - PV utility fixed
        - PV utility tracking

    :rooftop combines the available ERAA profiles:
        - PV industrial rooftop
        - PV residential rooftop

    If only one underlying ERAA profile exists, that profile is used
    directly.

    If both underlying ERAA profiles exist, their capacity factors
    are averaged equally for every timestep and weather scenario.
    """
    function aggregate_solar_profiles(
        zone::String,
        solar_type::Symbol,
    )

        if solar_type == :photovoltaic

            file_suffixes = [
                "PV_utility_fixed",
                "PV_utility_tracking",
            ]

        elseif solar_type == :rooftop

            file_suffixes = [
                "PV_industrial_rooftop",
                "PV_residential_rooftop",
            ]

        else
            error(
                "Unknown solar_type: $solar_type"
            )
        end


        solar_profiles = DataFrame[]


        # --------------------------------------------------------
        # READ ALL AVAILABLE ERAA SUBTYPE PROFILES
        # --------------------------------------------------------

        for file_suffix in file_suffixes

            file = joinpath(
                capacity_factors_folder,
                "$(zone)_CapacityFactors_$(file_suffix)_2035.csv",
            )

            # Not every ERAA market node contains every solar
            # subtype. Missing subtypes are therefore ignored.
            if !isfile(file)
                continue
            end

            data = CSV.read(
                file,
                DataFrame;
                header=11,
            )

            @assert nrow(data) == n_timesteps """
            Expected $n_timesteps timesteps in:
            $file

            Found $(nrow(data)).
            """

            for ws in scenario_cols
                @assert Symbol(ws) in propertynames(data) """
                Scenario column $ws not found in:
                $file
                """
            end

            push!(
                solar_profiles,
                data,
            )
        end


        # --------------------------------------------------------
        # NO ERAA PROFILE FOUND
        # --------------------------------------------------------

        if isempty(solar_profiles)
            return nothing
        end


        # --------------------------------------------------------
        # EXACTLY ONE ERAA PROFILE FOUND
        # --------------------------------------------------------

        # No aggregation is necessary.
        if length(solar_profiles) == 1
            return solar_profiles[1]
        end


        # --------------------------------------------------------
        # MULTIPLE ERAA PROFILES FOUND
        # --------------------------------------------------------

        # Take the unweighted mean of the available ERAA capacity
        # factors for every timestep and weather scenario.
        aggregated = deepcopy(
            solar_profiles[1]
        )

        for ws in scenario_cols

            col = Symbol(ws)

            aggregated[!, col] .= 0.0

            for data in solar_profiles
                aggregated[!, col] .+=
                    data[!, col]
            end

            aggregated[!, col] ./=
                length(solar_profiles)
        end


        return aggregated
    end


    # ============================================================
    # WIND ONSHORE
    # ============================================================

    for asset in filter(
        a -> endswith(a, "_Wind_Onshore"),
        profile_assets,
    )

        zone = replace(
            asset,
            "_Wind_Onshore" => "",
        )

        file = joinpath(
            capacity_factors_folder,
            "$(zone)_CapacityFactors_Wind_Onshore_2035.csv",
        )

        data = read_capacity_factor(
            file,
            asset,
        )

        add_profile!(
            asset,
            data,
        )
    end


    # ============================================================
    # WIND OFFSHORE
    # ============================================================

    for asset in filter(
        a -> occursin("_Wind_Offshore", a),
        profile_assets,
    )

        # Examples:
        #
        # BE00_Wind_Offshore
        # DKE1_Wind_Offshore
        # DKE1_Wind_Offshore_DKKF
        # DE00_Wind_Offshore_DEKF
        #
        # Special offshore assets use the ERAA offshore
        # availability profile of their corresponding market node.

        zone = first(
            split(
                asset,
                "_Wind_Offshore",
            )
        )

        file = joinpath(
            capacity_factors_folder,
            "$(zone)_CapacityFactors_Wind_Offshore_2035.csv",
        )

        data = read_capacity_factor(
            file,
            asset,
        )

        add_profile!(
            asset,
            data,
        )
    end


    # ============================================================
    # SOLAR PHOTOVOLTAIC
    # ============================================================

    for asset in filter(
        a -> endswith(
            a,
            "_Solar_Photovoltaic",
        ),
        profile_assets,
    )

        zone = replace(
            asset,
            "_Solar_Photovoltaic" => "",
        )

        solar_data = aggregate_solar_profiles(
            zone,
            :photovoltaic,
        )


        if solar_data === nothing
            error(
                """
                No ERAA Solar_Photovoltaic availability data
                was found for $asset.

                Expected at least one of:
                $(zone)_CapacityFactors_PV_utility_fixed_2035.csv
                $(zone)_CapacityFactors_PV_utility_tracking_2035.csv

                The profile is required by:
                $profiles_file
                """
            )
        end


        add_profile!(
            asset,
            solar_data,
        )
    end


    # ============================================================
    # SOLAR ROOFTOP
    # ============================================================

    for asset in filter(
        a -> endswith(
            a,
            "_Solar_Rooftop",
        ),
        profile_assets,
    )

        zone = replace(
            asset,
            "_Solar_Rooftop" => "",
        )

        solar_data = aggregate_solar_profiles(
            zone,
            :rooftop,
        )


        if solar_data === nothing
            error(
                """
                No ERAA Solar_Rooftop availability data
                was found for $asset.

                Expected at least one of:
                $(zone)_CapacityFactors_PV_industrial_rooftop_2035.csv
                $(zone)_CapacityFactors_PV_residential_rooftop_2035.csv

                The profile is required by:
                $profiles_file
                """
            )
        end


        add_profile!(
            asset,
            solar_data,
        )
    end


    # ============================================================
    # COMBINE PROFILES
    # ============================================================

    if isempty(profiles_list_full)
        error(
            """
            profiles_list_full is empty.

            No matching wind or solar ERAA availability
            profiles were generated for the assets in:
            $profiles_file
            """
        )
    end


    all_profiles_df = vcat(
        profiles_list_full...
    )


    sort!(
        all_profiles_df,
        [
            :year,
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
        "Generated $(length(generated_profiles)) ERAA availability profiles:"
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
        "profiles-availability.csv",
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
        "Availability profiles written to:"
    )
    println(
        output_file
    )
end


prepare_availability_data()