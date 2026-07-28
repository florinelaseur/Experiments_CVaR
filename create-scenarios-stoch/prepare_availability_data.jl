using DataFrames
using CSV

cd(@__DIR__)

function prepare_availability_data()
    input_data_file = "Availability_timeseries_raw/"
    capacities_file = input_data_file * "capacities.csv"
    df_capacities = CSV.read(capacities_file, DataFrame)
    profiles_file = input_data_file * "assets_profiles.csv"
    df_profiles = CSV.read(profiles_file, DataFrame)
    n_timesteps = 8760

    #prefix_countries = ["AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "UK"]
    country_map = Dict(
        "AT" => ["AT00"],
        "BE" => ["BE00"],
        "BG" => ["BG00"],
        "CH" => ["CH00"],
        "CY" => ["CY00"],
        "CZ" => ["CZ00"],
        "DE" => ["DE00"],
        "DK" => ["DKE1", "DKW1"],
        "EE" => ["EE00"],
        "ES" => ["ES00"],
        "FI" => ["FI00"],
        "FR" => ["FR00"],
        "GR" => ["GR00", "GR03"],
        "HR" => ["HR00"],
        "HU" => ["HU00"],
        "IE" => ["IE00"],
        "IT" => ["ITN1", "ITCN", "ITCS", "ITS1", "ITCA", "ITSI", "ITSA"],
        "LT" => ["LT00"],
        "LU" => ["LUG1"],
        "LV" => ["LV00"],
        "MT" => ["MT00"],
        "NL" => ["NL00"],
        "NO" => ["NOM1", "NON1", "NOS1", "NOS2", "NOS3"],
        "PL" => ["PL00"],
        "PT" => ["PT00"],
        "RO" => ["RO00"],
        "SE" => ["SE01", "SE02", "SE03", "SE04"],
        "SI" => ["SI00"],
        "SK" => ["SK00"],
        "UK" => ["UK00", "UKNI"]
    )
    prefixes = collect(keys(country_map))
    n_scenarios = 5
    profiles_list_full = DataFrame[]
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:n_scenarios]

    for prefix in prefixes
        #profiles = filter(row -> startswith(row.asset, prefix), df_profiles)
        sub_prefixes = country_map[prefix]

        # wind onshore and wind offshore
        if "$(prefix)_Wind_Onshore" in df_profiles.asset
            if length(country_map[prefix]) == 1
                sub_pref = sub_prefixes[1]
                wind_onshore_data = CSV.read(input_data_file * "/$(sub_pref)_CapacityFactors_Wind_Onshore_2030.csv", DataFrame; header=11)

            else
                availability_sum = nothing
                capacity_sum = 0
                for sub in sub_prefixes
                    file_sub = input_data_file * "/$(sub)_CapacityFactors_Wind_Onshore_2030.csv"
                    if isfile(file_sub)
                        df = CSV.read(file_sub, DataFrame; header=11)
                        vals = df_capacities[
                            (df_capacities.Market_Node.==sub).&(df_capacities.Technology.=="Wind onshore"),
                            :Value
                        ]
                        cap = isempty(vals) ? 0 : only(vals)
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end
                end

                if availability_sum === nothing
                    error("No data found for wind onshore for $prefix")
                end
                wind_onshore_data = deepcopy(availability_sum)
                if capacity_sum < 1e-6
                    @error("capacity onshore 0 for $prefix")
                end
                for ws in scenario_cols
                    wind_onshore_data[!, Symbol(ws)] ./= capacity_sum
                end

            end
            for (s, ws) in enumerate(scenario_cols)
                push!(profiles_list_full, DataFrame(
                    year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                    timestep=1:n_timesteps,
                    scenario=fill(s, n_timesteps),
                    profile_name="$(prefix)_Wind_Onshore",
                    value=wind_onshore_data[!, Symbol(ws)],
                ))
            end

        end

        if "$(prefix)_Wind_Offshore" in df_profiles.asset
            if length(country_map[prefix]) == 1 && prefix != "SI"
                sub_pref = sub_prefixes[1]
                wind_offshore_data = CSV.read(input_data_file * "/$(sub_pref)_CapacityFactors_Wind_Offshore_2030.csv", DataFrame; header=11)

            else
                availability_sum = nothing
                capacity_sum = 0
                for sub in sub_prefixes
                    file_sub = input_data_file * "/$(sub)_CapacityFactors_Wind_Offshore_2030.csv"
                    if isfile(file_sub)
                        df = CSV.read(file_sub, DataFrame; header=11)
                        vals = df_capacities[
                            (df_capacities.Market_Node.==sub).&(df_capacities.Technology.=="Wind offshore"),
                            :Value
                        ]

                        cap = isempty(vals) ? 0 : only(vals)
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end
                end

                if availability_sum === nothing && prefix != "SI"
                    error("No data found for wind offshore for $prefix")
                end
                if prefix != "SI" # treated later
                    wind_offshore_data = deepcopy(availability_sum)
                    if capacity_sum < 1e-6
                        #@error("capacity offshore 0 for $prefix")
                        for ws in scenario_cols
                            wind_offshore_data[!, Symbol(ws)] .= 0
                        end
                    else
                        for ws in scenario_cols
                            wind_offshore_data[!, Symbol(ws)] ./= capacity_sum
                        end
                    end
                end

            end
            if prefix != "SI"
                for (s, ws) in enumerate(scenario_cols)
                    push!(profiles_list_full, DataFrame(
                        year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                        timestep=1:n_timesteps,
                        scenario=fill(s, n_timesteps),
                        profile_name="$(prefix)_Wind_Offshore",
                        value=wind_offshore_data[!, Symbol(ws)],
                    ))
                end
            end
        end

        if "$(prefix)_Solar" in df_profiles.asset
            if length(country_map[prefix]) == 1
                sub_pref = sub_prefixes[1]
                file_sub1 = input_data_file * "/$(sub_pref)_CapacityFactors_PV_utility_fixed_2030.csv"
                file_sub2 = input_data_file * "/$(sub_pref)_CapacityFactors_PV_utility_tracking_2030.csv"
                file_sub3 = input_data_file * "/$(sub_pref)_CapacityFactors_PV_industrial_rooftop_2030.csv"
                file_sub4 = input_data_file * "/$(sub_pref)_CapacityFactors_PV_residential_rooftop_2030.csv"
                availability_sum = nothing
                capacity_sum = 0
                if isfile(file_sub1)
                    df = CSV.read(file_sub1, DataFrame; header=11)
                    vals = df_capacities[
                        (df_capacities.Market_Node.==sub_pref).&(df_capacities.Technology.=="Solar PV utility non-tracking"),
                        :Value
                    ]

                    cap = isempty(vals) ? 0 : only(vals)
                    capacity_sum += cap
                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                        end
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                        end
                    end
                end
                if isfile(file_sub2)
                    df = CSV.read(file_sub2, DataFrame; header=11)
                    vals = df_capacities[
                        (df_capacities.Market_Node.==sub_pref).&(df_capacities.Technology.=="Solar PV utility tracking"),
                        :Value
                    ]

                    cap = isempty(vals) ? 0 : only(vals)
                    capacity_sum += cap
                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                        end
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                        end
                    end
                end
                if isfile(file_sub3)
                    df = CSV.read(file_sub3, DataFrame; header=11)
                    vals = df_capacities[
                        (df_capacities.Market_Node.==sub_pref).&(df_capacities.Technology.=="Solar PV rooftop industrial"),
                        :Value
                    ]

                    cap = isempty(vals) ? 0 : only(vals)
                    capacity_sum += cap
                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                        end
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                        end
                    end
                end
                if isfile(file_sub4)
                    df = CSV.read(file_sub4, DataFrame; header=11)
                    vals = df_capacities[
                        (df_capacities.Market_Node.==sub_pref).&(df_capacities.Technology.=="Solar PV rooftop residential"),
                        :Value
                    ]
                    cap = isempty(vals) ? 0 : only(vals)
                    capacity_sum += cap
                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                        end
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                        end
                    end
                end
                if availability_sum === nothing
                    error("No data found for solar for $prefix")
                end
                solar_data = deepcopy(availability_sum)
                if capacity_sum < 1e-6
                    @error("capacity 0 for $prefix")
                end
                for ws in scenario_cols
                    solar_data[!, Symbol(ws)] ./= capacity_sum
                end

            else
                availability_sum = nothing
                capacity_sum = 0
                for sub in sub_prefixes
                    file_sub1 = input_data_file * "/$(sub)_CapacityFactors_PV_utility_fixed_2030.csv"
                    file_sub2 = input_data_file * "/$(sub)_CapacityFactors_PV_utility_tracking_2030.csv"
                    file_sub3 = input_data_file * "/$(sub)_CapacityFactors_PV_industrial_rooftop_2030.csv"
                    file_sub4 = input_data_file * "/$(sub)_CapacityFactors_PV_residential_rooftop_2030.csv"
                    if isfile(file_sub1)
                        df = CSV.read(file_sub1, DataFrame; header=11)
                        vals = df_capacities[
                            (df_capacities.Market_Node.==sub).&(df_capacities.Technology.=="Solar PV utility non-tracking"),
                            :Value
                        ]
                        cap = isempty(vals) ? 0 : only(vals)
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end
                    if isfile(file_sub2)
                        df = CSV.read(file_sub2, DataFrame; header=11)
                        cap = only(df_capacities[(df_capacities.Market_Node.==sub).&&(df_capacities.Technology.=="Solar PV utility tracking"), :Value])
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end
                    if isfile(file_sub3)
                        df = CSV.read(file_sub3, DataFrame; header=11)
                        vals = df_capacities[
                            (df_capacities.Market_Node.==sub).&(df_capacities.Technology.=="Solar PV rooftop industrial"),
                            :Value
                        ]

                        cap = isempty(vals) ? 0 : only(vals)
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end
                    if isfile(file_sub4)
                        df = CSV.read(file_sub4, DataFrame; header=11)
                        cap = only(df_capacities[(df_capacities.Market_Node.==sub).&&(df_capacities.Technology.=="Solar PV rooftop residential"), :Value])
                        capacity_sum += cap
                        if availability_sum === nothing
                            availability_sum = deepcopy(df)
                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .= availability_sum[!, Symbol(ws)] .* cap
                            end
                        else
                            @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                            for ws in scenario_cols
                                availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)] .* cap
                            end
                        end
                    end


                end

                if availability_sum === nothing
                    error("No data found for solar for $prefix")
                end
                solar_data = deepcopy(availability_sum)
                if capacity_sum < 1e-6
                    @error("capacity 0 for $prefix")
                end
                for ws in scenario_cols
                    solar_data[!, Symbol(ws)] ./= capacity_sum
                end

            end
            for (s, ws) in enumerate(scenario_cols)
                push!(profiles_list_full, DataFrame(
                    year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                    timestep=1:n_timesteps,
                    scenario=fill(s, n_timesteps),
                    profile_name="$(prefix)_Solar",
                    value=solar_data[!, Symbol(ws)],
                ))
            end

        end


        # hydro_inflow_data_daily = CSV.read(input_data_file * "/NL00_Hydro_Inflows_HRR_$(year).csv", DataFrame)
        # hydro_inflow_data_hourly = DataFrame()
        # for ws in scenario_cols # transform from daily to hourly
        #     hydro_inflow_data_hourly[!, ws] = repeat(hydro_inflow_data_daily[!, ws], inner=24)
    end

    # now add also OBZLL
    profiles_list_obzll = Vector{DataFrame}(undef, n_scenarios)
    av_data_obzll = CSV.read(input_data_file * "/OBZLL_wind.csv", DataFrame)

    for s in 1:n_scenarios
        profiles_list_obzll[s] = DataFrame(
            year=av_data_obzll.year,
            timestep=av_data_obzll.timestep,
            scenario=fill(s, nrow(av_data_obzll)),
            profile_name=av_data_obzll.profile_name,
            value=av_data_obzll.value,
        )
    end
    # and SI wind offshore (which is missing)
    profiles_list_si = Vector{DataFrame}(undef, n_scenarios)
    av_data_si = CSV.read(input_data_file * "/SI_wind.csv", DataFrame)

    for s in 1:n_scenarios
        profiles_list_si[s] = DataFrame(
            year=av_data_si.year,
            timestep=av_data_si.timestep,
            scenario=fill(s, nrow(av_data_si)),
            profile_name=av_data_si.profile_name,
            value=av_data_si.value,
        )
    end



    all_profiles_df = vcat(profiles_list_full..., profiles_list_obzll..., profiles_list_si...) # vertical concatenation
    sort!(all_profiles_df, [:year, :scenario, :profile_name, :timestep])
    CSV.write("input_data/profiles-availability.csv", all_profiles_df; writeheader=true)
end

prepare_availability_data()