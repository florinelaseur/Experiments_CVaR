using DataFrames
using CSV

cd(@__DIR__)

function prepare_inflow_data()
    input_data_file = "Inflow_timeseries_raw/"
    profiles_file = input_data_file * "assets_profiles.csv"
    df_profiles = CSV.read(profiles_file, DataFrame)
    n_timesteps = 8760
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
        "GR" => ["GR00"], # i don't have capacity for 03, i just ignore it
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
        "UK" => ["UK00"]
    )
    prefixes = collect(keys(country_map))
    n_scenarios = 5
    profiles_list_full = DataFrame[]
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:36] # i take them all to compute the maximum across the 36 scenarios

    for prefix in prefixes
        sub_prefixes = country_map[prefix]

        if "$(prefix)_Hydro" in df_profiles.asset

            availability_sum = nothing
            for sub in sub_prefixes
                file_sub = input_data_file * "/$(sub)_Hydro_Inflows_HRR_2030.csv"
                if isfile(file_sub)
                    df = CSV.read(file_sub, DataFrame)

                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)]
                        end
                    end
                end
            end

            if availability_sum === nothing && prefix != "LV"
                error("No data found for HRR for $prefix")
            end
            if prefix != "LV" # treated later
                # now let us take profiles        
                max_val = maximum(
                    maximum(availability_sum[!, Symbol(ws)]) for ws in scenario_cols
                )

                if max_val == 0
                    @warn "Max value is zero for $prefix"
                else
                    for ws in scenario_cols
                        availability_sum[!, Symbol(ws)] ./= max_val
                    end
                end
                hydro_inflow_data_hourly = DataFrame()
                for ws in scenario_cols # transform from daily to hourly
                    hydro_inflow_data_hourly[!, Symbol(ws)] = repeat(availability_sum[!, Symbol(ws)], inner=24)
                end

                for (s, ws) in enumerate(scenario_cols[1:n_scenarios])
                    push!(profiles_list_full, DataFrame(
                        year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                        timestep=1:n_timesteps,
                        scenario=fill(s, n_timesteps),
                        profile_name="$(prefix)_Hydro",
                        value=hydro_inflow_data_hourly[!, Symbol(ws)],
                    ))
                end
            end
        end

        if "$(prefix)_Hydro_Reservoir" in df_profiles.asset

            availability_sum = nothing
            for sub in sub_prefixes
                file_sub = input_data_file * "/$(sub)_Hydro_Inflows_HRI_2030.csv"
                if isfile(file_sub)
                    df = CSV.read(file_sub, DataFrame)

                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)]
                        end
                    end
                end
            end

            if availability_sum === nothing
                error("No data found for HRI for $prefix")
            end
            # now let us take profiles        
            max_val = maximum(
                maximum(availability_sum[!, Symbol(ws)]) for ws in scenario_cols
            )

            if max_val == 0
                @warn "Max value is zero for $prefix"
            else
                for ws in scenario_cols
                    availability_sum[!, Symbol(ws)] ./= max_val
                end
            end
            hydro_reservoir_inflow_data_hourly = DataFrame()
            for ws in scenario_cols # transform from weekly to hourly
                vals = repeat(availability_sum[!, Symbol(ws)], inner=24 * 7)
                hydro_reservoir_inflow_data_hourly[!, Symbol(ws)] = vals[1:8760]
            end

            for (s, ws) in enumerate(scenario_cols[1:n_scenarios])
                push!(profiles_list_full, DataFrame(
                    year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                    timestep=1:n_timesteps,
                    scenario=fill(s, n_timesteps),
                    profile_name="$(prefix)_Hydro_Reservoir_Inflow",
                    value=hydro_reservoir_inflow_data_hourly[!, Symbol(ws)],
                ))
            end

        end

        if "$(prefix)_Pump_Hydro_Open" in df_profiles.asset
            availability_sum = nothing
            for sub in sub_prefixes
                file_sub = input_data_file * "/$(sub)_Hydro_Inflows_HOL_2030.csv"
                if isfile(file_sub)
                    df = CSV.read(file_sub, DataFrame)

                    if availability_sum === nothing
                        availability_sum = deepcopy(df)
                    else
                        @assert nrow(availability_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                        for ws in scenario_cols
                            availability_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)]
                        end
                    end
                end
            end

            if availability_sum === nothing
                error("No data found for HOL for $prefix")
            end
            # now let us take profiles        
            max_val = maximum(
                maximum(availability_sum[!, Symbol(ws)]) for ws in scenario_cols
            )

            if max_val == 0
                @warn "Max value is zero for $prefix"
            else
                for ws in scenario_cols
                    availability_sum[!, Symbol(ws)] ./= max_val
                end
            end
            hydro_pump_inflow_data_hourly = DataFrame()
            for ws in scenario_cols # transform from weekly to hourly
                vals = repeat(availability_sum[!, Symbol(ws)], inner=24 * 7)
                hydro_pump_inflow_data_hourly[!, Symbol(ws)] = vals[1:8760]
            end

            for (s, ws) in enumerate(scenario_cols[1:n_scenarios])
                push!(profiles_list_full, DataFrame(
                    year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                    timestep=1:n_timesteps,
                    scenario=fill(s, n_timesteps),
                    profile_name="$(prefix)_Pump_Hydro_Open_Inflow",
                    value=hydro_pump_inflow_data_hourly[!, Symbol(ws)],
                ))
            end

        end


    end

    # and LV Hydro (which is missing)
    profiles_list_lv = Vector{DataFrame}(undef, n_scenarios)
    av_data_lv = CSV.read(input_data_file * "/LV_HRR.csv", DataFrame)

    for s in 1:n_scenarios
        profiles_list_lv[s] = DataFrame(
            year=av_data_lv.year,
            timestep=av_data_lv.timestep,
            scenario=fill(s, nrow(av_data_lv)),
            profile_name=av_data_lv.profile_name,
            value=av_data_lv.value,
        )
    end



    all_profiles_df = vcat(profiles_list_full..., profiles_list_lv...) # vertical concatenation
    sort!(all_profiles_df, [:year, :scenario, :profile_name, :timestep])
    CSV.write("input_data/profiles-inflow.csv", all_profiles_df; writeheader=true)
end

prepare_inflow_data()