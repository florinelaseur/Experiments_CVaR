using DataFrames
using CSV

cd(@__DIR__)

function prepare_demand_data()
    input_data_file = "Demand_timeseries_raw/"

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
    n_countries = length(prefixes)

    n_scenarios = 5
    profiles_list_full = Vector{DataFrame}(undef, n_countries * n_scenarios)
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:36] # i take them all to compute the maximum across the 36 scenarios

    for (i, prefix) in enumerate(prefixes)
        sub_prefixes = country_map[prefix]
        #wind_offshore_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_Wind_Offshore_$(year).csv", DataFrame; header=11) # skipping first 10 rows
        #demand_data = CSV.read(input_data_file * "/$(prefix)00_Demand_total_2030_National Trends.csv", DataFrame)
        # wind_onshore_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_Wind_Onshore_$(year).csv", DataFrame; header=11)
        # solar_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_PV_utility_tracking_$(year).csv", DataFrame; header=11)
        # hydro_inflow_data_daily = CSV.read(input_data_file * "/availability-data/NL00_Hydro_Inflows_HRR_$(year).csv", DataFrame)
        # hydro_inflow_data_hourly = DataFrame()
        # for ws in scenario_cols # transform from daily to hourly
        #     hydro_inflow_data_hourly[!, ws] = repeat(hydro_inflow_data_daily[!, ws], inner=24)
        # end
        demand_sum = nothing

        for sub in sub_prefixes
            file = input_data_file * "/$(sub)_Demand_total_2030_National Trends.csv"

            df = CSV.read(file, DataFrame)

            if demand_sum === nothing
                demand_sum = deepcopy(df)
            else
                @assert nrow(demand_sum) == nrow(df) "Mismatch in timesteps for $prefix"

                for ws in scenario_cols
                    demand_sum[!, Symbol(ws)] .+= df[!, Symbol(ws)]
                end
            end
        end

        if demand_sum === nothing
            error("No data found for $prefix")
        end

        # now let us take profiles        
        max_val = maximum(
            maximum(demand_sum[!, Symbol(ws)]) for ws in scenario_cols
        )

        if max_val == 0
            @warn "Max value is zero for $prefix"
        else
            for ws in scenario_cols
                demand_sum[!, Symbol(ws)] ./= max_val
            end
        end


        n_timesteps = nrow(demand_sum) # 8760
        profiles_list = Vector{DataFrame}(undef, n_scenarios) # 7 scenario dataframes
        for (s, ws) in enumerate(scenario_cols[1:n_scenarios])
            profiles_list[s] = DataFrame(
                year=fill(2050, n_timesteps), # in OBZ case we use year 2050, profiles are taken from 2030
                timestep=1:n_timesteps,
                scenario=fill(s, n_timesteps),
                # solar=solar_data[!, Symbol(ws)],
                # wind_offshore=wind_offshore_data[!, Symbol(ws)],
                # wind_onshore=wind_onshore_data[!, Symbol(ws)],
                profile_name="$(prefix)_E_Demand",
                value=demand_sum[!, Symbol(ws)],
                # hydro_inflow=hydro_inflow_data_hourly[!, Symbol(ws)],
            )
        end
        start_index = (i - 1) * n_scenarios + 1
        stop_index = i * n_scenarios
        profiles_list_full[start_index:stop_index] = profiles_list
    end

    # now add also OBZLL
    profiles_list_obzll = Vector{DataFrame}(undef, n_scenarios)
    demand_data_obzll = CSV.read(input_data_file * "/OBZLL_Demand.csv", DataFrame)

    for s in 1:n_scenarios
        profiles_list_obzll[s] = DataFrame(
            year=demand_data_obzll.year,
            timestep=demand_data_obzll.timestep,
            scenario=fill(s, nrow(demand_data_obzll)),
            profile_name=demand_data_obzll.profile_name,
            value=demand_data_obzll.value,
        )
    end


    all_profiles_df = vcat(profiles_list_full..., profiles_list_obzll...) # vertical concatenation
    sort!(all_profiles_df, [:year, :scenario, :profile_name, :timestep])
    CSV.write("input_data/profiles-demand.csv", all_profiles_df; writeheader=true)
end

prepare_demand_data()