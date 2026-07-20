using DataFrames
using CSV

cd(@__DIR__)

function prepare_data()
    input_data_file = "Timeseries_raw/"
    n_scenarios = 36
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:n_scenarios] # i take them all to compute the maximum across the 36 scenarios

    wind_offshore_data = CSV.read(input_data_file * "/NL00_CapacityFactors_Wind_Offshore_2030.csv", DataFrame; header=11) # skipping first 10 rows
    demand_data = CSV.read(input_data_file * "/NL00_Demand_total_2030_National Trends.csv", DataFrame)
    wind_onshore_data = CSV.read(input_data_file * "/NL00_CapacityFactors_Wind_Onshore_2030.csv", DataFrame; header=11)
    solar_data = CSV.read(input_data_file * "/NL00_CapacityFactors_PV_utility_tracking_2030.csv", DataFrame; header=11)

    # now let us take profiles for demand       
    max_val = maximum(
        maximum(demand_data[!, Symbol(ws)]) for ws in scenario_cols
    )

    if max_val == 0
        @warn "Max value is zero for $prefix"
    else
        for ws in scenario_cols
            demand_data[!, Symbol(ws)] ./= max_val
        end
    end


    n_timesteps = nrow(demand_data) # 8760
    profiles_list = Vector{DataFrame}(undef, n_scenarios) # 10 scenario dataframes
    for (s, ws) in enumerate(scenario_cols)
        profiles_list[s] = DataFrame(
            year=fill(2030, n_timesteps),
            timestep=1:n_timesteps,
            scenario=fill(s, n_timesteps),
            solar=solar_data[!, Symbol(ws)],
            wind_offshore=wind_offshore_data[!, Symbol(ws)],
            wind_onshore=wind_onshore_data[!, Symbol(ws)],
            demand=demand_data[!, Symbol(ws)],
        )
    end

    all_profiles_df = vcat(profiles_list...)# vertical concatenation
    CSV.write("input_data/profiles-wide_$(n_scenarios)scenarios.csv", all_profiles_df; writeheader=true)
end

prepare_data()