using DataFrames
using CSV

input_data_file = "C:/Users/fjlaseur/Tulipa/Experiments_CVaR/base-input-data/create-scenarios/"

wind_offshore_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_Wind_Offshore_2028.csv", DataFrame; header=11)
demand_data = CSV.read(input_data_file * "/demand-data/NL00_Demand_total_2028_National Trends.csv", DataFrame)
wind_onshore_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_Wind_Onshore_2028.csv", DataFrame; header=11)
solar_data = CSV.read(input_data_file * "/availability-data/NL00_CapacityFactors_PV_utility_tracking_2028.csv", DataFrame; header=11)
hydro_inflow_data_daily = CSV.read(input_data_file * "/availability-data/NL00_Hydro_Inflows_HRR_2028.csv", DataFrame)

scenario_cols = ["WS" * lpad(string(i), 2, '0') for i in 1:36]

hydro_inflow_data_hourly = DataFrame()
for ws in scenario_cols
    hydro_inflow_data_hourly[!, ws] = repeat(hydro_inflow_data_daily[!, ws], inner=24)
end

n_timesteps = nrow(wind_offshore_data)
profiles_list = Vector{DataFrame}(undef, length(scenario_cols))


for (s, ws) in enumerate(scenario_cols)
    profiles_list[s] = DataFrame(
        milestone_year=fill(2028, n_timesteps),
        scenario=fill(s, n_timesteps),
        timestep=1:n_timesteps,
        solar=solar_data[!, Symbol(ws)],
        wind_offshore=wind_offshore_data[!, Symbol(ws)],
        wind_onshore=wind_onshore_data[!, Symbol(ws)],
        demand=demand_data[!, Symbol(ws)],
        hydro_inflow=hydro_inflow_data_hourly[!, Symbol(ws)],
    )
end

profiles_df = vcat(profiles_list...)

CSV.write(joinpath(input_data_file, "profiles-wide.csv"), profiles_df; writeheader=true)