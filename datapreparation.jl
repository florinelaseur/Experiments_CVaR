cd(@__DIR__)
# using Pkg: Pkg
# Pkg.activate(".")
# Pkg.instantiate()

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using HiGHS: HiGHS
using Gurobi: Gurobi
using Distances: Distances
using CSV: CSV
using Statistics: Statistics
using JuMP: JuMP
using TOML: TOML
using Plots
using Random
using DataFrames

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

function main()

    config = TOML.parsefile("config.toml")
    input_data_path = config["simulation"]["scenario_data"]

    profiles_list_full = Vector{DataFrame}(undef, 144)
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:36]

    for (i, year) in enumerate([2028, 2030, 2033, 2035])

        wind_offshore_data = CSV.read(joinpath(input_data_path, "availability-data", "NL00_CapacityFactors_Wind_Offshore_$(year).csv"), DataFrame; header=11)
        demand_data = CSV.read(joinpath(input_data_path, "demand-data", "NL00_Demand_total_$(year)_National Trends.csv"), DataFrame)
        wind_onshore_data = CSV.read(joinpath(input_data_path, "availability-data", "NL00_CapacityFactors_Wind_Onshore_$(year).csv"), DataFrame; header=11)
        solar_data = CSV.read(joinpath(input_data_path, "availability-data", "NL00_CapacityFactors_PV_utility_tracking_$(year).csv"), DataFrame; header=11)
        hydro_inflow_data_daily = CSV.read(joinpath(input_data_path, "availability-data", "NL00_Hydro_Inflows_HRR_$(year).csv"), DataFrame)

        hydro_inflow_data_hourly = DataFrame()
        for ws in scenario_cols
            hydro_inflow_data_hourly[!, ws] = repeat(hydro_inflow_data_daily[!, ws], inner=24)
        end

        n_timesteps = nrow(wind_offshore_data)
        profiles_list = Vector{DataFrame}(undef, length(scenario_cols))


        for (s, ws) in enumerate(scenario_cols)
            profiles_list[s] = DataFrame(
                milestone_year=fill(year, n_timesteps),
                scenario=fill(s, n_timesteps),
                timestep=1:n_timesteps,
                solar=solar_data[!, Symbol(ws)],
                wind_offshore=wind_offshore_data[!, Symbol(ws)],
                wind_onshore=wind_onshore_data[!, Symbol(ws)],
                demand=demand_data[!, Symbol(ws)],
                hydro_inflow=hydro_inflow_data_hourly[!, Symbol(ws)],
            )
        end
        start_index = (i - 1) * 36 + 1
        stop_index = i * 36
        profiles_list_full[start_index:stop_index] = profiles_list
    end

    all_profiles_df = vcat(profiles_list_full...)
    all_profiles_df[!, :milestone_year] .= 2030
    n_timesteps_per_scenario = div(nrow(all_profiles_df), 144)
    all_profiles_df[!, :scenario] .= repeat(1:144, inner=n_timesteps_per_scenario)

    for col in [:demand, :hydro_inflow]
        max_val = maximum(all_profiles_df[!, col])
        all_profiles_df[!, col] ./= max_val
    end

    CSV.write(joinpath(input_data_path, "profiles-wide-all-scenarios.csv"), all_profiles_df; writeheader=true)
end

main()

# config = TOML.parsefile("C:/Users/fjlaseur/Tulipa/Experiments_CVaR/config.toml")
# number_of_scenarios = config["simulation"]["number_of_scenarios"] #no more than 144
# profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
# CSV.write(joinpath(input_data_file, "profiles-wide.csv"), profiles_df; writeheader=true)