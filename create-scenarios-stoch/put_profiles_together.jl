using DataFrames
using CSV

cd(@__DIR__)

demand = CSV.read("input_data/profiles-demand.csv", DataFrame)
availability = CSV.read("input_data/profiles-availability.csv", DataFrame)
inflow = CSV.read("input_data/profiles-inflow.csv", DataFrame)
all_profiles_df = vcat(demand, availability, inflow) # vertical concatenation
sort!(all_profiles_df, [:year, :scenario, :profile_name, :timestep])
CSV.write("input_data/profiles.csv", all_profiles_df; writeheader=true)

df_wide = unstack(all_profiles_df, 
    [:year, :timestep, :scenario],
    :profile_name,
    :value
)
CSV.write("input_data/profiles_wide.csv", df_wide; writeheader=true)
