using DataFrames
using CSV

cd(@__DIR__)

function discover_eraa_root()
    candidates = [
        joinpath(homedir(), "Nextcloud", "ExperimentData", "ERAA"),
        joinpath(@__DIR__, "..", "Nextcloud", "ExperimentData", "ERAA"),
        joinpath(@__DIR__, "Nextcloud", "ExperimentData", "ERAA"),
    ]

    for path in candidates
        if isdir(path)
            return path
        end
    end

    error("Could not locate the ERAA data root. Checked: $(join(candidates, ", "))")
end

function discover_country_codes(availability_dir, demand_dir)
    country_codes = Set{String}()

    for file in readdir(availability_dir)
        if endswith(file, ".csv")
            m = match(r"^([A-Za-z]{2})", file)
            if !isnothing(m)
                push!(country_codes, uppercase(m.captures[1]))
            end
        end
    end

    for file in readdir(demand_dir)
        if endswith(file, ".csv") && occursin("Demand_total", file)
            m = match(r"^([A-Za-z]{2})", file)
            if !isnothing(m)
                push!(country_codes, uppercase(m.captures[1]))
            end
        end
    end

    return sort(collect(country_codes))
end

function find_matching_file(dir::String, pattern::String)
    matches = filter(f -> occursin(pattern, f), readdir(dir))
    if isempty(matches)
        return nothing
    end
    return joinpath(dir, first(matches))
end

function read_capacity_factor_file(path::String)
    return CSV.read(path, DataFrame; header=11)
end

function read_demand_file(path::String)
    return CSV.read(path, DataFrame)
end

function read_hydro_inflow_file(path::String)
    return CSV.read(path, DataFrame)
end

function expand_to_length(values::AbstractVector, target_length::Int)
    if length(values) == target_length
        return collect(values)
    end
    if isempty(values)
        return fill(0.0, target_length)
    end
    repeats = cld(target_length, length(values))
    expanded = repeat(values, outer=repeats)
    return collect(expanded[1:target_length])
end

function build_country_profiles(country_code::String, eraa_root::String, years::Vector{Int}, scenario_cols::Vector{String})
    availability_dir = joinpath(eraa_root, "AvailabilityData", "Capacity Factors_250716")
    demand_dir = joinpath(eraa_root, "DemandData", "Demand timeseries")
    hydro_dir = joinpath(eraa_root, "AvailabilityData", "Hydro Inflows_250704")

    profiles_list_full = Vector{DataFrame}(undef, 144)

    for (i, year) in enumerate(years)
        solar_path = nothing
        wind_onshore_path = nothing
        wind_offshore_path = nothing
        demand_path = nothing
        hydro_path = nothing

        for file in readdir(availability_dir)
            if occursin("$(country_code)", file) && occursin("$(year)", file)
                if occursin("CapacityFactors_PV_utility_tracking", file)
                    solar_path = joinpath(availability_dir, file)
                elseif occursin("CapacityFactors_PV_utility_fixed", file)
                    solar_path = joinpath(availability_dir, file)
                elseif occursin("CapacityFactors_PV_residential_rooftop", file)
                    solar_path = joinpath(availability_dir, file)
                elseif occursin("CapacityFactors_PV_industrial_rooftop", file)
                    solar_path = joinpath(availability_dir, file)
                elseif occursin("CapacityFactors_Wind_Onshore", file)
                    wind_onshore_path = joinpath(availability_dir, file)
                elseif occursin("CapacityFactors_Wind_Offshore", file)
                    wind_offshore_path = joinpath(availability_dir, file)
                end
            end
        end

        for file in readdir(demand_dir)
            if occursin("$(country_code)", file) && occursin("$(year)", file) && occursin("Demand_total", file)
                demand_path = joinpath(demand_dir, file)
                break
            end
        end

        for file in readdir(hydro_dir)
            if occursin("$(country_code)", file) && occursin("$(year)", file) && occursin("Hydro_Inflows", file)
                hydro_path = joinpath(hydro_dir, file)
                break
            end
        end

        if isnothing(solar_path) || isnothing(demand_path)
            @warn "Skipping $(country_code) for $(year): missing solar or demand file"
            continue
        end

        wind_offshore_data = isnothing(wind_offshore_path) ? nothing : read_capacity_factor_file(wind_offshore_path)
        wind_onshore_data = isnothing(wind_onshore_path) ? nothing : read_capacity_factor_file(wind_onshore_path)
        solar_data = read_capacity_factor_file(solar_path)
        demand_data = read_demand_file(demand_path)
        hydro_inflow_daily = isnothing(hydro_path) ? nothing : read_hydro_inflow_file(hydro_path)

        hydro_inflow_hourly = DataFrame()
        if !isnothing(hydro_inflow_daily)
            for ws in scenario_cols
                daily_vals = hydro_inflow_daily[!, Symbol(ws)]
                hydro_inflow_hourly[!, ws] = expand_to_length(repeat(daily_vals, inner=24), nrow(solar_data))
            end
        end

        n_timesteps = nrow(solar_data)
        profiles_list = Vector{DataFrame}(undef, length(scenario_cols))

        for (s, ws) in enumerate(scenario_cols)
            row_data = Dict(
                :milestone_year => fill(year, n_timesteps),
                :scenario => fill(s, n_timesteps),
                :timestep => 1:n_timesteps,
                Symbol("$(country_code)_Solar") => solar_data[!, Symbol(ws)],
            )

            if !isnothing(wind_onshore_data)
                row_data[Symbol("$(country_code)_Wind_Onshore")] = wind_onshore_data[!, Symbol(ws)]
            end
            if !isnothing(wind_offshore_data)
                row_data[Symbol("$(country_code)_Wind_Offshore")] = wind_offshore_data[!, Symbol(ws)]
            end

            row_data[Symbol("$(country_code)_E_Demand")] = demand_data[!, Symbol(ws)]

            if !isnothing(hydro_inflow_daily)
                row_data[Symbol("$(country_code)_Hydro")] = hydro_inflow_hourly[!, Symbol(ws)]
            end

            profiles_list[s] = DataFrame(row_data)
        end

        start_index = (i - 1) * 36 + 1
        stop_index = i * 36
        profiles_list_full[start_index:stop_index] = profiles_list
    end

    # Remove any unused placeholder entries left by skipped years.
    nonmissing_profiles = [df for df in profiles_list_full if !isnothing(df)]
    if isempty(nonmissing_profiles)
        return DataFrame()
    end

    all_profiles_df = vcat(nonmissing_profiles...)
    all_profiles_df[!, :milestone_year] .= 2030
    n_timesteps_per_scenario = div(nrow(all_profiles_df), 144)
    all_profiles_df[!, :scenario] .= repeat(1:144, inner=n_timesteps_per_scenario)

    return all_profiles_df
end

function normalize_profile_columns!(df::DataFrame)
    for col_name in names(df)
        name = string(col_name)
        if !endswith(name, "_E_Demand") && !endswith(name, "_Hydro")
            continue
        end

        max_val = maximum(df[!, col_name])
        if ismissing(max_val) || max_val == 0
            @warn "Skipping normalization for $(name): maximum value is zero or missing"
            continue
        end

        df[!, col_name] ./= max_val
    end

    return df
end

function prepare_eu_profiles()
    eraa_root = discover_eraa_root()
    availability_dir = joinpath(eraa_root, "AvailabilityData", "Capacity Factors_250716")
    demand_dir = joinpath(eraa_root, "DemandData", "Demand timeseries")
    countries = discover_country_codes(availability_dir, demand_dir)
    scenario_cols = ["WS" * lpad(string(j), 2, '0') for j in 1:36]
    years = [2028, 2030, 2033, 2035]

    country_profiles = DataFrame()
    for country in countries
        country_df = build_country_profiles(country, eraa_root, years, scenario_cols)
        if isempty(country_df)
            continue
        end

        asset_cols = select(country_df, Not([:milestone_year, :scenario, :timestep]))
        if isempty(country_profiles)
            country_profiles = select(country_df, [:milestone_year, :scenario, :timestep])
        end

        for col in names(asset_cols)
            country_profiles[!, col] = asset_cols[!, col]
        end
    end

    if isempty(country_profiles)
        error("No profile data could be generated from the ERAA input files")
    end

    normalize_profile_columns!(country_profiles)

    output_path = joinpath(@__DIR__, "create-scenarios", "profiles-wide-all-scenarios-EU.csv")
    CSV.write(output_path, country_profiles; writeheader=true)
    println("Wrote EU scenario profiles to $(output_path)")
end

prepare_eu_profiles()
