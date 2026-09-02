# read asset.csv and pick up ERAA raw files that end with 2035.csv
# build long profiles (year,timestep,scenario,profile_name,value) for WS01..WS36
# then pivot to wide and save profiles-wide.csv in the requested output folder

using CSV
using DataFrames
using FilePathsBase: mkpath
using Glob

ERAA_DIR = "/Users/floorlaseur/Nextcloud/ExperimentData/EU-input-data/ERAA"
ASSET_CSV = "/Users/floorlaseur/Nextcloud/ExperimentData/EU-input-data/TYNDP/Outputs/tulipa_input_north_sea_TYNDP_NT_ERAA_2035/asset.csv"
OUT_DIR = "/Users/floorlaseur/Nextcloud/ExperimentData/EU-input-data/TYNDP/Outputs/tulipa_input_north_sea_TYNDP_NT_ERAA_2035"
OUT_WIDE = joinpath(OUT_DIR, "profiles-wide.csv")
OUT_LONG = joinpath(OUT_DIR, "profiles-long.csv") # optional, useful for debugging

# scenario column names expected
SCENARIO_COLS = ["WS" * lpad(string(i), 2, '0') for i in 1:36]

function try_read_2035(file)
    # many ERAA CSVs include 10 header rows -> try reading with header=11 first, fallback to default
    try
        return CSV.read(file, DataFrame; header=11)
    catch
        try
            return CSV.read(file, DataFrame)
        catch e
            @warn "Failed to read $file : $e"
            return nothing
        end
    end
end

function find_2035_files()
    # find all files under ERAA_DIR that end with 2035.csv
    all = collect(Glob.glob(joinpath(ERAA_DIR, "**", "*2035.csv")))
    return sort(all)
end

function match_asset_to_file(asset::AbstractString, files::Vector{String})
    # asset like "NL00_Wind_Onshore" -> tokens = ["NL00","Wind","Onshore"]
    toks = split(asset, "_")
    if isempty(toks)
        return nothing
    end
    prefix = toks[1]
    tail = join(toks[2:end], "_")
    # candidate files must include prefix; prefer those that contain the tail tokens
    candidates = filter(f -> occursin(prefix, basename(f)), files)
    if isempty(candidates)
        return nothing
    end
    # rank candidates by how many tail tokens they match
    best = nothing
    best_score = -1
    subtoks = toks[2:end]
    for f in candidates
        name = lowercase(basename(f))
        score = 0
        for t in subtoks
            if occursin(lowercase(t), name)
                score += 1
            end
        end
        if score > best_score
            best = f
            best_score = score
        end
    end
    return best
end

function build_profiles()
    mkpath(OUT_DIR)
    df_assets = CSV.read(ASSET_CSV, DataFrame)
    if !haskey(df_assets, :asset)
        error("asset.csv must contain 'asset' column")
    end
    assets = unique(df_assets.asset)

    files2035 = find_2035_files()
    @info "Found $(length(files2035)) ERAA files for 2035"

    profiles_long = DataFrame(year=Int[], timestep=Int[], scenario=Int[], profile_name=String[], value=Float64[])

    for asset in assets
        f = match_asset_to_file(asset, files2035)
        if f === nothing
            @info "No 2035 ERAA file matched for asset $asset -> skipping"
            continue
        end
        @info "Asset $asset -> matched file $(basename(f))"

        df = try_read_2035(f)
        if df === nothing
            @warn "Could not read file for asset $asset -> skipping"
            continue
        end

        # identify scenario columns present in this file (subset of SCENARIO_COLS)
        present = filter(c -> Symbol(c) in propertynames(df), SCENARIO_COLS)
        if isempty(present)
            @warn "No WS## scenario columns found in $(basename(f)) -> skipping"
            continue
        end

        # determine timestep column if present, otherwise assume 1:nrow(df)
        if :timestep in propertynames(df)
            timesteps = collect(df.timestep)
        elseif :Timestep in propertynames(df)
            timesteps = collect(df.Timestep)
        elseif :hour in propertynames(df)
            timesteps = collect(df.hour)
        else
            timesteps = collect(1:nrow(df))
        end

        n = length(timesteps)
        # ensure each scenario column has length n
        for sc in present
            col = Symbol(sc)
            vals = convert(Vector{Union{Missing,Real}}, df[!, col])
            # coerce missings -> NaN then skip if all missing
            if length(vals) != n
                @warn "Scenario column $sc in $(basename(f)) has unexpected length; skipping this scenario"
                continue
            end
            # convert to Float64 with missing handling
            valsf = Array{Float64}(undef, n)
            for i in 1:n
                v = vals[i]
                if v === missing
                    valsf[i] = NaN
                else
                    valsf[i] = Float64(v)
                end
            end
            scen_index = parse(Int, last(sc)) isa Int ? nothing : nothing # placeholder
            # determine scenario number from sc ("WS01" -> 1)
            scen_num = parse(Int, replace(sc, "WS" => ""))

            # append rows
            for t in 1:n
                push!(profiles_long, (year=2035, timestep=timesteps[t], scenario=scen_num, profile_name=asset, value=valsf[t]))
            end
        end
    end

    if nrow(profiles_long) == 0
        error("No profiles produced - nothing to write")
    end

    # sort and write long CSV (useful)
    sort!(profiles_long, [:year, :scenario, :profile_name, :timestep])
    CSV.write(OUT_LONG, profiles_long; writeheader=true)

    # pivot to wide
    df_wide = unstack(profiles_long, [:year, :timestep, :scenario], :profile_name, :value)
    # write wide CSV (matches the format of provided profiles-wide.csv: year,timestep,<profiles...>,scenario may be last column)
    CSV.write(OUT_WIDE, df_wide; writeheader=true)
    @info "Wrote wide profiles to $OUT_WIDE (rows=$(nrow(df_wide)), cols=$(ncol(df_wide)))"

end