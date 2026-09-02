using DataFrames
using CSV
using Statistics
using Plots
import PlotlyJS

cd(@__DIR__)

input_file = "input_data/profiles_wide.csv"
output_dir = "analysis_outputs"
mkpath(output_dir)

const INK = "#25313C"
const BLUE = "#3666A0"
const ORANGE = "#D97706"
const GREY = "#8A96A3"
const GREY_LIGHT = "#E8ECF0"

default(
    background_color=:white,
    foreground_color_subplot=INK,
    fontfamily="sans-serif",
    gridalpha=0.18,
    legend_background_color=:transparent,
    legend_foreground_color=:transparent,
    titlelocation=:left,
    titlefontsize=14,
)

profiles = CSV.read(input_file, DataFrame)

id_cols = ["year", "timestep", "scenario"]
value_cols = setdiff(String.(names(profiles)), id_cols)

function canonical_series_name(raw::AbstractString)
    normalized = uppercase(strip(String(raw)))
    if normalized in ("E_DEMAND", "DEMAND")
        return "E_Demand"
    elseif normalized in ("SOLAR", "SUNPV", "PV")
        return "Solar"
    elseif normalized in ("WIND_ONSHORE", "WINDON", "WIND_ON")
        return "Wind_Onshore"
    elseif normalized in ("WIND_OFFSHORE", "WINDOFF", "WIND_OFF")
        return "Wind_Offshore"
    end
    return strip(String(raw))
end

function parse_profile_column(col::AbstractString)
    parts = split(String(col), "_")
    length(parts) < 2 && return nothing

    country = strip(parts[1])
    series_parts = parts[2:end]

    # New inputs append a projection year as the final token, e.g. AUS_WINDON_2050.
    if !isempty(series_parts) && occursin(r"^\d{4}$", series_parts[end])
        series_parts = series_parts[1:(end-1)]
    end
    isempty(series_parts) && return nothing

    return (country=String(country), series=canonical_series_name(join(series_parts, "_")))
end

meta_rows = NamedTuple{(:column, :country, :series)}[]
for col in value_cols
    parsed = parse_profile_column(col)
    if parsed !== nothing
        push!(meta_rows, (column=col, country=parsed.country, series=parsed.series))
    end
end
meta = DataFrame(meta_rows)

vre_series = Set([
    "Solar",
    "Wind_Onshore",
    "Wind_Offshore",
])

demand_cols = meta.column[meta.series .== "E_Demand"]
renewable_cols = meta.column[in.(meta.series, Ref(vre_series))]
scenarios = sort(unique(profiles.scenario))
countries = sort(unique(meta.country))

mapping_file = joinpath("input_data", "ERAA-scenario-mapping.csv")
scenario_mapping = CSV.read(mapping_file, DataFrame)

id_col = "Weather scenario ID"
model_col = "C3S climate projection model"
target_year_col = "C3S climate projection target year"

mapping_by_id = Dict(
    strip(string(r[id_col])) => (
        model=strip(string(r[model_col])),
        target_year=Int(r[target_year_col]),
    ) for r in eachrow(scenario_mapping)
)

function short_model_name(model::AbstractString)
    m = match(r"\(([^)]+)\)\s*$", model)
    return m === nothing ? String(model) : String(m.captures[1])
end

scenario_to_weather_year = Dict{eltype(scenarios),Union{Missing,Int}}()
scenario_to_model = Dict{eltype(scenarios),String}()
scenario_to_model_short = Dict{eltype(scenarios),String}()
scenario_to_label = Dict{eltype(scenarios),String}()
for sc in scenarios
    ws_id = startswith(string(sc), "WS") ? string(sc) : "WS$(sc)"
    if haskey(mapping_by_id, ws_id)
        m = mapping_by_id[ws_id]
        scenario_to_weather_year[sc] = m.target_year
        scenario_to_model[sc] = m.model
        scenario_to_model_short[sc] = short_model_name(m.model)
        scenario_to_label[sc] = "$(ws_id) · $(short_model_name(m.model)) · $(m.target_year)"
    else
        # Fallback keeps downstream output working if an unexpected scenario ID appears.
        scenario_to_weather_year[sc] = missing
        scenario_to_model[sc] = "Mapping missing"
        scenario_to_model_short[sc] = "Unknown"
        scenario_to_label[sc] = "$(ws_id) (mapping missing)"
    end
end

label_for(s) = get(scenario_to_label, s, "Scenario $(s)")
weather_year_for(s) = get(scenario_to_weather_year, s, missing)
model_for(s) = get(scenario_to_model, s, "Mapping missing")
model_short_for(s) = get(scenario_to_model_short, s, "Unknown")

month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

country_code_to_iso3 = Dict(
    "AT" => "AUT",
    "AUS" => "AUT",
    "BE" => "BEL",
    "BEL" => "BEL",
    "BG" => "BGR",
    "BLK" => "BGR",
    "HR" => "HRV",
    "CY" => "CYP",
    "CZ" => "CZE",
    "CZE" => "CZE",
    "DK" => "DNK",
    "DEN" => "DNK",
    "EE" => "EST",
    "FI" => "FIN",
    "FIN" => "FIN",
    "FR" => "FRA",
    "FRA" => "FRA",
    "DE" => "DEU",
    "GER" => "DEU",
    "GR" => "GRC",
    "HU" => "HUN",
    "IE" => "IRL",
    "IRE" => "IRL",
    "IT" => "ITA",
    "ITA" => "ITA",
    "LV" => "LVA",
    "LT" => "LTU",
    "BLT" => "LTU",
    "LU" => "LUX",
    "MT" => "MLT",
    "NL" => "NLD",
    "NED" => "NLD",
    "PL" => "POL",
    "POL" => "POL",
    "PT" => "PRT",
    "POR" => "PRT",
    "RO" => "ROU",
    "SK" => "SVK",
    "SKO" => "SVK",
    "SI" => "SVN",
    "ES" => "ESP",
    "SPA" => "ESP",
    "SE" => "SWE",
    "SWE" => "SWE",
    "UK" => "GBR",
    "UKI" => "GBR",
    "NO" => "NOR",
    "NOR" => "NOR",
    "CH" => "CHE",
    "SWI" => "CHE",
)

function iso3_country_code(country::AbstractString)
    code = uppercase(strip(String(country)))
    if haskey(country_code_to_iso3, code)
        return country_code_to_iso3[code]
    elseif length(code) == 3
        return code
    end
    return nothing
end

mapped_codes = Set(keys(country_code_to_iso3))
unmapped_country_codes = sort([c for c in countries if !(uppercase(strip(String(c))) in mapped_codes)])
if isempty(unmapped_country_codes)
    println("Country-code mapping check: all $(length(countries)) country codes are explicitly mapped.")
else
    println("WARNING: Unmapped country codes detected in profiles_wide: $(join(unmapped_country_codes, ", ")).")
    println("These will use ISO-3 fallback where possible; consider adding explicit entries to country_code_to_iso3.")
end

function month_from_timestep(timestep::Int)
    day_of_year = ((timestep - 1) ÷ 24) + 1
    return searchsortedlast(month_starts, day_of_year)
end

function safe_colmean(df::DataFrame, cols::Vector{String})
    isempty(cols) && return NaN
    return mean(Matrix(df[:, cols]))
end

function timestep_index(df::DataFrame, cols::Vector{String})
    isempty(cols) && return fill(NaN, nrow(df))
    return vec(mean(Matrix(df[:, cols]), dims=2))
end

function country_cols(meta::DataFrame, country::String, allowed_series::Set{String})
    return meta.column[(meta.country .== country) .& in.(meta.series, Ref(allowed_series))]
end

function longest_true_run(values::AbstractVector{Bool})
    longest = 0
    current = 0
    for value in values
        current = value ? current + 1 : 0
        longest = max(longest, current)
    end
    return longest
end

function densest_window(values::AbstractVector{Bool}, width::Int)
    window_width = min(width, length(values))
    counts = cumsum(vcat(0, Int.(values)))
    window_counts = counts[(window_width+1):end] .- counts[1:(end-window_width)]
    start_index = argmax(window_counts)
    return start_index, start_index + window_width - 1, window_counts[start_index]
end

@assert longest_true_run([false, true, true, false, true]) == 2
@assert densest_window([true, false, true, true], 2) == (3, 4, 2)

scenario_rows = NamedTuple[]
ts_rows = NamedTuple[]

global_renew_ts = Float64[]
global_demand_ts = Float64[]

for sc in scenarios
    df_s = profiles[profiles.scenario .== sc, :]
    sort!(df_s, :timestep)

    renew_ts = timestep_index(df_s, renewable_cols)
    demand_ts = timestep_index(df_s, demand_cols)

    append!(global_renew_ts, renew_ts)
    append!(global_demand_ts, demand_ts)

    renew_mean = mean(renew_ts)
    renew_std = std(renew_ts)
    renew_p10 = quantile(renew_ts, 0.10)
    renew_p90 = quantile(renew_ts, 0.90)

    demand_mean = mean(demand_ts)
    demand_std = std(demand_ts)
    demand_p10 = quantile(demand_ts, 0.10)
    demand_p90 = quantile(demand_ts, 0.90)

    for i in eachindex(renew_ts)
        push!(ts_rows, (
            scenario=sc,
            weather_year=weather_year_for(sc),
            climate_model=model_for(sc),
            climate_model_short=model_short_for(sc),
            scenario_label=label_for(sc),
            timestep=df_s.timestep[i],
            month=month_from_timestep(df_s.timestep[i]),
            renewable_index=renew_ts[i],
            demand_index=demand_ts[i],
            pressure_proxy=demand_ts[i] - renew_ts[i],
        ))
    end

    push!(scenario_rows, (
        scenario=sc,
        weather_year=weather_year_for(sc),
        climate_model=model_for(sc),
        climate_model_short=model_short_for(sc),
        scenario_label=label_for(sc),
        renewable_mean=renew_mean,
        renewable_std=renew_std,
        renewable_p10=renew_p10,
        renewable_p90=renew_p90,
        renewable_range_p90_p10=renew_p90 - renew_p10,
        demand_mean=demand_mean,
        demand_std=demand_std,
        demand_p10=demand_p10,
        demand_p90=demand_p90,
        demand_range_p90_p10=demand_p90 - demand_p10,
    ))
end

scenario_summary = DataFrame(scenario_rows)
ts_metrics = DataFrame(ts_rows)

global_low_renew_thr = quantile(global_renew_ts, 0.10)
global_high_demand_thr = quantile(global_demand_ts, 0.90)
ts_metrics.stress_proxy = (ts_metrics.renewable_index .<= global_low_renew_thr) .&
                          (ts_metrics.demand_index .>= global_high_demand_thr)

stress_rows = NamedTuple[]
for sc in scenarios
    ts_s = ts_metrics[ts_metrics.scenario .== sc, :]
    stress_hours = count(ts_s.stress_proxy)
    _, _, max_7d_stress_hours = densest_window(ts_s.stress_proxy, 7 * 24)
    push!(stress_rows, (
        scenario=sc,
        stress_hours=stress_hours,
        stress_share=stress_hours / nrow(ts_s),
        longest_stress_run=longest_true_run(ts_s.stress_proxy),
        max_7d_stress_hours=max_7d_stress_hours,
        pressure_p95=quantile(ts_s.pressure_proxy, 0.95),
    ))
end

stress_summary = DataFrame(stress_rows)
scenario_summary = leftjoin(scenario_summary, stress_summary, on=:scenario)

country_rows = NamedTuple[]
for sc in scenarios
    df_s = profiles[profiles.scenario .== sc, :]
    for c in countries
        r_cols = country_cols(meta, c, vre_series)
        d_cols = country_cols(meta, c, Set(["E_Demand"]))

        renew_mean = safe_colmean(df_s, r_cols)
        demand_mean = safe_colmean(df_s, d_cols)

        if !isnan(renew_mean)
            push!(country_rows, (
                scenario=sc,
                weather_year=weather_year_for(sc),
                climate_model=model_for(sc),
                climate_model_short=model_short_for(sc),
                scenario_label=label_for(sc),
                country=c,
                renewable_mean=renew_mean,
                demand_mean=demand_mean,
            ))
        end
    end
end

country_scenario_summary = DataFrame(country_rows)

country_scenario_extremes = combine(
    groupby(country_scenario_summary, :country),
    :renewable_mean => minimum => :min_renewable_across_scenarios,
    :renewable_mean => maximum => :max_renewable_across_scenarios,
)
country_scenario_extremes.variation = country_scenario_extremes.max_renewable_across_scenarios .- country_scenario_extremes.min_renewable_across_scenarios
sort!(country_scenario_extremes, :variation, rev=true)

technology_rows = NamedTuple[]
for sc in scenarios
    df_s = profiles[profiles.scenario .== sc, :]
    for series_name in sort(unique(meta.series))
        cols = meta.column[meta.series .== series_name]
        vals = vec(Matrix(df_s[:, cols]))
        push!(technology_rows, (
            scenario=sc,
            weather_year=weather_year_for(sc),
            scenario_label=label_for(sc),
            series=series_name,
            mean_value=mean(vals),
            std_value=std(vals),
            p10=quantile(vals, 0.10),
            p90=quantile(vals, 0.90),
        ))
    end
end
technology_summary = DataFrame(technology_rows)

scenario_legend = DataFrame(
    scenario=scenarios,
    weather_year=[weather_year_for(sc) for sc in scenarios],
    climate_model=[model_for(sc) for sc in scenarios],
    climate_model_short=[model_short_for(sc) for sc in scenarios],
    scenario_label=[label_for(sc) for sc in scenarios],
)

monthly_g = groupby(
    ts_metrics,
    [:scenario, :weather_year, :climate_model, :climate_model_short, :scenario_label, :month],
)
monthly_summary = combine(
    monthly_g,
    :renewable_index => mean => :renewable_mean,
    :demand_index => mean => :demand_mean,
    :pressure_proxy => mean => :pressure_mean,
    :stress_proxy => sum => :stress_hours,
)
monthly_q = combine(
    monthly_g,
    :renewable_index => (x -> quantile(x, 0.10)) => :renewable_p10,
    :renewable_index => (x -> quantile(x, 0.90)) => :renewable_p90,
    :demand_index => (x -> quantile(x, 0.10)) => :demand_p10,
    :demand_index => (x -> quantile(x, 0.90)) => :demand_p90,
)
monthly_summary = leftjoin(
    monthly_summary,
    monthly_q,
    on=[:scenario, :weather_year, :climate_model, :climate_model_short, :scenario_label, :month],
)
sort!(monthly_summary, [:scenario, :month])
monthly_summary.month_name = month_names[monthly_summary.month]

monthly_pressure_envelope = combine(
    groupby(monthly_summary, :month),
    :pressure_mean => (x -> quantile(x, 0.10)) => :pressure_p10,
    :pressure_mean => median => :pressure_median,
    :pressure_mean => (x -> quantile(x, 0.90)) => :pressure_p90,
)
sort!(monthly_pressure_envelope, :month)
monthly_pressure_envelope.month_name = month_names[monthly_pressure_envelope.month]

country_ranking_summary = combine(
    groupby(country_scenario_summary, :country),
    :renewable_mean => mean => :avg_renewable_mean,
    :demand_mean => mean => :avg_demand_mean,
    :renewable_mean => minimum => :min_renewable_mean,
    :renewable_mean => maximum => :max_renewable_mean,
)
country_ranking_summary.renewable_variation = country_ranking_summary.max_renewable_mean .- country_ranking_summary.min_renewable_mean
sort!(country_ranking_summary, :renewable_variation, rev=true)

CSV.write(joinpath(output_dir, "scenario_summary.csv"), scenario_summary)
CSV.write(joinpath(output_dir, "scenario_legend.csv"), scenario_legend)
CSV.write(joinpath(output_dir, "country_scenario_summary.csv"), country_scenario_summary)
CSV.write(joinpath(output_dir, "country_scenario_extremes.csv"), country_scenario_extremes)
CSV.write(joinpath(output_dir, "country_ranking_summary.csv"), country_ranking_summary)
CSV.write(joinpath(output_dir, "technology_summary.csv"), technology_summary)
CSV.write(joinpath(output_dir, "monthly_summary.csv"), monthly_summary)
CSV.write(joinpath(output_dir, "monthly_pressure_envelope.csv"), monthly_pressure_envelope)
CSV.write(joinpath(output_dir, "timestep_metrics.csv"), ts_metrics)

most_renewable = scenario_summary.scenario[argmax(scenario_summary.renewable_mean)]
least_renewable = scenario_summary.scenario[argmin(scenario_summary.renewable_mean)]
most_variable_renew = scenario_summary.scenario[argmax(scenario_summary.renewable_range_p90_p10)]
most_stress = scenario_summary.scenario[argmax(scenario_summary.stress_hours)]
least_stress = scenario_summary.scenario[argmin(scenario_summary.stress_hours)]

worst_stress_hours = maximum(scenario_summary.stress_hours)
median_stress_hours = median(scenario_summary.stress_hours)
tail_multiple = median_stress_hours > 0 ? worst_stress_hours / median_stress_hours : Inf

top_countries_var = first(country_scenario_extremes, min(10, nrow(country_scenario_extremes)))

key_scenario_maps = [
    (slug="most_stress_hours", description="VRE availability in the highest-stress proxy scenario", scenario=most_stress),
    (slug="least_stress_hours", description="VRE availability in the lowest-stress proxy scenario", scenario=least_stress),
]

map_files = ["plot_eu_choropleth_$(item.slug).html" for item in key_scenario_maps]

report_file = joinpath(output_dir, "eda_findings.md")
open(report_file, "w") do io
    println(io, "# Why uncertainty belongs in the optimization")
    println(io)
    println(io, "## Management takeaway")
    println(io, "- The planning decision is made once, but it must perform across **$(length(scenarios)) weather realisations** from $(length(unique(scenario_legend.climate_model_short))) climate models and $(length(unique(scenario_legend.weather_year))) target years.")
    println(io, "- The median realisation contains **$(round(Int, median_stress_hours)) joint-stress proxy hours**; the worst contains **$(worst_stress_hours)**—**$(round(tail_multiple, digits=1))× the median**.")
    println(io, "- An average profile is useful context but a poor stress test: it blends away the low-VRE/high-demand combinations that drive flexibility needs.")
    println(io, "- Stochastic optimization treats the scenarios as a portfolio of plausible futures, choosing one first-stage plan and allowing scenario-specific operations after uncertainty is revealed.")
    println(io)
    println(io, "## Suggested storyline for managers")
    println(io, "1. **The decision:** we commit capital before knowing which weather realisation will occur.")
    println(io, "2. **The false comfort:** the average year looks smooth, but it is a composite future that never actually occurs.")
    println(io, "3. **The tension:** most scenarios have modest stress, while a small tail creates much greater exposure.")
    println(io, "4. **Make it tangible:** the difficult scenario is not one isolated hour; its stress clusters into demanding weeks.")
    println(io, "5. **The resolution:** optimize the investment once against the full scenario set, then test the resulting plan's cost, reliability, and regret in every scenario.")
    println(io)
    println(io, "A non-technical analogy: choosing a system from the average profile is like choosing an umbrella from annual average rainfall. The average says little about the downpour that determines whether the umbrella works.")
    println(io)
    println(io, "## Slide-ready visual sequence")
    println(io, "1. plot_scenario_risk_matrix.png — introduce the 3 × 12 scenario portfolio and show that risk is not a smooth time trend.")
    println(io, "2. plot_stress_hours_tail.png — reveal the long tail and quantify the worst-versus-median gap.")
    println(io, "3. plot_monthly_pressure_envelope.png — contrast the scenario range with the single median line.")
    println(io, "4. plot_worst_scenario_week.png — make the abstract tail scenario concrete as a difficult operating week.")
    println(io, "5. plot_country_vre_uncertainty.png — show where scenario choice changes the geographic picture most.")
    println(io, "6. executive_summary_dashboard.png — use as a leave-behind; present the individual plots when speaking live.")
    println(io)
    println(io, "## Key scenario maps")
    for (item, file_name) in zip(key_scenario_maps, map_files)
        println(io, "- $(item.description): $(file_name)")
    end
    println(io)
    println(io, "## What is joint-stress exposure?")
    println(io, "- In plain terms, it counts how many hours a scenario faces a difficult combination: **renewable availability is unusually low while demand is unusually high**.")
    println(io, "- How it is calculated:")
    println(io, "- Build Europe-wide hourly indices: an unweighted mean renewable index and an unweighted mean demand index across all countries.")
    println(io, "- Mark an hour as joint-stress if renewable is in the lowest 10% (pooled p10) and demand is in the highest 10% (pooled p90), then count those marked hours per scenario.")
    println(io, "- So, a scenario's joint-stress exposure is simply the **number of marked hours** in that scenario.")
    println(io)
    println(io, "## What the EDA says")
    println(io, "- Highest joint-stress exposure: **$(label_for(most_stress))** with **$(worst_stress_hours) hours**.")
    println(io, "- Lowest joint-stress exposure: **$(label_for(least_stress))**.")
    println(io, "- Highest annual mean unweighted VRE availability: **$(label_for(most_renewable))**.")
    println(io, "- Lowest annual mean unweighted VRE availability: **$(label_for(least_renewable))**.")
    println(io, "- Largest within-scenario hourly VRE p90-p10 range: **$(label_for(most_variable_renew))**.")
    println(io)
    println(io, "Countries with the largest scenario-to-scenario range in annual mean unweighted VRE availability:")
    for r in eachrow(top_countries_var)
        println(io, "- $(r.country): variation = $(round(r.variation, digits = 4))")
    end
    println(io)
    println(io, "## What this EDA cannot prove")
    println(io, "- The VRE and demand measures are **unweighted screening indices**, not MW or MWh. Capacity, technology, and country weights are not available here.")
    println(io, "- A stress-proxy hour is one where the Europe-wide unweighted VRE index is at or below its pooled p10 and the unweighted demand index is at or above its pooled p90.")
    println(io, "- Hydro and inflow profiles are kept out of the VRE index because they have different physical meanings; inspect them separately in technology_summary.csv.")
    println(io, "- Scenarios are shown with equal visual weight because no probabilities were supplied. The charts are scenario comparisons, not probability estimates.")
    println(io, "- This profile-only EDA does not quantify investment adequacy, reliability, congestion, imports, or the value of stochastic optimization.")
    println(io)
    println(io, "## Evidence to add from optimization results")
    println(io, "- **Value of the stochastic solution (VSS):** expected cost of the average-profile plan when tested across all scenarios minus expected cost of the stochastic plan.")
    println(io, "- **Regret distribution:** scenario cost gap versus the scenario-perfect plan; report median, p90, and worst case.")
    println(io, "- **Reliability:** expected and worst-case unserved energy, scarcity hours, and reserve shortfall.")
    println(io, "- **Risk appetite:** compare expected-cost and CVaR-efficient plans to show the price of tail protection.")
    println(io)
    println(io, "## Dataset and reproducibility")
    println(io, "- Planning year(s): $(join(sort(unique(profiles.year)), ", "))")
    println(io, "- Rows: $(nrow(profiles)); columns: $(ncol(profiles)); countries: $(length(countries)).")
    scenario_sizes = combine(groupby(profiles, :scenario), nrow => :n).n
    println(io, "- Scenarios: $(length(scenarios)); timesteps per scenario: $(minimum(scenario_sizes)) to $(maximum(scenario_sizes)).")
    println(io, "- Scenario mapping: $(mapping_file).")
    println(io, "- scenario_legend.csv includes scenario, climate model, and projection target year.")
    println(io, "- monthly_pressure_envelope.csv contains the cross-scenario monthly p10, median, and p90 pressure proxy.")
end

# Plot 1: Scenario structure and stress exposure
model_order = unique(scenario_legend.climate_model_short)
weather_year_order = sort(unique(collect(skipmissing(scenario_legend.weather_year))))
risk_matrix = fill(NaN, length(model_order), length(weather_year_order))
for r in eachrow(scenario_summary)
    ismissing(r.weather_year) && continue
    model_index = findfirst(==(r.climate_model_short), model_order)
    year_index = findfirst(==(r.weather_year), weather_year_order)
    risk_matrix[model_index, year_index] = r.stress_hours
end

p1 = heatmap(
    weather_year_order,
    1:length(model_order),
    risk_matrix,
    yticks=(1:length(model_order), model_order),
    color=cgrad([GREY_LIGHT, ORANGE]),
    clims=(0, worst_stress_hours),
    colorbar_title="Hours",
    title="Scenario risk is irregular across climate models and target years",
    xlabel="",
    ylabel="",
    xticks=(weather_year_order, string.(weather_year_order)),
    framestyle=:box,
    grid=false,
    size=(1100, 430),
)
for r in eachrow(scenario_summary)
    ismissing(r.weather_year) && continue
    model_index = findfirst(==(r.climate_model_short), model_order)
    label_color = r.stress_hours >= 0.55 * worst_stress_hours ? :white : INK
    annotate!(p1, r.weather_year, model_index, text(string(r.stress_hours), 8, label_color))
end
savefig(p1, joinpath(output_dir, "plot_scenario_risk_matrix.png"))

# Plot 2: Ordered tail of stress exposure
stress_rank = sort(scenario_summary, :stress_hours)
tail_cutoff = quantile(stress_rank.stress_hours, 0.90)
bar_colors = ifelse.(stress_rank.stress_hours .>= tail_cutoff, ORANGE, GREY_LIGHT)
p2 = bar(
    1:nrow(stress_rank),
    stress_rank.stress_hours,
    color=bar_colors,
    linecolor=:transparent,
    label=false,
    title="The worst weather future creates $(round(tail_multiple, digits=1))× the median stress exposure\nBars = joint-stress proxy hours; all $(length(scenarios)) scenarios shown once; top decile highlighted",
    xlabel="Scenarios ordered from lowest to highest exposure",
    ylabel="",
    xticks=(1:5:nrow(stress_rank), string.(1:5:nrow(stress_rank))),
    ylim=(0, 1.28 * worst_stress_hours),
    framestyle=:semi,
    grid=:y,
    size=(1100, 540),
)
hline!(
    p2,
    [median_stress_hours],
    color=BLUE,
    linestyle=:dash,
    linewidth=2,
    label="Median = $(round(Int, median_stress_hours)) hours",
)
for (position, rank) in enumerate((nrow(stress_rank)-2):nrow(stress_rank))
    annotate!(
        p2,
        rank,
        stress_rank.stress_hours[rank] + 3 + 5 * (position - 1),
        text("$(stress_rank.stress_hours[rank]) h", 8, INK),
    )
end
plot!(p2, legend=:topleft)
savefig(p2, joinpath(output_dir, "plot_stress_hours_tail.png"))

# Plot 3: Monthly uncertainty envelope instead of 36 overlapping lines
worst_monthly = monthly_summary[monthly_summary.scenario .== most_stress, :]
p3 = plot(
    monthly_pressure_envelope.month,
    monthly_pressure_envelope.pressure_median,
    ribbon=(
        monthly_pressure_envelope.pressure_median .- monthly_pressure_envelope.pressure_p10,
        monthly_pressure_envelope.pressure_p90 .- monthly_pressure_envelope.pressure_median,
    ),
    color=BLUE,
    fillcolor=BLUE,
    fillalpha=0.18,
    linewidth=3,
    label="Scenario median and p10-p90 band",
    title="The average profile hides the monthly pressure range\nDemand minus unweighted VRE availability; focused vertical scale",
    xlabel="Month",
    ylabel="",
    xticks=(1:12, month_names),
    framestyle=:semi,
    grid=:y,
    size=(1250, 520),
)
plot!(
    p3,
    worst_monthly.month,
    worst_monthly.pressure_mean,
    color=ORANGE,
    linestyle=:dash,
    linewidth=3,
    marker=:circle,
    markersize=4,
    label="Highest-stress scenario: $(label_for(most_stress))",
)
plot!(p3, legend=:outertop)
savefig(p3, joinpath(output_dir, "plot_monthly_pressure_envelope.png"))

# Plot 4: Make the tail scenario tangible as a difficult week
worst_ts = ts_metrics[ts_metrics.scenario .== most_stress, :]
sort!(worst_ts, :timestep)
week_start, week_end, week_stress_hours = densest_window(worst_ts.stress_proxy, 7 * 24)
worst_week = worst_ts[week_start:week_end, :]
week_hour = 0:(nrow(worst_week)-1)
stress_points = findall(worst_week.stress_proxy)

p4 = plot(
    week_hour,
    worst_week.demand_index,
    color=BLUE,
    linewidth=2.5,
    label="Demand index",
    title="A difficult week is a sustained mismatch, not one bad hour\n$(label_for(most_stress)): $(week_stress_hours) joint-stress proxy hours in the densest 7-day window",
    xlabel="Hour in selected week",
    ylabel="",
    xticks=(0:24:144, ["Day $(day)" for day in 1:7]),
    framestyle=:semi,
    grid=:y,
    size=(1250, 520),
)
plot!(
    p4,
    week_hour,
    worst_week.renewable_index,
    color=GREY,
    linestyle=:dash,
    linewidth=2.5,
    label="VRE availability index",
)
scatter!(
    p4,
    week_hour[stress_points],
    worst_week.demand_index[stress_points],
    color=ORANGE,
    marker=:diamond,
    markersize=5,
    label="Joint-stress proxy hour",
)
plot!(p4, legend=:outertopright)
savefig(p4, joinpath(output_dir, "plot_worst_scenario_week.png"))

# Plot 5: Show geographic exposure as ranges, not an invalid energy balance
country_plot = top_countries_var[end:-1:1, :]
country_positions = 1:nrow(country_plot)
country_min = country_plot.min_renewable_across_scenarios
country_max = country_plot.max_renewable_across_scenarios
p5 = plot(
    title="Geography changes where uncertainty matters most\nTop countries by scenario range in annual mean unweighted VRE availability",
    xlabel="Annual mean VRE availability index",
    yticks=(country_positions, country_plot.country),
    ylabel="",
    framestyle=:semi,
    grid=:x,
    size=(1000, 590),
)
for i in country_positions
    plot!(p5, [country_min[i], country_max[i]], [i, i], color=GREY, linewidth=3, label=false)
end
scatter!(
    p5,
    country_min,
    country_positions,
    markercolor=:white,
    markerstrokecolor=GREY,
    markerstrokewidth=2,
    markersize=6,
    label="Lowest scenario",
)
scatter!(
    p5,
    country_max,
    country_positions,
    color=BLUE,
    markersize=6,
    label="Highest scenario",
)
plot!(p5, legend=:bottomright)
savefig(p5, joinpath(output_dir, "plot_country_vre_uncertainty.png"))

# Interactive maps: two contrasting scenarios with a common color scale
map_zmin = minimum(country_scenario_summary.renewable_mean)
map_zmax = maximum(country_scenario_summary.renewable_mean)
for item in key_scenario_maps
    cs = country_scenario_summary[country_scenario_summary.scenario .== item.scenario, [:country, :renewable_mean]]
    locations = String[]
    z_values = Float64[]
    hover_text = String[]

    for r in eachrow(cs)
        country_code = uppercase(strip(String(r.country)))
        iso3 = iso3_country_code(country_code)
        if iso3 !== nothing && !isnan(r.renewable_mean)
            push!(locations, iso3)
            push!(z_values, r.renewable_mean)
            push!(hover_text, "$(country_code): $(round(r.renewable_mean, digits=4))")
        end
    end

    if isempty(locations)
        @warn "No EU country data found for choropleth map" scenario=item.scenario
        continue
    end

    trace = PlotlyJS.choropleth(
        locationmode="ISO-3",
        locations=locations,
        z=z_values,
        text=hover_text,
        zmin=map_zmin,
        zmax=map_zmax,
        colorscale="YlGnBu",
        marker=PlotlyJS.attr(line=PlotlyJS.attr(color="white", width=0.6)),
        colorbar=PlotlyJS.attr(title="Unweighted VRE index"),
        hovertemplate="%{text}<extra></extra>",
    )

    layout = PlotlyJS.Layout(
        title="$(item.description)<br><sup>$(label_for(item.scenario))</sup>",
        geo=PlotlyJS.attr(
            scope="europe",
            projection=PlotlyJS.attr(type="mercator"),
            fitbounds="locations",
            showcountries=true,
            showcoastlines=true,
            showland=true,
            landcolor="rgb(245, 245, 245)",
        ),
        width=1100,
        height=700,
        margin=PlotlyJS.attr(l=20, r=20, t=80, b=20),
    )

    fig = PlotlyJS.plot(trace, layout)
    out_file = joinpath(output_dir, "plot_eu_choropleth_$(item.slug).html")
    PlotlyJS.savefig(fig, out_file)
end

# Two self-reinforcing visuals for a leave-behind; use individual charts when presenting live.
dashboard = plot(
    p2,
    p3,
    layout=grid(2, 1, heights=[0.52, 0.48]),
    size=(1300, 1050),
    plot_title="Why uncertainty belongs in the optimization",
    plot_titlefontsize=18,
)
savefig(dashboard, joinpath(output_dir, "executive_summary_dashboard.png"))

println("EDA completed.")
println("Results written to: $(abspath(output_dir))")
println("Main report: $(joinpath(output_dir, "eda_findings.md"))")
