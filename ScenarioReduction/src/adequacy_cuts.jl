# Per-scenario capacity-adequacy cuts for SD screening.
#
# Motivation: the SD loop fixes ONE investment and solves the operational problem
# against the selected scenarios. Most mean-centered samples are infeasible purely
# because firm+VRE capacity cannot meet demand at peak-demand / low-renewable hours.
# That failure mode is a NECESSARY linear condition we can check in microseconds,
# so we use it to (a) reject doomed samples before the expensive LP and (b) shift
# the sampling centre onto the feasible frontier.
#
# Derivation (MW-space). The model's hourly e_demand balance is an EQUALITY
#   Σ_g flow(g→e_demand) (− optional loads) == peak_demand · demand_h.
# Each inflow is capped by its capacity constraint flow(g→) ≤ availability_{g,h}·x_g
# (thermal availability = 1). Summing the per-source upper bounds gives an upper
# bound on deliverable supply. Dropping the optional electrolyzer/charging loads
# only shrinks the required side. Hence:
#   supply_ub_h(x) = x_ccgt + x_ocgt + a_solar·x_solar + a_won·x_wind
#                  + a_woff·x_wind_offshore + x_battery + HYDRO_CAP + ENS_CAP
#   required_h     = PEAK_DEMAND · demand_h
# If supply_ub_h(x) < required_h for ANY hour, x is PROVABLY infeasible (the most
# optimistic supply can't meet the least optimistic demand). This is a sound
# one-sided filter: it never rejects a truly feasible portfolio, only LP-screens
# the survivors.
#
# This file depends only on DataFrames + base so it can be included in the
# solver-free unit-test environment. `feasibility_center` (a tiny LP) and the
# CSV writers reference `JuMP`/`CSV` from the includer's scope (exactly like
# src/utils.jl references TIO/JuMP/DuckDB without importing them) — so DO NOT add
# `using JuMP`/`using CSV` here, or the test include would fail.

using DataFrames: DataFrame, nrow, eachrow

# `investment_mapping.jl` (defining INVESTABLE_ASSETS, sample_vector_by_asset) is
# included before this file by src/utils.jl and by the test setup module.

"""Adequacy parameters read from the input data (no hardcoding at the call site)."""
struct AdequacyParams
    peak_demand::Float64   # demand multiplier (e_demand peak_demand in asset-milestone.csv)
    hydro_cap::Float64     # MW, hydro_reservoir output capacity
    ens_cap::Float64       # MW, energy-not-served slack capacity
end

AdequacyParams(; peak_demand=1.5, hydro_cap=0.1, ens_cap=2.0) =
    AdequacyParams(peak_demand, hydro_cap, ens_cap)

"""Compact set of adequacy cuts for one scenario: `A·x ≥ b`, x in INVESTABLE_ASSETS order."""
struct AdequacyCuts
    scenario::Int
    A::Matrix{Float64}        # (n_kept × length(assets)) LHS coefficients
    b::Vector{Float64}        # RHS
    timesteps::Vector{Int}    # kept demanding hours (for inspection / CSV)
    demand::Vector{Float64}   # raw demand at kept hours
    a_solar::Vector{Float64}
    a_won::Vector{Float64}
    a_woff::Vector{Float64}
end

# LHS coefficient row in INVESTABLE_ASSETS order
# [ccgt, ocgt, solar, wind, wind_offshore, electrolizer, battery].
# Thermal & battery deliver up to their full invested MW (coeff 1); VRE is scaled
# by availability; electrolyzer is a load (does not supply e_demand) → coeff 0.
function _cut_coeff_row(a_solar::Float64, a_won::Float64, a_woff::Float64)
    return Float64[1.0, 1.0, a_solar, a_won, a_woff, 0.0, 1.0]
end

# Return indices of the non-dominated ("demanding") hours. Hour h is dominated by
# h' iff h' is at least as hard in every coordinate (demand ≥, each availability ≤)
# and strictly harder in at least one; exact ties keep the lowest index.
function _nondominated_indices(
    demand::Vector{Float64},
    a_solar::Vector{Float64},
    a_won::Vector{Float64},
    a_woff::Vector{Float64},
)
    n = length(demand)
    keep = trues(n)
    @inbounds for i in 1:n
        keep[i] || continue
        for j in 1:n
            (i == j || !keep[j]) && continue
            harder_eq = demand[j] >= demand[i] &&
                        a_solar[j] <= a_solar[i] &&
                        a_won[j]   <= a_won[i] &&
                        a_woff[j]  <= a_woff[i]
            harder_eq || continue
            strict = demand[j] > demand[i] ||
                     a_solar[j] < a_solar[i] ||
                     a_won[j]   < a_won[i] ||
                     a_woff[j]  < a_woff[i]
            if strict || j < i        # j dominates i (or identical, keep lower index)
                keep[i] = false
                break
            end
        end
    end
    return findall(keep)
end

"""
    build_adequacy_cuts(profiles_wide_df, scenario_id, params; assets=INVESTABLE_ASSETS)

Build the non-dominated demanding-hour adequacy cuts for one scenario from the wide
hourly profile table (columns `scenario, timestep, solar, wind_onshore,
wind_offshore, demand`). Returns an [`AdequacyCuts`](@ref).
"""
function build_adequacy_cuts(
    profiles_wide_df::DataFrame,
    scenario_id::Integer,
    params::AdequacyParams;
    assets=INVESTABLE_ASSETS,
)
    sdf = profiles_wide_df[profiles_wide_df.scenario .== scenario_id, :]
    nrow(sdf) > 0 || error("No profile rows for scenario $scenario_id")

    timesteps = Int.(sdf.timestep)
    demand   = Float64.(sdf.demand)
    a_solar  = Float64.(sdf.solar)
    a_won    = Float64.(sdf.wind_onshore)
    a_woff   = Float64.(sdf.wind_offshore)

    keep = _nondominated_indices(demand, a_solar, a_won, a_woff)

    nA = length(assets)
    A = Matrix{Float64}(undef, length(keep), nA)
    b = Vector{Float64}(undef, length(keep))
    for (r, h) in enumerate(keep)
        A[r, :] = _cut_coeff_row(a_solar[h], a_won[h], a_woff[h])
        b[r] = params.peak_demand * demand[h] - params.hydro_cap - params.ens_cap
    end

    return AdequacyCuts(
        Int(scenario_id), A, b,
        timesteps[keep], demand[keep], a_solar[keep], a_won[keep], a_woff[keep],
    )
end

"""
    passes_adequacy(x_mw, cuts; tol=1e-6)

`true` iff investment `x_mw` (MW, INVESTABLE_ASSETS order) satisfies every cut, i.e.
is NOT provably infeasible for the scenario. `false` ⇒ definitely infeasible.
"""
function passes_adequacy(x_mw::AbstractVector{<:Real}, cuts::AdequacyCuts; tol=1e-6)
    size(cuts.A, 2) == length(x_mw) ||
        error("x has length $(length(x_mw)); cuts expect $(size(cuts.A, 2))")
    supply = cuts.A * collect(Float64, x_mw)
    @inbounds for r in eachindex(cuts.b)
        supply[r] + tol < cuts.b[r] && return false
    end
    return true
end

"""`true` iff `x_mw` passes every scenario's cuts. Also returns the first binding scenario."""
function adequacy_verdict(x_mw::AbstractVector{<:Real}, cuts_list; tol=1e-6)
    for cuts in cuts_list
        passes_adequacy(x_mw, cuts; tol) || return (passed=false, binding_scenario=cuts.scenario)
    end
    return (passed=true, binding_scenario=0)
end

# NOTE: `feasibility_center` (the minimal-shift LP) lives in `adequacy_center.jl`
# because it uses JuMP *macros*, which expand at include time and so require JuMP
# in scope. Keeping it out of this file lets `adequacy_cuts.jl` be included in the
# solver-free unit-test environment.

"""
    feasibility_center_max_optima(per_scenario_df, scenario_ids; assets=INVESTABLE_ASSETS)

Zero-LP fallback centre: element-wise max over the given scenarios' own optima
(MW). Because the cut coefficients are ≥ 0 and each scenario optimum is feasible
for that scenario, the element-wise max satisfies every scenario's cuts.
"""
function feasibility_center_max_optima(
    per_scenario_df::DataFrame,
    scenario_ids;
    assets=INVESTABLE_ASSETS,
)
    rows = per_scenario_df[in.(per_scenario_df.scenario, Ref(collect(scenario_ids))), :]
    nrow(rows) > 0 || error("No per-scenario rows for scenarios $(collect(scenario_ids))")
    return Float64[maximum(rows[!, Symbol(a)]) for a in assets]
end

"""Read adequacy parameters from the input tables on `connection` (references TIO/DuckDB)."""
function read_adequacy_params(connection)
    asset = DataFrame(TIO.get_table(connection, "asset"))
    capof(name) = Float64(only(asset[string.(asset.asset) .== name, :capacity]))
    peak = try
        am = DataFrame(TIO.get_table(connection, "asset_milestone"))
        Float64(only(am[string.(am.asset) .== "e_demand", :peak_demand]))
    catch
        1.5
    end
    return AdequacyParams(peak, capof("hydro_reservoir"), capof("ens"))
end

# ---- CSV persistence (reference CSV/DataFrame from includer scope) ----

"""Write the demanding-hour frontier for all scenarios to a CSV."""
function save_adequacy_cuts_csv(path::String, cuts_list; assets=INVESTABLE_ASSETS)
    df = DataFrame(;
        scenario=Int[], timestep=Int[], demand=Float64[],
        a_solar=Float64[], a_won=Float64[], a_woff=Float64[], rhs=Float64[],
    )
    for cuts in cuts_list
        for r in eachindex(cuts.b)
            push!(df, (cuts.scenario, cuts.timesteps[r], cuts.demand[r],
                       cuts.a_solar[r], cuts.a_won[r], cuts.a_woff[r], cuts.b[r]))
        end
    end
    CSV.write(path, df)
    return df
end

"""Write the chosen sampling centre (μ*), the per-asset shift, and LP status to a CSV."""
function save_feasibility_center_csv(
    path::String, assets, mean_mw, center, shift, status,
)
    df = DataFrame(;
        asset=collect(assets),
        bounds_mean=Float64.(mean_mw),
        center=center === nothing ? fill(NaN, length(assets)) : Float64.(center),
        shift=shift === nothing ? fill(NaN, length(assets)) : Float64.(shift),
        lp_status=fill(string(status), length(assets)),
    )
    CSV.write(path, df)
    return df
end
