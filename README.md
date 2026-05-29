# Experiments Two-stage Stochastic Optimization with Tulipa

Experiments to test two-stage stochastic formulation with representative periods using [TulipaEnergyModel.jl](https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/)

## Tulipa Setup

Follow the [Tutorial Setup](https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/10-tutorials/11-setting-up/) and clone the current repo to get to work

## The case study runs in main.jl and calls various other files.
Please be aware that a number of scenarios <144 is required before testing. This is due to the size of the problem. The scenarios are randomly selected. Run the benchmark and case studies for this starting set. Then implement a scenario reduction method as you wish. Run the benchmark and case studies for the reduced set and compare results with the starting set. 

## Units of measurement

The model uses the following units of measurement throughout:

- Power: GW
- Energy: GWh
- Cost: MEUR
- Time: hours
- Efficiency: per unit (0 to 1)

## Scenario reduction & adequacy-cut screening

The stochastic-dominance screening, the capacity-adequacy cuts, the feasibility mean-shift,
and the reject-to-target sampling are documented in
[`ScenarioReduction/SCREENING_NOTES.md`](ScenarioReduction/SCREENING_NOTES.md).

### Reference for the adequacy-cut formula

The adequacy cuts (`ScenarioReduction/src/adequacy_cuts.jl`) are derived directly from
**TulipaEnergyModel's consumer balance constraint**, in which the demand term is
`demand_profile × peak_demand`:

> ∑ inflow − ∑ outflow  {=, ≥}  `demand_profile` · `peak_demand`

References:

- Mathematical Formulation (consumer balance), TulipaEnergyModel docs:
  <https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/40-scientific-foundation/40-formulation/>
- Source: `TulipaEnergyModel.jl/src/constraints/consumer.jl` —
  `incoming_flow − outgoing_flow − demand_agg * row.peak_demand ∈ balance_sense`
  (the demand profile is aggregated over the time block with `Statistics.mean`; under
  full-hourly `dummy_cluster!` that block is one hour).
- `peak_demand` field schema (`input-schemas.json`): *"Value that multiplies the demand
  profile time series."*

From this, the per-hour adequacy cut bounds every inflow by its capacity
(`flow ≤ availability · investment`; thermal availability = 1) and requires the maximum
possible supply to meet `peak_demand · demand_profile` at each (non-dominated) demanding
hour — a **sound necessary condition** for the consumer balance to be satisfiable. See
`SCREENING_NOTES.md` §3.1 for the full derivation. The quasi-random sampler's inverse-CDF
uses the Beasley–Springer–Moro approximation (referenced in
`ScenarioReduction/src/sampling.jl`).


