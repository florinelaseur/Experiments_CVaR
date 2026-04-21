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


