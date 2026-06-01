cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()

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
using Statistics

using DataFrames
solver = Gurobi
case_name = "0_HourlyBenchmark"
balance_df = CSV.read(joinpath(@__DIR__, "outputs", "$case_name", string(solver), "cons_balance_consumer.csv"), DataFrame)
flow_df = CSV.read(joinpath(@__DIR__, "outputs", "$case_name", string(solver), "var_flow.csv"), DataFrame)

using DataFrames

function revenue_per_timestep(balance_df::DataFrame, flow_df::DataFrame)
    join_cols = [
        :milestone_year,
        :rep_period,
        :time_block_start,
        :time_block_end,
        :to_asset,
    ]

    prices = select(
        balance_df,
        :asset => :to_asset,
        :milestone_year,
        :rep_period,
        :time_block_start,
        :time_block_end,
        :dual_balance_consumer => :price,
    )

    flows = select(
        flow_df,
        :from_asset,
        :to_asset,
        :milestone_year,
        :rep_period,
        :time_block_start,
        :time_block_end,
        :solution => :capacity,
    )

    joined = innerjoin(flows, prices, on=join_cols)

    joined[!, :revenue_component] = joined.price .* joined.capacity

    recovery_df = combine(
        groupby(joined, [
            :milestone_year,
            :rep_period,
            :time_block_start,
            :time_block_end,
        ]),
        :revenue_component => sum => :revenue,
    )

    return recovery_df, joined
end

