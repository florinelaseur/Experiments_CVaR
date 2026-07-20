
cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")

# Load the required packages
import TulipaIO as TIO
import DuckDB
import CSV
using DataFrames

function main()
    input_data_path = "input_tables/"
    # set up the connection and read the data
    connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(connection_benchmark, input_data_path)

    profiles = TIO.get_table(connection_benchmark, "profiles")
    assets = TIO.get_table(connection_benchmark, "asset_milestone")

    demand_assets = filter(row -> occursin("E_Demand", row.asset), assets)

    println("N of E demands ", nrow(demand_assets))

    df = innerjoin(
        profiles,
        select(demand_assets, [:asset, :peak_demand]),
        on=:profile_name => :asset
    )
    df.hourly_demand = df.value .* df.peak_demand

    total_demand = sum(df.hourly_demand)

    demand_assets_H = filter(row -> occursin("H_Demand", row.asset), assets)

    println("N of H demands ", nrow(demand_assets_H))

    total_demand_H = sum(demand_assets_H.peak_demand .* 8760)

    println("Total demand E = ", total_demand)
    println("Total demand H = ", total_demand_H)

    println("Total demand ", total_demand + total_demand_H)









    return nothing
end

main()