cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO
import TulipaClustering
import DuckDB
import Gurobi
import Distances
import CSV
using DataFrames
using Random
using DuckDB: DBInterface, DuckDB
using Plots

user_input_dir = "obz_data_raw/"

connection = DBInterface.connect(DuckDB.DB, "obz4.db")

TulipaIO.read_csv_folder(
    connection,
    user_input_dir,
    replace_if_exists=true,
)
DuckDB.query(connection,
    "ALTER TABLE profiles_wide
        ADD scenario Int32;",
)
DuckDB.query(
    connection,
    "
    UPDATE profiles_wide
    SET scenario = 0;
"
)
TulipaClustering.transform_wide_to_long!(connection, "profiles_wide", "pivot_profiles"; exclude_columns=["year", "timestep", "scenario"])
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE profiles AS
    FROM pivot_profiles
    ORDER BY profile_name, year, timestep
    "
)
# I have added investment in batteries and electrolyzers
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE asset AS
    SELECT
        name AS asset,
        type,
        capacity,
        capacity_storage_energy,
        is_seasonal,
        CASE
            WHEN LOWER(name) LIKE '%wind_onshore%'
              OR LOWER(name) LIKE '%wind_offshore%'
              OR LOWER(name) LIKE '%solar%'
              OR LOWER(name) LIKE '%coal%'
              OR LOWER(name) LIKE '%ocgt%'
              OR LOWER(name) LIKE '%gas%'
              OR LOWER(name) LIKE '%nuclear%'
              OR LOWER(name) LIKE '%battery%'
              OR LOWER(name) LIKE '%electrolyzer%'
            THEN 'simple'
            ELSE 'none'
        END AS investment_method,
        false as storage_method_energy,
        false AS investment_integer
    FROM (
        FROM assets_consumer_basic_data
        UNION BY NAME
        FROM assets_conversion_basic_data
        UNION BY NAME
        FROM assets_producer_basic_data
        UNION BY NAME
        FROM assets_storage_basic_data
    )
    ORDER BY asset
    ",
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE t_asset_yearly AS
    FROM (
        FROM assets_consumer_yearly_data
        UNION BY NAME
        FROM assets_conversion_yearly_data
        UNION BY NAME
        FROM assets_producer_yearly_data
        UNION BY NAME
        FROM assets_storage_yearly_data
    )
    ",
)
# investment_costs
# wind_onshore = 77356.32865703155
# wind_offshore = 119732.61777406993
# solar = 34342.98027538492
# battery = 77577.8503057667
# for capacity --> battery = 77577.8503057667 * 4 = 310311.401223067
# coal = 420000.0
# ocgt = 55000.0
# gas = 95000.0     
# nuclear = 950000.0
# eleectrolyzer 800000.0

DuckDB.query(
    connection,
    "
    CREATE OR REPLACE TABLE asset_commission AS
    SELECT
        tay.name AS asset,
        tay.year AS commission_year,
        CASE
            WHEN LOWER(tay.name) LIKE '%wind_onshore%'  THEN 77356.32865703155
            WHEN LOWER(tay.name) LIKE '%wind_offshore%' THEN 119732.61777406993
            WHEN LOWER(tay.name) LIKE '%solar%'          THEN 34342.98027538492
            WHEN LOWER(tay.name) LIKE '%coal%'           THEN 420000.0
            WHEN LOWER(tay.name) LIKE '%ocgt%'            THEN 55000.0
            WHEN LOWER(tay.name) LIKE '%gas%'             THEN 95000.0
            WHEN LOWER(tay.name) LIKE '%nuclear%'         THEN 950000.0
            WHEN LOWER(tay.name) LIKE '%battery%'         THEN 77577.8503057667
            WHEN LOWER(tay.name) LIKE '%electrolyzer%'    THEN 800000.0
            ELSE NULL
        END AS investment_cost,
        CASE 
        WHEN LOWER(tay.name) LIKE '%pl%' THEN 3 * a.capacity
        ELSE 2 * a.capacity END AS investment_limit,
    FROM t_asset_yearly tay
    JOIN asset a
    ON a.asset = tay.name
    ORDER BY asset;
    "
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE asset_milestone AS
    SELECT
        name AS asset,
        year AS milestone_year,
        peak_demand,
        initial_storage_level,
        storage_inflows,
        CASE
            WHEN LOWER(name) LIKE '%wind_onshore%'
              OR LOWER(name) LIKE '%wind_offshore%'
              OR LOWER(name) LIKE '%solar%'
              OR LOWER(name) LIKE '%coal%'
              OR LOWER(name) LIKE '%ocgt%'
              OR LOWER(name) LIKE '%gas%'
              OR LOWER(name) LIKE '%nuclear%'
              OR LOWER(name) LIKE '%battery%'
              OR LOWER(name) LIKE '%electrolyzer%'
            THEN true
            ELSE false
        END AS investable,
    FROM t_asset_yearly
    ORDER by asset
    "
)

DuckDB.query(
    connection,
    "
    CREATE OR REPLACE TABLE asset_both AS
    SELECT
        tay.name AS asset,
        tay.year AS milestone_year,
        tay.year AS commission_year, -- same year, different semantic meaning
        CASE
            WHEN a.investment_method = 'simple' THEN 0
            ELSE tay.initial_units
        END AS initial_units,
        tay.initial_storage_units
    FROM t_asset_yearly tay
    JOIN asset a
    ON a.asset = tay.name
    ORDER BY asset;
    "
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow AS
    SELECT
        from_asset,
        to_asset,
        carrier,
        capacity,
        is_transport,
    FROM (
        FROM flows_assets_connections_basic_data
        UNION BY NAME
        FROM flows_transport_assets_basic_data
    )
    ORDER BY from_asset, to_asset
    ",
)

# flows
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow AS
    SELECT
        from_asset,
        to_asset,
        carrier,
        capacity,
        is_transport,
    FROM (
        FROM flows_assets_connections_basic_data
        UNION BY NAME
        FROM flows_transport_assets_basic_data
    )
    ORDER BY from_asset, to_asset
    ",
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE t_flow_yearly AS
    FROM (
        FROM flows_assets_connections_yearly_data
        UNION BY NAME
        FROM flows_transport_assets_yearly_data
    )
    ",
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_commission AS
    SELECT
        from_asset,
        to_asset,
        year AS commission_year,
        efficiency AS producer_efficiency,
    FROM t_flow_yearly
    ORDER by from_asset, to_asset
    "
)
# Voll --> 20000 
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_milestone AS
    SELECT
        from_asset,
        to_asset,
        year AS milestone_year,
        CASE
            WHEN ends_with(from_asset, '_ENS') THEN 20000
            WHEN LOWER(from_asset) LIKE '%smr%' THEN 20000
        ELSE variable_cost
END AS operational_cost
    FROM t_flow_yearly
    ORDER by from_asset, to_asset
    "
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_both AS
    SELECT
        t_flow_yearly.from_asset,
        t_flow_yearly.to_asset,
        t_flow_yearly.year AS milestone_year,
        t_flow_yearly.year AS commission_year,
        t_flow_yearly.initial_export_units,
        t_flow_yearly.initial_import_units,
    FROM t_flow_yearly
    LEFT JOIN flow
      ON flow.from_asset = t_flow_yearly.from_asset
      AND flow.to_asset = t_flow_yearly.to_asset
    WHERE flow.is_transport = TRUE -- flow_both must only contain transport flows
    ORDER by t_flow_yearly.from_asset, t_flow_yearly.to_asset
    "
)
# profiles # for now i can remove them
# DuckDB.query(
#     connection,
#     "CREATE OR REPLACE TABLE assets_timeframe_profiles AS
#     SELECT
#       asset,
#       commission_year AS year,
#       profile_type,
#       profile_name
#     FROM assets_storage_min_max_reservoir_level_profiles
#     ORDER BY asset, year, profile_name
#     ",
# )
# ASK DIEGO ABOUT THOSE
# # asset partitions
# DuckDB.query(
#     connection,
#     "CREATE OR REPLACE TABLE assets_rep_periods_partitions AS
#     SELECT
#         t.name AS asset,
#         t.year,
#         CASE WHEN t.partition::integer > 4 THEN t.partition::integer ELSE 1 END::varchar(255) AS partition,
#         rep_periods_data.rep_period,
#         'uniform' AS specification,
#     FROM t_asset_yearly AS t
#     LEFT JOIN rep_periods_data
#         ON t.year = rep_periods_data.year
#     ORDER BY asset, t.year, rep_period
#     ",
# )

# # flow partitions
# DuckDB.query(
#     connection,
#     "CREATE OR REPLACE TABLE flows_rep_periods_partitions AS
#     SELECT
#         flow.from_asset,
#         flow.to_asset,
#         t_from.year,
#         t_from.rep_period,
#         'uniform' AS specification,
#         IF(
#             flow.is_transport,
#             greatest(t_from.partition::int, t_to.partition::int),
#             least(t_from.partition::int, t_to.partition::int)
#         )::varchar(255) AS partition,
#     FROM flow
#     LEFT JOIN assets_rep_periods_partitions AS t_from
#         ON flow.from_asset = t_from.asset
#     LEFT JOIN assets_rep_periods_partitions AS t_to
#         ON flow.to_asset = t_to.asset
#         AND t_from.year = t_to.year
#         AND t_from.rep_period = t_to.rep_period
#     ",
# )

# timeframe profiles # for now I can remove them
# TulipaClustering.transform_wide_to_long!(
#     connection,
#     "min_max_reservoir_levels",
#     "pivot_min_max_reservoir_levels",
# )

# period_duration = 24

# DuckDB.query(
#     connection,
#     "
#     CREATE OR REPLACE TABLE profiles_timeframe AS
#     WITH cte_split_profiles AS (
#         SELECT
#             profile_name,
#             year,
#             1 + (timestep - 1) // $period_duration  AS period,
#             1 + (timestep - 1)  % $period_duration AS timestep,
#             value,
#         FROM pivot_min_max_reservoir_levels
#     )
#     SELECT
#         cte_split_profiles.profile_name,
#         cte_split_profiles.year,
#         cte_split_profiles.year AS milestone_year,
#         cte_split_profiles.period,
#         AVG(cte_split_profiles.value) AS value, -- Computing the average aggregation
#     FROM cte_split_profiles
#     GROUP BY
#         cte_split_profiles.profile_name,
#         cte_split_profiles.year,
#         cte_split_profiles.period
#     ORDER BY
#         cte_split_profiles.profile_name,
#         cte_split_profiles.year,
#         cte_split_profiles.period
#     ",
# )

# TEM.populate_with_defaults!(connection) this can run only after clustering

# save the tables in csv to be reread everytime

output_dir = joinpath(@__DIR__, "input_tables_nostoragecapinvestment")
mkpath(output_dir)

tables_df = DataFrame(DuckDB.query(
    connection,
    "SELECT table_name
     FROM information_schema.tables
     WHERE table_schema = 'main'"
))

table_names = tables_df.table_name
tables_to_keep = ["profiles", "asset", "asset_both", "asset_commission", "asset_milestone", "assets_profiles", "assets_rep_periods_partitions", "flows_rep_periods_partitions", "assets_timeframe_profiles", "flow", "flow_both", "flow_commission", "flow_milestone", "year_data", "profiles_timeframe", "profiles_wide"]
for t in table_names
    if t in tables_to_keep
        outfile = joinpath(output_dir, "$(t).csv")
        DuckDB.query(
            connection,
            "COPY $t TO '$outfile' (HEADER, DELIMITER ',');"
        )
    end
end


close(connection)