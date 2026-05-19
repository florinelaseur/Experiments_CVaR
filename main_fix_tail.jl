# Copyright (c) 2025: Diego Tejada and contributors
#
# Use of this source code is governed by an Apache 2.0 license that can be found
# in the LICENSE.md file or at https://opensource.org/license/apache-2-0.

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

Random.seed!(19990907)

using DataFrames

# helper functions
@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

distance_map = Dict(
    :Euclidean => Distances.Euclidean(),
    :SqEuclidean => Distances.SqEuclidean(),
    :CosineDist => Distances.CosineDist(),
    :Cityblock => Distances.Cityblock(),
    :Chebyshev => Distances.Chebyshev(),
)

# Read and transform user input files to Tulipa input files
config = TOML.parsefile("config.toml")
input_data_path = config["simulation"]["input_data"]
use_ratio = config["clustering"]["use_ratio"]
heuristic_distance = config["clustering"]["heuristic_distance"]
fix_level_storage = config["simulation"]["fix_level_storage"]
representative_periods = config["simulation"]["representative_periods"]
solvers = [Symbol(el) for el in config["simulation"]["solvers"]]
lambda = config["simulation"]["risk_aversion_weight_lambda"]
alpha = config["simulation"]["risk_aversion_confidence_level"]
number_of_scenarios = config["simulation"]["number_of_scenarios"]
run_benchmark = config["simulation"]["run_benchmark"]


## for new scenarios

profiles_path = joinpath(@__DIR__, "create-scenarios", "profiles-wide-all-scenarios.csv")
all_profiles_df = CSV.read(profiles_path, DataFrame)
profiles_df = get_scenario_set(all_profiles_df, number_of_scenarios)
selected_scenarios = sort(unique(profiles_df.scenario))
mapping = Dict(old => new for (new, old) in enumerate(selected_scenarios))
profiles_df[!, :scenario] = [mapping[s] for s in profiles_df.scenario]
CSV.write(joinpath(input_data_path, "profiles-wide.csv"), profiles_df; writeheader=true)

df_stochastic_scenario = DataFrame(;
    scenario=sort(unique(profiles_df.scenario)),
    probability=fill(1.0 / number_of_scenarios, number_of_scenarios),
)
CSV.write(joinpath(input_data_path, "stochastic-scenario.csv"), df_stochastic_scenario; writeheader=true)

## to keep scenarios

# profiles_df = CSV.read(joinpath(input_data_path, "profiles-wide.csv"), DataFrame)
# df_stochastic_scenario = CSV.read(joinpath(input_data_path, "stochastic-scenario.csv"), DataFrame)

case_studies_info = CSV.read(
    "case-studies-info.csv",
    DataFrame;
    types=Dict(
        :base_name => String,
        :period_duration => Int,
        :method => Symbol,
        :distance => Symbol,
        :weight_type => Symbol,
        :niters => Int,
        :learning_rate => Float64,
        :stochastic_method => Symbol,
        :run_case => Bool,
    ),
)

enable_names = true
direct_model = false
results_df = DataFrame(;
    base_name=String[],
    rp=Int[],
    solver=Symbol[],
    time_to_cluster=Float64[],
    time_to_read=Float64[],
    time_to_create=Float64[],
    time_to_solve=Float64[],
    time_to_save=Float64[],
    objective_value=Float64[],
    termination_status=String[],
    num_constraints=Int[],
    num_variables=Int[],
    time_to_resolve_benchmark=Float64[],
    objective_value_resolve_benchmark=Float64[],
    termination_status_resolve_benchmark=String[],
    num_loss_of_load_e_demand=Int[],
    num_loss_of_load_h2_demand=Int[],
    water_borrowed=Float64[],
    value_at_risk_threshold_mu=Float64[],
)

# function main()
# optimize for the base case study (0_HourlyBenchmark)
# set up the connection and read the data
connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
TIO.read_csv_folder(connection_benchmark, input_data_path)
profiles_wide = TIO.get_table(connection_benchmark, "profiles_wide")
n_scenarios = length(unique(profiles_wide.scenario))
# To make number of rps comparable with per and cross scenario
# we consider the case that n_rps is not divisible by the number of scenarios
#representative_periods .= n_scenarios .* round.(Int, representative_periods ./ n_scenarios)

#    if run_benchmark
@info "Running the base case study (0_HourlyBenchmark)"
base_name = "0_HourlyBenchmark"

# set up the connection and read the data
connection_benchmark = DuckDB.DBInterface.connect(DuckDB.DB)
TIO.read_csv_folder(connection_benchmark, input_data_path)
# update the CSV input data for Tulipa from the config file info
DuckDB.query(
    connection_benchmark,
    "
    UPDATE model_parameters -- tables are with underscore in DuckDB world
    SET
        risk_aversion_weight_lambda = $(lambda) ,
        risk_aversion_confidence_level_alpha = $(alpha);
    ",
)
# transform the profiles data from wide to long
TC.transform_wide_to_long!(
    connection_benchmark,
    "profiles_wide",
    "profiles";
    exclude_columns=["scenario", "milestone_year", "timestep"],
)

layout = TC.ProfilesTableLayout(;
    year=:milestone_year,
    cols_to_groupby=[:milestone_year, :scenario],
)
time_to_cluster = @elapsed TC.dummy_cluster!(connection_benchmark; layout=layout)
TEM.populate_with_defaults!(connection_benchmark)
DuckDB.query(connection_benchmark, "UPDATE asset SET is_seasonal = false")

time_to_read = @elapsed energy_problem_benchmark = TEM.EnergyProblem(connection_benchmark)

#       for solver in solvers
solver = :Gurobi
optimizer, parameters = get_solver_parameters(solver)

@info "Creating the model for the base case study (0_HourlyBenchmark) with $solver"
time_to_create = @elapsed TEM.create_model!(
    energy_problem_benchmark;
    optimizer=optimizer,
    optimizer_parameters=parameters,
    model_file_name="",
    enable_names=enable_names,
    direct_model=direct_model,
)

output_folder = joinpath(@__DIR__, "outputs-tailfix", base_name, string(solver))
mkpath(output_folder)

@info "Solving the model and saving the solution for the base case study (0_HourlyBenchmark) with $solver"
time_to_solve = @elapsed TEM.solve_model!(energy_problem_benchmark)
#        mu_value =
#            JuMP.value(energy_problem_benchmark.variables[:value_at_risk_threshold_mu].container)
time_to_save = @elapsed TEM.save_solution!(energy_problem_benchmark)
TEM.export_solution_to_csv_files(output_folder, energy_problem_benchmark)

df_base_cost = export_base_cost(energy_problem_benchmark, output_folder)
#kan geen expressions optellen, check hoe operational cost in worst case tail cost wordt gepakt 
df_operational_cost_per_scenario = export_operational_cost_per_scenario(energy_problem_benchmark, output_folder)
df_total_cost_per_scenario = export_total_cost_per_scenario(energy_problem_benchmark, output_folder)

mu_value_df = TIO.get_table(connection_benchmark, "var_value_at_risk_threshold_mu")
mu_value = if nrow(mu_value_df) == 0
    NaN
else
    only(mu_value_df.solution)
end

tol = 1e-5

df_tail_scenarios = copy(df_total_cost_per_scenario)

df_tail_scenarios[!, :solution] =
    max.(0.0, df_tail_scenarios.total_cost .- mu_value)

df_tail_scenarios = filter(
    row -> row.total_cost >= mu_value - tol,
    df_tail_scenarios,
)

df_tail_scenarios = df_tail_scenarios[:, [:id, :scenario, :probability, :solution]]

CSV.write(
    joinpath(output_folder, "tail_scenarios.csv"),
    df_tail_scenarios;
    writeheader=true,
)
#put this into a function later
worst_case_row = df_operational_cost_per_scenario[
    argmax(df_operational_cost_per_scenario.operational_cost),
    :
]

df_worst_case_tail_cost = DataFrame(
    scenario=[worst_case_row.scenario],
    operational_cost=[worst_case_row.operational_cost],
)

CSV.write(joinpath(output_folder, "worst-case-tail-cost.csv"), df_worst_case_tail_cost; writeheader=true)
#put this into a function later
df_sorted = sort(df_operational_cost_per_scenario, :operational_cost)
middle_idx = ceil(Int, nrow(df_sorted) / 2)
average_case_row = df_sorted[middle_idx, :]
df_average_case_cost = DataFrame(
    scenario=[average_case_row.scenario],
    operational_cost=[average_case_row.operational_cost],
)

CSV.write(joinpath(output_folder, "average-case-cost.csv"), df_average_case_cost; writeheader=true,)

plot_operational_cost_per_scenario(df_operational_cost_per_scenario, output_folder)


var_flow_df = TIO.get_table(connection_benchmark, "var_flow")
flow_ens = filter(row -> row.from_asset == "ens" && row.to_asset == "e_demand", var_flow_df)
flow_smr_ccs =
    filter(row -> row.from_asset == "smr_ccs" && row.to_asset == "h2_demand", var_flow_df)
water_borrowed = filter(
    row -> row.from_asset == "water_borrower" && row.to_asset == "hydro_reservoir",
    var_flow_df,
)

# count steps with loss of load
n_lol_ens = count(row -> row.solution > 0.0, eachrow(flow_ens))
n_lol_smr_cca = count(row -> row.solution > 0.0, eachrow(flow_smr_ccs))

# count how much water_borrowed
amount_water_borrowed_b = sum(water_borrowed.solution)

new_results_row = (
    base_name=base_name,
    rp=1,
    solver=solver,
    time_to_cluster=0.0,
    time_to_read=time_to_read,
    time_to_create=time_to_create,
    time_to_solve=time_to_solve,
    time_to_save=time_to_save,
    objective_value=energy_problem_benchmark.objective_value,
    termination_status=string(energy_problem_benchmark.termination_status),
    num_constraints=JuMP.num_constraints(
        energy_problem_benchmark.model;
        count_variable_in_set_constraints=false,
    ),
    num_variables=JuMP.num_variables(energy_problem_benchmark.model),
    time_to_resolve_benchmark=0.0,
    objective_value_resolve_benchmark=0.0,
    termination_status_resolve_benchmark="",
    num_loss_of_load_e_demand=n_lol_ens,
    num_loss_of_load_h2_demand=n_lol_smr_cca,
    water_borrowed=amount_water_borrowed_b,
    value_at_risk_threshold_mu=mu_value,
)
push!(results_df, new_results_row)
CSV.write(joinpath(output_folder, "results.csv"), results_df; writeheader=true)
#     end
# end

# tol = 1e-6

# tail_scenarios_path = joinpath(@__DIR__, "outputs-tailfix", "0_hourlyBenchmark", "Gurobi", "var_tail_excess_slack_xi.csv")
# tail_scenarios_df = CSV.read(tail_scenarios_path, DataFrame)


# df_check = leftjoin(
#     df_operational_cost_per_scenario,
#     tail_scenarios_df;
#     on=:scenario,
#     makeunique=true,
# )

# rename!(df_check, :solution => :xi)

# df_check[!, :mu] .= mu_value
# df_check[!, :cost_minus_mu] = df_check.operational_cost .- df_check.mu
# df_check[!, :xi_should_be_at_least] = max.(0.0, df_check.cost_minus_mu)
# df_check[!, :violation] = df_check.xi .+ tol .< df_check.xi_should_be_at_least

# sort!(df_check, :operational_cost, rev=true)

# CSV.write(joinpath(output_folder, "debug_cvar_constraints.csv"), df_check)
# df_check[:, [:scenario, :operational_cost, :mu, :xi, :cost_minus_mu, :xi_should_be_at_least, :violation]]

# obj_breakdown = TIO.get_table(connection_benchmark, "obj_breakdown")
# display(obj_breakdown)