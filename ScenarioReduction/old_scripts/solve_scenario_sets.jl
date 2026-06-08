# Full stochastic TEM solve for a configurable scenario list, then for all scenarios.
#
# Edits prepare_scenario_subset! overwrite profiles-wide.csv — this script backs up
# and restores the input folder between runs.
#
# Usage (from repo root):
#   julia --project=. ScenarioReduction/old_scripts/solve_scenario_sets.jl
#
# Edit the config block below. Paste undominated ids from analyze_scenario_dominance.jl output.

cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()
Pkg.add(url="https://github.com/TulipaEnergy/TulipaEnergyModel.jl", rev="227a80f7907e2c7178edb0697874cfb6666ad644")

import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
using DuckDB: DuckDB
using Gurobi: Gurobi
using JuMP: JuMP
using Distances: Distances
using CSV: CSV
using DataFrames: DataFrame
using TOML: TOML

include(joinpath(@__DIR__, "..", "..", "utils", "functions.jl"))
include(joinpath(@__DIR__, "..", "..", "utils", "constants.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))

# --- config (edit here) ---
const SELECTED_SCENARIOS = [25, 26, 27, 28]   # undominated set from dominance analysis
const RUN_SELECTED = true
const RUN_ALL = true

const CLUSTERING_MODE = :cluster              # :dummy = full hourly | :cluster = TC.cluster!
const REPRESENTATIVE_PERIODS = 90            # only when CLUSTERING_MODE == :cluster
const PERIOD_DURATION = 24                   # only when CLUSTERING_MODE == :cluster

const SOLVER = :Gurobi
# --- end config ---

const REPO_ROOT = joinpath(@__DIR__, "..", "..")
const CONFIG = TOML.parsefile(joinpath(REPO_ROOT, "config.toml"))
const INPUT_DATA_PATH = joinpath(REPO_ROOT, CONFIG["simulation"]["input_data"])
const LAMBDA = CONFIG["simulation"]["risk_aversion_weight_lambda"]
const ALPHA = CONFIG["simulation"]["risk_aversion_confidence_level"]
const USE_RATIO = CONFIG["clustering"]["use_ratio"]
const OUTPUT_CSV = joinpath(@__DIR__, "..", "outputs", "solve_scenario_sets.csv")
const BACKUP_DIR = joinpath(@__DIR__, "..", "outputs", "_input_backup")

function backup_input!(backup_dir::String, input_data_path::String)
    mkpath(backup_dir)
    for name in ("profiles-wide.csv", "stochastic-scenario.csv")
        src = joinpath(input_data_path, name)
        isfile(src) || error("Missing input file: $src")
        cp(src, joinpath(backup_dir, name); force=true)
    end
    return backup_dir
end

function restore_input!(backup_dir::String, input_data_path::String)
    for name in ("profiles-wide.csv", "stochastic-scenario.csv")
        src = joinpath(backup_dir, name)
        isfile(src) || error("Missing backup file: $src")
        cp(src, joinpath(input_data_path, name); force=true)
    end
    return nothing
end

function all_scenario_ids_from_backup(backup_dir::String)
    profiles = CSV.read(joinpath(backup_dir, "profiles-wide.csv"), DataFrame)
    return sort(unique(Int.(profiles.scenario)))
end

function solve_kwargs()
    base = (
        solver=SOLVER,
        lambda=LAMBDA,
        alpha=ALPHA,
        use_ratio=USE_RATIO,
        clustering_mode=CLUSTERING_MODE,
    )
    if CLUSTERING_MODE == :cluster
        return (; base..., representative_periods=REPRESENTATIVE_PERIODS, period_duration=PERIOD_DURATION)
    end
    return base
end

function run_solve(label::String, scenario_ids::Vector{Int})
    println("\n=== $label ($(length(scenario_ids)) scenarios, clustering=$CLUSTERING_MODE) ===")
    println("  ids: $scenario_ids")
    result = solve_scenarios(scenario_ids, INPUT_DATA_PATH; solve_kwargs()...)
    println("  status: $(result.termination_status)")
    println("  objective: $(result.objective_value)")
    return (
        label=label,
        clustering_mode=string(CLUSTERING_MODE),
        n_scenarios=length(scenario_ids),
        scenario_ids=join(result.scenario_ids, ","),
        objective_value=result.objective_value,
        termination_status=result.termination_status,
    )
end

function main()
    backup_input!(BACKUP_DIR, INPUT_DATA_PATH)
    all_ids = all_scenario_ids_from_backup(BACKUP_DIR)
    println("Input backup: $BACKUP_DIR")
    println("All scenarios in profiles-wide.csv: $all_ids")

    rows = NamedTuple[]
    if RUN_SELECTED
        push!(rows, run_solve("selected", collect(Int.(SELECTED_SCENARIOS))))
        restore_input!(BACKUP_DIR, INPUT_DATA_PATH)
        println("Restored full input after selected run.")
    end
    if RUN_ALL
        push!(rows, run_solve("all", all_ids))
        restore_input!(BACKUP_DIR, INPUT_DATA_PATH)
        println("Restored full input after all-scenarios run.")
    end

    isempty(rows) && error("Nothing to run: set RUN_SELECTED or RUN_ALL to true")

    mkpath(dirname(OUTPUT_CSV))
    CSV.write(OUTPUT_CSV, DataFrame(rows))
    println("\nWrote $(OUTPUT_CSV)")
    return rows
end

main()
