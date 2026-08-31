cd(@__DIR__)

ENV["GKSwstype"] = "100"

using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV: CSV
using TOML: TOML
using DataFrames
using Plots
using StatsPlots

@info "Including helper functions"
include("utils/functions.jl")
include("utils/constants.jl")

function run_experiments()
    config_path = joinpath(@__DIR__, "config.toml")
    original_config = TOML.parsefile(config_path)
    n_seeds = original_config["simulation"]["seeds"]
    seeds = collect(1:n_seeds)
    scenario_sizes = copy(original_config["simulation"]["scenarios_starting_set_sizes"],)
    representative_periods = copy(original_config["simulation"]["representative_periods"],)
    main_cc_path = joinpath(@__DIR__, "main-NL-ScSeRP-hourly.jl")

    log = DataFrame(
        seed=Int[],
        number_of_scenarios=Int[],
        representative_periods=String[],
        success=Bool[],
        elapsed=Float64[],
    )

    try
        for n in scenario_sizes
            for seed in seeds
                @info "Running experiment with seed=$seed, number_of_scenarios=$n, representative_periods=$representative_periods"

                config = deepcopy(original_config)
                config["simulation"]["number_of_scenarios"] = n
                config["simulation"]["representative_periods"] = copy(representative_periods)
                config["simulation"]["run_benchmark"] = false

                open(config_path, "w") do io
                    TOML.print(io, config)
                end

                env = copy(ENV)
                env["EXPERIMENT_SEED"] = string(seed)
                t = time()
                success = true

                cmd = Cmd(`$(Base.julia_cmd()) --project=$(@__DIR__) $main_cc_path`; env=env, dir=@__DIR__,)

                try
                    run(cmd)
                catch error
                    success = false
                    @warn "Experiment failed for seed=$seed, number_of_scenarios=$n, representative_periods=$representative_periods with error: $error" exception = error
                end

                elapsed = time() - t

                push!(log, (
                    seed=seed,
                    number_of_scenarios=n,
                    representative_periods=string(representative_periods),
                    success=success,
                    elapsed=elapsed,
                ))
                # if seed == last(seeds)
                #     plot_comparison_runtime(joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "results_ScSeRP_N$(n)_seed$(seed).csv"))
                # end
            end
        end
    finally
        open(config_path, "w") do io
            TOML.print(io, original_config)
        end
    end
    out = joinpath(homedir(), "Nextcloud", "ExperimentData", "NL-output-data", "experiment_log_ScSeRP.csv")
    mkpath(dirname(out))
    CSV.write(out, log)

    return log
end

run_experiments()