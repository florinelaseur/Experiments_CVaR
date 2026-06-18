# Run batch experiments for main.jl, rasra.jl, and main_fixed_rasra.jl across multiple seeds and scenario counts

# Experiment values
const SEEDS = [1, 2, 3, 4, 5, 6, 7]
const N_SCENARIOS = [10, 20, 30]

# J (size of the scenario subset) per N value for RASRA
const N_TO_J = Dict(
    10 => 5,
    20 => 10,
    30 => 15,
)

const RUN_MAIN = true
const RUN_RASRA = true
const RUN_MAIN_FIXED_RASRA = true   # Only works if RUN_RASRA = true

using TOML: TOML

const PROJECT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const CONFIG_PATH = joinpath(PROJECT_DIR, "config.toml")
const MAIN_PATH = joinpath(PROJECT_DIR, "main.jl")
const RASRA_PATH = joinpath(PROJECT_DIR, "rasra.jl")
const MAIN_FIXED_RASRA_PATH = joinpath(PROJECT_DIR, "main_fixed_rasra.jl")

# Helpers
"""Patch config.toml with the given key-value pairs under [simulation]."""
function patch_config!(pairs::Pair{String}...)
    config = TOML.parsefile(CONFIG_PATH)
    for (k, v) in pairs
        config["simulation"][k] = v
    end
    open(CONFIG_PATH, "w") do io
        TOML.print(io, config)
    end
end

"""Run a Julia script as a child process with the given seed injected via ENV.
Returns (success::Bool, elapsed_seconds::Float64)."""
function run_script(script_path::String, seed::Int; label::String = "")
    t = time()
    env = copy(ENV)
    env["EXPERIMENT_SEED"] = string(seed)

    cmd = Cmd(
        `$(Base.julia_cmd()) --project=$(PROJECT_DIR) $(script_path)`;
        env = env,
        dir = PROJECT_DIR,
    )

    @info "[$label] Starting"
    proc = run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = true)
    elapsed = time() - t
    success = proc.exitcode == 0

    if success
        @info "[$label] Finished in $(round(elapsed; digits=1))s"
    else
        @warn "[$label] FAILED (exit code $(proc.exitcode)) after $(round(elapsed; digits=1))s"
    end

    return success, elapsed
end

struct RunRecord
    script :: String
    seed :: Int
    n :: Int
    j :: Int
    success :: Bool
    elapsed :: Float64
end

# Main loop
function run_all_experiments()
    # Validate N_TO_J up front
    if RUN_RASRA || RUN_MAIN_FIXED_RASRA
        for n in N_SCENARIOS
            if !haskey(N_TO_J, n)
                error("N_TO_J has no entry for N=$n")
            end
            j = N_TO_J[n]
            if j >= n
                error("N_TO_J[$n] = $j but J must be strictly less than N")
            end
        end
    end

    if RUN_MAIN_FIXED_RASRA && !RUN_RASRA
        error("RUN_MAIN_FIXED_RASRA = true requires RUN_RASRA = true")
    end

    n_scripts = RUN_MAIN + RUN_RASRA + RUN_MAIN_FIXED_RASRA
    total = length(SEEDS) * length(N_SCENARIOS) * n_scripts
    log = RunRecord[]
    run_idx = 0

    for seed in SEEDS, n in N_SCENARIOS
        @info "seed=$seed, N=$n"

        # Full benchmark
        if RUN_MAIN
            run_idx += 1
            patch_config!("number_of_scenarios" => n)
            label = "main.jl [$run_idx/$total] N=$n seed=$seed"
            ok, dt = run_script(MAIN_PATH, seed; label = label)
            push!(log, RunRecord("main.jl", seed, n, 0, ok, dt))
        end

        # RASRA reduction
        rasra_ok = false
        if RUN_RASRA
            run_idx += 1
            j = N_TO_J[n]
            patch_config!("number_of_scenarios" => n, "rasra_size_of_j" => j)
            label = "rasra.jl [$run_idx/$total] N=$n J=$j seed=$seed"
            rasra_ok, dt = run_script(RASRA_PATH, seed; label = label)
            push!(log, RunRecord("rasra.jl", seed, n, j, rasra_ok, dt))
        end

        # Re-solve on main with fixed investment decisions
        if RUN_MAIN_FIXED_RASRA
            run_idx += 1
            if !rasra_ok
                @warn "Skipping main_fixed_rasra.jl for seed=$seed N=$n because rasra.jl failed"
                push!(log, RunRecord("fixed_rasra.jl", seed, n, 0, false, 0.0))
            else
                patch_config!("number_of_scenarios" => n)
                label = "main_fixed_rasra.jl [$run_idx/$total] N=$n seed=$seed"
                ok, dt = run_script(MAIN_FIXED_RASRA_PATH, seed; label = label)
                push!(log, RunRecord("fixed_rasra.jl", seed, n, 0, ok, dt))
            end
        end
    end

    # Summary table
    println("\n", "="^76)
    println("Experiment Summary")
    println("="^76)
    println(rpad("Script", 18), rpad("Seed", 8), rpad("N", 6), rpad("J", 6), rpad("Status", 10), "Time (s)")
    println("-"^76)
    for r in log
        status = r.success ? "OK" : "FAILED"
        j_str = r.j == 0 ? "-" : string(r.j)
        println(
            rpad(r.script, 18),
            rpad(string(r.seed), 8),
            rpad(string(r.n), 6),
            rpad(j_str, 6),
            rpad(status, 10),
            round(r.elapsed; digits = 1),
        )
    end
    println("="^76)

    n_ok = count(r -> r.success, log)
    n_failed = count(r -> !r.success, log)
    total_time = sum(r.elapsed for r in log)
    @info "Done: $n_ok succeeded, $n_failed failed, " * "total wall time $(round(total_time / 60; digits=1)) min"

    return log
end

run_all_experiments()