"""
Simple Python Script with no dependencies used for running the experiments.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

def run_experiment(n_scenarios: int, target_k: int, seed: int, base_out_dir: Path):
    # Define a clean, unique folder name for this run
    exp_name = f"N_{n_scenarios}_K_{target_k}_seed_{seed}"
    out_dir = base_out_dir / exp_name
    # results_csv = out_dir / "results.csv"

    # Pipeline Feature: Skip if already successfully run
    if out_dir.exists():
        print(f"⏭️  Skipping {exp_name}: '{out_dir}' already exists.")
        return

    # Create the isolated output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = out_dir /  "run.log"

    # Construct the Environment Variables
    env = os.environ.copy()
    env["NUMBER_OF_SCENARIOS"] = str(n_scenarios)
    env["TARGET_SCENARIOS"] = str(target_k)
    env["EXPERIMENT_SEED"] = str(seed)
    env["OUTPUT_DIR"] = str(out_dir)

    print(f"\n🚀 Starting Experiment: {exp_name}")
    start_time = time.time()

    # Open a subprocess, streaming output to both the console and a log file simultaneously
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            ["julia", "main.jl"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read the buffer line-by-line in real time
        for line in process.stdout:
            sys.stdout.write(line)   # Print to console
            log_file.write(line)     # Append to log
            
        exit_code = process.wait()
        if exit_code != 0:
            print("WARNING: non-zero exit code encountered!")


    end_time = time.time()
    runtime = end_time - start_time

    # Save the runtime to a simple text file for easy parsing later
    with open(out_dir / "runtime.txt", "w") as f:
        f.write(f"Runtime (seconds): {runtime:.2f}\n")
        f.write(f"Exit code: {exit_code}\n")


    print(f"✅ Finished {exp_name} in {runtime:.2f} seconds.\n")
    print("-" * 60)


def main():
    # --- EXPERIMENT DEFINITIONS ---
    scenario_pools = [30, 
                      20, 
                      10
                    ]
    seeds = [1, 2, 3, 4, 5, 6, 7]
    base_out_dir = Path(__file__).parent / "pipeline-results"
    experiment_name = "2026-06-17_001"
    base_out_dir = base_out_dir / experiment_name

    print("=" * 60)
    print(" IPDSR BATCH EXPERIMENT PIPELINE")
    print("=" * 60)

    # Calculate combinations and trigger runs
    for seed in seeds:
        for n in scenario_pools:
            k = n // 2  # As per methodology: target scenarios = exactly half of initial pool
            run_experiment(n_scenarios=n, target_k=k, seed=seed, base_out_dir=base_out_dir)
            
    print("🎉 All experiments completed successfully!")

if __name__ == "__main__":
    main()