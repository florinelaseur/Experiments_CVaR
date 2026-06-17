"""
Simple Python Script with no dependencies used for running the experiments.
"""

import os
import subprocess
import time
import sys
import threading
from pathlib import Path

log_lock = threading.Lock()

class MasterLogger:
    """
    Intercepts all Python standard output and errors. 
    Writes everything to BOTH the terminal AND a master log file for the entire batch.
    """
    def __init__(self, log_filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        try:
            # Force OS to write to SSD to prevent OOM data loss
            os.fsync(self.log_file.fileno())
        except OSError:
            pass


def system_monitor(stop_event, log_file, interval: float = 5.0):
    """
    Background thread that monitors Linux RAM and Swap usage.
    Running this in Python guarantees it will never be blocked by Julia/Gurobi.
    """
    while not stop_event.is_set():
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                
                mem_total = mem_avail = swap_total = swap_free = 0.0
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        val = float(parts[1]) / (1024**2) # Convert kB to GB
                        if line.startswith("MemTotal:"): mem_total = val
                        elif line.startswith("MemAvailable:"): mem_avail = val
                        elif line.startswith("SwapTotal:"): swap_total = val
                        elif line.startswith("SwapFree:"): swap_free = val
                
                mem_used = mem_total - mem_avail
                swap_used = swap_total - swap_free
                
                # Format to match Julia's @info style
                log_str = f"[python] RAM: {mem_used:.2f}GB / {mem_total:.2f}GB | Swap: {swap_used:.2f}GB / {swap_total:.2f}GB\n"
                
                # Thread-safe forced write
                with log_lock:
                    sys.stdout.write(log_str)
                    sys.stdout.flush()
                    
                    if log_file and not log_file.closed:
                        log_file.write(log_str)
                        log_file.flush()
                        os.fsync(log_file.fileno())
        except Exception as e:
            pass # Fail silently if we temporarily can't read meminfo
            
        # Wait 1 second, but break immediately if the process ends
        stop_event.wait(interval)


def run_experiment(n_scenarios: int, target_k: int, seed: int, base_out_dir: Path):
    # Define a clean, unique folder name for this run
    exp_name = f"N_{n_scenarios}_K_{target_k}_seed_{seed}"
    out_dir = base_out_dir / exp_name

    # Pipeline Feature: Skip if already successfully run
    if out_dir.exists():
        print(f"⏭️  Skipping {exp_name}: '{out_dir}' already exists.")
        return

    # Create the isolated output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = out_dir / "run.log"

    # Construct the Environment Variables
    env = os.environ.copy()
    env["NUMBER_OF_SCENARIOS"] = str(n_scenarios)
    env["TARGET_SCENARIOS"] = str(target_k)
    env["EXPERIMENT_SEED"] = str(seed)
    env["OUTPUT_DIR"] = str(out_dir)

    # Helper to print strictly to the experiment's local run.log AND the Master sys.stdout
    def log_print(msg, lf):
        out_msg = msg + "\n"
        with log_lock:
            sys.stdout.write(out_msg)
            sys.stdout.flush()
            lf.write(out_msg)
            lf.flush()
            os.fsync(lf.fileno())

    # Open the log file BEFORE starting so we can capture the Python setup prints
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        
        log_print(f"\n🚀 Starting Experiment: {exp_name}", log_file)
        start_time = time.time()

        # 1. Start the unblockable Python system monitor
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(target=system_monitor, args=(stop_monitor, log_file, 5.0), daemon=True)
        monitor_thread.start()

        # 2. Start the Julia subprocess
        process = subprocess.Popen(
            ["julia", "--threads=4", "main.jl"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 3. Read the buffer line-by-line in real time
        for line in process.stdout:
            with log_lock:
                sys.stdout.write(line)   # Print to console (and Master Log!)
                sys.stdout.flush()       # Force console update
                
                log_file.write(line)     # Append to local experiment run.log
                log_file.flush()         # Force Python to flush its internal RAM buffer
                os.fsync(log_file.fileno()) # Force OS to physically write to SSD
            
        exit_code = process.wait()
        
        # 4. Stop the Python monitor cleanly
        stop_monitor.set()
        monitor_thread.join(timeout=2.0)
        
        if exit_code != 0:
            log_print("WARNING: non-zero exit code encountered!", log_file)

        end_time = time.time()
        runtime = end_time - start_time

        # Save the runtime to a simple text file for easy parsing later
        with open(out_dir / "runtime.txt", "w", encoding="utf-8") as f:
            f.write(f"Runtime (seconds): {runtime:.2f}\n")
            f.write(f"Exit code: {exit_code}\n")

        log_print(f"✅ Finished {exp_name} in {runtime:.2f} seconds.\n", log_file)
        log_print("-" * 60, log_file)


def main():
    # --- EXPERIMENT DEFINITIONS ---
    scenario_pools = [30, 20, 10]
    seeds = [1, 2, 3, 4, 5, 6, 7]
    base_out_dir = Path(__file__).parent / "pipeline-results"
    experiment_name = "2026-06-17_007"
    base_out_dir = base_out_dir / experiment_name

    # Create base dir before setting up master logger
    base_out_dir.mkdir(parents=True, exist_ok=True)

    master_log_path = base_out_dir / "master_pipeline.log"
    logger = MasterLogger(master_log_path)
    sys.stdout = logger
    sys.stderr = logger

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