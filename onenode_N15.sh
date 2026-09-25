#!/bin/bash
#SBATCH --job-name=scserp_N15
#SBATCH --partition=compute
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=04:00:00
#SBATCH --account=research-eemcs-st
#SBATCH --output=slurm-%x-draw%a-%A.out

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export EXPERIMENT_N=15
export EXPERIMENT_DRAW=$SLURM_ARRAY_TASK_ID
export GKSwstype=100
export GRB_LICENSE_FILE=/apps/generic/gurobi/12.0.0/linux64/lib/gurobi.lic

cd $HOME/Experiments_CVaR
julia +1.10.9 --version
srun julia +1.10.9 --project=. main-NL-ScSeRP-hourly-outlier-delftblue-job.jl