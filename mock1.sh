#!/bin/bash
#SBATCH --job-name=scserp_onenode_N10D1
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=01:00:00
#SBATCH --account=research-eemcs-st
#SBATCH --output=slurm-%x-%j.out

module load julia
# module load gurobi

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export EXPERIMENT_DRAW=1
export GKSwstype=100
export GRB_LICENSE_FILE=/apps/generic/gurobi/12.0.0/linux64/lib/gurobi.lic

cd $HOME/Experiments_CVaR
julia --version
srun julia --project=. main-NL-ScSeRP-hourly-outlier-delftblue.jl