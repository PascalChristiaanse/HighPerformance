#!/bin/bash
#SBATCH --job-name=exp_b_n2000_b32
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_b/run_n2000_b32_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with shared memory (default for experiment b)
echo "=== Experiment B: Matrix size=2000, Block size=32, Memory=shared ==="
srun ./power_gpu --size 2000 --blocksize 32 --max_iteration 100 --use_shared 1
