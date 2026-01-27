#!/bin/bash
#SBATCH --job-name=exp_d_n2000_b64_global
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_d/run_n2000_b64_global_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run experiment
echo "=== Experiment D: Memory Type Comparison ==="
echo "Matrix size: 2000, Block size: 64, Memory: global"
echo ""

srun ./power_gpu --size 2000 --blocksize 64 --max_iteration 100 --use_shared 0
