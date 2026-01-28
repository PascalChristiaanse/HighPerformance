#!/bin/bash
#SBATCH --job-name=exp_c_n4000_b100
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n4000_b100_manual_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with manual memory transfers
echo "=== Experiment C: Speedup Analysis (Manual Memory Transfers) ==="
echo "Matrix size: 4000, Block size: 100"
echo "Memory management: manual (cudaMalloc + cudaMemcpy)"
echo ""

srun ./power_gpu --size 4000 --blocksize 100 --max_iteration 100 --use_shared 1 --use_unified 0
