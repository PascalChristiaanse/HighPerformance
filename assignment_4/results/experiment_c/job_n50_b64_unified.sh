#!/bin/bash
#SBATCH --job-name=exp_c_u_n50_b64
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n50_b64_unified_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with CUDA Unified Memory (cudaMallocManaged)
echo "=== Experiment C: Speedup Analysis (Unified Memory) ==="
echo "Matrix size: 50, Block size: 64"
echo "Memory management: unified (cudaMallocManaged)"
echo ""

srun ./power_gpu --size 50 --blocksize 64 --max_iteration 100 --use_shared 1 --use_unified 1
