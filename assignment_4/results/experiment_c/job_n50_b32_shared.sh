#!/bin/bash
#SBATCH --job-name=exp_c_shared_n50_b32
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n50_b32_shared_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with SHARED memory kernel (fast on-chip cache)
echo "=== Memory Type: SHARED GPU Memory (On-chip) ==="
echo "Matrix size: 50, Block size: 32"
echo "Kernel: Av_Product_Shared (tiled with shared memory)"
echo ""

srun ./power_gpu --size 50 --blocksize 32 --max_iteration 100 --use_shared 1 --use_unified 0
