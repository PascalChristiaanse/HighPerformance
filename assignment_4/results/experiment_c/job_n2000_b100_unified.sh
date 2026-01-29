#!/bin/bash
#SBATCH --job-name=exp_c_unified_n2000_b100
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n2000_b100_unified_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with UNIFIED Memory (cudaMallocManaged - automatic migration)
echo "=== Memory Type: UNIFIED Memory (cudaMallocManaged) ==="
echo "Matrix size: 2000, Block size: 100"
echo "Kernel: Av_Product_Shared with unified memory allocation"
echo ""

srun ./power_gpu --size 2000 --blocksize 100 --max_iteration 100 --use_shared 1 --use_unified 1
