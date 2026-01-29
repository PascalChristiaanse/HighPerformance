#!/bin/bash
#SBATCH --job-name=exp_c_global_n4000_b64
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n4000_b64_global_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with GLOBAL memory kernel (direct VRAM access)
echo "=== Memory Type: GLOBAL GPU Memory (VRAM) ==="
echo "Matrix size: 4000, Block size: 64"
echo "Kernel: Av_Product_Global (direct global memory access)"
echo ""

srun ./power_gpu --size 4000 --blocksize 64 --max_iteration 100 --use_shared 0 --use_unified 0
