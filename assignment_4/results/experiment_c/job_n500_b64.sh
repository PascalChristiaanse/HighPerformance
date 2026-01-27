#!/bin/bash
#SBATCH --job-name=exp_c_n500_b64
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/experiment_c/run_n500_b64_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with shared memory for speedup measurements
echo "=== Experiment C: Speedup Analysis ==="
echo "Matrix size: 500, Block size: 64"
echo ""
echo "The output includes:"
echo "  - cpu_time: CPU execution time"
echo "  - gpu_time_with_memcpy: GPU time including memory transfers"
echo "  - gpu_time_no_memcpy: GPU compute-only time"
echo "  - memcpy_time: Time spent on memory copies"
echo "  - speedup_with_memcpy: CPU_time / GPU_time_with_memcpy"
echo "  - speedup_no_memcpy: CPU_time / GPU_compute_only_time"
echo ""

srun ./power_gpu --size 500 --blocksize 64 --max_iteration 100 --use_shared 1
