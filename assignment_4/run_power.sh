#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=results/power_gpu_%j.out

# Load required modules
module load 2024r1 nvhpc

# Create results directory if it doesn't exist
mkdir -p results

# Default parameters (can be overridden via environment variables)
SIZE=${SIZE:-1000}
BLOCKSIZE=${BLOCKSIZE:-32}
MAX_ITER=${MAX_ITER:-100}
USE_SHARED=${USE_SHARED:-1}

echo "Running power_gpu with:"
echo "  Matrix size: ${SIZE}"
echo "  Block size: ${BLOCKSIZE}"
echo "  Max iterations: ${MAX_ITER}"
echo "  Use shared memory: ${USE_SHARED}"
echo ""

# Run the program
srun ./power_gpu --size ${SIZE} --blocksize ${BLOCKSIZE} --max_iteration ${MAX_ITER} --use_shared ${USE_SHARED}
