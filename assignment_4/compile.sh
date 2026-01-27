#!/bin/bash
# Compile script for power_gpu.cu
# Run this on a GPU node or use sbatch to compile

module load 2024r1 nvhpc

echo "Compiling power_gpu.cu..."
nvcc -O3 -o power_gpu power_gpu.cu

if [ $? -eq 0 ]; then
    echo "Compilation successful! Executable: power_gpu"
else
    echo "Compilation failed!"
    exit 1
fi
