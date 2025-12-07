#!/bin/bash
#
#SBATCH --job-name="MM_scaling"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-Courses-WI4049TU

module load 2024r1 openmpi

CSV_FILE="/home/pchristiaanse/HighPerformance/assignment_1/MM_product/mm_performance.csv"
EXECUTABLE="/home/pchristiaanse/HighPerformance/assignment_1/MM_product/MM-product"

# Write fresh CSV header
echo "method,matrix_size,num_procs,time_ms" > $CSV_FILE

# 1. Run sequential methods ONCE (single process)
echo "=========================================="
echo "Running sequential methods (basic, eigen)"
echo "=========================================="
srun --ntasks=1 $EXECUTABLE --mode=sequential

# 2. Run parallel SUMMA with varying process counts
echo ""
echo "=========================================="
echo "Running parallel SUMMA scaling test"
echo "=========================================="
for P in 1 2 8 24 48 64; do
    echo ""
    echo "--- P=$P processes ---"
    srun --ntasks=$P $EXECUTABLE --mode=parallel
done

echo ""
echo "=========================================="
echo "All runs completed."
echo "Results saved to: $CSV_FILE"
echo "=========================================="
