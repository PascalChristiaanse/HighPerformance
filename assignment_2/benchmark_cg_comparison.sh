#!/bin/bash
#
# Benchmark script for Exercise 1.2.13: Conjugate Gradient Comparison
# Compares CG algorithm with the best previous algorithm (optimized SOR)
# 
# Key measurements:
# - Time per iteration for both algorithms
# - Total iterations to convergence (CG should be ~125 for test problem)
# - Total solve time comparison
#

set -e

# Configuration
BUILD_DIR="build/debug-mpicxx-ninja"
EXECUTABLE="${BUILD_DIR}/PAR_Poisson"
INPUT_FILE="input.dat"
OUTPUT_DIR="benchmark_results/cg_comparison"

# Test parameters
GRID_SIZE=100            # Standard test problem size
NUM_PROCS=(1 2 4 8)      # Process counts to test
MAX_ITER=10000           # High enough to allow convergence
RUNS_PER_CONFIG=3        # Number of runs to average
OMEGA=1.8                # Best omega for SOR on this problem

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Results file
RESULTS_FILE="${OUTPUT_DIR}/cg_comparison_results.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

echo "=============================================="
echo "Exercise 1.2.13: CG vs SOR Comparison Benchmark"
echo "=============================================="
echo ""
echo "Comparing:"
echo "  - Conjugate Gradient (CG): No omega tuning, single-step iteration"
echo "  - Optimized SOR: omega=$OMEGA, stride-2 optimized loop"
echo ""
echo "Key verification: CG should converge in ~125 iterations for test problem"
echo ""

# Check executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo -e "${RED}Error: Executable not found at ${EXECUTABLE}${NC}"
    echo "Please build the project first: cd ${BUILD_DIR} && ninja"
    exit 1
fi

# Write CSV header
echo "solver,num_procs,run,iterations,total_time_s,time_per_iter_us,converged" > "${RESULTS_FILE}"

# Function to run a single benchmark
run_benchmark() {
    local solver=$1
    local nprocs=$2
    local run_num=$3
    
    local solver_args=""
    if [ "$solver" = "cg" ]; then
        solver_args="--solver cg"
    else
        solver_args="--solver sor --omega ${OMEGA} --optimized-loop"
    fi
    
    # Run the solver and capture output
    local output
    output=$(mpirun --oversubscribe -np ${nprocs} ${EXECUTABLE} \
        ${solver_args} \
        --grid-size ${GRID_SIZE} \
        --max-iter ${MAX_ITER} \
        ${INPUT_FILE} 2>&1)
    
    # Extract timing from output
    local time_sec
    time_sec=$(echo "$output" | grep "Elapsed time:" | head -1 | awk '{print $3}')
    
    # Extract iteration count
    local iterations
    iterations=$(echo "$output" | grep "Number of iterations:" | head -1 | awk '{print $4}')
    
    # Extract convergence status
    local converged
    converged=$(echo "$output" | grep "Converged:" | head -1 | awk '{print $2}')
    
    if [ -z "$time_sec" ] || [ -z "$iterations" ]; then
        echo "ERROR"
        echo "${solver},${nprocs},${run_num},ERROR,ERROR,ERROR,no" >> "${RESULTS_FILE}"
        return 1
    fi
    
    # Calculate time per iteration in microseconds
    local time_per_iter_us
    time_per_iter_us=$(echo "scale=6; ${time_sec} * 1000000 / ${iterations}" | bc)
    
    echo "${solver},${nprocs},${run_num},${iterations},${time_sec},${time_per_iter_us},${converged}" >> "${RESULTS_FILE}"
    echo "$time_sec $iterations"
}

# Main benchmark loop
total_tests=$((${#NUM_PROCS[@]} * 2 * RUNS_PER_CONFIG))
current_test=0

echo ""
echo "Running benchmarks..."
echo "Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "Total test configurations: ${total_tests}"
echo ""

for nprocs in "${NUM_PROCS[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing with ${nprocs} MPI processes${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Test CG
    echo -e "\n${CYAN}Conjugate Gradient:${NC}"
    cg_times=""
    cg_iters=""
    for run in $(seq 1 ${RUNS_PER_CONFIG}); do
        echo -n "  Run ${run}/${RUNS_PER_CONFIG}... "
        result=$(run_benchmark "cg" ${nprocs} ${run})
        if [ "$result" != "ERROR" ]; then
            time_val=$(echo "$result" | awk '{print $1}')
            iter_val=$(echo "$result" | awk '{print $2}')
            echo -e "${GREEN}✓${NC} ${time_val}s, ${iter_val} iterations"
            cg_times="${cg_times} ${time_val}"
            cg_iters="${cg_iters} ${iter_val}"
        else
            echo -e "${RED}✗ FAILED${NC}"
        fi
    done
    
    # Test SOR (optimized)
    echo -e "\n${CYAN}Optimized SOR (omega=${OMEGA}):${NC}"
    sor_times=""
    sor_iters=""
    for run in $(seq 1 ${RUNS_PER_CONFIG}); do
        echo -n "  Run ${run}/${RUNS_PER_CONFIG}... "
        result=$(run_benchmark "sor" ${nprocs} ${run})
        if [ "$result" != "ERROR" ]; then
            time_val=$(echo "$result" | awk '{print $1}')
            iter_val=$(echo "$result" | awk '{print $2}')
            echo -e "${GREEN}✓${NC} ${time_val}s, ${iter_val} iterations"
            sor_times="${sor_times} ${time_val}"
            sor_iters="${sor_iters} ${iter_val}"
        else
            echo -e "${RED}✗ FAILED${NC}"
        fi
    done
    
    echo ""
done

echo -e "\n${GREEN}Benchmark complete!${NC}"
echo "Results saved to: ${RESULTS_FILE}"
echo ""

# Generate summary
{
    echo "=============================================="
    echo "CG vs SOR Comparison - Summary"
    echo "=============================================="
    echo ""
    echo "Test Configuration:"
    echo "  Grid size: ${GRID_SIZE}x${GRID_SIZE}"
    echo "  SOR omega: ${OMEGA}"
    echo "  Runs per configuration: ${RUNS_PER_CONFIG}"
    echo ""
    echo "Expected CG iterations for test problem: ~125"
    echo ""
    echo "Raw results saved to: ${RESULTS_FILE}"
    echo ""
    echo "Run plot_cg_comparison.py to generate analysis and plots."
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo ""
echo "To generate plots, run:"
echo "  python3 plot_cg_comparison.py"
