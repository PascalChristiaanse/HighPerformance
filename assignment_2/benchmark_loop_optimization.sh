#!/bin/bash
#
# Benchmark script for Exercise 1.2.9: Loop Optimization
# Compares regular loop (with parity check) vs stride-2 optimized loop
#

set -e

# Configuration
BUILD_DIR="build/debug-mpicxx-ninja"
EXECUTABLE="${BUILD_DIR}/PAR_Poisson"
INPUT_FILE="input.dat"
OUTPUT_DIR="benchmark_results/loop_optimization"

# Test parameters
GRID_SIZES=(100 200 400)
NUM_PROCS=(2 4 8 )
MAX_ITER=100           # Fixed iterations for fair timing comparison
OMEGA=1.88             # Use SOR for faster convergence
RUNS_PER_CONFIG=3      # Number of runs to average

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Results file
RESULTS_FILE="${OUTPUT_DIR}/loop_optimization_results.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

echo "=============================================="
echo "Exercise 1.2.9: Loop Optimization Benchmark"
echo "=============================================="
echo ""
echo "Comparing:"
echo "  - Regular loop: checks (globalX + globalY) % 2 == parity for each point"
echo "  - Optimized loop: stride-2 iteration, no parity check needed"
echo ""

# Check executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo -e "${RED}Error: Executable not found at ${EXECUTABLE}${NC}"
    echo "Please build the project first: cd ${BUILD_DIR} && ninja"
    exit 1
fi

# Write CSV header
echo "grid_size,num_procs,loop_type,run,time_seconds,iterations" > "${RESULTS_FILE}"

# Function to run a single benchmark
run_benchmark() {
    local grid_size=$1
    local nprocs=$2
    local optimized=$3
    local run_num=$4
    
    local loop_type="regular"
    local opt_flag=""
    if [ "$optimized" = "1" ]; then
        loop_type="optimized"
        opt_flag="--optimized-loop"
    fi
    
    # Run the solver and capture output
    local output
    output=$(mpirun --oversubscribe -np ${nprocs} ${EXECUTABLE} \
        --solver sor \
        --omega ${OMEGA} \
        --grid-size ${grid_size} \
        --max-iter ${MAX_ITER} \
        ${opt_flag} \
        ${INPUT_FILE} 2>&1)
    
    # Extract timing from output (look for "Elapsed time: X.XXXXXX s")
    local time_sec
    time_sec=$(echo "$output" | grep "Elapsed time:" | head -1 | awk '{print $3}')
    
    # Extract iteration count
    local iterations
    iterations=$(echo "$output" | grep "Number of iterations:" | head -1 | awk '{print $4}')
    
    if [ -z "$time_sec" ]; then
        time_sec="ERROR"
    fi
    if [ -z "$iterations" ]; then
        iterations="ERROR"
    fi
    
    echo "${grid_size},${nprocs},${loop_type},${run_num},${time_sec},${iterations}" >> "${RESULTS_FILE}"
    echo "$time_sec"
}

# Function to calculate average
calc_avg() {
    local sum=0
    local count=0
    for val in "$@"; do
        if [[ "$val" != "ERROR" ]]; then
            sum=$(echo "$sum + $val" | bc -l)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 0 ]; then
        echo "scale=6; $sum / $count" | bc -l
    else
        echo "ERROR"
    fi
}

# Main benchmark loop
echo ""
echo "Starting benchmarks..."
echo ""

# Summary data for final report
declare -A regular_times
declare -A optimized_times

for grid_size in "${GRID_SIZES[@]}"; do
    echo -e "${BLUE}=== Grid Size: ${grid_size} x ${grid_size} ===${NC}"
    
    for nprocs in "${NUM_PROCS[@]}"; do
        # Skip if more procs than grid points per dimension
        if [ $nprocs -gt $((grid_size * grid_size / 4)) ]; then
            echo -e "  ${YELLOW}Skipping ${nprocs} procs (too many for grid size)${NC}"
            continue
        fi
        
        echo -n "  Processors: ${nprocs} ... "
        
        # Run regular loop benchmarks
        regular_runs=()
        for run in $(seq 1 ${RUNS_PER_CONFIG}); do
            t=$(run_benchmark ${grid_size} ${nprocs} 0 ${run})
            regular_runs+=("$t")
        done
        
        # Run optimized loop benchmarks
        optimized_runs=()
        for run in $(seq 1 ${RUNS_PER_CONFIG}); do
            t=$(run_benchmark ${grid_size} ${nprocs} 1 ${run})
            optimized_runs+=("$t")
        done
        
        # Calculate averages
        avg_regular=$(calc_avg "${regular_runs[@]}")
        avg_optimized=$(calc_avg "${optimized_runs[@]}")
        
        # Store for summary
        key="${grid_size}_${nprocs}"
        regular_times[$key]=$avg_regular
        optimized_times[$key]=$avg_optimized
        
        # Calculate speedup
        if [[ "$avg_regular" != "ERROR" && "$avg_optimized" != "ERROR" ]]; then
            speedup=$(echo "scale=3; $avg_regular / $avg_optimized" | bc -l)
            improvement=$(echo "scale=6; ($avg_regular - $avg_optimized) / $avg_regular * 100" | bc -l)
            
            if (( $(echo "$speedup > 1.0" | bc -l) )); then
                echo -e "${GREEN}Regular: ${avg_regular}s, Optimized: ${avg_optimized}s, Speedup: ${speedup}x (+${improvement}%)${NC}"
            else
                echo -e "${YELLOW}Regular: ${avg_regular}s, Optimized: ${avg_optimized}s, Speedup: ${speedup}x (${improvement}%)${NC}"
            fi
        else
            echo -e "${RED}Error in timing${NC}"
        fi
    done
    echo ""
done

# Generate summary report
echo "=============================================="
echo "Generating Summary Report..."
echo "=============================================="

{
    echo "=========================================="
    echo "Exercise 1.2.9: Loop Optimization Results"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  - Solver: SOR (omega = ${OMEGA})"
    echo "  - Max iterations: ${MAX_ITER}"
    echo "  - Runs per configuration: ${RUNS_PER_CONFIG}"
    echo ""
    echo "Optimization Description:"
    echo "  Regular loop: Iterates over all grid points, uses if-statement to check parity"
    echo "  Optimized loop: Uses stride-2 iteration, calculates correct starting y-offset"
    echo ""
    echo "Results Summary (times in seconds):"
    echo ""
    printf "%-10s %-8s %-12s %-12s %-10s %-12s\n" "Grid" "Procs" "Regular" "Optimized" "Speedup" "Improvement"
    printf "%-10s %-8s %-12s %-12s %-10s %-12s\n" "----" "-----" "-------" "---------" "-------" "-----------"
    
    for grid_size in "${GRID_SIZES[@]}"; do
        for nprocs in "${NUM_PROCS[@]}"; do
            key="${grid_size}_${nprocs}"
            reg=${regular_times[$key]}
            opt=${optimized_times[$key]}
            
            if [[ -n "$reg" && -n "$opt" && "$reg" != "ERROR" && "$opt" != "ERROR" ]]; then
                speedup=$(echo "scale=3; $reg / $opt" | bc -l)
                improvement=$(echo "scale=6; ($reg - $opt) / $reg * 100" | bc -l)
                printf "%-10s %-8s %-12.6f %-12.6f %-10.3f %+.2f%%\n" \
                    "${grid_size}x${grid_size}" "${nprocs}" "$reg" "$opt" "$speedup" "$improvement"
            fi
        done
    done
    
    echo ""
    echo "Analysis:"
    echo "  - The optimized loop eliminates the parity check inside the inner loop"
    echo "  - This reduces branch mispredictions and improves cache efficiency"
    echo "  - Expected improvement: 5-15% depending on architecture and cache behavior"
    echo ""
    echo "Raw data saved to: ${RESULTS_FILE}"
    
} | tee "${SUMMARY_FILE}"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
echo "Results saved to:"
echo "  - CSV data: ${RESULTS_FILE}"
echo "  - Summary: ${SUMMARY_FILE}"
