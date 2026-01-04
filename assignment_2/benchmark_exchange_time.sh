#!/bin/bash
#
# Benchmark script for Exercise 1.2.10: Exchange Time Analysis
# Measures time spent in Exchange_Borders vs computation as a function of
# problem size and number of processes.
#

set -e

# Configuration
BUILD_DIR="build/debug-mpicxx-ninja"
EXECUTABLE="${BUILD_DIR}/PAR_Poisson"
INPUT_FILE="input.dat"
OUTPUT_DIR="benchmark_results/exchange_time"

# Test parameters
GRID_SIZES=(32 50 100 200 400 800)
NUM_PROCS=(2 4 8 16)
MAX_ITER=50            # Fixed iterations for timing
OMEGA=1.88             # Use SOR
RUNS_PER_CONFIG=3      # Number of runs to average

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
RESULTS_FILE="${OUTPUT_DIR}/exchange_time_results.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

echo "=============================================="
echo "Exercise 1.2.10: Exchange Time Analysis"
echo "=============================================="
echo ""
echo "Measuring time spent in boundary exchange vs computation"
echo "as a function of problem size and number of processes."
echo ""

# Check executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo -e "${RED}Error: Executable not found at ${EXECUTABLE}${NC}"
    echo "Please build the project first: cd ${BUILD_DIR} && ninja"
    exit 1
fi

# Write CSV header
echo "grid_size,num_procs,run,total_time,step_time,exchange_time,reduction_time,exchange_ratio,points_per_proc" > "${RESULTS_FILE}"

# Function to run a single benchmark and extract timing
run_benchmark() {
    local grid_size=$1
    local nprocs=$2
    local run_num=$3
    
    # Run the solver with verbose timing and capture output
    local output
    output=$(mpirun --oversubscribe -np ${nprocs} ${EXECUTABLE} \
        --solver sor \
        --omega ${OMEGA} \
        --grid-size ${grid_size} \
        --max-iter ${MAX_ITER} \
        --verbose-timing \
        ${INPUT_FILE} 2>&1)
    
    # Extract timing values from verbose output
    # Format: "║   Compute time:               0.002546 s ( 94.3%)           ║"
    local total_time step_time exchange_time reduction_time
    
    total_time=$(echo "$output" | grep "Total time (t):" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    step_time=$(echo "$output" | grep "Compute time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    exchange_time=$(echo "$output" | grep "Exchange time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    reduction_time=$(echo "$output" | grep "Reduction time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    
    # Handle missing values
    [ -z "$total_time" ] && total_time="0"
    [ -z "$step_time" ] && step_time="0"
    [ -z "$exchange_time" ] && exchange_time="0"
    [ -z "$reduction_time" ] && reduction_time="0"
    
    # Calculate exchange ratio (exchange_time / step_time)
    local exchange_ratio
    if [[ "$step_time" != "0" && "$step_time" != "ERROR" ]]; then
        exchange_ratio=$(echo "scale=6; $exchange_time / $step_time" | bc -l)
    else
        exchange_ratio="0"
    fi
    
    # Calculate points per process
    local points_per_proc=$((grid_size * grid_size / nprocs))
    
    echo "${grid_size},${nprocs},${run_num},${total_time},${step_time},${exchange_time},${reduction_time},${exchange_ratio},${points_per_proc}" >> "${RESULTS_FILE}"
    
    # Return values for averaging (tab-separated)
    echo "${step_time}	${exchange_time}	${reduction_time}	${total_time}"
}

# Function to calculate average of a specific field
calc_avg() {
    local sum=0
    local count=0
    for val in "$@"; do
        if [[ "$val" != "0" && "$val" != "ERROR" ]]; then
            sum=$(echo "$sum + $val" | bc -l)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 0 ]; then
        echo "scale=6; $sum / $count" | bc -l
    else
        echo "0"
    fi
}

# Main benchmark loop
echo ""
echo "Starting benchmarks..."
echo ""

# Summary data for final report
declare -A step_times
declare -A exchange_times
declare -A reduction_times
declare -A total_times

for grid_size in "${GRID_SIZES[@]}"; do
    echo -e "${BLUE}=== Grid Size: ${grid_size} x ${grid_size} ===${NC}"
    
    for nprocs in "${NUM_PROCS[@]}"; do
        # Skip invalid configurations
        if [ $nprocs -gt $((grid_size * grid_size / 4)) ]; then
            echo -e "  ${YELLOW}Skipping ${nprocs} procs (too many for grid size)${NC}"
            continue
        fi
        
        # Skip 1 proc for exchange time measurement (no exchange needed)
        # But still run it for comparison
        
        echo -n "  Processors: ${nprocs} ... "
        
        # Run benchmarks
        step_runs=()
        exchange_runs=()
        reduction_runs=()
        total_runs=()
        
        for run in $(seq 1 ${RUNS_PER_CONFIG}); do
            result=$(run_benchmark ${grid_size} ${nprocs} ${run})
            step_runs+=($(echo "$result" | cut -f1))
            exchange_runs+=($(echo "$result" | cut -f2))
            reduction_runs+=($(echo "$result" | cut -f3))
            total_runs+=($(echo "$result" | cut -f4))
        done
        
        # Calculate averages
        avg_step=$(calc_avg "${step_runs[@]}")
        avg_exchange=$(calc_avg "${exchange_runs[@]}")
        avg_reduction=$(calc_avg "${reduction_runs[@]}")
        avg_total=$(calc_avg "${total_runs[@]}")
        
        # Store for summary
        key="${grid_size}_${nprocs}"
        step_times[$key]=$avg_step
        exchange_times[$key]=$avg_exchange
        reduction_times[$key]=$avg_reduction
        total_times[$key]=$avg_total
        
        # Calculate and display ratio
        if [[ "$avg_step" != "0" ]]; then
            ratio=$(echo "scale=3; $avg_exchange / $avg_step" | bc -l)
            ratio_pct=$(echo "scale=1; $ratio * 100" | bc -l)
            
            if (( $(echo "$ratio >= 0.9 && $ratio <= 1.1" | bc -l) )); then
                echo -e "${RED}Step: ${avg_step}s, Exchange: ${avg_exchange}s, Ratio: ${ratio} (${ratio_pct}%) *** BALANCED ***${NC}"
            elif (( $(echo "$ratio > 1.0" | bc -l) )); then
                echo -e "${YELLOW}Step: ${avg_step}s, Exchange: ${avg_exchange}s, Ratio: ${ratio} (${ratio_pct}%) [Exchange dominant]${NC}"
            else
                echo -e "${GREEN}Step: ${avg_step}s, Exchange: ${avg_exchange}s, Ratio: ${ratio} (${ratio_pct}%) [Compute dominant]${NC}"
            fi
        else
            echo -e "${CYAN}Step: ${avg_step}s, Exchange: ${avg_exchange}s${NC}"
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
    echo "Exercise 1.2.10: Exchange Time Analysis"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  - Solver: SOR (omega = ${OMEGA})"
    echo "  - Max iterations: ${MAX_ITER}"
    echo "  - Runs per configuration: ${RUNS_PER_CONFIG}"
    echo ""
    echo "Question: For which number of processes and/or problem size"
    echo "          is exchange time approximately equal to computation time?"
    echo ""
    echo "Results Summary (times in seconds):"
    echo ""
    printf "%-10s %-6s %-10s %-10s %-12s %-10s %-14s\n" \
        "Grid" "Procs" "Step" "Exchange" "Reduction" "Ratio" "Points/Proc"
    printf "%-10s %-6s %-10s %-10s %-12s %-10s %-14s\n" \
        "----" "-----" "----" "--------" "---------" "-----" "-----------"
    
    for grid_size in "${GRID_SIZES[@]}"; do
        for nprocs in "${NUM_PROCS[@]}"; do
            key="${grid_size}_${nprocs}"
            step=${step_times[$key]}
            exch=${exchange_times[$key]}
            redu=${reduction_times[$key]}
            
            if [[ -n "$step" && -n "$exch" && "$step" != "0" ]]; then
                ratio=$(echo "scale=3; $exch / $step" | bc -l)
                points_per_proc=$((grid_size * grid_size / nprocs))
                
                # Mark configurations where ratio is close to 1.0
                marker=""
                if (( $(echo "$ratio >= 0.8 && $ratio <= 1.2" | bc -l) )); then
                    marker=" ***"
                fi
                
                printf "%-10s %-6s %-10.6f %-10.6f %-12.6f %-10.3f %-14d%s\n" \
                    "${grid_size}x${grid_size}" "${nprocs}" "$step" "$exch" "$redu" "$ratio" "$points_per_proc" "$marker"
            fi
        done
    done
    
    echo ""
    echo "Legend:"
    echo "  Ratio = Exchange Time / Step Time"
    echo "  ***   = Ratio close to 1.0 (exchange ≈ computation)"
    echo ""
    echo "Analysis:"
    echo ""
    echo "The exchange time depends on:"
    echo "  - Number of boundary points to exchange (perimeter of subdomain)"
    echo "  - MPI latency and bandwidth"
    echo ""
    echo "The computation time depends on:"
    echo "  - Number of interior points (area of subdomain)"
    echo "  - CPU performance"
    echo ""
    echo "Surface-to-volume ratio:"
    echo "  - For a subdomain of size n×n: boundary ~ 4n, interior ~ n²"
    echo "  - Ratio of boundary/interior ~ 4/n"
    echo "  - Exchange time becomes significant when subdomain is small"
    echo ""
    echo "Expected crossover point:"
    echo "  - When exchange_time ≈ step_time"
    echo "  - This typically occurs when subdomains become too small"
    echo "  - Adding more processes beyond this point hurts performance"
    echo ""
    echo "Raw data saved to: ${RESULTS_FILE}"
    
} | tee "${SUMMARY_FILE}"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
echo "Results saved to:"
echo "  - CSV data: ${RESULTS_FILE}"
echo "  - Summary: ${SUMMARY_FILE}"

# Generate a simple plot data file for gnuplot or similar
PLOT_FILE="${OUTPUT_DIR}/plot_data.dat"
{
    echo "# Grid_Size Num_Procs Step_Time Exchange_Time Ratio Points_Per_Proc"
    for grid_size in "${GRID_SIZES[@]}"; do
        for nprocs in "${NUM_PROCS[@]}"; do
            key="${grid_size}_${nprocs}"
            step=${step_times[$key]}
            exch=${exchange_times[$key]}
            
            if [[ -n "$step" && -n "$exch" && "$step" != "0" ]]; then
                ratio=$(echo "scale=6; $exch / $step" | bc -l)
                points_per_proc=$((grid_size * grid_size / nprocs))
                echo "$grid_size $nprocs $step $exch $ratio $points_per_proc"
            fi
        done
        echo ""  # Blank line between grid sizes for gnuplot
    done
} > "${PLOT_FILE}"

echo "  - Plot data: ${PLOT_FILE}"
