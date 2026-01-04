#!/bin/bash

# Benchmark script for PAR_Poisson solver - Sweeps per Exchange Investigation
# Tests how multiple sweeps between border exchanges affects convergence
# NOTE: Only applies to GS and SOR solvers (CG uses single-step iteration)

OUTPUT_CSV="benchmark_sweeps_results.csv"
EXECUTABLE="./build/debug-mpicxx-ninja/PAR_Poisson"
CONFIG_FILE="input.dat"

# Solver parameters - MUST use SOR or GS (not CG - sweeps don't apply to CG)
SOLVER="sor"
OMEGA=1.99
NUM_PROCS=8
TOPOLOGY="4x2"
GRID_SIZE=100

# Maximum iterations (high enough to allow convergence)
MAX_ITER=100000

# Maximum runtime per test in seconds (300 = 5 minutes)
MAX_RUNTIME=300

# Sweeps per exchange to test
SWEEPS_VALUES=(1 2 3 4 5 6 7 8 9 10)

# Number of runs per configuration (for averaging)
NUM_RUNS=3

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first with: cmake --build build/debug-mpicxx-ninja"
    exit 1
fi

# Create CSV header
echo "sweeps_per_exchange,run,iterations,total_sweep_pairs,time_s,converged,final_residual,time_per_iter_ms,time_per_sweep_ms" > "$OUTPUT_CSV"
echo ""
echo "=============================================="
echo "Sweeps per Exchange Benchmark"
echo "=============================================="
echo "Started at $(date)"
echo "Results will be saved to $OUTPUT_CSV"
echo ""
echo "Parameters:"
echo "  Solver: $SOLVER (omega=$OMEGA)"
echo "  Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "  Processes: $NUM_PROCS"
echo "  Topology: $TOPOLOGY"
echo "  Sweeps per exchange: ${SWEEPS_VALUES[*]}"
echo "  Max iterations: $MAX_ITER"
echo "  Runs per config: $NUM_RUNS"
echo "  Max runtime per test: ${MAX_RUNTIME}s"
echo "=============================================="
echo ""

# Function to run a single benchmark
run_benchmark() {
    local sweeps=$1
    local run_num=$2
    
    echo -n "  Run $run_num: sweeps=$sweeps..."
    
    # Run the solver with timeout
    output=$(timeout $MAX_RUNTIME mpirun --use-hwthread-cpus -np $NUM_PROCS "$EXECUTABLE" \
        --solver "$SOLVER" \
        --omega "$OMEGA" \
        --grid-size "$GRID_SIZE" \
        --max-iter "$MAX_ITER" \
        --topology "$TOPOLOGY" \
        --sweeps-per-exchange "$sweeps" \
        "$CONFIG_FILE" 2>&1)
    
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo " TIMEOUT (>${MAX_RUNTIME}s)"
        echo "$sweeps,$run_num,TIMEOUT,TIMEOUT,no,N/A,N/A" >> "$OUTPUT_CSV"
        return 1
    fi
    
    # Extract time from "Elapsed time: X.XXXXXX s" line
    time_s=$(echo "$output" | grep -oP "Elapsed time: \K[0-9.]+")
    
    # Extract iterations from "Number of iterations: X" line
    iterations=$(echo "$output" | grep -oP "Number of iterations: \K[0-9]+")
    
    # Extract total sweep-pairs from "Total sweep-pairs: X" line
    total_sweep_pairs=$(echo "$output" | grep -oP "Total sweep-pairs: \K[0-9]+")
    
    # Extract convergence status
    converged=$(echo "$output" | grep -oP "Converged: \K(yes|no)")
    
    # Extract final residual
    final_residual=$(echo "$output" | grep -oP "Final residual: \K[0-9.e+-]+")
    
    # Calculate time per iteration (outer loop)
    if [ -n "$time_s" ] && [ -n "$iterations" ] && [ "$iterations" -gt 0 ]; then
        time_per_iter_ms=$(echo "scale=6; ($time_s * 1000) / $iterations" | bc)
    else
        time_per_iter_ms="N/A"
    fi
    
    # Calculate time per sweep-pair (the real work unit - should be constant!)
    if [ -n "$time_s" ] && [ -n "$total_sweep_pairs" ] && [ "$total_sweep_pairs" -gt 0 ]; then
        time_per_sweep_ms=$(echo "scale=6; ($time_s * 1000) / $total_sweep_pairs" | bc)
    else
        time_per_sweep_ms="N/A"
    fi
    
    if [ -n "$iterations" ]; then
        echo " $iterations iters ($total_sweep_pairs sweeps), ${time_s}s, ${time_per_sweep_ms}ms/sweep"
        echo "$sweeps,$run_num,$iterations,$total_sweep_pairs,$time_s,$converged,$final_residual,$time_per_iter_ms,$time_per_sweep_ms" >> "$OUTPUT_CSV"
        return 0
    else
        echo " FAILED"
        echo "$sweeps,$run_num,FAILED,FAILED,FAILED,no,N/A,N/A,N/A" >> "$OUTPUT_CSV"
        return 1
    fi
}

# Main benchmark loop
for sweeps in "${SWEEPS_VALUES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Testing sweeps per exchange: $sweeps"
    echo "----------------------------------------"
    
    for run in $(seq 1 $NUM_RUNS); do
        run_benchmark "$sweeps" "$run"
    done
done

echo ""
echo "=============================================="
echo "Benchmark completed at $(date)"
echo "Results saved to $OUTPUT_CSV"
echo "=============================================="

# Display summary table
echo ""
echo "Summary (averages):"
echo ""
echo "Sweeps | Avg Iterations | Avg Time (s) | Avg Time/Iter (ms) | Converged"
echo "-------|----------------|--------------|--------------------|-----------"

for sweeps in "${SWEEPS_VALUES[@]}"; do
    result=$(grep "^$sweeps," "$OUTPUT_CSV" | \
        awk -F',' '{
            if ($3 != "TIMEOUT" && $3 != "FAILED") {
                sum_iter += $3
                sum_time += $4
                sum_tpi += $7
                count++
                conv = $5
            }
        } END {
            if (count > 0) {
                printf "%6d | %14.1f | %12.4f | %18.4f | %s", '$sweeps', sum_iter/count, sum_time/count, sum_tpi/count, conv
            } else {
                print "'$sweeps' | N/A | N/A | N/A | no"
            }
        }')
    echo "$result"
done

echo ""
echo "Full results:"
cat "$OUTPUT_CSV" | column -t -s','

echo ""
echo "Done! Run 'python plot_sweeps_benchmark.py' to visualize the results."
