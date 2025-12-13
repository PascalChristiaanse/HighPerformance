#!/bin/bash

# Benchmark script for PAR_Poisson solver - Convergence vs Grid Size
# Tests how the number of iterations to convergence depends on problem size
# Uses the same stopping criterion (precision goal from input.dat) for all sizes

OUTPUT_CSV="benchmark_convergence_results.csv"
EXECUTABLE="./build/debug-mpicxx-ninja/PAR_Poisson"
CONFIG_FILE="input.dat"

# Solver parameters
OMEGA=1.95
SOLVER="sor"
NUM_PROCS=4
TOPOLOGY="2x2"

# Grid sizes to test
GRID_SIZES=(200 400 800 1000)

# Maximum iterations (high enough to allow convergence)
MAX_ITER=100000

# Maximum runtime per test in seconds (300 = 5 minutes)
MAX_RUNTIME=300

# Number of runs per grid size (for averaging)
NUM_RUNS=3

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first with: cmake --build build/debug-mpicxx-ninja"
    exit 1
fi

# Create CSV header
echo "grid_size,run,iterations,time_s,converged,final_residual" > "$OUTPUT_CSV"
echo ""
echo "=============================================="
echo "Convergence vs Grid Size Benchmark"
echo "=============================================="
echo "Started at $(date)"
echo "Results will be saved to $OUTPUT_CSV"
echo ""
echo "Parameters:"
echo "  Solver: $SOLVER"
echo "  Omega: $OMEGA"
echo "  Processes: $NUM_PROCS"
echo "  Topology: $TOPOLOGY"
echo "  Grid sizes: ${GRID_SIZES[*]}"
echo "  Max iterations: $MAX_ITER"
echo "  Runs per size: $NUM_RUNS"
echo "  Max runtime per test: ${MAX_RUNTIME}s"
echo "=============================================="
echo ""

# Function to run a single benchmark
run_benchmark() {
    local grid_size=$1
    local run_num=$2
    
    echo -n "  Run $run_num: grid=${grid_size}x${grid_size}..."
    
    # Run the solver with timeout
    output=$(timeout $MAX_RUNTIME mpirun --use-hwthread-cpus -np $NUM_PROCS "$EXECUTABLE" \
        --solver "$SOLVER" \
        --omega "$OMEGA" \
        --grid-size "$grid_size" \
        --max-iter "$MAX_ITER" \
        --topology "$TOPOLOGY" \
        "$CONFIG_FILE" 2>&1)
    
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo " TIMEOUT (>${MAX_RUNTIME}s)"
        echo "$grid_size,$run_num,TIMEOUT,TIMEOUT,no,N/A" >> "$OUTPUT_CSV"
        return 1
    fi
    
    # Extract time from "Elapsed time: X.XXXXXX s" line
    time_s=$(echo "$output" | grep -oP "Elapsed time: \K[0-9.]+")
    
    # Extract iterations from "Number of iterations: X" line
    iterations=$(echo "$output" | grep -oP "Number of iterations: \K[0-9]+")
    
    # Extract convergence status
    converged=$(echo "$output" | grep -oP "Converged: \K(yes|no)")
    
    # Extract final residual
    final_residual=$(echo "$output" | grep -oP "Final residual: \K[0-9.e+-]+")
    
    if [ -n "$iterations" ]; then
        echo " $iterations iterations, ${time_s}s, converged=$converged"
        echo "$grid_size,$run_num,$iterations,$time_s,$converged,$final_residual" >> "$OUTPUT_CSV"
        return 0
    else
        echo " FAILED"
        echo "$grid_size,$run_num,FAILED,FAILED,no,N/A" >> "$OUTPUT_CSV"
        return 1
    fi
}

# Main benchmark loop
for grid_size in "${GRID_SIZES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Testing grid size: ${grid_size}x${grid_size}"
    echo "----------------------------------------"
    
    for run in $(seq 1 $NUM_RUNS); do
        run_benchmark "$grid_size" "$run"
    done
done

echo ""
echo "=============================================="
echo "Benchmark completed at $(date)"
echo "Results saved to $OUTPUT_CSV"
echo "=============================================="

# Display summary table
echo ""
echo "Summary:"
echo ""

# Calculate averages
echo "Grid Size | Avg Iterations | Avg Time (s) | Converged"
echo "----------|----------------|--------------|----------"

for grid_size in "${GRID_SIZES[@]}"; do
    # Extract all values for this grid size
    result=$(grep "^$grid_size," "$OUTPUT_CSV" | \
        awk -F',' '{
            if ($3 != "TIMEOUT" && $3 != "FAILED") {
                sum_iter += $3
                sum_time += $4
                count++
                conv = $5
            }
        } END {
            if (count > 0) {
                printf "%d | %.1f | %.4f | %s", '$grid_size', sum_iter/count, sum_time/count, conv
            } else {
                print "'$grid_size' | N/A | N/A | no"
            }
        }')
    echo "$result"
done

echo ""
echo "Full results:"
cat "$OUTPUT_CSV" | column -t -s','

# Scaling analysis
echo ""
echo "=============================================="
echo "Scaling Analysis"
echo "=============================================="
echo ""
echo "Theoretical expectation: For a grid of size N×N, the number of iterations"
echo "should scale approximately as O(N²) for simple iterative methods like"
echo "Gauss-Seidel, or O(N) for optimized SOR with optimal omega."
echo ""
echo "Iteration count ratios (relative to 200x200):"

# Get baseline iterations for 200x200
baseline=$(grep "^200," "$OUTPUT_CSV" | awk -F',' '{if ($3 != "TIMEOUT" && $3 != "FAILED") {sum+=$3; count++}} END {if(count>0) print sum/count; else print 0}')

if [ -n "$baseline" ] && [ "$baseline" != "0" ]; then
    for grid_size in "${GRID_SIZES[@]}"; do
        avg_iter=$(grep "^$grid_size," "$OUTPUT_CSV" | awk -F',' '{if ($3 != "TIMEOUT" && $3 != "FAILED") {sum+=$3; count++}} END {if(count>0) print sum/count; else print 0}')
        if [ -n "$avg_iter" ] && [ "$avg_iter" != "0" ]; then
            ratio=$(echo "scale=2; $avg_iter / $baseline" | bc)
            size_ratio=$(echo "scale=2; $grid_size / 200" | bc)
            echo "  ${grid_size}x${grid_size}: ${avg_iter} iterations (ratio: ${ratio}x, size ratio: ${size_ratio}x)"
        fi
    done
fi

echo ""
echo "Done!"
