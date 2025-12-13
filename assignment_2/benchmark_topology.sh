#!/bin/bash

# Benchmark script for PAR_Poisson solver - Topology Scaling Investigation
# Tests different processor topologies (4x1, 2x2, 1x4) with different grid sizes
# Uses fixed iterations to measure time per iteration: t(n) = α + β*n

OUTPUT_CSV="benchmark_topology_results.csv"
EXECUTABLE="./build/debug-mpicxx-ninja/PAR_Poisson"
CONFIG_FILE="input.dat"

# Solver parameters
OMEGA=1.95
SOLVER="sor"
NUM_PROCS=4

# Grid sizes to test
GRID_SIZES=(200 400 800)

# Iteration counts to test (for fitting t(n) = α + β*n)
ITERATION_COUNTS=(50 100 200 500)

# Processor topologies to test (format: "PxQ")
TOPOLOGIES=("4x1" "2x2" "1x4")

# Maximum runtime per test in seconds (180 = 3 minutes)
MAX_RUNTIME=180

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first with: cmake --build build/debug-mpicxx-ninja"
    exit 1
fi

# Create CSV header
echo "topology,grid_size,iterations,time_s,time_per_iter_ms,converged" > "$OUTPUT_CSV"
echo ""
echo "=============================================="
echo "Topology Scaling Benchmark"
echo "=============================================="
echo "Started at $(date)"
echo "Results will be saved to $OUTPUT_CSV"
echo ""
echo "Parameters:"
echo "  Solver: $SOLVER"
echo "  Omega: $OMEGA"
echo "  Processes: $NUM_PROCS"
echo "  Topologies: ${TOPOLOGIES[*]}"
echo "  Grid sizes: ${GRID_SIZES[*]}"
echo "  Iteration counts: ${ITERATION_COUNTS[*]}"
echo "  Max runtime per test: ${MAX_RUNTIME}s"
echo "=============================================="
echo ""

# Function to run a single benchmark
run_benchmark() {
    local topology=$1
    local grid_size=$2
    local max_iter=$3
    
    echo -n "  Testing topology=$topology, grid=${grid_size}x${grid_size}, iter=$max_iter..."
    
    # Run the solver with timeout
    output=$(timeout $MAX_RUNTIME mpirun --use-hwthread-cpus -np $NUM_PROCS "$EXECUTABLE" \
        --solver "$SOLVER" \
        --omega "$OMEGA" \
        --grid-size "$grid_size" \
        --max-iter "$max_iter" \
        --topology "$topology" \
        "$CONFIG_FILE" 2>&1)
    
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo " TIMEOUT (>${MAX_RUNTIME}s)"
        echo "$topology,$grid_size,$max_iter,TIMEOUT,N/A,no" >> "$OUTPUT_CSV"
        return 1
    fi
    
    # Extract time from "Elapsed time: X.XXXXXX s" line
    time_s=$(echo "$output" | grep -oP "Elapsed time: \K[0-9.]+")
    
    # Extract actual iterations from "Number of iterations: X" line
    actual_iterations=$(echo "$output" | grep -oP "Number of iterations: \K[0-9]+")
    
    # Extract convergence status
    converged=$(echo "$output" | grep -oP "Converged: \K(yes|no)")
    
    # Calculate time per iteration in milliseconds
    if [ -n "$time_s" ] && [ -n "$actual_iterations" ] && [ "$actual_iterations" -gt 0 ]; then
        time_per_iter_ms=$(echo "scale=6; ($time_s * 1000) / $actual_iterations" | bc)
    else
        time_per_iter_ms="N/A"
    fi
    
    if [ -n "$time_s" ]; then
        echo " ${time_s}s (${time_per_iter_ms}ms/iter)"
        echo "$topology,$grid_size,$actual_iterations,$time_s,$time_per_iter_ms,$converged" >> "$OUTPUT_CSV"
        return 0
    else
        echo " FAILED"
        echo "$topology,$grid_size,$max_iter,FAILED,N/A,$converged" >> "$OUTPUT_CSV"
        return 1
    fi
}

# Main benchmark loop
for topology in "${TOPOLOGIES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Testing topology: $topology"
    echo "----------------------------------------"
    
    for grid_size in "${GRID_SIZES[@]}"; do
        echo ""
        echo "  Grid size: ${grid_size}x${grid_size}"
        
        for max_iter in "${ITERATION_COUNTS[@]}"; do
            run_benchmark "$topology" "$grid_size" "$max_iter"
            
            # Check if we should skip larger iteration counts for this grid size
            # (if the previous run was close to timeout)
            if [ $? -ne 0 ]; then
                echo "    Skipping remaining iteration counts for this grid size"
                break
            fi
        done
    done
done

echo ""
echo "=============================================="
echo "Benchmark completed at $(date)"
echo "Results saved to $OUTPUT_CSV"
echo "=============================================="

# Display summary table
echo ""
echo "Summary (time per iteration in ms):"
echo ""

# Create a pivot-style summary
echo "Topology | Grid Size | Avg Time/Iter (ms)" 
echo "---------|-----------|-------------------"

for topology in "${TOPOLOGIES[@]}"; do
    for grid_size in "${GRID_SIZES[@]}"; do
        # Extract all time_per_iter values for this topology/grid combination
        avg_time=$(grep "^$topology,$grid_size," "$OUTPUT_CSV" | \
            awk -F',' '{if ($5 != "N/A" && $5 != "") sum+=$5; count++} END {if (count>0) printf "%.4f", sum/count; else print "N/A"}')
        printf "%-8s | %-9s | %s\n" "$topology" "${grid_size}x${grid_size}" "$avg_time"
    done
done

echo ""
echo "Full results:"
cat "$OUTPUT_CSV" | column -t -s','

# Additional analysis: Calculate α and β for t(n) = α + β*n using linear regression
echo ""
echo "=============================================="
echo "Linear Regression Analysis: t(n) = α + β*n"
echo "=============================================="
echo ""

for topology in "${TOPOLOGIES[@]}"; do
    echo "Topology: $topology"
    echo "-----------------"
    
    for grid_size in "${GRID_SIZES[@]}"; do
        # Extract iterations and times for this topology/grid
        data=$(grep "^$topology,$grid_size," "$OUTPUT_CSV" | \
            awk -F',' '{if ($4 != "TIMEOUT" && $4 != "FAILED") print $3, $4}')
        
        if [ -n "$data" ]; then
            # Simple linear regression using awk
            result=$(echo "$data" | awk '
            {
                n++
                sum_x += $1
                sum_y += $2
                sum_xy += $1 * $2
                sum_x2 += $1 * $1
            }
            END {
                if (n >= 2) {
                    denom = n * sum_x2 - sum_x * sum_x
                    if (denom != 0) {
                        beta = (n * sum_xy - sum_x * sum_y) / denom
                        alpha = (sum_y - beta * sum_x) / n
                        printf "  Grid %dx%d: α = %.6f s, β = %.6f s/iter (%.4f ms/iter)\n", 
                               '$grid_size', '$grid_size', alpha, beta, beta*1000
                    } else {
                        print "  Grid '$grid_size'x'$grid_size': Insufficient variance in data"
                    }
                } else {
                    print "  Grid '$grid_size'x'$grid_size': Not enough data points"
                }
            }')
            echo "$result"
        else
            echo "  Grid ${grid_size}x${grid_size}: No valid data"
        fi
    done
    echo ""
done

echo "Done!"
