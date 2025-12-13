#!/bin/bash

# Benchmark script for PAR_Poisson solver
# Tests with 1, 2, 4, 8, 16 cores and logs results to CSV

OUTPUT_CSV="benchmark_results.csv"
EXECUTABLE="./build/debug-mpicxx-ninja/PAR_Poisson"
CONFIG_FILE="input.dat"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first with: cmake --build build/debug-mpicxx-ninja"
    exit 1
fi

# Create CSV header
echo "cores,time_ms,iterations,converged" > "$OUTPUT_CSV"
echo "Benchmark started at $(date)"
echo "Results will be saved to $OUTPUT_CSV"
echo "----------------------------------------"

# Test with 2^n cores: 1, 2, 4, 8, 16
for n in 0 1 2 3 4; do
    cores=$((2**n))
    
    echo "Running with $cores core(s)..."
    
    # Run the solver and capture output
    output=$(mpirun --use-hwthread-cpus -np $cores "$EXECUTABLE" "$CONFIG_FILE" 2>&1)
    
    # Extract time from "Elapsed time: X.XXXXXX s" line
    time_s=$(echo "$output" | grep -oP "Elapsed time: \K[0-9.]+")
    
    # Convert to milliseconds
    if [ -n "$time_s" ]; then
        time_ms=$(echo "$time_s * 1000" | bc)
    else
        time_ms="N/A"
    fi
    
    # Extract iterations from "Number of iterations: X" line
    iterations=$(echo "$output" | grep -oP "Number of iterations: \K[0-9]+")
    
    # Extract convergence status
    converged=$(echo "$output" | grep -oP "Converged: \K(yes|no)")
    
    # Log to console
    echo "  Cores: $cores, Time: ${time_ms}ms, Iterations: $iterations, Converged: $converged"
    
    # Append to CSV
    echo "$cores,$time_ms,$iterations,$converged" >> "$OUTPUT_CSV"
done

echo "----------------------------------------"
echo "Benchmark completed at $(date)"
echo "Results saved to $OUTPUT_CSV"

# Display the CSV contents
echo ""
echo "Summary:"
cat "$OUTPUT_CSV" | column -t -s','
