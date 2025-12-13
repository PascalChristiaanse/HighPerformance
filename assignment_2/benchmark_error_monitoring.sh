#!/bin/bash

# Benchmark script for PAR_Poisson solver - Error vs Iteration Monitoring
# Monitors the residual (error) as a function of iteration number for 800x800 grid

OUTPUT_DIR="error_monitoring"
EXECUTABLE="./build/debug-mpicxx-ninja/PAR_Poisson"
CONFIG_FILE="input.dat"

# Solver parameters
OMEGA=1.8
SOLVER="cg"
NUM_PROCS=8
TOPOLOGY="4x2"
GRID_SIZE=800

# Maximum iterations (high enough to allow convergence)
MAX_ITER=100000

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first with: cmake --build build/debug-mpicxx-ninja"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Error vs Iteration Monitoring"
echo "=============================================="
echo "Started at $(date)"
echo ""
echo "Parameters:"
echo "  Solver: $SOLVER"
echo "  Omega: $OMEGA"
echo "  Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "  Processes: $NUM_PROCS"
echo "  Topology: $TOPOLOGY"
echo "  Max iterations: $MAX_ITER"
echo "=============================================="
echo ""

OUTPUT_CSV="${OUTPUT_DIR}/error_vs_iteration_${GRID_SIZE}.csv"

echo "Running solver and recording per-iteration telemetry..."
echo ""

# Run the solver with verbose timing to capture per-iteration residuals
output=$(mpirun --use-hwthread-cpus -np $NUM_PROCS "$EXECUTABLE" \
    --solver "$SOLVER" \
    --omega "$OMEGA" \
    --grid-size "$GRID_SIZE" \
    --max-iter "$MAX_ITER" \
    --topology "$TOPOLOGY" \
    --verbose-timing \
    --timing-output "$OUTPUT_CSV" \
    "$CONFIG_FILE" 2>&1)

echo "$output"

# Check if the CSV was created
if [ -f "$OUTPUT_CSV" ]; then
    echo ""
    echo "=============================================="
    echo "Results saved to $OUTPUT_CSV"
    echo "=============================================="
    
    # Show first and last few lines
    echo ""
    echo "First 10 iterations:"
    head -11 "$OUTPUT_CSV" | column -t -s','
    
    echo ""
    echo "Last 10 iterations:"
    tail -10 "$OUTPUT_CSV" | column -t -s','
    
    # Count total iterations
    total_lines=$(($(wc -l < "$OUTPUT_CSV") - 1))
    echo ""
    echo "Total iterations recorded: $total_lines"
else
    echo ""
    echo "ERROR: CSV file was not created!"
    echo "The solver output was:"
    echo "$output"
fi

echo ""
echo "=============================================="
echo "Benchmark completed at $(date)"
echo "=============================================="
echo ""
echo "To plot the results, run:"
echo "  python plot_error_monitoring.py"
