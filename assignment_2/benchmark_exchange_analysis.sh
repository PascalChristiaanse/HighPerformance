#!/bin/bash
#
# Benchmark script for Exercise 1.2.11: Exchange_Borders Analysis
# Determines latency/overhead and bandwidth for point-to-point communication.
# Uses same configurations as 1.2.3: 4x1 and 2x2 topologies with grid sizes 200, 400, 800.
#

set -e

# Configuration
BUILD_DIR="build/debug-mpicxx-ninja"
EXECUTABLE="${BUILD_DIR}/PAR_Poisson"
INPUT_FILE="input.dat"
OUTPUT_DIR="benchmark_results/exchange_analysis"

# Test parameters (same as exercise 1.2.3)
GRID_SIZES=( 64p 256 512 1024)
TOPOLOGIES=("4x1" "2x2")
NUM_PROCS=4
MAX_ITER=100          # Fixed iterations for timing
OMEGA=1.95            # Use SOR (as specified in 1.2.3)
RUNS_PER_CONFIG=3     # Number of runs to average

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Results file
RESULTS_FILE="${OUTPUT_DIR}/exchange_analysis_results.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

echo "=============================================="
echo "Exercise 1.2.11: Exchange_Borders Analysis"
echo "=============================================="
echo ""
echo "Analyzing latency, bandwidth, and data communication"
echo "for Exchange_Borders function."
echo ""
echo "Configurations (same as 1.2.3):"
echo "  Topologies: ${TOPOLOGIES[*]}"
echo "  Grid sizes: ${GRID_SIZES[*]}"
echo "  Omega: $OMEGA"
echo "  Processes: $NUM_PROCS"
echo ""

# Check executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "Error: Executable not found at ${EXECUTABLE}"
    echo "Please build the project first"
    exit 1
fi

# Write CSV header
# Include fields for calculating bandwidth: data transferred and exchange time
echo "topology,grid_size,run,total_time,step_time,exchange_time,reduction_time,iterations,local_nx,local_ny,data_per_exchange_bytes,exchange_count" > "${RESULTS_FILE}"

# Function to calculate data transferred per exchange
# In 2D domain decomposition:
# - X-direction: exchange rows of size local_ny
# - Y-direction: exchange columns of size local_nx
# Each exchange involves send+recv with potentially 4 neighbors
calculate_data() {
    local grid_size=$1
    local topology=$2
    
    # Parse topology
    local px=$(echo $topology | cut -d'x' -f1)
    local py=$(echo $topology | cut -d'x' -f2)
    
    # Local subdomain size (excluding ghost cells)
    local local_nx=$((grid_size / px))
    local local_ny=$((grid_size / py))
    
    # Data exchanged per direction (8 bytes per double):
    # X-direction: 2 rows of local_ny elements (send+recv both directions)
    # Y-direction: 2 columns of local_nx elements (send+recv both directions)
    # Factor of 2 for bidirectional communication
    local x_data=$((2 * 2 * local_ny * 8))  # 2 directions, 2 borders (send+recv), local_ny elements
    local y_data=$((2 * 2 * local_nx * 8))  # 2 directions, 2 borders (send+recv), local_nx elements
    
    # For interior processes, data = x_data + y_data
    # Edge processes have fewer neighbors, but we report worst case (interior)
    local total_data=$((x_data + y_data))
    
    echo "${local_nx},${local_ny},${total_data}"
}

# Function to run a single benchmark
run_benchmark() {
    local grid_size=$1
    local topology=$2
    local run_num=$3
    
    # Run the solver with verbose timing
    local output
    output=$(mpirun --oversubscribe -np ${NUM_PROCS} ${EXECUTABLE} \
        --solver sor \
        --omega ${OMEGA} \
        --grid-size ${grid_size} \
        --max-iter ${MAX_ITER} \
        --topology ${topology} \
        --verbose-timing \
        ${INPUT_FILE} 2>&1)
    
    # Extract timing values
    local total_time step_time exchange_time reduction_time iterations
    
    total_time=$(echo "$output" | grep "Total time (t):" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    step_time=$(echo "$output" | grep "Compute time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    exchange_time=$(echo "$output" | grep "Exchange time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    reduction_time=$(echo "$output" | grep "Reduction time:" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    iterations=$(echo "$output" | grep "Iterations (n):" | head -1 | sed 's/.*: *//' | awk '{print $1}')
    
    # Handle missing values
    [ -z "$total_time" ] && total_time="0"
    [ -z "$step_time" ] && step_time="0"
    [ -z "$exchange_time" ] && exchange_time="0"
    [ -z "$reduction_time" ] && reduction_time="0"
    [ -z "$iterations" ] && iterations="${MAX_ITER}"
    
    # Calculate data transferred
    local data_info
    data_info=$(calculate_data $grid_size $topology)
    local local_nx=$(echo $data_info | cut -d',' -f1)
    local local_ny=$(echo $data_info | cut -d',' -f2)
    local data_per_exchange=$(echo $data_info | cut -d',' -f3)
    
    # Each iteration has one exchange
    local exchange_count=$iterations
    
    echo "${topology},${grid_size},${run_num},${total_time},${step_time},${exchange_time},${reduction_time},${iterations},${local_nx},${local_ny},${data_per_exchange},${exchange_count}" >> "${RESULTS_FILE}"
    
    echo "$total_time $exchange_time"
}

# Run benchmarks
for topology in "${TOPOLOGIES[@]}"; do
    echo -e "${CYAN}Testing topology: ${topology}${NC}"
    
    for grid_size in "${GRID_SIZES[@]}"; do
        echo -n "  Grid ${grid_size}x${grid_size}: "
        
        for run in $(seq 1 ${RUNS_PER_CONFIG}); do
            result=$(run_benchmark $grid_size $topology $run)
            total=$(echo $result | awk '{print $1}')
            exch=$(echo $result | awk '{print $2}')
            echo -n "run$run(${total}s,exch=${exch}s) "
        done
        echo ""
    done
    echo ""
done

echo -e "${GREEN}Benchmark complete!${NC}"
echo "Results saved to: ${RESULTS_FILE}"

# Generate summary
echo "" >> "${SUMMARY_FILE}"
echo "Exchange Analysis Summary - $(date)" >> "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "This benchmark measures Exchange_Borders behavior to determine:" >> "${SUMMARY_FILE}"
echo "  - Latency (α): Fixed overhead per communication" >> "${SUMMARY_FILE}"
echo "  - Bandwidth (β): Data transfer rate" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Communication model: t_exchange = α + data_size/β" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Run plot_exchange_analysis.py to analyze results." >> "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"
