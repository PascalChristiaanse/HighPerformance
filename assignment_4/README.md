# GPU Power Method - Lab Exercise 3

This project implements the Power Method for computing the dominant eigenvalue of a matrix using CUDA GPU acceleration.

## Project Structure

```
assignment_4/
├── power_gpu.cu          # Main CUDA program
├── compile.sh            # Compilation script
├── run_power.sh          # Generic SLURM job script
├── experiment_b.py       # Experiment: vary matrix size & block size
├── experiment_c.py       # Experiment: speedup analysis
├── experiment_d.py       # Experiment: shared vs global memory
├── analysis_b.py         # Analysis script for experiment B
├── analysis_c.py         # Analysis script for experiment C
├── analysis_d.py         # Analysis script for experiment D
├── results/              # Directory for experiment output files
└── analysis_output/      # Directory for plots and tables
```

## Quick Start

### 1. Compile the CUDA Program

On the cluster, first compile the program:

```bash
# Option 1: Request an interactive GPU node
srun --partition=gpu-a100-small --gres=gpu --time=00:30:00 --pty /bin/bash
module load 2024r1 nvhpc
nvcc -O3 -o power_gpu power_gpu.cu

# Option 2: Use the compile script
sbatch compile.sh
```

### 2. Run a Single Test

```bash
# Basic run with defaults
srun --partition=gpu-a100-small --gres=gpu ./power_gpu

# With custom parameters
srun --partition=gpu-a100-small --gres=gpu ./power_gpu --size 2000 --blocksize 64

# Show help
./power_gpu --help
```

### 3. Run Experiments

```bash
# Run all experiments for question B (matrix size & block size)
python3 experiment_b.py

# Run all experiments for question C (speedup analysis)
python3 experiment_c.py

# Run all experiments for question D (shared vs global memory)
python3 experiment_d.py
```

### 4. Analyze Results

After experiments complete, run the analysis scripts:

```bash
# Analyze experiment B results
python3 analysis_b.py

# Analyze experiment C results
python3 analysis_c.py

# Analyze experiment D results
python3 analysis_d.py
```

## Command Line Options

The `power_gpu` program accepts the following arguments:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--size` | `-n` | Matrix size (N×N) | 5000 |
| `--blocksize` | `-b` | Threads per block | 32 |
| `--max_iteration` | `-m` | Maximum iterations | 100 |
| `--use_shared` | `-s` | Use shared memory (1) or global (0) | 1 |
| `--use_unified` | `-u` | Use unified memory (1) or manual (0) | 0 |
| `--help` | `-h` | Show help message | - |

### Memory Management Modes

- **Manual (default)**: Uses `cudaMalloc` and `cudaMemcpy` for explicit memory transfers
- **Unified Memory**: Uses `cudaMallocManaged` for automatic memory management

```bash
# Manual memory management (default)
./power_gpu --size 2000 --use_unified 0

# Unified memory (cudaMallocManaged)
./power_gpu --size 2000 --use_unified 1
```

## Output Format

The program outputs both human-readable results and a CSV line for automated parsing:

```
=== CSV_OUTPUT ===
matrix_size,block_size,memory_mode,cpu_time,gpu_time_with_memcpy,gpu_time_no_memcpy,...
2000,32,shared,1.234567,0.012345,0.010234,...
```

## Lab Exercise Questions Mapping

| Question | Description | Experiment Script | Analysis Script |
|----------|-------------|-------------------|-----------------|
| Step 1 (a) | Shared vs Global Memory | `experiment_d.py` | `analysis_d.py` |
| Step 2 (b) | Matrix size & threads | `experiment_b.py` | `analysis_b.py` |
| Step 3 (c) | Speedup with/without memcpy | `experiment_c.py` | `analysis_c.py` |
| Step 4 (d) | Explain performance | - | `analysis_d.py` |

## SLURM Configuration

All experiments use:
- **Partition**: `gpu-a100-small`
- **Time limit**: 10 minutes per job
- **Account**: `Education-EEMCS-Courses-WI4049TU`

## Dependencies

- CUDA Toolkit (loaded via `module load 2024r1 nvhpc`)
- Python 3.x
- matplotlib (optional, for plots)
- pandas (optional, for data handling)

## Algorithm

The Power Method iteratively computes:

1. **W = A × V** - Matrix-vector multiplication
2. **||W||** - Compute norm of W
3. **V = W / ||W||** - Normalize
4. **λ = V · W** - Compute eigenvalue estimate

Repeat until |λ_new - λ_old| < ε or max iterations reached.

## GPU Implementation

The GPU implementation uses:
- **Av_Product**: Tiled matrix-vector multiplication with shared memory
- **FindNormW**: Parallel reduction for computing squared norm
- **NormalizeW**: Parallel element-wise normalization
- **ComputeLamda**: Parallel reduction for dot product

## Troubleshooting

### Compilation fails on login node
Use an interactive GPU node or submit compilation as a batch job.

### "sbatch: command not found"
You're not on the cluster. The experiment scripts will still generate job files you can submit later.

### No results in analysis
Make sure all SLURM jobs have completed before running analysis scripts.
Check `squeue -u $USER` for pending jobs.
