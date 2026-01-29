#!/usr/bin/env python3
"""
Experiment C: Compare all three memory types for GPU speedup

This script generates and submits SLURM jobs to measure speedup for:
1. Global GPU Memory: Direct VRAM access (Av_Product_Global kernel)
2. Shared GPU Memory: Fast on-chip cache (Av_Product_Shared kernel)
3. Unified Memory: CPU-managed memory with automatic migration (cudaMallocManaged)

All speedups are compared against the CPU implementation.
Results are saved in the results/experiment_c/ directory.
"""

import os
import subprocess
import itertools
from pathlib import Path

# Configuration
MATRIX_SIZES = [50, 500, 2000, 4000]
BLOCK_SIZES = [32, 64, 100]
MAX_ITERATIONS = 100
RESULTS_DIR = Path("results/experiment_c")

# SLURM job template for Global GPU Memory (global kernel, manual transfers)
SLURM_TEMPLATE_GLOBAL = """#!/bin/bash
#SBATCH --job-name=exp_c_global_n{size}_b{blocksize}
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output={results_dir}/run_n{size}_b{blocksize}_global_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with GLOBAL memory kernel (direct VRAM access)
echo "=== Memory Type: GLOBAL GPU Memory (VRAM) ==="
echo "Matrix size: {size}, Block size: {blocksize}"
echo "Kernel: Av_Product_Global (direct global memory access)"
echo ""

srun ./power_gpu --size {size} --blocksize {blocksize} --max_iteration {max_iter} --use_shared 0 --use_unified 0
"""

# SLURM job template for Shared GPU Memory (shared kernel, manual transfers)
SLURM_TEMPLATE_SHARED = """#!/bin/bash
#SBATCH --job-name=exp_c_shared_n{size}_b{blocksize}
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output={results_dir}/run_n{size}_b{blocksize}_shared_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with SHARED memory kernel (fast on-chip cache)
echo "=== Memory Type: SHARED GPU Memory (On-chip) ==="
echo "Matrix size: {size}, Block size: {blocksize}"
echo "Kernel: Av_Product_Shared (tiled with shared memory)"
echo ""

srun ./power_gpu --size {size} --blocksize {blocksize} --max_iteration {max_iter} --use_shared 1 --use_unified 0
"""

# SLURM job template for Unified Memory (shared kernel, unified allocation)
SLURM_TEMPLATE_UNIFIED = """#!/bin/bash
#SBATCH --job-name=exp_c_unified_n{size}_b{blocksize}
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output={results_dir}/run_n{size}_b{blocksize}_unified_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with UNIFIED Memory (cudaMallocManaged - automatic migration)
echo "=== Memory Type: UNIFIED Memory (cudaMallocManaged) ==="
echo "Matrix size: {size}, Block size: {blocksize}"
echo "Kernel: Av_Product_Shared with unified memory allocation"
echo ""

srun ./power_gpu --size {size} --blocksize {blocksize} --max_iteration {max_iter} --use_shared 1 --use_unified 1
"""


def ensure_directories():
    """Create necessary directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR.absolute()}")


def check_executable():
    """Check if power_gpu executable exists."""
    if not Path("power_gpu").exists():
        print("ERROR: power_gpu executable not found!")
        print("Please compile first using:")
        print("  sbatch compile.sh")
        print("  or on a GPU node: bash compile.sh")
        return False
    return True


def generate_job_script(size, blocksize, memory_type='shared'):
    """Generate a SLURM job script for given parameters.
    
    memory_type: 'global', 'shared', or 'unified'
    """
    if memory_type == 'global':
        script_content = SLURM_TEMPLATE_GLOBAL.format(
            size=size,
            blocksize=blocksize,
            max_iter=MAX_ITERATIONS,
            results_dir=RESULTS_DIR
        )
        script_path = RESULTS_DIR / f"job_n{size}_b{blocksize}_global.sh"
    elif memory_type == 'unified':
        script_content = SLURM_TEMPLATE_UNIFIED.format(
            size=size,
            blocksize=blocksize,
            max_iter=MAX_ITERATIONS,
            results_dir=RESULTS_DIR
        )
        script_path = RESULTS_DIR / f"job_n{size}_b{blocksize}_unified.sh"
    else:  # shared (default)
        script_content = SLURM_TEMPLATE_SHARED.format(
            size=size,
            blocksize=blocksize,
            max_iter=MAX_ITERATIONS,
            results_dir=RESULTS_DIR
        )
        script_path = RESULTS_DIR / f"job_n{size}_b{blocksize}_shared.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def submit_job(script_path):
    """Submit a SLURM job and return the job ID."""
    try:
        result = subprocess.run(
            ['sbatch', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            return job_id
        else:
            print(f"Error submitting job: {result.stderr}")
            return None
    except FileNotFoundError:
        print("sbatch command not found. Are you on the cluster?")
        return None


def main():
    """Main function to generate and submit all experiment C jobs."""
    print("=" * 65)
    print("Experiment C: Compare All Three Memory Types (vs CPU)")
    print("=" * 65)
    
    ensure_directories()
    
    if not check_executable():
        print("\nGenerating job scripts anyway (you can submit after compiling)...")
    
    # Generate all combinations
    experiments = list(itertools.product(MATRIX_SIZES, BLOCK_SIZES))
    print(f"\nTotal parameter combinations: {len(experiments)}")
    print(f"Matrix sizes: {MATRIX_SIZES}")
    print(f"Block sizes: {BLOCK_SIZES}")
    print()
    print("Memory types being compared (all vs CPU):")
    print("  1. GLOBAL Memory  - Direct GPU VRAM access (Av_Product_Global kernel)")
    print("  2. SHARED Memory  - Fast on-chip GPU cache (Av_Product_Shared kernel)")
    print("  3. UNIFIED Memory - cudaMallocManaged (automatic CPU-GPU migration)")
    print()
    
    job_ids = []
    
    # Part 1: Global memory jobs
    print("--- [1/3] Generating GLOBAL Memory Jobs ---")
    for size, blocksize in experiments:
        script_path = generate_job_script(size, blocksize, memory_type='global')
        print(f"Generated: {script_path.name}", end=" ")
        
        job_id = submit_job(script_path)
        if job_id:
            job_ids.append(job_id)
            print(f"-> Job ID: {job_id}")
        else:
            print("-> Not submitted (run sbatch manually)")
    
    # Part 2: Shared memory jobs
    print("\n--- [2/3] Generating SHARED Memory Jobs ---")
    for size, blocksize in experiments:
        script_path = generate_job_script(size, blocksize, memory_type='shared')
        print(f"Generated: {script_path.name}", end=" ")
        
        job_id = submit_job(script_path)
        if job_id:
            job_ids.append(job_id)
            print(f"-> Job ID: {job_id}")
        else:
            print("-> Not submitted (run sbatch manually)")
    
    # Part 3: Unified memory jobs
    print("\n--- [3/3] Generating UNIFIED Memory Jobs ---")
    for size, blocksize in experiments:
        script_path = generate_job_script(size, blocksize, memory_type='unified')
        print(f"Generated: {script_path.name}", end=" ")
        
        job_id = submit_job(script_path)
        if job_id:
            job_ids.append(job_id)
            print(f"-> Job ID: {job_id}")
        else:
            print("-> Not submitted (run sbatch manually)")
    
    print()
    print("=" * 65)
    total_jobs = len(experiments) * 3  # global + shared + unified
    if job_ids:
        print(f"Submitted {len(job_ids)} jobs (out of {total_jobs} total)")
        print(f"Monitor with: squeue -u $USER")
        print(f"Results will be in: {RESULTS_DIR}/")
    else:
        print(f"Job scripts generated ({total_jobs} total). Submit manually with:")
        print(f"  for f in {RESULTS_DIR}/job_*.sh; do sbatch $f; done")
    print("=" * 65)
    
    # Save experiment configuration
    config_path = RESULTS_DIR / "experiment_config.txt"
    with open(config_path, 'w') as f:
        f.write("Experiment C Configuration: Three Memory Types Comparison\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Matrix sizes: {MATRIX_SIZES}\n")
        f.write(f"Block sizes: {BLOCK_SIZES}\n")
        f.write(f"Max iterations: {MAX_ITERATIONS}\n")
        f.write(f"Total experiments: {total_jobs}\n\n")
        f.write("Memory Types:\n")
        f.write("  1. GLOBAL ({} jobs) - use_shared=0, use_unified=0\n".format(len(experiments)))
        f.write("     Kernel: Av_Product_Global (direct VRAM access)\n")
        f.write("  2. SHARED ({} jobs) - use_shared=1, use_unified=0\n".format(len(experiments)))
        f.write("     Kernel: Av_Product_Shared (tiled with on-chip shared memory)\n")
        f.write("  3. UNIFIED ({} jobs) - use_shared=1, use_unified=1\n".format(len(experiments)))
        f.write("     Allocation: cudaMallocManaged (automatic migration)\n\n")
        f.write("Speedup = CPU_time / GPU_time_with_memcpy\n")
        if job_ids:
            f.write(f"\nJob IDs: {', '.join(job_ids)}\n")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
