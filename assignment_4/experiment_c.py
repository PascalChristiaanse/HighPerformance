#!/usr/bin/env python3
"""
Experiment C: Calculate speedups with and without memory copy

This script generates and submits SLURM jobs to measure:
- Speedup (i): excluding memory copy between CPU and GPU
- Speedup (ii): including time of memory copies

Uses the same parameter combinations as experiment_b for comprehensive speedup analysis.
Results are saved in the results/experiment_c/ directory.
"""

import os
import subprocess
import itertools
from pathlib import Path

# Configuration - same as experiment_b for fair comparison
MATRIX_SIZES = [50, 500, 2000, 4000]
BLOCK_SIZES = [32, 64, 100]
MAX_ITERATIONS = 100
RESULTS_DIR = Path("results/experiment_c")

# SLURM job template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=exp_c_n{size}_b{blocksize}
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output={results_dir}/run_n{size}_b{blocksize}_%j.out

# Load required modules
module load 2025 cuda/12 gcc

# Run with shared memory for speedup measurements
echo "=== Experiment C: Speedup Analysis ==="
echo "Matrix size: {size}, Block size: {blocksize}"
echo ""
echo "The output includes:"
echo "  - cpu_time: CPU execution time"
echo "  - gpu_time_with_memcpy: GPU time including memory transfers"
echo "  - gpu_time_no_memcpy: GPU compute-only time"
echo "  - memcpy_time: Time spent on memory copies"
echo "  - speedup_with_memcpy: CPU_time / GPU_time_with_memcpy"
echo "  - speedup_no_memcpy: CPU_time / GPU_compute_only_time"
echo ""

srun ./power_gpu --size {size} --blocksize {blocksize} --max_iteration {max_iter} --use_shared 1
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


def generate_job_script(size, blocksize):
    """Generate a SLURM job script for given parameters."""
    script_content = SLURM_TEMPLATE.format(
        size=size,
        blocksize=blocksize,
        max_iter=MAX_ITERATIONS,
        results_dir=RESULTS_DIR
    )
    
    script_path = RESULTS_DIR / f"job_n{size}_b{blocksize}.sh"
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
    print("=" * 60)
    print("Experiment C: Speedup Analysis (with/without memory copy)")
    print("=" * 60)
    
    ensure_directories()
    
    if not check_executable():
        print("\nGenerating job scripts anyway (you can submit after compiling)...")
    
    # Generate all combinations
    experiments = list(itertools.product(MATRIX_SIZES, BLOCK_SIZES))
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Matrix sizes: {MATRIX_SIZES}")
    print(f"Block sizes: {BLOCK_SIZES}")
    print()
    print("Speedup metrics measured:")
    print("  (i)  Speedup excluding memory copy = CPU_time / GPU_compute_only")
    print("  (ii) Speedup including memory copy = CPU_time / GPU_total_time")
    print()
    
    job_ids = []
    for size, blocksize in experiments:
        script_path = generate_job_script(size, blocksize)
        print(f"Generated: {script_path.name}", end=" ")
        
        job_id = submit_job(script_path)
        if job_id:
            job_ids.append(job_id)
            print(f"-> Job ID: {job_id}")
        else:
            print("-> Not submitted (run sbatch manually)")
    
    print()
    print("=" * 60)
    if job_ids:
        print(f"Submitted {len(job_ids)} jobs")
        print(f"Monitor with: squeue -u $USER")
        print(f"Results will be in: {RESULTS_DIR}/")
    else:
        print("Job scripts generated. Submit manually with:")
        print(f"  for f in {RESULTS_DIR}/job_*.sh; do sbatch $f; done")
    print("=" * 60)
    
    # Save experiment configuration
    config_path = RESULTS_DIR / "experiment_config.txt"
    with open(config_path, 'w') as f:
        f.write("Experiment C Configuration\n")
        f.write("=" * 40 + "\n")
        f.write(f"Matrix sizes: {MATRIX_SIZES}\n")
        f.write(f"Block sizes: {BLOCK_SIZES}\n")
        f.write(f"Max iterations: {MAX_ITERATIONS}\n")
        f.write(f"Memory mode: shared\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write("\nSpeedup metrics:\n")
        f.write("  (i)  speedup_no_memcpy = cpu_time / gpu_time_no_memcpy\n")
        f.write("  (ii) speedup_with_memcpy = cpu_time / gpu_time_with_memcpy\n")
        if job_ids:
            f.write(f"\nJob IDs: {', '.join(job_ids)}\n")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
