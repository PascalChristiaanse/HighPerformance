#!/usr/bin/env python3
"""
Run experiments to find crossover points where communication time equals computation time.

Three experiments:
1. Fixed P=4: Vary problem size to find where T_comm ≈ T_compute
2. Fixed 1000x1000: Vary P to find where T_comm ≈ T_compute  
3. Surface: Vary both problem size and P to create crossover surface

Note: Total processor time should not exceed 2 minutes!
"""

import subprocess
import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import math

# SLURM default settings
SLURM_DEFAULTS = {
    'partition': 'compute',
    'time': '00:10:00',
    'nodes': 1,
    'account': 'Education-EEMCS-Courses-WI4049TU',
    'mem_per_cpu': '1G',
}

# Experiment 1: Fixed P=4, vary problem size
# We expect comm ~ O(n) and compute ~ O(n^2/P), so crossover at n ~ P
FIXED_P_SIZES = [50, 75, 100, 150, 200, 300, 400]
FIXED_P_CONFIG = {"name": "2x2", "px": 2, "py": 2, "np": 4}

# Experiment 2: Fixed size 1000x1000, vary P
# Need process counts that form valid grids
FIXED_SIZE = {"name": "1000x1000", "dim_x": 1000, "dim_y": 1000}

# All process configurations (up to 48)
# Note: 8x4 causes MPI_ERR_TRUNCATE in MPI_Fempois, using 4x8 instead
ALL_P_CONFIGS = [
    {"name": "1x1", "px": 1, "py": 1, "np": 1},
    {"name": "2x1", "px": 2, "py": 1, "np": 2},
    {"name": "2x2", "px": 2, "py": 2, "np": 4},
    {"name": "4x2", "px": 4, "py": 2, "np": 8},
    {"name": "4x4", "px": 4, "py": 4, "np": 16},
    {"name": "4x8", "px": 4, "py": 8, "np": 32},  # was 8x4 but that causes truncation errors
    {"name": "6x8", "px": 6, "py": 8, "np": 48},  # was 8x6
    # For extrapolation predictions (larger configs)
    {"name": "8x8", "px": 8, "py": 8, "np": 64},
    {"name": "8x16", "px": 8, "py": 16, "np": 128},  # was 16x8
    {"name": "16x16", "px": 16, "py": 16, "np": 256},
]

# Experiment 3 (Surface): Vary both problem size and P
SURFACE_SIZES = [100, 150, 200, 250, 300, 350, 400, 800]
SURFACE_MAX_P = 48


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if check and result.returncode != 0:
        print(f"Error running: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def generate_grid(work_dir, px, py, dim_x, dim_y):
    """Generate grid files using GridDist."""
    cmd = f"./GridDist {px} {py} {dim_x} {dim_y}"
    run_command(cmd, cwd=work_dir)


def run_solver(work_dir, np, telemetry_file, precision=None, max_iter=None,
               use_slurm=False, slurm_opts=None, oversubscribe=False):
    """Run MPI_Fempois with given parameters."""
    solver_args = f"-t {telemetry_file}"
    if precision is not None:
        solver_args += f" -p {precision}"
    if max_iter is not None:
        solver_args += f" -m {max_iter}"
    
    if use_slurm:
        result = run_solver_slurm(work_dir, np, solver_args, slurm_opts)
    else:
        oversub = " --oversubscribe" if oversubscribe else ""
        cmd = f"mpirun -np {np}{oversub} ./MPI_Fempois {solver_args}"
        result = run_command(cmd, cwd=work_dir)
    
    return result


def run_solver_slurm(work_dir, np, solver_args, slurm_opts):
    """Run MPI_Fempois using SLURM's srun."""
    opts = slurm_opts or {}
    
    srun_cmd = "srun"
    
    if opts.get('partition'):
        srun_cmd += f" --partition={opts['partition']}"
    if opts.get('time'):
        srun_cmd += f" --time={opts['time']}"
    if opts.get('account'):
        srun_cmd += f" --account={opts['account']}"
    if opts.get('nodes'):
        srun_cmd += f" --nodes={opts['nodes']}"
    
    srun_cmd += f" --ntasks={np}"
    
    if opts.get('ntasks_per_node'):
        srun_cmd += f" --ntasks-per-node={min(np, opts['ntasks_per_node'])}"
    if opts.get('mem_per_cpu'):
        srun_cmd += f" --mem-per-cpu={opts['mem_per_cpu']}"
    
    if opts.get('extra_args'):
        srun_cmd += f" {opts['extra_args']}"
    
    srun_cmd += f" ./MPI_Fempois {solver_args}"
    
    return run_command(srun_cmd, cwd=work_dir)


def parse_telemetry(telemetry_path):
    """Parse the telemetry CSV file and return a dictionary."""
    data = {}
    with open(telemetry_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2 and not row[0].startswith('#'):
                key, value = row
                try:
                    if '.' in value or 'e' in value.lower():
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
    return data


def run_single_experiment(work_dir, config, dim_x, dim_y, output_dir, exp_name,
                          use_slurm=False, slurm_opts=None, max_iter=None, oversubscribe=False):
    """Run a single experiment and return results."""
    # Generate grid
    generate_grid(work_dir, config['px'], config['py'], dim_x, dim_y)
    
    # Create telemetry filename
    telemetry_file = output_dir / f"telemetry_{exp_name}.csv"
    
    # Run solver (limit iterations for large problems to save time)
    run_solver(work_dir, config['np'], telemetry_file,
               max_iter=max_iter,
               use_slurm=use_slurm, slurm_opts=slurm_opts, oversubscribe=oversubscribe)
    
    # Parse results
    results = parse_telemetry(telemetry_file)
    results['config_name'] = config['name']
    results['dim_x'] = dim_x
    results['dim_y'] = dim_y
    results['np'] = config['np']
    
    return results


def run_experiment_fixed_p(work_dir, output_dir, use_slurm, slurm_opts, sizes=None, oversubscribe=False):
    """Experiment 1: Fixed P=4, varying problem size."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Fixed P=4, varying problem size")
    print("=" * 60)
    
    sizes = sizes or FIXED_P_SIZES
    config = FIXED_P_CONFIG
    results = []
    
    for size in sizes:
        dim_x = dim_y = size
        exp_name = f"fixedP_{config['name']}_{size}x{size}"
        print(f"  Running {size}x{size} with {config['name']}...", end=' ', flush=True)
        
        try:
            # Limit iterations for larger problems to stay within time budget
            max_iter = min(500, 5000)  # Cap iterations
            
            result = run_single_experiment(
                work_dir, config, dim_x, dim_y, output_dir, exp_name,
                use_slurm=use_slurm, slurm_opts=slurm_opts, max_iter=max_iter,
                oversubscribe=oversubscribe
            )
            results.append(result)
            
            t_compute = result.get('time_compute_avg', 0)
            t_comm = result.get('time_comm_neighbors_avg', 0) + result.get('time_comm_global_avg', 0)
            ratio = t_comm / t_compute if t_compute > 0 else float('inf')
            
            print(f"compute={t_compute:.4f}s, comm={t_comm:.4f}s, ratio={ratio:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
    
    return results


def run_experiment_fixed_size(work_dir, output_dir, use_slurm, slurm_opts, configs=None, oversubscribe=False):
    """Experiment 2: Fixed size 1000x1000, varying P."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Fixed size 1000x1000, varying P")
    print("=" * 60)
    
    configs = configs or [c for c in ALL_P_CONFIGS if c['np'] <= 48]
    dim_x, dim_y = FIXED_SIZE['dim_x'], FIXED_SIZE['dim_y']
    results = []
    
    for config in configs:
        exp_name = f"fixedSize_{config['name']}_{dim_x}x{dim_y}"
        print(f"  Running {dim_x}x{dim_y} with P={config['np']} ({config['name']})...", end=' ', flush=True)
        
        try:
            # Limit iterations to stay within time budget
            # Larger P means more communication overhead per iteration
            max_iter = min(200, 5000)
            
            result = run_single_experiment(
                work_dir, config, dim_x, dim_y, output_dir, exp_name,
                use_slurm=use_slurm, slurm_opts=slurm_opts, max_iter=max_iter,
                oversubscribe=oversubscribe
            )
            results.append(result)
            
            t_compute = result.get('time_compute_avg', 0)
            t_comm = result.get('time_comm_neighbors_avg', 0) + result.get('time_comm_global_avg', 0)
            ratio = t_comm / t_compute if t_compute > 0 else float('inf')
            
            print(f"compute={t_compute:.4f}s, comm={t_comm:.4f}s, ratio={ratio:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
    
    return results


def run_experiment_surface(work_dir, output_dir, use_slurm, slurm_opts, sizes=None, max_p=48, oversubscribe=False):
    """Experiment 3: Vary both problem size and P to create surface data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Surface - varying both problem size and P")
    print("=" * 60)
    
    sizes = sizes or SURFACE_SIZES
    configs = [c for c in ALL_P_CONFIGS if c['np'] <= max_p]
    results = []
    
    for size in sizes:
        dim_x = dim_y = size
        print(f"\n  Problem size: {size}x{size}")
        print("-" * 50)
        
        for config in configs:
            exp_name = f"surface_{config['name']}_{size}x{size}"
            print(f"    P={config['np']:3d} ({config['name']:5s})...", end=' ', flush=True)
            
            try:
                # Adjust iterations based on problem size and P
                # Larger problems need fewer iterations to get meaningful timing
                if size >= 1000:
                    max_iter = 100
                elif size >= 500:
                    max_iter = 150
                else:
                    max_iter = 200
                
                result = run_single_experiment(
                    work_dir, config, dim_x, dim_y, output_dir, exp_name,
                    use_slurm=use_slurm, slurm_opts=slurm_opts, max_iter=max_iter,
                    oversubscribe=oversubscribe
                )
                results.append(result)
                
                t_compute = result.get('time_compute_avg', 0)
                t_comm = result.get('time_comm_neighbors_avg', 0) + result.get('time_comm_global_avg', 0)
                ratio = t_comm / t_compute if t_compute > 0 else float('inf')
                
                print(f"compute={t_compute:.4f}s, comm={t_comm:.4f}s, ratio={ratio:.3f}")
            except Exception as e:
                print(f"FAILED: {e}")
    
    return results


def save_results(exp1_results, exp2_results, exp3_results, output_dir):
    """Save all results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to JSON
    json_file = output_dir / f"crossover_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'experiment1_fixed_p': exp1_results,
            'experiment2_fixed_size': exp2_results,
            'experiment3_surface': exp3_results,
            'timestamp': timestamp,
        }, f, indent=2)
    print(f"\nResults saved to: {json_file}")
    
    # Save CSV for experiment 1
    if exp1_results:
        csv_file = output_dir / f"crossover_fixedP_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=exp1_results[0].keys())
            writer.writeheader()
            writer.writerows(exp1_results)
    
    # Save CSV for experiment 2
    if exp2_results:
        csv_file = output_dir / f"crossover_fixedSize_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=exp2_results[0].keys())
            writer.writeheader()
            writer.writerows(exp2_results)
    
    # Save CSV for experiment 3 (surface)
    if exp3_results:
        csv_file = output_dir / f"crossover_surface_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=exp3_results[0].keys())
            writer.writeheader()
            writer.writerows(exp3_results)
    
    return json_file


def print_summary(exp1_results, exp2_results, exp3_results):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: Communication vs Computation Crossover Analysis")
    print("=" * 70)
    
    if exp1_results:
        print("\nExperiment 1: Fixed P=4, varying problem size")
        print("-" * 60)
        print(f"{'Size':<12} {'Compute(s)':<12} {'Comm(s)':<12} {'Ratio':<10} {'Crossover?':<10}")
        print("-" * 60)
        
        for r in sorted(exp1_results, key=lambda x: x['dim_x']):
            size = f"{r['dim_x']}x{r['dim_y']}"
            t_compute = r.get('time_compute_avg', 0)
            t_comm = r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)
            ratio = t_comm / t_compute if t_compute > 0 else float('inf')
            crossover = "YES" if 0.8 < ratio < 1.2 else ("comm>" if ratio > 1 else "comp>")
            print(f"{size:<12} {t_compute:<12.4f} {t_comm:<12.4f} {ratio:<10.3f} {crossover:<10}")
    
    if exp2_results:
        print("\nExperiment 2: Fixed size 1000x1000, varying P")
        print("-" * 60)
        print(f"{'P':<8} {'Config':<10} {'Compute(s)':<12} {'Comm(s)':<12} {'Ratio':<10} {'Crossover?':<10}")
        print("-" * 60)
        
        for r in sorted(exp2_results, key=lambda x: x['np']):
            t_compute = r.get('time_compute_avg', 0)
            t_comm = r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)
            ratio = t_comm / t_compute if t_compute > 0 else float('inf')
            crossover = "YES" if 0.8 < ratio < 1.2 else ("comm>" if ratio > 1 else "comp>")
            print(f"{r['np']:<8} {r['config_name']:<10} {t_compute:<12.4f} {t_comm:<12.4f} {ratio:<10.3f} {crossover:<10}")
    
    if exp3_results:
        print("\nExperiment 3: Surface (varying both n and P)")
        print("-" * 70)
        
        # Group by size
        sizes = sorted(set(r['dim_x'] for r in exp3_results))
        for size in sizes:
            size_results = [r for r in exp3_results if r['dim_x'] == size]
            crossover_p = None
            prev_ratio = None
            for r in sorted(size_results, key=lambda x: x['np']):
                t_compute = r.get('time_compute_avg', 0)
                t_comm = r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)
                ratio = t_comm / t_compute if t_compute > 0 else float('inf')
                if prev_ratio is not None and prev_ratio < 1 and ratio >= 1:
                    crossover_p = r['np']
                prev_ratio = ratio
            
            status = f"crossover at P≈{crossover_p}" if crossover_p else "no crossover in range"
            print(f"  {size}x{size}: {status}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run crossover experiments')
    parser.add_argument('--work-dir', type=str, default='.',
                        help='Working directory containing MPI_Fempois')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--exp1-only', action='store_true',
                        help='Run only experiment 1 (fixed P)')
    parser.add_argument('--exp2-only', action='store_true',
                        help='Run only experiment 2 (fixed size 1000x1000)')
    parser.add_argument('--exp3-only', action='store_true',
                        help='Run only experiment 3 (surface - varying both n and P)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Problem sizes for experiments (default varies by experiment)')
    parser.add_argument('--surface-sizes', type=int, nargs='+', default=None,
                        help='Problem sizes for surface experiment (default: 100 250 500 1000 1250 1500)')
    parser.add_argument('--max-p', type=int, default=48,
                        help='Maximum number of processes (default: 48)')
    parser.add_argument('--oversubscribe', action='store_true',
                        help='Allow mpirun to oversubscribe (run more processes than cores)')
    
    # SLURM options
    slurm_group = parser.add_argument_group('SLURM options')
    slurm_group.add_argument('--slurm', action='store_true',
                             help='Run experiments using SLURM')
    slurm_group.add_argument('--partition', '-p', type=str,
                             default=SLURM_DEFAULTS['partition'],
                             help=f"SLURM partition (default: {SLURM_DEFAULTS['partition']})")
    slurm_group.add_argument('--time', '-t', type=str,
                             default=SLURM_DEFAULTS['time'],
                             help=f"Time limit (default: {SLURM_DEFAULTS['time']})")
    slurm_group.add_argument('--account', '-A', type=str,
                             default=SLURM_DEFAULTS['account'],
                             help='SLURM account')
    slurm_group.add_argument('--nodes', '-N', type=int, default=1,
                             help='Number of nodes')
    slurm_group.add_argument('--ntasks-per-node', type=int, default=48,
                             help='Max tasks per node (default: 48)')
    slurm_group.add_argument('--mem-per-cpu', type=str,
                             default=SLURM_DEFAULTS['mem_per_cpu'],
                             help='Memory per CPU')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir).resolve()
    output_dir = work_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    slurm_opts = {
        'partition': args.partition,
        'time': args.time,
        'account': args.account,
        'nodes': args.nodes,
        'ntasks_per_node': args.ntasks_per_node,
        'mem_per_cpu': args.mem_per_cpu,
    }
    
    print(f"Working directory: {work_dir}")
    print(f"Output directory: {output_dir}")
    print(f"SLURM mode: {'enabled' if args.slurm else 'disabled (using mpirun)'}")
    if args.slurm:
        print(f"  Partition: {args.partition}")
        print(f"  Account: {args.account}")
    
    exp1_results = []
    exp2_results = []
    exp3_results = []
    
    # Determine which experiments to run
    run_exp1 = not args.exp2_only and not args.exp3_only
    run_exp2 = not args.exp1_only and not args.exp3_only
    run_exp3 = args.exp3_only or (not args.exp1_only and not args.exp2_only)
    
    # If a specific experiment is requested, only run that one
    if args.exp1_only:
        run_exp1, run_exp2, run_exp3 = True, False, False
    if args.exp2_only:
        run_exp1, run_exp2, run_exp3 = False, True, False
    if args.exp3_only:
        run_exp1, run_exp2, run_exp3 = False, False, True
    
    # Run experiments
    if run_exp1:
        sizes = args.sizes or FIXED_P_SIZES
        exp1_results = run_experiment_fixed_p(
            work_dir, output_dir, args.slurm, slurm_opts, sizes,
            oversubscribe=args.oversubscribe
        )
    
    if run_exp2:
        # Filter configs based on max_p
        configs = [c for c in ALL_P_CONFIGS if c['np'] <= args.max_p]
        exp2_results = run_experiment_fixed_size(
            work_dir, output_dir, args.slurm, slurm_opts, configs,
            oversubscribe=args.oversubscribe
        )
    
    if run_exp3:
        surface_sizes = args.surface_sizes or SURFACE_SIZES
        exp3_results = run_experiment_surface(
            work_dir, output_dir, args.slurm, slurm_opts, 
            sizes=surface_sizes, max_p=args.max_p,
            oversubscribe=args.oversubscribe
        )
    
    # Save and summarize
    save_results(exp1_results, exp2_results, exp3_results, output_dir)
    print_summary(exp1_results, exp2_results, exp3_results)
    
    print("\nRun 'python scripts/analyze_crossover.py' to analyze and plot results")


if __name__ == '__main__':
    main()
