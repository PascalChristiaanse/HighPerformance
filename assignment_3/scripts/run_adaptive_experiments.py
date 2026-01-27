#!/usr/bin/env python3
"""
Run experiments comparing adaptive vs uniform grids.

Investigates whether adaptive grids (with more gridpoints near source points)
lead to faster convergence and/or affect computing time.

Tests sizes: 100×100, 200×200, 400×400, etc.
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

# Grid sizes to test
GRID_SIZES = [100, 200, 400, 800]

# Process configurations to test (use reasonable defaults)
P_CONFIGS = [
    {"name": "1x1", "px": 1, "py": 1, "np": 1},
    {"name": "2x2", "px": 2, "py": 2, "np": 4},
    {"name": "4x4", "px": 4, "py": 4, "np": 16},
]


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
        print("Error running: {}".format(cmd))
        print("stdout: {}".format(result.stdout))
        print("stderr: {}".format(result.stderr))
        raise RuntimeError("Command failed: {}".format(cmd))
    return result


def generate_grid(work_dir, px, py, dim_x, dim_y, adaptive=False):
    """Generate grid files using GridDist.
    
    Args:
        adaptive: If True, use the 'adapt' option to create adaptive grid
    """
    if adaptive:
        cmd = "./GridDist {} {} {} {} adapt".format(px, py, dim_x, dim_y)
    else:
        cmd = "./GridDist {} {} {} {}".format(px, py, dim_x, dim_y)
    run_command(cmd, cwd=work_dir)


def run_solver(work_dir, np, telemetry_file, precision=None, max_iter=None,
               use_slurm=False, slurm_opts=None, oversubscribe=False):
    """Run MPI_Fempois with given parameters."""
    solver_args = "-t {}".format(telemetry_file)
    if precision is not None:
        solver_args += " -p {}".format(precision)
    if max_iter is not None:
        solver_args += " -m {}".format(max_iter)
    
    if use_slurm:
        result = run_solver_slurm(work_dir, np, solver_args, slurm_opts)
    else:
        oversub = " --oversubscribe" if oversubscribe else ""
        cmd = "mpirun -np {}{} ./MPI_Fempois {}".format(np, oversub, solver_args)
        result = run_command(cmd, cwd=work_dir)
    
    return result


def run_solver_slurm(work_dir, np, solver_args, slurm_opts):
    """Run MPI_Fempois using SLURM's srun."""
    opts = slurm_opts or {}
    
    srun_cmd = "srun"
    
    if opts.get('partition'):
        srun_cmd += " --partition={}".format(opts['partition'])
    if opts.get('time'):
        srun_cmd += " --time={}".format(opts['time'])
    if opts.get('account'):
        srun_cmd += " --account={}".format(opts['account'])
    if opts.get('nodes'):
        srun_cmd += " --nodes={}".format(opts['nodes'])
    
    srun_cmd += " --ntasks={}".format(np)
    
    if opts.get('ntasks_per_node'):
        srun_cmd += " --ntasks-per-node={}".format(min(np, opts['ntasks_per_node']))
    if opts.get('mem_per_cpu'):
        srun_cmd += " --mem-per-cpu={}".format(opts['mem_per_cpu'])
    
    if opts.get('extra_args'):
        srun_cmd += " {}".format(opts['extra_args'])
    
    srun_cmd += " ./MPI_Fempois {}".format(solver_args)
    
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
                          adaptive=False, use_slurm=False, slurm_opts=None, 
                          max_iter=None, precision=None, oversubscribe=False):
    """Run a single experiment and return results."""
    # Generate grid (adaptive or uniform)
    generate_grid(work_dir, config['px'], config['py'], dim_x, dim_y, adaptive=adaptive)
    
    # Create telemetry filename
    telemetry_file = output_dir / "telemetry_{}.csv".format(exp_name)
    
    # Run solver
    run_solver(work_dir, config['np'], telemetry_file,
               precision=precision, max_iter=max_iter,
               use_slurm=use_slurm, slurm_opts=slurm_opts, oversubscribe=oversubscribe)
    
    # Parse results
    results = parse_telemetry(telemetry_file)
    results['config_name'] = config['name']
    results['dim_x'] = dim_x
    results['dim_y'] = dim_y
    results['np'] = config['np']
    results['adaptive'] = adaptive
    
    return results


def run_adaptive_comparison(work_dir, output_dir, use_slurm, slurm_opts, 
                             sizes=None, configs=None, precision=None,
                             max_iter=None, oversubscribe=False):
    """Run comparison experiments for uniform vs adaptive grids."""
    print("\n" + "=" * 70)
    print("ADAPTIVE GRID EXPERIMENTS")
    print("Comparing uniform grids vs adaptive grids (more points near sources)")
    print("=" * 70)
    
    sizes = sizes or GRID_SIZES
    configs = configs or P_CONFIGS
    
    all_results = []
    
    for size in sizes:
        dim_x = dim_y = size
        print("\n" + "-" * 60)
        print("Grid size: {}x{}".format(size, size))
        print("-" * 60)
        
        for config in configs:
            print("\n  P={} ({})".format(config['np'], config['name']))
            
            # Run uniform grid
            exp_name = "uniform_{}_{}_{}x{}".format(config['name'], size, size, size)
            print("    Uniform grid...", end=' ', flush=True)
            
            try:
                result_uniform = run_single_experiment(
                    work_dir, config, dim_x, dim_y, output_dir, exp_name,
                    adaptive=False, use_slurm=use_slurm, slurm_opts=slurm_opts,
                    max_iter=max_iter, precision=precision, oversubscribe=oversubscribe
                )
                all_results.append(result_uniform)
                
                iters_u = result_uniform.get('iterations', 0)
                time_u = result_uniform.get('time_total', 0)
                print("iters={}, time={:.4f}s".format(iters_u, time_u))
            except Exception as e:
                print("FAILED: {}".format(e))
                continue
            
            # Run adaptive grid
            exp_name = "adaptive_{}_{}_{}x{}".format(config['name'], size, size, size)
            print("    Adaptive grid...", end=' ', flush=True)
            
            try:
                result_adaptive = run_single_experiment(
                    work_dir, config, dim_x, dim_y, output_dir, exp_name,
                    adaptive=True, use_slurm=use_slurm, slurm_opts=slurm_opts,
                    max_iter=max_iter, precision=precision, oversubscribe=oversubscribe
                )
                all_results.append(result_adaptive)
                
                iters_a = result_adaptive.get('iterations', 0)
                time_a = result_adaptive.get('time_total', 0)
                print("iters={}, time={:.4f}s".format(iters_a, time_a))
                
                # Print comparison
                if iters_u > 0 and time_u > 0:
                    iter_ratio = iters_a / iters_u
                    time_ratio = time_a / time_u
                    print("    -> Adaptive vs Uniform: iters {:.2f}x, time {:.2f}x".format(
                        iter_ratio, time_ratio))
                    
            except Exception as e:
                print("FAILED: {}".format(e))
    
    return all_results


def save_results(results, output_dir):
    """Save experiment results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / "adaptive_results_{}.json".format(timestamp)
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': timestamp,
            'description': 'Adaptive vs uniform grid comparison'
        }, f, indent=2)
    
    print("\nResults saved to: {}".format(output_file))
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Run adaptive grid experiments')
    parser.add_argument('--work-dir', type=str, default='.',
                        help='Working directory containing executables')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Grid sizes to test (default: 100 200 400 800)')
    parser.add_argument('--precision', type=float, default=1e-6,
                        help='Convergence precision (default: 1e-6)')
    parser.add_argument('--max-iter', type=int, default=None,
                        help='Maximum iterations (default: no limit)')
    
    # SLURM options
    parser.add_argument('--slurm', action='store_true',
                        help='Use SLURM for job submission')
    parser.add_argument('--partition', type=str, default=SLURM_DEFAULTS['partition'],
                        help='SLURM partition')
    parser.add_argument('--time', type=str, default=SLURM_DEFAULTS['time'],
                        help='SLURM time limit')
    parser.add_argument('--account', type=str, default=SLURM_DEFAULTS['account'],
                        help='SLURM account')
    parser.add_argument('--nodes', type=int, default=SLURM_DEFAULTS['nodes'],
                        help='Number of nodes')
    parser.add_argument('--ntasks-per-node', type=int, default=None,
                        help='Tasks per node')
    parser.add_argument('--mem-per-cpu', type=str, default=SLURM_DEFAULTS['mem_per_cpu'],
                        help='Memory per CPU')
    
    # Local MPI options
    parser.add_argument('--oversubscribe', action='store_true',
                        help='Allow oversubscription when running locally with mpirun')
    
    args = parser.parse_args()
    
    # Setup paths
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = work_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify executables exist
    if not (work_dir / "GridDist").exists():
        print("Error: GridDist not found in {}".format(work_dir))
        sys.exit(1)
    if not (work_dir / "MPI_Fempois").exists():
        print("Error: MPI_Fempois not found in {}".format(work_dir))
        sys.exit(1)
    
    # SLURM options
    slurm_opts = {
        'partition': args.partition,
        'time': args.time,
        'account': args.account,
        'nodes': args.nodes,
        'ntasks_per_node': args.ntasks_per_node,
        'mem_per_cpu': args.mem_per_cpu,
    }
    
    print("=" * 70)
    print("ADAPTIVE GRID EXPERIMENT")
    print("=" * 70)
    print("Work directory: {}".format(work_dir))
    print("Output directory: {}".format(output_dir))
    print("Grid sizes: {}".format(args.sizes or GRID_SIZES))
    print("Precision: {}".format(args.precision))
    print("Max iterations: {}".format(args.max_iter or 'unlimited'))
    print("Using SLURM: {}".format(args.slurm))
    
    # Run experiments
    results = run_adaptive_comparison(
        work_dir, output_dir,
        use_slurm=args.slurm,
        slurm_opts=slurm_opts,
        sizes=args.sizes,
        precision=args.precision,
        max_iter=args.max_iter,
        oversubscribe=args.oversubscribe
    )
    
    # Save results
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
