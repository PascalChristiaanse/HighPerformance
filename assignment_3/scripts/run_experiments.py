#!/usr/bin/env python3
"""
Run experiments for Exercise 4.1: Time analysis of MPI_Fempois
Analyzes time spent in different phases:
- Computation
- Neighbor communication (Exchange_Borders)
- Global communication (MPI_Allreduce)
- Idle time

Configurations:
- 4 processes: 4x1 (414) and 2x2 (422)
- Problem sizes: 100x100, 200x200, 400x400
"""

import subprocess
import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
import argparse

# Configuration
CONFIGURATIONS = [
    {"name": "4x1", "px": 4, "py": 1, "np": 4},
    {"name": "2x2", "px": 2, "py": 2, "np": 4},
]

PROBLEM_SIZES = [
    {"name": "100x100", "dim_x": 100, "dim_y": 100},
    {"name": "200x200", "dim_x": 200, "dim_y": 200},
    {"name": "400x400", "dim_x": 400, "dim_y": 400},
]

NUM_RUNS = 3  # Number of runs per configuration for averaging


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
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


def run_solver(work_dir, np, telemetry_file, precision=None, max_iter=None):
    """Run MPI_Fempois with given parameters."""
    cmd = f"mpirun -np {np} ./MPI_Fempois -t {telemetry_file}"
    if precision is not None:
        cmd += f" -p {precision}"
    if max_iter is not None:
        cmd += f" -m {max_iter}"
    
    result = run_command(cmd, cwd=work_dir)
    return result


def parse_telemetry(telemetry_path):
    """Parse the telemetry CSV file and return a dictionary."""
    data = {}
    with open(telemetry_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2 and not row[0].startswith('#'):
                key, value = row
                try:
                    # Try to convert to appropriate type
                    if '.' in value or 'e' in value.lower():
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
    return data


def run_experiment(work_dir, config, problem_size, run_id, output_dir):
    """Run a single experiment and return the results."""
    print(f"  Running {config['name']} with {problem_size['name']} (run {run_id + 1})...")
    
    # Generate grid
    generate_grid(
        work_dir,
        config['px'], config['py'],
        problem_size['dim_x'], problem_size['dim_y']
    )
    
    # Create telemetry filename
    telemetry_file = output_dir / f"telemetry_{config['name']}_{problem_size['name']}_run{run_id}.csv"
    
    # Run solver
    run_solver(work_dir, config['np'], telemetry_file)
    
    # Parse results
    results = parse_telemetry(telemetry_file)
    results['configuration'] = config['name']
    results['problem_size'] = problem_size['name']
    results['run_id'] = run_id
    results['dim_x'] = problem_size['dim_x']
    results['dim_y'] = problem_size['dim_y']
    
    return results


def aggregate_results(all_results):
    """Aggregate results from multiple runs."""
    # Group by configuration and problem size
    grouped = {}
    for result in all_results:
        key = (result['configuration'], result['problem_size'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    # Calculate averages
    aggregated = []
    for key, runs in grouped.items():
        config, size = key
        avg_result = {
            'configuration': config,
            'problem_size': size,
            'num_runs': len(runs),
        }
        
        # Average numeric values
        numeric_keys = [
            'time_solve', 'time_compute_avg', 'time_comm_neighbors_avg',
            'time_comm_global_avg', 'time_idle_est', 'time_setup',
            'time_total', 'time_io', 'iterations', 'global_vertices',
            'compute_fraction', 'comm_neighbors_fraction', 'comm_global_fraction'
        ]
        
        for nkey in numeric_keys:
            if nkey in runs[0]:
                values = [r[nkey] for r in runs if nkey in r]
                avg_result[f'{nkey}_avg'] = sum(values) / len(values)
                avg_result[f'{nkey}_min'] = min(values)
                avg_result[f'{nkey}_max'] = max(values)
        
        aggregated.append(avg_result)
    
    return aggregated


def save_results(all_results, aggregated, output_dir):
    """Save results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = output_dir / f"all_results_{timestamp}.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(all_results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"All results saved to: {all_results_file}")
    
    # Save aggregated results
    agg_results_file = output_dir / f"aggregated_results_{timestamp}.csv"
    if aggregated:
        keys = aggregated[0].keys()
        with open(agg_results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(aggregated)
        print(f"Aggregated results saved to: {agg_results_file}")
    
    # Save as JSON for easier programmatic access
    json_file = output_dir / f"results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'aggregated': aggregated,
            'timestamp': timestamp,
            'configurations': CONFIGURATIONS,
            'problem_sizes': PROBLEM_SIZES,
            'num_runs': NUM_RUNS
        }, f, indent=2)
    print(f"JSON results saved to: {json_file}")
    
    return json_file


def print_summary(aggregated):
    """Print a summary table of the results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY - Time Distribution Analysis")
    print("=" * 80)
    
    # Header
    print(f"\n{'Config':<8} {'Size':<10} {'Solve(s)':<10} {'Compute':<10} {'Neighbors':<10} {'Global':<10} {'Idle':<10}")
    print("-" * 80)
    
    for result in sorted(aggregated, key=lambda x: (x['configuration'], x['problem_size'])):
        solve = result.get('time_solve_avg', 0)
        compute = result.get('time_compute_avg_avg', 0)
        neighbors = result.get('time_comm_neighbors_avg_avg', 0)
        global_comm = result.get('time_comm_global_avg_avg', 0)
        idle = result.get('time_idle_est_avg', 0)
        
        print(f"{result['configuration']:<8} {result['problem_size']:<10} "
              f"{solve:<10.4f} {compute:<10.4f} {neighbors:<10.4f} "
              f"{global_comm:<10.4f} {idle:<10.4f}")
    
    # Fraction summary
    print("\n" + "-" * 80)
    print("TIME FRACTIONS (as percentage of solve time)")
    print("-" * 80)
    print(f"\n{'Config':<8} {'Size':<10} {'Compute %':<12} {'Neighbors %':<12} {'Global %':<12}")
    print("-" * 60)
    
    for result in sorted(aggregated, key=lambda x: (x['configuration'], x['problem_size'])):
        compute_frac = result.get('compute_fraction_avg', 0) * 100
        neighbors_frac = result.get('comm_neighbors_fraction_avg', 0) * 100
        global_frac = result.get('comm_global_fraction_avg', 0) * 100
        
        print(f"{result['configuration']:<8} {result['problem_size']:<10} "
              f"{compute_frac:<12.2f} {neighbors_frac:<12.2f} {global_frac:<12.2f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Run MPI_Fempois experiments')
    parser.add_argument('--work-dir', type=str, default='.',
                        help='Working directory containing MPI_Fempois')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--runs', type=int, default=NUM_RUNS,
                        help=f'Number of runs per configuration (default: {NUM_RUNS})')
    parser.add_argument('--configs', type=str, nargs='+', 
                        choices=['4x1', '2x2'], default=['4x1', '2x2'],
                        help='Configurations to run')
    parser.add_argument('--sizes', type=str, nargs='+',
                        choices=['100x100', '200x200', '400x400'],
                        default=['100x100', '200x200', '400x400'],
                        help='Problem sizes to run')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir).resolve()
    output_dir = work_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Filter configurations and sizes based on arguments
    configs = [c for c in CONFIGURATIONS if c['name'] in args.configs]
    sizes = [s for s in PROBLEM_SIZES if s['name'] in args.sizes]
    
    print(f"Working directory: {work_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configurations: {[c['name'] for c in configs]}")
    print(f"Problem sizes: {[s['name'] for s in sizes]}")
    print(f"Runs per configuration: {args.runs}")
    print()
    
    # Run experiments
    all_results = []
    total_experiments = len(configs) * len(sizes) * args.runs
    current = 0
    
    for config in configs:
        for size in sizes:
            for run_id in range(args.runs):
                current += 1
                print(f"[{current}/{total_experiments}] ", end='')
                try:
                    result = run_experiment(work_dir, config, size, run_id, output_dir)
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
    
    # Aggregate and save results
    aggregated = aggregate_results(all_results)
    json_file = save_results(all_results, aggregated, output_dir)
    
    # Print summary
    print_summary(aggregated)
    
    print(f"\nExperiments complete! Results saved to {output_dir}")
    return json_file


if __name__ == '__main__':
    main()
