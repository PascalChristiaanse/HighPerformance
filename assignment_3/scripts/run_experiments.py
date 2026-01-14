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
import time
from pathlib import Path
from datetime import datetime
import argparse
import tempfile

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

# SLURM default settings
SLURM_DEFAULTS = {
    'partition': 'compute',
    'time': '01:00:00',
    'nodes': 1,
    'ntasks_per_node': 4,
    'account': 'Education-EEMCS-Courses-WI4049TU',
    'mem_per_cpu': '1G',
}


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
               use_slurm=False, slurm_opts=None):
    """Run MPI_Fempois with given parameters."""
    solver_args = f"-t {telemetry_file}"
    if precision is not None:
        solver_args += f" -p {precision}"
    if max_iter is not None:
        solver_args += f" -m {max_iter}"
    
    if use_slurm:
        result = run_solver_slurm(work_dir, np, solver_args, slurm_opts)
    else:
        cmd = f"mpirun -np {np} ./MPI_Fempois {solver_args}"
        result = run_command(cmd, cwd=work_dir)
    
    return result


def run_solver_slurm(work_dir, np, solver_args, slurm_opts):
    """Run MPI_Fempois using SLURM's srun (interactive mode)."""
    opts = slurm_opts or {}
    
    # Build srun command for interactive execution
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
        srun_cmd += f" --ntasks-per-node={opts['ntasks_per_node']}"
    if opts.get('cpus_per_task'):
        srun_cmd += f" --cpus-per-task={opts['cpus_per_task']}"
    if opts.get('mem'):
        srun_cmd += f" --mem={opts['mem']}"
    if opts.get('mem_per_cpu'):
        srun_cmd += f" --mem-per-cpu={opts['mem_per_cpu']}"
    if opts.get('constraint'):
        srun_cmd += f" --constraint={opts['constraint']}"
    
    # Add any extra SLURM options
    if opts.get('extra_args'):
        srun_cmd += f" {opts['extra_args']}"
    
    srun_cmd += f" ./MPI_Fempois {solver_args}"
    
    return run_command(srun_cmd, cwd=work_dir)


def submit_batch_job(work_dir, np, solver_args, slurm_opts, job_name="fempois"):
    """Submit a batch job to SLURM and wait for completion."""
    opts = slurm_opts or {}
    
    # Create batch script
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --ntasks={np}
"""
    
    if opts.get('partition'):
        script_content += f"#SBATCH --partition={opts['partition']}\n"
    if opts.get('time'):
        script_content += f"#SBATCH --time={opts['time']}\n"
    if opts.get('account'):
        script_content += f"#SBATCH --account={opts['account']}\n"
    if opts.get('nodes'):
        script_content += f"#SBATCH --nodes={opts['nodes']}\n"
    if opts.get('ntasks_per_node'):
        script_content += f"#SBATCH --ntasks-per-node={opts['ntasks_per_node']}\n"
    if opts.get('cpus_per_task'):
        script_content += f"#SBATCH --cpus-per-task={opts['cpus_per_task']}\n"
    if opts.get('mem'):
        script_content += f"#SBATCH --mem={opts['mem']}\n"
    if opts.get('constraint'):
        script_content += f"#SBATCH --constraint={opts['constraint']}\n"
    
    script_content += f"""
cd {work_dir}
srun ./MPI_Fempois {solver_args}
"""
    
    # Write batch script
    script_path = work_dir / f"{job_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Submit job
    result = run_command(f"sbatch --parsable {script_path}", cwd=work_dir)
    job_id = result.stdout.strip()
    print(f"    Submitted batch job {job_id}")
    
    # Wait for job completion
    wait_for_job(job_id)
    
    # Clean up
    script_path.unlink(missing_ok=True)
    
    return result


def wait_for_job(job_id, poll_interval=5):
    """Wait for a SLURM job to complete."""
    while True:
        result = run_command(f"squeue -j {job_id} -h -o %T", check=False)
        status = result.stdout.strip()
        
        if not status or status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
            break
        
        time.sleep(poll_interval)
    
    # Check final job status
    result = run_command(f"sacct -j {job_id} -o State -n -X", check=False)
    final_status = result.stdout.strip().split('\n')[0].strip() if result.stdout else 'UNKNOWN'
    
    if final_status not in ['COMPLETED']:
        raise RuntimeError(f"Job {job_id} ended with status: {final_status}")


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


def run_experiment(work_dir, config, problem_size, run_id, output_dir,
                   use_slurm=False, slurm_opts=None):
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
    run_solver(work_dir, config['np'], telemetry_file,
               use_slurm=use_slurm, slurm_opts=slurm_opts)
    
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
    
    # SLURM options
    slurm_group = parser.add_argument_group('SLURM options')
    slurm_group.add_argument('--slurm', action='store_true',
                             help='Run experiments using SLURM (srun)')
    slurm_group.add_argument('--partition', '-p', type=str,
                             default=SLURM_DEFAULTS['partition'],
                             help=f"SLURM partition (default: {SLURM_DEFAULTS['partition']})")
    slurm_group.add_argument('--time', '-t', type=str,
                             default=SLURM_DEFAULTS['time'],
                             help=f"Time limit per job (default: {SLURM_DEFAULTS['time']})")
    slurm_group.add_argument('--account', '-A', type=str,
                             default=SLURM_DEFAULTS['account'],
                             help='SLURM account/project for billing')
    slurm_group.add_argument('--nodes', '-N', type=int,
                             default=SLURM_DEFAULTS['nodes'],
                             help=f"Number of nodes (default: {SLURM_DEFAULTS['nodes']})")
    slurm_group.add_argument('--ntasks-per-node', type=int,
                             default=SLURM_DEFAULTS['ntasks_per_node'],
                             help=f"Tasks per node (default: {SLURM_DEFAULTS['ntasks_per_node']})")
    slurm_group.add_argument('--cpus-per-task', type=int, default=None,
                             help='CPUs per task (optional)')
    slurm_group.add_argument('--mem', type=str, default=None,
                             help='Memory per node (e.g., 4G, 16000M)')
    slurm_group.add_argument('--mem-per-cpu', type=str,
                             default=SLURM_DEFAULTS['mem_per_cpu'],
                             help=f"Memory per CPU (default: {SLURM_DEFAULTS['mem_per_cpu']})")
    slurm_group.add_argument('--constraint', type=str, default=None,
                             help='Node feature constraint')
    slurm_group.add_argument('--slurm-extra', type=str, default=None,
                             help='Extra SLURM arguments (passed directly to srun)')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir).resolve()
    output_dir = work_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Filter configurations and sizes based on arguments
    configs = [c for c in CONFIGURATIONS if c['name'] in args.configs]
    sizes = [s for s in PROBLEM_SIZES if s['name'] in args.sizes]
    
    # Build SLURM options dictionary
    slurm_opts = {
        'partition': args.partition,
        'time': args.time,
        'account': args.account,
        'nodes': args.nodes,
        'ntasks_per_node': args.ntasks_per_node,
        'cpus_per_task': args.cpus_per_task,
        'mem': args.mem,
        'mem_per_cpu': args.mem_per_cpu,
        'constraint': args.constraint,
        'extra_args': args.slurm_extra,
    }
    
    print(f"Working directory: {work_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configurations: {[c['name'] for c in configs]}")
    print(f"Problem sizes: {[s['name'] for s in sizes]}")
    print(f"Runs per configuration: {args.runs}")
    if args.slurm:
        print(f"SLURM mode: enabled")
        print(f"  Partition: {args.partition}")
        print(f"  Time limit: {args.time}")
        if args.account:
            print(f"  Account: {args.account}")
    else:
        print(f"Running locally with mpirun")
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
                    result = run_experiment(work_dir, config, size, run_id, output_dir,
                                            use_slurm=args.slurm, slurm_opts=slurm_opts)
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
