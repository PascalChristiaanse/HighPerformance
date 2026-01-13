#!/usr/bin/env python3
"""
Exercise 4.2: Communication Volume Analysis

Analyzes the amount of data sent/received each iteration between processes
and compares measurements with theoretical predictions for stripe-wise
and block-wise partitioning.

Theoretical analysis:
- For n×n grid with P processes:
  * Stripe partition (P×1): boundary = n points → O(n) doubles per neighbor
  * Block partition (√P × √P): boundary = n/√P points → O(n/√P) doubles per neighbor

For triangulated grids:
- Each interior point shares ~6 edges with neighbors
- Boundary points between subdomains form a "strip" of ghost nodes
- Stripe: ~2*n boundary points (one strip, two directions)
- Block: ~4*(n/√P) boundary points (four edges of block)
"""

import os
import sys
import subprocess
import argparse
import math
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def parse_telemetry(filepath):
    """Parse telemetry CSV file into dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or ',' not in line:
                continue
            key, value = line.split(',', 1)
            try:
                if '.' in value or 'e' in value.lower():
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value
    return data

def theoretical_comm_stripe(n, P):
    """
    Theoretical communication volume for stripe-wise partition.
    
    For P×1 stripe partition of n×n grid:
    - Each process except boundary has 2 neighbors
    - Each boundary has ~n points (actually n+1 for FEM due to shared vertices)
    - Each point is a double (8 bytes)
    
    Returns: approximate number of doubles sent per process per iteration
    """
    # Interior processes have 2 neighbors, boundary processes have 1
    # Average neighbors = 2*(P-2)/P + 2*1/P = (2P-2)/P ≈ 2 for large P
    # Communication per neighbor boundary ≈ n points
    return 2 * n  # approximate total send per process

def theoretical_comm_block(n, P):
    """
    Theoretical communication volume for block-wise partition.
    
    For √P × √P block partition of n×n grid:
    - Interior processes have 4 neighbors (could be more in triangulated mesh)
    - Each boundary has ~n/√P points
    
    Returns: approximate number of doubles sent per process per iteration
    """
    sqrt_P = math.sqrt(P)
    boundary_length = n / sqrt_P
    # Average 4 neighbors for interior (corners have 2, edges have 3, interior have 4)
    # In triangulated mesh, diagonal neighbors may also exist
    avg_neighbors = 4  # conservative estimate
    return avg_neighbors * boundary_length

def run_experiment(workdir, Px, Py, dim_x, dim_y, output_file):
    """Run a single experiment and return telemetry."""
    P = Px * Py
    
    # Generate grid
    result = subprocess.run(
        ['./GridDist', str(Px), str(Py), str(dim_x), str(dim_y)],
        cwd=workdir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"GridDist failed: {result.stderr}")
        return None
    
    # Run solver
    result = subprocess.run(
        ['mpirun', '-np', str(P), './MPI_Fempois', '-t', output_file],
        cwd=workdir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"MPI_Fempois failed: {result.stderr}")
        return None
    
    return parse_telemetry(os.path.join(workdir, output_file))

def main():
    parser = argparse.ArgumentParser(description='Analyze communication volume')
    parser.add_argument('--workdir', default='.', help='Working directory')
    parser.add_argument('--sizes', nargs='+', default=['100x100', '200x200', '400x400'],
                        help='Problem sizes (NxN format)')
    parser.add_argument('--output', default='results/comm_volume_analysis.txt',
                        help='Output file for analysis')
    args = parser.parse_args()
    
    workdir = os.path.abspath(args.workdir)
    
    # Ensure results directory exists
    os.makedirs(os.path.join(workdir, 'results'), exist_ok=True)
    
    # Configurations: (Px, Py, description)
    configs = [
        (4, 1, 'stripe_4x1'),
        (2, 2, 'block_2x2'),
    ]
    
    results = []
    
    print("=" * 80)
    print("COMMUNICATION VOLUME ANALYSIS")
    print("=" * 80)
    print()
    
    for size_str in args.sizes:
        dim_x, dim_y = map(int, size_str.split('x'))
        n = dim_x  # assuming square grid
        n_squared = n * n
        
        print(f"\n{'='*60}")
        print(f"Problem size: {dim_x}×{dim_y} (n² = {n_squared} grid points)")
        print(f"{'='*60}")
        
        for Px, Py, config_name in configs:
            P = Px * Py
            partition_type = 'stripe' if Px == P or Py == P else 'block'
            
            print(f"\n  Configuration: {Px}×{Py} ({partition_type})")
            print(f"  {'-'*50}")
            
            output_file = f'telemetry_{config_name}_{dim_x}x{dim_y}.csv'
            data = run_experiment(workdir, Px, Py, dim_x, dim_y, output_file)
            
            if data is None:
                print("    FAILED")
                continue
            
            # Extract measured values
            send_per_iter_total = data.get('send_count_per_iter_total', 0)
            send_per_iter_avg = send_per_iter_total / P
            send_per_iter_max = data.get('send_count_per_iter_max', 0)
            send_per_iter_min = data.get('send_count_per_iter_min', 0)
            min_neighbors = data.get('min_neighbors', 0)
            max_neighbors = data.get('max_neighbors', 0)
            iterations = data.get('iterations', 0)
            bytes_per_iter = data.get('bytes_per_iter_sent', 0)
            
            # Theoretical predictions
            if partition_type == 'stripe':
                theoretical = theoretical_comm_stripe(n, P)
            else:
                theoretical = theoretical_comm_block(n, P)
            
            # Print results
            print(f"    Measured:")
            print(f"      Send count/iter (total across all procs): {send_per_iter_total}")
            print(f"      Send count/iter (per proc avg): {send_per_iter_avg:.1f}")
            print(f"      Send count/iter (per proc min): {send_per_iter_min}")
            print(f"      Send count/iter (per proc max): {send_per_iter_max}")
            print(f"      Neighbors per proc: {min_neighbors}-{max_neighbors}")
            print(f"      Bytes per iteration (total): {bytes_per_iter}")
            print(f"      Iterations: {iterations}")
            print(f"    Theoretical (approx doubles/proc):")
            print(f"      Expected: ~{theoretical:.0f}")
            print(f"      Ratio (measured/theoretical): {send_per_iter_avg/theoretical:.2f}")
            
            results.append({
                'size': size_str,
                'n': n,
                'n_squared': n_squared,
                'config': config_name,
                'partition': partition_type,
                'P': P,
                'Px': Px,
                'Py': Py,
                'send_total': send_per_iter_total,
                'send_avg': send_per_iter_avg,
                'send_min': send_per_iter_min,
                'send_max': send_per_iter_max,
                'theoretical': theoretical,
                'ratio': send_per_iter_avg / theoretical if theoretical > 0 else 0,
                'min_neighbors': min_neighbors,
                'max_neighbors': max_neighbors,
                'iterations': iterations,
                'bytes_per_iter': bytes_per_iter,
            })
    
    # Print summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: Communication Volume Scaling")
    print("=" * 80)
    
    print("\nTheoretical predictions:")
    print("  - Stripe partition (P×1): Communication ∝ O(n)")
    print("    Each process sends/receives ~2n doubles per iteration")
    print("  - Block partition (√P×√P): Communication ∝ O(n/√P)")
    print("    Each process sends/receives ~4*(n/√P) doubles per iteration")
    print()
    
    # Group by partition type
    for partition in ['stripe', 'block']:
        print(f"\n{partition.upper()} PARTITION:")
        print(f"  {'Size':<12} {'n':<8} {'Measured':<15} {'Theoretical':<15} {'Ratio':<10}")
        print(f"  {'-'*60}")
        for r in results:
            if r['partition'] == partition:
                print(f"  {r['size']:<12} {r['n']:<8} {r['send_avg']:<15.1f} {r['theoretical']:<15.1f} {r['ratio']:<10.2f}")
    
    # Verify scaling
    print("\n" + "=" * 80)
    print("SCALING VERIFICATION")
    print("=" * 80)
    
    stripe_results = [r for r in results if r['partition'] == 'stripe']
    block_results = [r for r in results if r['partition'] == 'block']
    
    if len(stripe_results) >= 2:
        print("\nStripe partition (should scale as O(n)):")
        for i in range(1, len(stripe_results)):
            r0, r1 = stripe_results[i-1], stripe_results[i]
            n_ratio = r1['n'] / r0['n']
            comm_ratio = r1['send_avg'] / r0['send_avg'] if r0['send_avg'] > 0 else 0
            print(f"  n: {r0['n']} → {r1['n']} (×{n_ratio:.1f}), "
                  f"comm: {r0['send_avg']:.0f} → {r1['send_avg']:.0f} (×{comm_ratio:.2f})")
            print(f"    Expected ratio: {n_ratio:.1f}, Actual ratio: {comm_ratio:.2f}")
    
    if len(block_results) >= 2:
        print("\nBlock partition (should scale as O(n/√P) = O(n) for fixed P):")
        for i in range(1, len(block_results)):
            r0, r1 = block_results[i-1], block_results[i]
            n_ratio = r1['n'] / r0['n']
            comm_ratio = r1['send_avg'] / r0['send_avg'] if r0['send_avg'] > 0 else 0
            print(f"  n: {r0['n']} → {r1['n']} (×{n_ratio:.1f}), "
                  f"comm: {r0['send_avg']:.0f} → {r1['send_avg']:.0f} (×{comm_ratio:.2f})")
            print(f"    Expected ratio: {n_ratio:.1f}, Actual ratio: {comm_ratio:.2f}")
    
    # Compare stripe vs block
    print("\n" + "=" * 80)
    print("STRIPE vs BLOCK COMPARISON")
    print("=" * 80)
    print("\nFor same problem size, block partition should have LESS communication")
    print("when P > 1 (since O(n/√P) < O(n) for P > 1)")
    print()
    
    for size in args.sizes:
        stripe_r = next((r for r in stripe_results if r['size'] == size), None)
        block_r = next((r for r in block_results if r['size'] == size), None)
        if stripe_r and block_r:
            reduction = (1 - block_r['send_avg'] / stripe_r['send_avg']) * 100 if stripe_r['send_avg'] > 0 else 0
            print(f"  {size}: Stripe={stripe_r['send_avg']:.0f}, Block={block_r['send_avg']:.0f} "
                  f"({reduction:+.1f}% with block)")
    
    # Write results to file
    output_path = os.path.join(workdir, args.output)
    with open(output_path, 'w') as f:
        f.write("Communication Volume Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Theoretical expressions:\n")
        f.write("  Stripe (P×1): ~2n doubles per process per iteration\n")
        f.write("  Block (√P×√P): ~4*(n/√P) doubles per process per iteration\n\n")
        f.write(f"{'Config':<15} {'Size':<10} {'n':<6} {'P':<4} {'Measured':<12} {'Theory':<12} {'Ratio':<8}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['config']:<15} {r['size']:<10} {r['n']:<6} {r['P']:<4} "
                    f"{r['send_avg']:<12.1f} {r['theoretical']:<12.1f} {r['ratio']:<8.2f}\n")
    
    print(f"\nResults written to {output_path}")

if __name__ == '__main__':
    main()
