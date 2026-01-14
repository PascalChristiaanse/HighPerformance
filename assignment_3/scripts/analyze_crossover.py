#!/usr/bin/env python3
"""
Analyze crossover experiments to find where communication time equals computation time.

Uses measured data to fit models and extrapolate crossover points:
1. For fixed P=4: Find problem size n where T_comm = T_compute
2. For fixed 1000x1000: Find P where T_comm = T_compute

Theoretical background:
- Computation: T_compute ~ O(n² * iterations / P) per iteration: O(n²/P)
- Neighbor communication: T_comm_neighbors ~ O(n/√P) for block, O(n) for stripe
- Global communication: T_comm_global ~ O(log(P)) per Allreduce

For CG solver, per iteration:
- Compute: ~O(n²/P) - matrix-vector products, vector operations
- Neighbor comm: ~O(n/√P) or O(n) - boundary exchange
- Global comm: ~O(1) with O(log P) latency - Allreduce for dot products
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

def load_latest_results(results_dir):
    """Load the most recent crossover results file."""
    pattern = str(results_dir / "crossover_results_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No crossover results found in {results_dir}")
    
    latest = max(files, key=os.path.getctime)
    print(f"Loading results from: {latest}")
    
    with open(latest, 'r') as f:
        return json.load(f)


def fit_power_law(x, y):
    """Fit y = a * x^b using linear regression on log-log scale."""
    # Filter out zeros/negatives
    mask = (np.array(x) > 0) & (np.array(y) > 0)
    x_filt = np.array(x)[mask]
    y_filt = np.array(y)[mask]
    
    if len(x_filt) < 2:
        return None, None, None
    
    log_x = np.log(x_filt)
    log_y = np.log(y_filt)
    
    # Linear regression: log(y) = log(a) + b*log(x)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]  # exponent
    a = np.exp(coeffs[1])  # coefficient
    
    # R² calculation
    y_pred = a * x_filt**b
    ss_res = np.sum((y_filt - y_pred)**2)
    ss_tot = np.sum((y_filt - np.mean(y_filt))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return a, b, r_squared


def analyze_fixed_p(results):
    """Analyze experiment 1: Fixed P=4, find crossover problem size."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Fixed P=4, varying problem size")
    print("=" * 70)
    
    if not results:
        print("No results for experiment 1")
        return None
    
    # Extract data
    sizes = []
    t_compute = []
    t_comm = []
    
    for r in results:
        n = r['dim_x']
        sizes.append(n)
        tc = r.get('time_compute_avg', 0)
        # Total communication = neighbors + global
        tcomm = r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)
        t_compute.append(tc)
        t_comm.append(tcomm)
    
    sizes = np.array(sizes)
    t_compute = np.array(t_compute)
    t_comm = np.array(t_comm)
    
    # Normalize by iterations for per-iteration analysis
    iterations = [r.get('iterations', 1) for r in results]
    t_compute_per_iter = t_compute / np.array(iterations)
    t_comm_per_iter = t_comm / np.array(iterations)
    
    print("\nMeasured data (per iteration):")
    print(f"{'n':<8} {'T_compute':<15} {'T_comm':<15} {'Ratio':<10}")
    print("-" * 50)
    for i, n in enumerate(sizes):
        ratio = t_comm_per_iter[i] / t_compute_per_iter[i] if t_compute_per_iter[i] > 0 else float('inf')
        print(f"{n:<8} {t_compute_per_iter[i]:<15.6f} {t_comm_per_iter[i]:<15.6f} {ratio:<10.4f}")
    
    # Fit power laws
    # T_compute ~ a * n^b (expect b ≈ 2 for O(n²/P))
    a_comp, b_comp, r2_comp = fit_power_law(sizes, t_compute_per_iter)
    # T_comm ~ c * n^d (expect d ≈ 0.5-1 depending on partition)
    a_comm, b_comm, r2_comm = fit_power_law(sizes, t_comm_per_iter)
    
    print("\nFitted models (T = a * n^b):")
    if a_comp is not None:
        print(f"  T_compute = {a_comp:.2e} * n^{b_comp:.3f}  (R² = {r2_comp:.4f})")
        print(f"    Expected exponent: ~2 (O(n²/P)), measured: {b_comp:.3f}")
    if a_comm is not None:
        print(f"  T_comm    = {a_comm:.2e} * n^{b_comm:.3f}  (R² = {r2_comm:.4f})")
        print(f"    Expected exponent: ~0.5-1 (O(n) or O(n/√P)), measured: {b_comm:.3f}")
    
    # Find crossover: a_comp * n^b_comp = a_comm * n^b_comm
    # n^(b_comp - b_comm) = a_comm / a_comp
    # n = (a_comm / a_comp)^(1/(b_comp - b_comm))
    crossover_n = None
    if a_comp and a_comm and abs(b_comp - b_comm) > 0.01:
        try:
            crossover_n = (a_comm / a_comp) ** (1 / (b_comp - b_comm))
            print(f"\n*** CROSSOVER ESTIMATE (T_comm = T_compute): n ≈ {crossover_n:.0f} ***")
            
            # Verify by computing times at crossover
            t_comp_cross = a_comp * crossover_n**b_comp
            t_comm_cross = a_comm * crossover_n**b_comm
            print(f"    At n={crossover_n:.0f}: T_compute ≈ {t_comp_cross:.6f}s, T_comm ≈ {t_comm_cross:.6f}s")
        except:
            print("\nCould not compute crossover (numerical issues)")
    else:
        # Find by interpolation
        ratios = t_comm_per_iter / t_compute_per_iter
        for i in range(len(ratios) - 1):
            if (ratios[i] < 1 and ratios[i+1] >= 1) or (ratios[i] > 1 and ratios[i+1] <= 1):
                # Linear interpolation
                crossover_n = sizes[i] + (sizes[i+1] - sizes[i]) * (1 - ratios[i]) / (ratios[i+1] - ratios[i])
                print(f"\n*** CROSSOVER ESTIMATE (interpolated): n ≈ {crossover_n:.0f} ***")
                break
    
    if crossover_n is None:
        ratios = t_comm_per_iter / t_compute_per_iter
        if all(r < 1 for r in ratios):
            print("\n*** Computation dominates for all tested sizes. Crossover at n < {:.0f} ***".format(min(sizes)))
        else:
            print("\n*** Communication dominates for all tested sizes. Crossover at n > {:.0f} ***".format(max(sizes)))
    
    return {
        'sizes': sizes.tolist(),
        't_compute': t_compute_per_iter.tolist(),
        't_comm': t_comm_per_iter.tolist(),
        'fit_compute': {'a': a_comp, 'b': b_comp, 'r2': r2_comp},
        'fit_comm': {'a': a_comm, 'b': b_comm, 'r2': r2_comm},
        'crossover_n': crossover_n,
    }


def analyze_fixed_size(results):
    """Analyze experiment 2: Fixed size, find crossover P."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Fixed size 1000x1000, varying P")
    print("=" * 70)
    
    if not results:
        print("No results for experiment 2")
        return None
    
    # Extract data
    procs = []
    t_compute = []
    t_comm = []
    t_comm_neighbors = []
    t_comm_global = []
    
    for r in results:
        p = r['np']
        procs.append(p)
        tc = r.get('time_compute_avg', 0)
        tcn = r.get('time_comm_neighbors_avg', 0)
        tcg = r.get('time_comm_global_avg', 0)
        t_compute.append(tc)
        t_comm_neighbors.append(tcn)
        t_comm_global.append(tcg)
        t_comm.append(tcn + tcg)
    
    procs = np.array(procs)
    t_compute = np.array(t_compute)
    t_comm = np.array(t_comm)
    t_comm_neighbors = np.array(t_comm_neighbors)
    t_comm_global = np.array(t_comm_global)
    
    # Normalize by iterations
    iterations = [r.get('iterations', 1) for r in results]
    t_compute_per_iter = t_compute / np.array(iterations)
    t_comm_per_iter = t_comm / np.array(iterations)
    t_comm_n_per_iter = t_comm_neighbors / np.array(iterations)
    t_comm_g_per_iter = t_comm_global / np.array(iterations)
    
    print("\nMeasured data (per iteration):")
    print(f"{'P':<6} {'T_compute':<12} {'T_comm_nb':<12} {'T_comm_gl':<12} {'T_comm_tot':<12} {'Ratio':<8}")
    print("-" * 65)
    for i, p in enumerate(procs):
        ratio = t_comm_per_iter[i] / t_compute_per_iter[i] if t_compute_per_iter[i] > 0 else float('inf')
        print(f"{p:<6} {t_compute_per_iter[i]:<12.6f} {t_comm_n_per_iter[i]:<12.6f} "
              f"{t_comm_g_per_iter[i]:<12.6f} {t_comm_per_iter[i]:<12.6f} {ratio:<8.4f}")
    
    # Fit models
    # T_compute ~ a / P (expect speedup with more processes)
    # T_comm ~ c * sqrt(P) or c * log(P)
    
    # Fit T_compute = a * P^b (expect b ≈ -1)
    a_comp, b_comp, r2_comp = fit_power_law(procs, t_compute_per_iter)
    # Fit T_comm = c * P^d
    a_comm, b_comm, r2_comm = fit_power_law(procs, t_comm_per_iter)
    
    print("\nFitted models (T = a * P^b):")
    if a_comp is not None:
        print(f"  T_compute = {a_comp:.2e} * P^{b_comp:.3f}  (R² = {r2_comp:.4f})")
        print(f"    Expected exponent: ~-1 (O(1/P)), measured: {b_comp:.3f}")
    if a_comm is not None:
        print(f"  T_comm    = {a_comm:.2e} * P^{b_comm:.3f}  (R² = {r2_comm:.4f})")
        print(f"    Expected exponent: ~0.5 (O(√P)), measured: {b_comm:.3f}")
    
    # Find crossover P
    crossover_p = None
    if a_comp and a_comm and abs(b_comp - b_comm) > 0.01:
        try:
            crossover_p = (a_comm / a_comp) ** (1 / (b_comp - b_comm))
            print(f"\n*** CROSSOVER ESTIMATE (T_comm = T_compute): P ≈ {crossover_p:.0f} ***")
            
            # Verify
            t_comp_cross = a_comp * crossover_p**b_comp
            t_comm_cross = a_comm * crossover_p**b_comm
            print(f"    At P={crossover_p:.0f}: T_compute ≈ {t_comp_cross:.6f}s, T_comm ≈ {t_comm_cross:.6f}s")
        except:
            print("\nCould not compute crossover (numerical issues)")
    else:
        # Find by interpolation
        ratios = t_comm_per_iter / t_compute_per_iter
        for i in range(len(ratios) - 1):
            if (ratios[i] < 1 and ratios[i+1] >= 1) or (ratios[i] > 1 and ratios[i+1] <= 1):
                # Linear interpolation in log space
                log_p = np.log(procs)
                crossover_log_p = log_p[i] + (log_p[i+1] - log_p[i]) * (1 - ratios[i]) / (ratios[i+1] - ratios[i])
                crossover_p = np.exp(crossover_log_p)
                print(f"\n*** CROSSOVER ESTIMATE (interpolated): P ≈ {crossover_p:.0f} ***")
                break
    
    if crossover_p is None:
        ratios = t_comm_per_iter / t_compute_per_iter
        if all(r < 1 for r in ratios):
            print("\n*** Computation dominates for all tested P. Crossover at P > {:.0f} ***".format(max(procs)))
        else:
            print("\n*** Communication dominates for all tested P. Crossover at P < {:.0f} ***".format(min(procs)))
    
    return {
        'procs': procs.tolist(),
        't_compute': t_compute_per_iter.tolist(),
        't_comm': t_comm_per_iter.tolist(),
        't_comm_neighbors': t_comm_n_per_iter.tolist(),
        't_comm_global': t_comm_g_per_iter.tolist(),
        'fit_compute': {'a': a_comp, 'b': b_comp, 'r2': r2_comp},
        'fit_comm': {'a': a_comm, 'b': b_comm, 'r2': r2_comm},
        'crossover_p': crossover_p,
    }


def theoretical_analysis():
    """Print theoretical analysis for reference."""
    print("\n" + "=" * 70)
    print("THEORETICAL ANALYSIS")
    print("=" * 70)
    
    print("""
For the Conjugate Gradient solver on an n×n grid with P processes:

Per iteration costs:
  - Computation: Matrix-vector product (SpMV) + vector operations
    T_compute ∝ n²/P  (each process handles n²/P grid points)
    
  - Neighbor communication: Ghost cell exchange
    T_comm_neighbors ∝ n/√P  (block partition boundary length)
    or T_comm_neighbors ∝ n   (stripe partition)
    
  - Global communication: 2× MPI_Allreduce for dot products
    T_comm_global ∝ log(P)  (reduction tree)

Crossover analysis:

1. Fixed P=4, varying n:
   T_compute = α·n²/P,  T_comm = β·n/√P + γ·log(P)
   
   For large n, T_compute dominates (O(n²) vs O(n))
   Crossover when: α·n²/P = β·n/√P
                   n = β·√P / α
   
   For P=4: n_crossover ≈ β/α · 2

2. Fixed n=1000, varying P:
   T_compute = α·n²/P,  T_comm = β·n/√P + γ·log(P)
   
   As P increases, T_compute decreases while T_comm increases
   Crossover when: α·n²/P = β·n/√P
                   P^(1/2) = α·n / β
                   P = (α·n/β)²
   
   For n=1000: P_crossover depends on α/β ratio (machine-dependent)
""")


def save_analysis(analysis1, analysis2, output_dir):
    """Save analysis results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"crossover_analysis_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump({
            'fixed_p_analysis': convert(analysis1),
            'fixed_size_analysis': convert(analysis2),
            'timestamp': timestamp,
        }, f, indent=2)
    
    print(f"\nAnalysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze crossover experiments')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing experiment results')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Specific results file to analyze')
    parser.add_argument('--theoretical', action='store_true',
                        help='Print theoretical analysis')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    
    if args.theoretical:
        theoretical_analysis()
    
    # Load results
    try:
        if args.results_file:
            with open(args.results_file, 'r') as f:
                data = json.load(f)
        else:
            data = load_latest_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python scripts/run_crossover_experiments.py' first to generate data")
        theoretical_analysis()
        return
    
    # Analyze
    exp1_results = data.get('experiment1_fixed_p', [])
    exp2_results = data.get('experiment2_fixed_size', [])
    
    analysis1 = analyze_fixed_p(exp1_results)
    analysis2 = analyze_fixed_size(exp2_results)
    
    # Print theoretical context
    theoretical_analysis()
    
    # Save analysis
    save_analysis(analysis1, analysis2, results_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if analysis1 and analysis1.get('crossover_n'):
        print(f"\n1. For P=4 processes:")
        print(f"   Communication equals computation at n ≈ {analysis1['crossover_n']:.0f}")
        print(f"   For n > {analysis1['crossover_n']:.0f}: computation dominates")
        print(f"   For n < {analysis1['crossover_n']:.0f}: communication dominates")
    
    if analysis2 and analysis2.get('crossover_p'):
        print(f"\n2. For 1000×1000 problem:")
        print(f"   Communication equals computation at P ≈ {analysis2['crossover_p']:.0f}")
        print(f"   For P < {analysis2['crossover_p']:.0f}: computation dominates")
        print(f"   For P > {analysis2['crossover_p']:.0f}: communication dominates")


if __name__ == '__main__':
    main()
