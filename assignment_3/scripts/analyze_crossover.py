#!/usr/bin/env python3
"""
Analyze crossover experiments to find where communication time equals computation time.

Creates three separate figures:
1. Experiment 1: Fixed P=4, varying problem size
2. Experiment 2 (1000x1000 only): Fixed size, varying P
3. Experiment 3 (Surface): 3D surface showing crossover as function of both n and P

Uses measured data to fit models and find/extrapolate crossover points.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

# Plotting availability: allow matplotlib without requiring scipy.
# SciPy is optional (used for interpolation/advanced root-finding).
HAS_MATPLOTLIB = False
HAS_SCIPY = False
HAS_PLOTTING = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from mpl_toolkits.mplot3d import Axes3D  # optional for 3D plotting
except Exception:
    Axes3D = None

try:
    from scipy.interpolate import griddata
    from scipy.optimize import brentq
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

HAS_PLOTTING = HAS_MATPLOTLIB
if not HAS_PLOTTING:
    print("Warning: matplotlib not available. Plotting disabled.")
else:
    if not HAS_SCIPY:
        print("Warning: scipy not available — 3D/interpolation features will be limited.")


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
    mask = (np.array(x) > 0) & (np.array(y) > 0)
    x_filt = np.array(x)[mask]
    y_filt = np.array(y)[mask]
    
    if len(x_filt) < 2:
        return None, None, None
    
    log_x = np.log(x_filt)
    log_y = np.log(y_filt)
    
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    
    y_pred = a * x_filt**b
    ss_res = np.sum((y_filt - y_pred)**2)
    ss_tot = np.sum((y_filt - np.mean(y_filt))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return a, b, r_squared


def find_crossover(a_comp, b_comp, a_comm, b_comm):
    """Find crossover point where a_comp * x^b_comp = a_comm * x^b_comm."""
    if a_comp and a_comm and abs(b_comp - b_comm) > 0.01:
        try:
            crossover = (a_comm / a_comp) ** (1 / (b_comp - b_comm))
            return crossover
        except:
            pass
    return None


def analyze_fixed_p(results, output_dir):
    """Analyze and plot experiment 1: Fixed P=4, varying problem size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fixed P=4, varying problem size")
    print("=" * 70)
    
    if not results:
        print("No results for experiment 1")
        return None
    
    # Extract data
    data = []
    for r in results:
        n = r['dim_x']
        iterations = r.get('iterations', 1)
        t_compute = r.get('time_compute_avg', 0) / iterations
        t_comm_nb = r.get('time_comm_neighbors_avg', 0) / iterations
        t_comm_gl = r.get('time_comm_global_avg', 0) / iterations
        t_comm = t_comm_nb + t_comm_gl
        data.append((n, t_compute, t_comm, t_comm_nb, t_comm_gl))
    
    data.sort(key=lambda x: x[0])
    sizes = np.array([d[0] for d in data])
    t_compute = np.array([d[1] for d in data])
    t_comm = np.array([d[2] for d in data])
    
    # Print table
    print("\nMeasured data (per iteration):")
    print(f"{'n':<10} {'T_compute':<15} {'T_comm':<15} {'Ratio':<10}")
    print("-" * 50)
    for d in data:
        ratio = d[2] / d[1] if d[1] > 0 else float('inf')
        print(f"{d[0]:<10} {d[1]:<15.6f} {d[2]:<15.6f} {ratio:<10.4f}")
    
    # Fit power laws
    a_comp, b_comp, r2_comp = fit_power_law(sizes, t_compute)
    a_comm, b_comm, r2_comm = fit_power_law(sizes, t_comm)
    
    print("\nFitted models (T = a * n^b):")
    if a_comp:
        print(f"  T_compute = {a_comp:.2e} * n^{b_comp:.3f}  (R² = {r2_comp:.4f})")
    if a_comm:
        print(f"  T_comm    = {a_comm:.2e} * n^{b_comm:.3f}  (R² = {r2_comm:.4f})")
    
    crossover_n = find_crossover(a_comp, b_comp, a_comm, b_comm)
    if crossover_n:
        print(f"\n*** CROSSOVER ESTIMATE: n ≈ {crossover_n:.0f} ***")
    
    # Plot
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(8.27/1.75, 8.27/2))
        
        ax.loglog(sizes, t_compute, 'bo-', markersize=8, linewidth=2, label='Computation')
        ax.loglog(sizes, t_comm, 'rs-', markersize=8, linewidth=2, label='Communication')
        
        # Plot fitted lines
        n_range = np.logspace(np.log10(min(sizes)*0.5), np.log10(max(sizes)*2), 100)
        if a_comp:
            ax.loglog(n_range, a_comp * n_range**b_comp, 'b--', alpha=0.5, 
                     label=f'Fit: {a_comp:.1e}·n^{b_comp:.2f}')
        if a_comm:
            ax.loglog(n_range, a_comm * n_range**b_comm, 'r--', alpha=0.5,
                     label=f'Fit: {a_comm:.1e}·n^{b_comm:.2f}')
        
        if crossover_n and min(sizes)*0.5 < crossover_n < max(sizes)*2:
            ax.axvline(crossover_n, color='green', linestyle=':', linewidth=2,
                      label=f'Crossover: n≈{crossover_n:.0f}')
        
        ax.set_xlabel('Problem Size (n)', fontsize=12)
        ax.set_ylabel('Mean Time per Iteration (s)', fontsize=12)
        # ax.set_title('Experiment 1: Fixed P=4, Varying Problem Size', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'exp1_fixed_p.png', dpi=150)
        fig.savefig(output_dir / 'exp1_fixed_p.pdf')
        print(f"\nPlot saved to: {output_dir / 'exp1_fixed_p.png'}")
        plt.close(fig)
    
    return {
        'sizes': sizes.tolist(),
        't_compute': t_compute.tolist(),
        't_comm': t_comm.tolist(),
        'fit_compute': {'a': a_comp, 'b': b_comp, 'r2': r2_comp},
        'fit_comm': {'a': a_comm, 'b': b_comm, 'r2': r2_comm},
        'crossover_n': crossover_n,
    }


def analyze_fixed_size(results, output_dir):
    """Analyze and plot experiment 2: Fixed size 1000x1000, varying P."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Fixed size 1000x1000, varying P")
    print("=" * 70)
    
    if not results:
        print("No results for experiment 2")
        return None
    
    # Extract data
    data = []
    for r in results:
        p = r['np']
        iterations = r.get('iterations', 1)
        t_compute = r.get('time_compute_avg', 0) / iterations
        t_comm_nb = r.get('time_comm_neighbors_avg', 0) / iterations
        t_comm_gl = r.get('time_comm_global_avg', 0) / iterations
        t_comm = t_comm_nb + t_comm_gl
        data.append((p, t_compute, t_comm, t_comm_nb, t_comm_gl))
    
    data.sort(key=lambda x: x[0])
    procs = np.array([d[0] for d in data])
    t_compute = np.array([d[1] for d in data])
    t_comm = np.array([d[2] for d in data])
    t_comm_nb = np.array([d[3] for d in data])
    t_comm_gl = np.array([d[4] for d in data])
    
    # Print table
    print("\nMeasured data (per iteration):")
    print(f"{'P':<6} {'T_compute':<12} {'T_comm_nb':<12} {'T_comm_gl':<12} {'T_comm':<12} {'Ratio':<8}")
    print("-" * 65)
    for d in data:
        ratio = d[2] / d[1] if d[1] > 0 else float('inf')
        print(f"{d[0]:<6} {d[1]:<12.6f} {d[3]:<12.6f} {d[4]:<12.6f} {d[2]:<12.6f} {ratio:<8.4f}")
    
    # Fit power laws
    a_comp, b_comp, r2_comp = fit_power_law(procs, t_compute)
    a_comm, b_comm, r2_comm = fit_power_law(procs, t_comm)
    
    print("\nFitted models (T = a * P^b):")
    if a_comp:
        print(f"  T_compute = {a_comp:.2e} * P^{b_comp:.3f}  (R² = {r2_comp:.4f})")
        print(f"    Expected: ~P^-1 (ideal scaling)")
    if a_comm:
        print(f"  T_comm    = {a_comm:.2e} * P^{b_comm:.3f}  (R² = {r2_comm:.4f})")
        print(f"    Expected: ~P^0.5 (boundary scaling)")
    
    crossover_p = find_crossover(a_comp, b_comp, a_comm, b_comm)
    if crossover_p:
        print(f"\n*** CROSSOVER ESTIMATE: P ≈ {crossover_p:.0f} ***")
    
    # Plot
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(procs, t_compute, 'bo-', markersize=8, linewidth=2, label='Computation')
        ax.loglog(procs, t_comm, 'rs-', markersize=8, linewidth=2, label='Total Communication')
        ax.loglog(procs, t_comm_nb, 'g^--', markersize=6, linewidth=1.5, alpha=0.7, label='Neighbor Comm')
        ax.loglog(procs, t_comm_gl, 'mv--', markersize=6, linewidth=1.5, alpha=0.7, label='Global Comm')
        
        # Plot fitted lines
        p_range = np.logspace(0, np.log10(max(procs)*2), 100)
        if a_comp:
            ax.loglog(p_range, a_comp * p_range**b_comp, 'b--', alpha=0.4)
        if a_comm:
            ax.loglog(p_range, a_comm * p_range**b_comm, 'r--', alpha=0.4)
        
        if crossover_p and crossover_p < max(procs)*2:
            ax.axvline(crossover_p, color='orange', linestyle=':', linewidth=2,
                      label=f'Crossover: P≈{crossover_p:.0f}')
        
        ax.set_xlabel('Number of Processes (P)', fontsize=12)
        ax.set_ylabel('Time per Iteration (s)', fontsize=12)
        # ax.set_title('Experiment 2: Fixed 1000×1000 Grid, Varying P', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'exp2_fixed_size_1000.png', dpi=150)
        fig.savefig(output_dir / 'exp2_fixed_size_1000.pdf')
        print(f"\nPlot saved to: {output_dir / 'exp2_fixed_size_1000.png'}")
        plt.close(fig)
    
    return {
        'procs': procs.tolist(),
        't_compute': t_compute.tolist(),
        't_comm': t_comm.tolist(),
        'fit_compute': {'a': a_comp, 'b': b_comp, 'r2': r2_comp},
        'fit_comm': {'a': a_comm, 'b': b_comm, 'r2': r2_comm},
        'crossover_p': crossover_p,
    }


def analyze_surface(results, output_dir):
    """Analyze and plot experiment 3: Surface showing crossover for varying n and P."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Surface - varying both n and P")
    print("=" * 70)
    
    if not results:
        print("No results for experiment 3")
        return None
    
    # Extract data
    sizes = sorted(set(r['dim_x'] for r in results))
    procs = sorted(set(r['np'] for r in results))
    
    print(f"\nProblem sizes tested: {sizes}")
    print(f"Process counts tested: {procs}")
    
    # Build data arrays
    data_points = []
    ratio_matrix = np.full((len(sizes), len(procs)), np.nan)
    
    print("\nCommunication/Computation Ratio (T_comm/T_compute):")
    print("-" * (12 + 10 * len(procs)))
    col_header = "n \\ P"
    header = "{:<12}".format(col_header) + "".join("{:<10}".format(p) for p in procs)
    print(header)
    print("-" * (12 + 10 * len(procs)))
    
    for i, size in enumerate(sizes):
        row = f"{size:<12}"
        for j, p in enumerate(procs):
            # Find matching result
            matching = [r for r in results if r['dim_x'] == size and r['np'] == p]
            if matching:
                r = matching[0]
                iterations = r.get('iterations', 1)
                t_compute = r.get('time_compute_avg', 0) / iterations
                t_comm = (r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)) / iterations
                ratio = t_comm / t_compute if t_compute > 0 else float('inf')
                ratio_matrix[i, j] = ratio
                data_points.append((size, p, t_compute, t_comm, ratio))
                row += f"{ratio:<10.3f}"
            else:
                row += f"{'---':<10}"
        print(row)
    
    # Find crossover points for each problem size
    print("\nCrossover points (T_comm = T_compute):")
    print("-" * 50)
    crossover_points = []
    
    for i, size in enumerate(sizes):
        ratios = ratio_matrix[i, :]
        valid_idx = ~np.isnan(ratios)
        if not np.any(valid_idx):
            continue
        
        valid_procs = np.array(procs)[valid_idx]
        valid_ratios = ratios[valid_idx]
        
        # Find where ratio crosses 1
        crossover_p = None
        for k in range(len(valid_ratios) - 1):
            if valid_ratios[k] < 1 and valid_ratios[k+1] >= 1:
                # Linear interpolation in log space
                log_p1, log_p2 = np.log(valid_procs[k]), np.log(valid_procs[k+1])
                r1, r2 = valid_ratios[k], valid_ratios[k+1]
                crossover_log_p = log_p1 + (log_p2 - log_p1) * (1 - r1) / (r2 - r1)
                crossover_p = np.exp(crossover_log_p)
                break
        
        if crossover_p:
            print(f"  n={size:5d}: P_crossover ≈ {crossover_p:.1f}")
            crossover_points.append((size, crossover_p))
        elif np.all(valid_ratios < 1):
            print(f"  n={size:5d}: Computation dominates for all P (crossover at P > {max(valid_procs)})")
        else:
            print(f"  n={size:5d}: Communication dominates for all P (crossover at P < {min(valid_procs)})")
    
    # Fit crossover curve: P_crossover = f(n)
    if len(crossover_points) >= 2:
        cross_n = np.array([cp[0] for cp in crossover_points])
        cross_p = np.array([cp[1] for cp in crossover_points])
        a_cross, b_cross, r2_cross = fit_power_law(cross_n, cross_p)
        if a_cross:
            print(f"\nCrossover curve fit: P_crossover = {a_cross:.2e} * n^{b_cross:.3f} (R² = {r2_cross:.4f})")
            # Estimate P_crossover for n = 1000 using the fitted curve
            try:
                n_est = 1000
                p_est = a_cross * (n_est ** b_cross)
                print(f"Estimated P_crossover for n={n_est}: P ≈ {p_est:.1f}")
            except Exception:
                pass
    
    # Plot
    if HAS_PLOTTING:
        # Create figure with 2 subplots: surface and crossover curve
        fig = plt.figure(figsize=(8.27/1.75, 8.27/1.75)) # A4 size
        
        # # 3D Surface plot of ratio
        # ax1 = fig.add_subplot(121, projection='3d')
        
        # # Create meshgrid for surface
        # N, P = np.meshgrid(sizes, procs)
        # Z = ratio_matrix.T  # Transpose to match meshgrid orientation
        
        # # Plot surface
        # surf = ax1.plot_surface(np.log10(N), np.log10(P), np.log10(Z + 0.001), 
        #                         cmap='RdYlGn_r', alpha=0.8, edgecolor='none')
        
        # # Add crossover plane at ratio = 1 (log10(1) = 0)
        # ax1.plot_surface(np.log10(N), np.log10(P), 
        #                 np.zeros_like(Z), alpha=0.3, color='blue')
        
        # ax1.set_xlabel('log₁₀(n)', fontsize=10)
        # ax1.set_ylabel('log₁₀(P)', fontsize=10)
        # ax1.set_zlabel('log₁₀(Ratio)', fontsize=10)
        # ax1.set_title('Communication/Computation Ratio Surface', fontsize=12)
        
        # # Add colorbar
        # cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
        # cbar.set_label('log₁₀(T_comm/T_compute)')
        
        # # 2D plot: Crossover curve
        ax2 = fig.add_subplot(111)
        
        # Plot data points colored by ratio
        for dp in data_points:
            n, p, t_comp, t_comm, ratio = dp
            color = 'blue' if ratio < 1 else 'red'
            ax2.scatter(n, p, c=color, s=50, alpha=0.6)
        
        # Plot crossover points
        if crossover_points:
            cross_n = [cp[0] for cp in crossover_points]
            cross_p = [cp[1] for cp in crossover_points]
            ax2.plot(cross_n, cross_p, 'go-', markersize=10, linewidth=2, 
                    label='Crossover line')
            
            # Extrapolate and plot fitted curve
            if len(crossover_points) >= 2:
                a_cross, b_cross, _ = fit_power_law(cross_n, cross_p)
                if a_cross:
                    n_ext = np.linspace(min(sizes)*0.8, max(sizes)*1.2, 100)
                    p_ext = a_cross * n_ext**b_cross
                    ax2.plot(n_ext, p_ext, 'g--', alpha=0.5, 
                            label=f'Fit: P={a_cross:.1e}·n^{b_cross:.2f}')
        
        ax2.set_xlabel('Problem Size (n)', fontsize=12)
        ax2.set_ylabel('Number of Processes (P)', fontsize=12)
        # ax2.set_title('Crossover Boundary\n(Blue: compute-bound, Red: comm-bound)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'exp3_surface.png', dpi=150)
        fig.savefig(output_dir / 'exp3_surface.pdf')
        print(f"\nPlot saved to: {output_dir / 'exp3_surface.png'}")
        plt.close(fig)
        
        # Additional plot: Heatmap of ratio
        fig2, ax = plt.subplots(figsize=(10, 8))
        
        # Use imshow for heatmap
        im = ax.imshow(np.log10(ratio_matrix + 0.001), aspect='auto', 
                       cmap='RdYlGn_r', origin='lower',
                       extent=[np.log10(min(procs))-0.2, np.log10(max(procs))+0.2,
                               np.log10(min(sizes))-0.2, np.log10(max(sizes))+0.2])
        
        # Add contour at ratio = 1
        if not np.all(np.isnan(ratio_matrix)):
            try:
                P_grid, N_grid = np.meshgrid(np.log10(procs), np.log10(sizes))
                cs = ax.contour(P_grid, N_grid, np.log10(ratio_matrix + 0.001), 
                               levels=[0], colors='black', linewidths=2)
                ax.clabel(cs, inline=True, fmt='ratio=1', fontsize=10)
            except:
                pass
        
        ax.set_xlabel('log₁₀(P)', fontsize=12)
        ax.set_ylabel('log₁₀(n)', fontsize=12)
        # ax.set_title('Communication/Computation Ratio Heatmap\n(Green: compute-bound, Red: comm-bound)', fontsize=14)
        
        cbar = fig2.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(T_comm/T_compute)')
        
        # Add tick labels
        ax.set_xticks(np.log10(procs))
        ax.set_xticklabels(procs)
        ax.set_yticks(np.log10(sizes))
        ax.set_yticklabels(sizes)
        
        plt.tight_layout()
        fig2.savefig(output_dir / 'exp3_heatmap.png', dpi=150)
        fig2.savefig(output_dir / 'exp3_heatmap.pdf')
        print(f"Heatmap saved to: {output_dir / 'exp3_heatmap.png'}")
        plt.close(fig2)
    
    return {
        'sizes': sizes,
        'procs': procs,
        'ratio_matrix': ratio_matrix.tolist(),
        'crossover_points': crossover_points,
    }


def save_analysis(analysis1, analysis2, analysis3, output_dir):
    """Save analysis results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"crossover_analysis_{timestamp}.json"
    
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
            'surface_analysis': convert(analysis3),
            'timestamp': timestamp,
        }, f, indent=2)
    
    print(f"\nAnalysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze crossover experiments')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing experiment results')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Specific results file to analyze')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    
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
        return
    
    # Analyze each experiment
    exp1_results = data.get('experiment1_fixed_p', [])
    exp2_results = data.get('experiment2_fixed_size', [])
    exp3_results = data.get('experiment3_surface', [])
    
    analysis1 = analyze_fixed_p(exp1_results, results_dir)
    analysis2 = analyze_fixed_size(exp2_results, results_dir)
    analysis3 = analyze_surface(exp3_results, results_dir)
    
    # Save analysis
    save_analysis(analysis1, analysis2, analysis3, results_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if analysis1 and analysis1.get('crossover_n'):
        print(f"\n1. Fixed P=4:")
        print(f"   Crossover at n ≈ {analysis1['crossover_n']:.0f}")
    
    if analysis2 and analysis2.get('crossover_p'):
        print(f"\n2. Fixed 1000×1000:")
        print(f"   Crossover at P ≈ {analysis2['crossover_p']:.0f}")
    
    if analysis3 and analysis3.get('crossover_points'):
        print(f"\n3. Surface crossover points:")
        for n, p in analysis3['crossover_points']:
            print(f"   n={n}: P_crossover ≈ {p:.1f}")


if __name__ == '__main__':
    main()
