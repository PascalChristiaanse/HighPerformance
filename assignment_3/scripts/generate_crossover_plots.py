#!/usr/bin/env python3
"""
Generate specific plots for crossover analysis report:
1. P=4 scatter plot showing compute vs comm times
2. Standalone heatmap (dot matrix) figure
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(results_dir):
    """Load the most recent crossover results."""
    import glob
    import os
    pattern = str(results_dir / "crossover_results_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No results found")
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    with open(latest, 'r') as f:
        return json.load(f)

def plot_fixed_p4(results, output_dir):
    """Create scatter plot for fixed P=4, varying problem size.
    
    Matches the plot style from analyze_crossover.py analyze_fixed_p().
    """
    exp3 = results.get('experiment3_surface', [])
    
    # Extract P=4 data
    data = []
    for r in exp3:
        if r['np'] == 4:
            n = r['dim_x']
            iterations = r.get('iterations', 1)
            t_compute = r.get('time_compute_avg', 0) / iterations
            t_comm = (r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)) / iterations
            data.append((n, t_compute, t_comm))
    
    if not data:
        print("No P=4 data found")
        return None
    
    data.sort(key=lambda x: x[0])
    sizes = np.array([d[0] for d in data])
    t_compute = np.array([d[1] for d in data])
    t_comm = np.array([d[2] for d in data])
    
    # Fit power laws (same as analyze_crossover.py)
    def fit_power_law(x, y):
        mask = (x > 0) & (y > 0)
        x_filt, y_filt = x[mask], y[mask]
        if len(x_filt) < 2:
            return None, None, None
        coeffs = np.polyfit(np.log(x_filt), np.log(y_filt), 1)
        b, a = coeffs[0], np.exp(coeffs[1])
        y_pred = a * x_filt**b
        ss_res = np.sum((y_filt - y_pred)**2)
        ss_tot = np.sum((y_filt - np.mean(y_filt))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return a, b, r2
    
    a_comp, b_comp, r2_comp = fit_power_law(sizes, t_compute)
    a_comm, b_comm, r2_comm = fit_power_law(sizes, t_comm)
    
    # Create plot (matching analyze_crossover.py style)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(sizes, t_compute, 'bo-', markersize=8, linewidth=2, label='Computation')
    ax.loglog(sizes, t_comm, 'rs-', markersize=8, linewidth=2, label='Communication')
    
    # Plot fitted lines
    n_range = np.logspace(np.log10(min(sizes)*0.5), np.log10(max(sizes)*2), 100)
    if a_comp:
        ax.loglog(n_range, a_comp * n_range**b_comp, 'b--', alpha=0.5,
                  label='Fit: {:.1e}·n^{:.2f}'.format(a_comp, b_comp))
    if a_comm:
        ax.loglog(n_range, a_comm * n_range**b_comm, 'r--', alpha=0.5,
                  label='Fit: {:.1e}·n^{:.2f}'.format(a_comm, b_comm))
    
    # Find and mark crossover point
    crossover_n = None
    if a_comp and a_comm and abs(b_comp - b_comm) > 0.01:
        crossover_n = (a_comm / a_comp) ** (1 / (b_comp - b_comm))
        if min(sizes)*0.5 < crossover_n < max(sizes)*2:
            ax.axvline(crossover_n, color='green', linestyle=':', linewidth=2,
                       label='Crossover: n≈{:.0f}'.format(crossover_n))
    
    ax.set_xlabel('Problem Size (n)', fontsize=12)
    ax.set_ylabel('Time per Iteration (s)', fontsize=12)
    ax.set_title('Experiment 1: Fixed P=4, Varying Problem Size', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'crossover_fixed_p4.png', dpi=150)
    fig.savefig(output_dir / 'crossover_fixed_p4.pdf')
    print(f"Saved: {output_dir / 'crossover_fixed_p4.png'}")
    plt.close(fig)
    
    return crossover_n

def plot_heatmap(results, output_dir):
    """Create standalone heatmap (dot matrix) figure."""
    exp3 = results.get('experiment3_surface', [])
    
    if not exp3:
        print("No experiment 3 data")
        return
    
    sizes = sorted(set(r['dim_x'] for r in exp3))
    procs = sorted(set(r['np'] for r in exp3))
    
    # Build ratio matrix
    ratio_matrix = np.full((len(sizes), len(procs)), np.nan)
    
    for r in exp3:
        i = sizes.index(r['dim_x'])
        j = procs.index(r['np'])
        iterations = r.get('iterations', 1)
        t_compute = r.get('time_compute_avg', 0) / iterations
        t_comm = (r.get('time_comm_neighbors_avg', 0) + r.get('time_comm_global_avg', 0)) / iterations
        if t_compute > 0:
            ratio_matrix[i, j] = t_comm / t_compute
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot as colored scatter (dot matrix)
    for i, n in enumerate(sizes):
        for j, p in enumerate(procs):
            ratio = ratio_matrix[i, j]
            if not np.isnan(ratio):
                # Color: green if compute-bound (ratio<1), red if comm-bound (ratio>1)
                if ratio < 0.5:
                    color = 'darkgreen'
                elif ratio < 1.0:
                    color = 'lightgreen'
                elif ratio < 2.0:
                    color = 'lightsalmon'
                else:
                    color = 'darkred'
                
                size = 300
                ax.scatter(j, i, c=color, s=size, edgecolors='black', linewidth=1)
                
                # Add ratio text
                fontsize = 8 if ratio < 10 else 7
                ax.text(j, i, f'{ratio:.2f}', ha='center', va='center', 
                        fontsize=fontsize, fontweight='bold',
                        color='white' if ratio < 0.5 or ratio > 2 else 'black')
    
    # Find and plot crossover line
    crossover_points = []
    for i, n in enumerate(sizes):
        ratios = ratio_matrix[i, :]
        for k in range(len(procs) - 1):
            if not np.isnan(ratios[k]) and not np.isnan(ratios[k+1]):
                if ratios[k] < 1 and ratios[k+1] >= 1:
                    # Interpolate
                    frac = (1 - ratios[k]) / (ratios[k+1] - ratios[k])
                    cross_j = k + frac
                    crossover_points.append((cross_j, i))
                    break
    
    if crossover_points:
        cross_x = [cp[0] for cp in crossover_points]
        cross_y = [cp[1] for cp in crossover_points]
        ax.plot(cross_x, cross_y, 'k-', linewidth=3, label='Crossover ($T_{comm}=T_{comp}$)')
        ax.plot(cross_x, cross_y, 'yo', markersize=10)
    
    ax.set_xticks(range(len(procs)))
    ax.set_xticklabels(procs)
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels(sizes)
    ax.set_xlabel('Number of Processes $P$', fontsize=12)
    ax.set_ylabel('Problem Size $n$', fontsize=12)
    ax.set_title('Communication/Computation Ratio ($T_{comm}/T_{comp}$)\nGreen: compute-bound, Red: comm-bound', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', edgecolor='black', label='Ratio < 0.5'),
        Patch(facecolor='lightgreen', edgecolor='black', label='0.5 ≤ Ratio < 1'),
        Patch(facecolor='lightsalmon', edgecolor='black', label='1 ≤ Ratio < 2'),
        Patch(facecolor='darkred', edgecolor='black', label='Ratio ≥ 2'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_xlim(-0.5, len(procs) - 0.5)
    ax.set_ylim(-0.5, len(sizes) - 0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'crossover_dotmatrix.png', dpi=150)
    fig.savefig(output_dir / 'crossover_dotmatrix.pdf')
    print(f"Saved: {output_dir / 'crossover_dotmatrix.png'}")
    plt.close(fig)

def main():
    results_dir = Path('results')
    results = load_results(results_dir)
    
    print("\nGenerating plots...")
    plot_fixed_p4(results, results_dir)
    plot_heatmap(results, results_dir)
    print("\nDone!")

if __name__ == '__main__':
    main()
