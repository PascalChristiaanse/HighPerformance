#!/usr/bin/env python3
"""
Analyze adaptive grid experiments.

Compares uniform vs adaptive grids in terms of:
1. Number of iterations to convergence
2. Computation time per iteration
3. Total time to convergence
4. Convergence rate (iterations vs problem size scaling)

Creates comparison plots and tables for the report.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

# Plotting availability
HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    print("Warning: matplotlib not available. Plotting disabled.")


def load_latest_results(results_dir):
    """Load the most recent adaptive results file."""
    pattern = str(results_dir / "adaptive_results_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No adaptive results found in {}".format(results_dir))
    
    latest = max(files, key=os.path.getctime)
    print("Loading results from: {}".format(latest))
    
    with open(latest, 'r') as f:
        return json.load(f)


def organize_results(results):
    """Organize results by grid size and processor count."""
    organized = {}
    
    for r in results:
        size = r['dim_x']
        np_count = r['np']
        adaptive = r.get('adaptive', False)
        grid_type = 'adaptive' if adaptive else 'uniform'
        
        key = (size, np_count)
        if key not in organized:
            organized[key] = {}
        organized[key][grid_type] = r
    
    return organized


def analyze_convergence(results, output_dir):
    """Analyze and compare convergence behavior."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS: Uniform vs Adaptive Grids")
    print("=" * 70)
    
    organized = organize_results(results)
    
    # Get unique sizes and process counts
    sizes = sorted(set(r['dim_x'] for r in results))
    procs = sorted(set(r['np'] for r in results))
    
    # Print comparison table
    print("\n" + "-" * 70)
    print("Iterations to Convergence")
    print("-" * 70)
    
    col_header = "Size"
    header = "{:<10}".format(col_header)
    for p in procs:
        header += "  P={:<6} (U/A)".format(p)
    print(header)
    print("-" * 70)
    
    # Store data for plotting
    iter_data = {'sizes': sizes, 'procs': procs, 'uniform': {}, 'adaptive': {}}
    time_data = {'sizes': sizes, 'procs': procs, 'uniform': {}, 'adaptive': {}}
    
    for size in sizes:
        row = "{:<10}".format("{}x{}".format(size, size))
        for p in procs:
            key = (size, p)
            if key in organized:
                u = organized[key].get('uniform', {})
                a = organized[key].get('adaptive', {})
                iter_u = u.get('iterations', 0)
                iter_a = a.get('iterations', 0)
                
                if iter_u > 0 and iter_a > 0:
                    ratio = iter_a / iter_u
                    row += "  {:>4}/{:<4} ({:.2f})".format(iter_u, iter_a, ratio)
                elif iter_u > 0:
                    row += "  {:>4}/---".format(iter_u)
                elif iter_a > 0:
                    row += "  ---/{:<4}".format(iter_a)
                else:
                    row += "  ---/---"
                
                # Store for plotting
                if p not in iter_data['uniform']:
                    iter_data['uniform'][p] = []
                    iter_data['adaptive'][p] = []
                    time_data['uniform'][p] = []
                    time_data['adaptive'][p] = []
                
                iter_data['uniform'][p].append(iter_u if iter_u > 0 else np.nan)
                iter_data['adaptive'][p].append(iter_a if iter_a > 0 else np.nan)
                
                time_u = u.get('time_total', 0)
                time_a = a.get('time_total', 0)
                time_data['uniform'][p].append(time_u if time_u > 0 else np.nan)
                time_data['adaptive'][p].append(time_a if time_a > 0 else np.nan)
            else:
                row += "  ---/---"
        print(row)
    
    # Print timing comparison
    print("\n" + "-" * 70)
    print("Total Time to Convergence (seconds)")
    print("-" * 70)
    
    header = "{:<10}".format("Size")
    for p in procs:
        header += "  P={:<6} (U/A)".format(p)
    print(header)
    print("-" * 70)
    
    for size in sizes:
        row = "{:<10}".format("{}x{}".format(size, size))
        for p in procs:
            key = (size, p)
            if key in organized:
                u = organized[key].get('uniform', {})
                a = organized[key].get('adaptive', {})
                time_u = u.get('time_total', 0)
                time_a = a.get('time_total', 0)
                
                if time_u > 0 and time_a > 0:
                    ratio = time_a / time_u
                    row += "  {:>5.2f}/{:<5.2f} ({:.2f})".format(time_u, time_a, ratio)
                elif time_u > 0:
                    row += "  {:>5.2f}/---".format(time_u)
                elif time_a > 0:
                    row += "  ---/{:<5.2f}".format(time_a)
                else:
                    row += "  ---/---"
            else:
                row += "  ---/---"
        print(row)
    
    # Print compute time per iteration
    print("\n" + "-" * 70)
    print("Compute Time per Iteration (seconds)")
    print("-" * 70)
    
    header = "{:<10}".format("Size")
    for p in procs:
        header += "  P={:<6} (U/A)".format(p)
    print(header)
    print("-" * 70)
    
    for size in sizes:
        row = "{:<10}".format("{}x{}".format(size, size))
        for p in procs:
            key = (size, p)
            if key in organized:
                u = organized[key].get('uniform', {})
                a = organized[key].get('adaptive', {})
                
                iter_u = u.get('iterations', 1)
                iter_a = a.get('iterations', 1)
                t_comp_u = u.get('time_compute_avg', 0) / max(iter_u, 1)
                t_comp_a = a.get('time_compute_avg', 0) / max(iter_a, 1)
                
                if t_comp_u > 0 and t_comp_a > 0:
                    ratio = t_comp_a / t_comp_u
                    row += "  {:.4f}/{:.4f} ({:.2f})".format(t_comp_u, t_comp_a, ratio)
                else:
                    row += "  ---/---"
            else:
                row += "  ---/---"
        print(row)
    
    return iter_data, time_data


def compute_speedup_summary(results):
    """Compute summary statistics for speedup."""
    print("\n" + "=" * 70)
    print("SPEEDUP SUMMARY")
    print("=" * 70)
    
    organized = organize_results(results)
    
    iter_ratios = []
    time_ratios = []
    
    for key, data in organized.items():
        u = data.get('uniform', {})
        a = data.get('adaptive', {})
        
        iter_u = u.get('iterations', 0)
        iter_a = a.get('iterations', 0)
        time_u = u.get('time_total', 0)
        time_a = a.get('time_total', 0)
        
        if iter_u > 0 and iter_a > 0:
            iter_ratios.append(iter_a / iter_u)
        if time_u > 0 and time_a > 0:
            time_ratios.append(time_a / time_u)
    
    if iter_ratios:
        print("\nIteration count (adaptive/uniform):")
        print("  Mean ratio:   {:.3f}".format(np.mean(iter_ratios)))
        print("  Median ratio: {:.3f}".format(np.median(iter_ratios)))
        print("  Min ratio:    {:.3f}".format(np.min(iter_ratios)))
        print("  Max ratio:    {:.3f}".format(np.max(iter_ratios)))
        
        if np.mean(iter_ratios) < 1:
            print("  -> Adaptive grids converge FASTER (fewer iterations)")
        else:
            print("  -> Adaptive grids converge SLOWER (more iterations)")
    
    if time_ratios:
        print("\nTotal time (adaptive/uniform):")
        print("  Mean ratio:   {:.3f}".format(np.mean(time_ratios)))
        print("  Median ratio: {:.3f}".format(np.median(time_ratios)))
        print("  Min ratio:    {:.3f}".format(np.min(time_ratios)))
        print("  Max ratio:    {:.3f}".format(np.max(time_ratios)))
        
        if np.mean(time_ratios) < 1:
            print("  -> Adaptive grids are FASTER overall")
        else:
            print("  -> Adaptive grids are SLOWER overall")
    
    return iter_ratios, time_ratios


def plot_comparison(iter_data, time_data, output_dir):
    """Create comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Plotting skipped (matplotlib not available)")
        return
    
    sizes = iter_data['sizes']
    procs = iter_data['procs']
    
    # Plot 1: Iterations vs problem size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Iterations
    ax1 = axes[0]
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors_u = plt.cm.Blues(np.linspace(0.4, 0.9, len(procs)))
    colors_a = plt.cm.Reds(np.linspace(0.4, 0.9, len(procs)))
    
    for i, p in enumerate(procs):
        if p in iter_data['uniform']:
            u_iters = iter_data['uniform'][p]
            a_iters = iter_data['adaptive'][p]
            
            ax1.plot(sizes, u_iters, marker=markers[i % len(markers)], 
                     color=colors_u[i], linestyle='-', linewidth=2, markersize=8,
                     label='Uniform P={}'.format(p))
            ax1.plot(sizes, a_iters, marker=markers[i % len(markers)],
                     color=colors_a[i], linestyle='--', linewidth=2, markersize=8,
                     label='Adaptive P={}'.format(p))
    
    ax1.set_xlabel('Grid Size (n)', fontsize=12)
    ax1.set_ylabel('Iterations to Convergence', fontsize=12)
    ax1.set_title('Convergence Speed: Uniform vs Adaptive', fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Right: Total time
    ax2 = axes[1]
    for i, p in enumerate(procs):
        if p in time_data['uniform']:
            u_time = time_data['uniform'][p]
            a_time = time_data['adaptive'][p]
            
            ax2.plot(sizes, u_time, marker=markers[i % len(markers)],
                     color=colors_u[i], linestyle='-', linewidth=2, markersize=8,
                     label='Uniform P={}'.format(p))
            ax2.plot(sizes, a_time, marker=markers[i % len(markers)],
                     color=colors_a[i], linestyle='--', linewidth=2, markersize=8,
                     label='Adaptive P={}'.format(p))
    
    ax2.set_xlabel('Grid Size (n)', fontsize=12)
    ax2.set_ylabel('Total Time (s)', fontsize=12)
    ax2.set_title('Total Time: Uniform vs Adaptive', fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'adaptive_comparison.png', dpi=150)
    fig.savefig(output_dir / 'adaptive_comparison.pdf')
    print("\nSaved: {}".format(output_dir / 'adaptive_comparison.png'))
    plt.close(fig)
    
    # Plot 2: Ratio plot (adaptive/uniform)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Iteration ratio
    ax3 = axes2[0]
    for i, p in enumerate(procs):
        if p in iter_data['uniform']:
            u_iters = np.array(iter_data['uniform'][p])
            a_iters = np.array(iter_data['adaptive'][p])
            ratio = a_iters / u_iters
            
            ax3.plot(sizes, ratio, marker=markers[i % len(markers)],
                     linewidth=2, markersize=8, label='P={}'.format(p))
    
    ax3.axhline(y=1.0, color='black', linestyle=':', linewidth=2, label='Equal')
    ax3.set_xlabel('Grid Size (n)', fontsize=12)
    ax3.set_ylabel('Iteration Ratio (Adaptive/Uniform)', fontsize=12)
    ax3.set_title('Convergence Speed Ratio', fontsize=14)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Right: Time ratio
    ax4 = axes2[1]
    for i, p in enumerate(procs):
        if p in time_data['uniform']:
            u_time = np.array(time_data['uniform'][p])
            a_time = np.array(time_data['adaptive'][p])
            ratio = a_time / u_time
            
            ax4.plot(sizes, ratio, marker=markers[i % len(markers)],
                     linewidth=2, markersize=8, label='P={}'.format(p))
    
    ax4.axhline(y=1.0, color='black', linestyle=':', linewidth=2, label='Equal')
    ax4.set_xlabel('Grid Size (n)', fontsize=12)
    ax4.set_ylabel('Time Ratio (Adaptive/Uniform)', fontsize=12)
    ax4.set_title('Total Time Ratio', fontsize=14)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'adaptive_ratio.png', dpi=150)
    fig2.savefig(output_dir / 'adaptive_ratio.pdf')
    print("Saved: {}".format(output_dir / 'adaptive_ratio.png'))
    plt.close(fig2)


def save_analysis(iter_data, time_data, iter_ratios, time_ratios, output_dir):
    """Save analysis results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / "adaptive_analysis_{}.json".format(timestamp)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    analysis = {
        'iter_data': convert(iter_data),
        'time_data': convert(time_data),
        'iter_ratios': convert(iter_ratios),
        'time_ratios': convert(time_ratios),
        'summary': {
            'mean_iter_ratio': float(np.mean(iter_ratios)) if iter_ratios else None,
            'mean_time_ratio': float(np.mean(time_ratios)) if time_ratios else None,
        },
        'timestamp': timestamp,
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\nAnalysis saved to: {}".format(output_file))


def main():
    parser = argparse.ArgumentParser(description='Analyze adaptive grid experiments')
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
        print("Error: {}".format(e))
        print("Run 'python scripts/run_adaptive_experiments.py' first to generate data")
        return
    
    results = data.get('results', [])
    if not results:
        print("No results found in data file")
        return
    
    # Analyze
    iter_data, time_data = analyze_convergence(results, results_dir)
    iter_ratios, time_ratios = compute_speedup_summary(results)
    
    # Plot
    plot_comparison(iter_data, time_data, results_dir)
    
    # Save
    save_analysis(iter_data, time_data, iter_ratios, time_ratios, results_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    if iter_ratios:
        mean_iter = np.mean(iter_ratios)
        if mean_iter < 0.9:
            print("\nAdaptive grids show FASTER convergence ({:.1f}% fewer iterations)".format(
                (1 - mean_iter) * 100))
        elif mean_iter > 1.1:
            print("\nAdaptive grids show SLOWER convergence ({:.1f}% more iterations)".format(
                (mean_iter - 1) * 100))
        else:
            print("\nAdaptive grids show SIMILAR convergence (within 10%)")
    
    if time_ratios:
        mean_time = np.mean(time_ratios)
        if mean_time < 0.9:
            print("Adaptive grids are FASTER overall ({:.1f}% less time)".format(
                (1 - mean_time) * 100))
        elif mean_time > 1.1:
            print("Adaptive grids are SLOWER overall ({:.1f}% more time)".format(
                (mean_time - 1) * 100))
        else:
            print("Adaptive grids have SIMILAR total time (within 10%)")


if __name__ == '__main__':
    main()
