#!/usr/bin/env python3
"""
Plot benchmark results: CG vs SOR comparison (Exercise 1.2.13)

Analyzes:
1. Time per iteration for both algorithms
2. Total iterations to convergence (CG expected: ~125)
3. Total solve time comparison
4. Scaling with number of MPI processes

Expected findings:
- CG has more expensive iterations (2 global dot products, more memory)
- CG requires no omega tuning and no red-black bookkeeping
- CG should converge in ~125 iterations for the test problem
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path('benchmark_results/cg_comparison')
RESULTS_FILE = RESULTS_DIR / 'cg_comparison_results.csv'
OUTPUT_DIR = RESULTS_DIR

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and preprocess the benchmark data."""
    df = pd.read_csv(RESULTS_FILE)
    
    # Filter out any error rows
    df = df[~df['iterations'].isin(['ERROR', 'FAILED'])].copy()
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['total_time_s'] = pd.to_numeric(df['total_time_s'])
    df['time_per_iter_us'] = pd.to_numeric(df['time_per_iter_us'])
    
    return df


def analyze_data(df):
    """Compute summary statistics grouped by solver and process count."""
    summary = df.groupby(['solver', 'num_procs']).agg({
        'iterations': ['mean', 'std', 'min', 'max'],
        'total_time_s': ['mean', 'std'],
        'time_per_iter_us': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    summary.columns = [
        'solver', 'num_procs',
        'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max',
        'total_time_mean', 'total_time_std',
        'time_per_iter_mean', 'time_per_iter_std'
    ]
    
    # Fill NaN std with 0
    summary = summary.fillna(0)
    
    return summary


def print_analysis(df, summary):
    """Print detailed analysis to console."""
    print("=" * 70)
    print("CG vs SOR Comparison Analysis (Exercise 1.2.13)")
    print("=" * 70)
    
    # Check CG iteration count
    cg_data = summary[summary['solver'] == 'cg']
    sor_data = summary[summary['solver'] == 'sor']
    
    print("\n" + "─" * 70)
    print("VERIFICATION: CG Iteration Count")
    print("─" * 70)
    print("Expected CG iterations for test problem: ~125")
    print("\nActual CG iterations:")
    for _, row in cg_data.iterrows():
        print(f"  {int(row['num_procs'])} procs: {row['iterations_mean']:.0f} iterations "
              f"(min={row['iterations_min']:.0f}, max={row['iterations_max']:.0f})")
    
    cg_avg_iter = cg_data['iterations_mean'].mean()
    if 120 <= cg_avg_iter <= 130:
        print(f"\n✓ CG iteration count ({cg_avg_iter:.0f}) matches expected ~125")
    else:
        print(f"\n⚠ CG iteration count ({cg_avg_iter:.0f}) differs from expected ~125")
    
    print("\n" + "─" * 70)
    print("TIME PER ITERATION COMPARISON")
    print("─" * 70)
    print("\n{:>8} {:>12} {:>14} {:>14} {:>12}".format(
        "Procs", "CG (μs)", "SOR (μs)", "CG/SOR Ratio", "CG Overhead"))
    print("-" * 62)
    
    for nprocs in sorted(cg_data['num_procs'].unique()):
        cg_row = cg_data[cg_data['num_procs'] == nprocs].iloc[0]
        sor_row = sor_data[sor_data['num_procs'] == nprocs].iloc[0]
        
        ratio = cg_row['time_per_iter_mean'] / sor_row['time_per_iter_mean']
        overhead = (ratio - 1) * 100
        
        print("{:>8d} {:>12.1f} {:>14.1f} {:>14.2f}x {:>11.1f}%".format(
            int(nprocs),
            cg_row['time_per_iter_mean'],
            sor_row['time_per_iter_mean'],
            ratio,
            overhead
        ))
    
    print("\n" + "─" * 70)
    print("TOTAL TIME TO CONVERGENCE")
    print("─" * 70)
    print("\n{:>8} {:>12} {:>14} {:>12} {:>14}".format(
        "Procs", "CG Time (s)", "SOR Time (s)", "CG Iters", "SOR Iters"))
    print("-" * 62)
    
    for nprocs in sorted(cg_data['num_procs'].unique()):
        cg_row = cg_data[cg_data['num_procs'] == nprocs].iloc[0]
        sor_row = sor_data[sor_data['num_procs'] == nprocs].iloc[0]
        
        print("{:>8d} {:>12.4f} {:>14.4f} {:>12.0f} {:>14.0f}".format(
            int(nprocs),
            cg_row['total_time_mean'],
            sor_row['total_time_mean'],
            cg_row['iterations_mean'],
            sor_row['iterations_mean']
        ))
    
    # Analysis of advantages
    print("\n" + "─" * 70)
    print("ANALYSIS SUMMARY")
    print("─" * 70)
    
    avg_cg_time_per_iter = cg_data['time_per_iter_mean'].mean()
    avg_sor_time_per_iter = sor_data['time_per_iter_mean'].mean()
    avg_ratio = avg_cg_time_per_iter / avg_sor_time_per_iter
    
    print(f"""
CG Algorithm Characteristics:
  • Requires 2 global dot products per iteration (MPI_Allreduce)
  • More memory usage (stores p, r, v vectors in addition to solution)
  • No relaxation parameter ω to tune
  • No red-black parity bookkeeping needed
  • Simpler implementation without color ordering

Performance Comparison:
  • Average CG time per iteration: {avg_cg_time_per_iter:.1f} μs
  • Average SOR time per iteration: {avg_sor_time_per_iter:.1f} μs  
  • CG is {avg_ratio:.2f}x slower per iteration
  
  • CG iterations: ~{cg_avg_iter:.0f}
  • SOR iterations: ~{sor_data['iterations_mean'].mean():.0f}
""")
    
    return cg_data, sor_data


def plot_comparison(df, summary, output_dir):
    """Generate comparison plots."""
    cg_data = summary[summary['solver'] == 'cg']
    sor_data = summary[summary['solver'] == 'sor']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme
    cg_color = '#2ecc71'  # Green for CG
    sor_color = '#3498db'  # Blue for SOR
    
    procs = sorted(cg_data['num_procs'].unique())
    x = np.arange(len(procs))
    width = 0.35
    
    # Plot 1: Time per iteration comparison
    ax1 = axes[0, 0]
    cg_times = [cg_data[cg_data['num_procs'] == p]['time_per_iter_mean'].values[0] for p in procs]
    cg_errs = [cg_data[cg_data['num_procs'] == p]['time_per_iter_std'].values[0] for p in procs]
    sor_times = [sor_data[sor_data['num_procs'] == p]['time_per_iter_mean'].values[0] for p in procs]
    sor_errs = [sor_data[sor_data['num_procs'] == p]['time_per_iter_std'].values[0] for p in procs]
    
    bars1 = ax1.bar(x - width/2, cg_times, width, label='CG', color=cg_color, 
                    yerr=cg_errs, capsize=5, alpha=0.8)
    bars2 = ax1.bar(x + width/2, sor_times, width, label='SOR (optimized)', color=sor_color,
                    yerr=sor_errs, capsize=5, alpha=0.8)
    
    ax1.set_xlabel('Number of MPI Processes', fontsize=11)
    ax1.set_ylabel('Time per Iteration (μs)', fontsize=11)
    ax1.set_title('Time per Iteration: CG vs Optimized SOR', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(procs)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (cg_t, sor_t) in enumerate(zip(cg_times, sor_times)):
        ratio = cg_t / sor_t
        ax1.annotate(f'{ratio:.2f}x', xy=(x[i], max(cg_t, sor_t) + max(cg_errs[i], sor_errs[i])),
                    ha='center', fontsize=9, color='#e74c3c')
    
    # Plot 2: Total iterations comparison
    ax2 = axes[0, 1]
    cg_iters = [cg_data[cg_data['num_procs'] == p]['iterations_mean'].values[0] for p in procs]
    cg_iter_errs = [cg_data[cg_data['num_procs'] == p]['iterations_std'].values[0] for p in procs]
    sor_iters = [sor_data[sor_data['num_procs'] == p]['iterations_mean'].values[0] for p in procs]
    sor_iter_errs = [sor_data[sor_data['num_procs'] == p]['iterations_std'].values[0] for p in procs]
    
    bars1 = ax2.bar(x - width/2, cg_iters, width, label='CG', color=cg_color,
                    yerr=cg_iter_errs, capsize=5, alpha=0.8)
    bars2 = ax2.bar(x + width/2, sor_iters, width, label='SOR (optimized)', color=sor_color,
                    yerr=sor_iter_errs, capsize=5, alpha=0.8)
    
    # Add expected CG iteration line
    ax2.axhline(y=125, color='#e74c3c', linestyle='--', linewidth=2, 
                label='Expected CG (~125)')
    
    ax2.set_xlabel('Number of MPI Processes', fontsize=11)
    ax2.set_ylabel('Iterations to Convergence', fontsize=11)
    ax2.set_title('Iterations to Convergence: CG vs SOR', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(procs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Total time comparison
    ax3 = axes[1, 0]
    cg_total = [cg_data[cg_data['num_procs'] == p]['total_time_mean'].values[0] for p in procs]
    cg_total_errs = [cg_data[cg_data['num_procs'] == p]['total_time_std'].values[0] for p in procs]
    sor_total = [sor_data[sor_data['num_procs'] == p]['total_time_mean'].values[0] for p in procs]
    sor_total_errs = [sor_data[sor_data['num_procs'] == p]['total_time_std'].values[0] for p in procs]
    
    bars1 = ax3.bar(x - width/2, cg_total, width, label='CG', color=cg_color,
                    yerr=cg_total_errs, capsize=5, alpha=0.8)
    bars2 = ax3.bar(x + width/2, sor_total, width, label='SOR (optimized)', color=sor_color,
                    yerr=sor_total_errs, capsize=5, alpha=0.8)
    
    ax3.set_xlabel('Number of MPI Processes', fontsize=11)
    ax3.set_ylabel('Total Time to Convergence (s)', fontsize=11)
    ax3.set_title('Total Solve Time: CG vs Optimized SOR', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(procs)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i, (cg_t, sor_t) in enumerate(zip(cg_total, sor_total)):
        if cg_t < sor_t:
            speedup = sor_t / cg_t
            label = f'CG {speedup:.1f}x faster'
        else:
            speedup = cg_t / sor_t
            label = f'SOR {speedup:.1f}x faster'
        ax3.annotate(label, xy=(x[i], max(cg_t, sor_t) + max(cg_total_errs[i], sor_total_errs[i]) + 0.01),
                    ha='center', fontsize=8, color='#333')
    
    # Plot 4: Ratio analysis (CG overhead)
    ax4 = axes[1, 1]
    time_ratios = [cg/sor for cg, sor in zip(cg_times, sor_times)]
    iter_ratios = [sor/cg for cg, sor in zip(cg_iters, sor_iters)]  # Inverted - fewer CG iterations is better
    total_ratios = [cg/sor for cg, sor in zip(cg_total, sor_total)]
    
    ax4.plot(procs, time_ratios, 'o-', label='Time per iter ratio (CG/SOR)', 
             color='#e74c3c', linewidth=2, markersize=8)
    ax4.plot(procs, iter_ratios, 's-', label='Iteration ratio (SOR/CG)', 
             color='#9b59b6', linewidth=2, markersize=8)
    ax4.plot(procs, total_ratios, '^-', label='Total time ratio (CG/SOR)', 
             color='#f39c12', linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax4.set_xlabel('Number of MPI Processes', fontsize=11)
    ax4.set_ylabel('Ratio', fontsize=11)
    ax4.set_title('Performance Ratios (CG vs SOR)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cg_comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cg_comparison_plot.pdf', bbox_inches='tight')
    print(f"\nPlots saved to {output_dir / 'cg_comparison_plot.png'}")
    plt.show()


def generate_latex_table(summary, output_dir):
    """Generate a LaTeX table for the report."""
    cg_data = summary[summary['solver'] == 'cg']
    sor_data = summary[summary['solver'] == 'sor']
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of Conjugate Gradient (CG) and optimized SOR solvers for the test problem (100×100 grid)}
\label{tab:cg_comparison}
\begin{tabular}{r|rr|rr|rr}
\toprule
& \multicolumn{2}{c|}{Time/Iter ($\mu$s)} & \multicolumn{2}{c|}{Iterations} & \multicolumn{2}{c}{Total Time (s)} \\
Procs & CG & SOR & CG & SOR & CG & SOR \\
\midrule
"""
    
    for nprocs in sorted(cg_data['num_procs'].unique()):
        cg_row = cg_data[cg_data['num_procs'] == nprocs].iloc[0]
        sor_row = sor_data[sor_data['num_procs'] == nprocs].iloc[0]
        
        latex += f"{int(nprocs)} & {cg_row['time_per_iter_mean']:.1f} & {sor_row['time_per_iter_mean']:.1f} "
        latex += f"& {cg_row['iterations_mean']:.0f} & {sor_row['iterations_mean']:.0f} "
        latex += f"& {cg_row['total_time_mean']:.4f} & {sor_row['total_time_mean']:.4f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    table_file = output_dir / 'cg_comparison_table.tex'
    with open(table_file, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to {table_file}")
    return latex


def main():
    """Main function to run the analysis."""
    # Check if results file exists
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found at {RESULTS_FILE}")
        print("Please run benchmark_cg_comparison.sh first.")
        return
    
    # Load and analyze data
    df = load_data()
    summary = analyze_data(df)
    
    # Print analysis
    print_analysis(df, summary)
    
    # Generate plots
    plot_comparison(df, summary, OUTPUT_DIR)
    
    # Generate LaTeX table
    latex = generate_latex_table(summary, OUTPUT_DIR)
    print("\n" + "─" * 70)
    print("LATEX TABLE")
    print("─" * 70)
    print(latex)


if __name__ == '__main__':
    main()
