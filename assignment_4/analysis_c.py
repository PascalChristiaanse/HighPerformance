#!/usr/bin/env python3
"""
Analysis C: Generate speedup plots and tables

This script parses the output files from experiment_c and generates:
- Speedup comparison plots (with vs without memory copy)
- Detailed speedup tables
- Analysis of memory transfer overhead

Usage:
    python analysis_c.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
"""

import os
import re
import argparse
from pathlib import Path
import csv

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")


def parse_output_file(filepath):
    """Parse a SLURM output file and extract CSV data."""
    results = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Look for CSV_OUTPUT section
    csv_match = re.search(r'=== CSV_OUTPUT ===\n(.+?)\n(.+?)$', content, re.MULTILINE)
    if csv_match:
        headers = csv_match.group(1).strip().split(',')
        values = csv_match.group(2).strip().split(',')
        
        for header, value in zip(headers, values):
            header = header.strip()
            value = value.strip()
            try:
                if '.' in value:
                    results[header] = float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    results[header] = int(value)
                else:
                    results[header] = value
            except ValueError:
                results[header] = value
    
    return results if results else None


def collect_results(results_dir):
    """Collect all results from output files."""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    for output_file in results_path.glob("run_*.out"):
        data = parse_output_file(output_file)
        if data:
            data['source_file'] = output_file.name
            results.append(data)
            print(f"Parsed: {output_file.name}")
    
    return results


def create_speedup_table(results, output_dir):
    """Create detailed speedup comparison table."""
    if not results:
        print("No results to create table from.")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort results
    sorted_results = sorted(results, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    
    # Calculate additional metrics
    for r in sorted_results:
        cpu_time = r.get('cpu_time', 0)
        gpu_with_mem = r.get('gpu_time_with_memcpy', 0)
        gpu_no_mem = r.get('gpu_time_no_memcpy', 0)
        memcpy_time = r.get('memcpy_time', 0)
        
        # Memory overhead percentage
        if gpu_with_mem > 0:
            r['memcpy_overhead_pct'] = (memcpy_time / gpu_with_mem) * 100
        else:
            r['memcpy_overhead_pct'] = 0
        
        # Speedup difference
        r['speedup_difference'] = r.get('speedup_no_memcpy', 0) - r.get('speedup_with_memcpy', 0)
    
    # Write detailed CSV
    columns = [
        'matrix_size', 'block_size', 'cpu_time', 
        'gpu_time_with_memcpy', 'gpu_time_no_memcpy', 'memcpy_time',
        'speedup_with_memcpy', 'speedup_no_memcpy', 
        'speedup_difference', 'memcpy_overhead_pct'
    ]
    
    csv_path = output_path / "experiment_c_speedup.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in sorted_results:
            writer.writerow(row)
    print(f"Speedup CSV saved to: {csv_path}")
    
    # Write LaTeX table
    latex_path = output_path / "experiment_c_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GPU Speedup Analysis: With and Without Memory Copy}\n")
        f.write("\\label{tab:experiment_c}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("N & Block & Speedup (i) & Speedup (ii) & $\\Delta$ & Memcpy \\% \\\\\n")
        f.write(" &  & (no memcpy) & (with memcpy) & & overhead \\\\\n")
        f.write("\\hline\n")
        
        for row in sorted_results:
            f.write(f"{row.get('matrix_size', 'N/A')} & ")
            f.write(f"{row.get('block_size', 'N/A')} & ")
            f.write(f"{row.get('speedup_no_memcpy', 0):.2f} & ")
            f.write(f"{row.get('speedup_with_memcpy', 0):.2f} & ")
            f.write(f"{row.get('speedup_difference', 0):.2f} & ")
            f.write(f"{row.get('memcpy_overhead_pct', 0):.1f}\\%")
            f.write(" \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to: {latex_path}")
    
    return sorted_results


def create_speedup_plots(results, output_dir):
    """Create speedup visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not results:
        print("No results to plot.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Organize data
    by_block_size = {}
    for r in results:
        b = r.get('block_size')
        if b not in by_block_size:
            by_block_size[b] = []
        by_block_size[b].append(r)
    
    # Plot 1: Speedup comparison (with vs without memcpy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    block_sizes = sorted(by_block_size.keys())
    
    for bs in block_sizes:
        data = sorted(by_block_size[bs], key=lambda x: x.get('matrix_size', 0))
        sizes = [d['matrix_size'] for d in data]
        speedup_with = [d['speedup_with_memcpy'] for d in data]
        speedup_no = [d['speedup_no_memcpy'] for d in data]
        
        ax1.plot(sizes, speedup_no, 's-', label=f'Compute only (B={bs})', linewidth=2)
        ax1.plot(sizes, speedup_with, 'o--', label=f'With memcpy (B={bs})', alpha=0.7)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=12)
    ax1.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
    ax1.set_title('Speedup: Compute Only vs Including Memory Copy', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='No speedup')
    ax1.set_xscale('log')
    
    # Plot 2: Memory copy overhead
    ax2 = axes[1]
    
    for bs in block_sizes:
        data = sorted(by_block_size[bs], key=lambda x: x.get('matrix_size', 0))
        sizes = [d['matrix_size'] for d in data]
        overhead = [d.get('memcpy_overhead_pct', 0) for d in data]
        
        ax2.plot(sizes, overhead, 'o-', label=f'Block size={bs}', linewidth=2)
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=12)
    ax2.set_ylabel('Memory Copy Overhead (%)', fontsize=12)
    ax2.set_title('Memory Transfer Overhead as Percentage of Total GPU Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_speedup_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Speedup comparison plot saved to: {plot_path}")
    plt.close()
    
    # Plot 2: Bar chart comparing speedups
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    matrix_sizes = sorted(set(r['matrix_size'] for r in results))
    x = range(len(matrix_sizes))
    width = 0.35
    
    # Average speedups across block sizes for simplicity
    speedup_with_avg = []
    speedup_no_avg = []
    
    for n in matrix_sizes:
        data_for_n = [r for r in results if r['matrix_size'] == n]
        speedup_with_avg.append(sum(d['speedup_with_memcpy'] for d in data_for_n) / len(data_for_n))
        speedup_no_avg.append(sum(d['speedup_no_memcpy'] for d in data_for_n) / len(data_for_n))
    
    bars1 = ax.bar([i - width/2 for i in x], speedup_no_avg, width, 
                   label='(i) Excluding memory copy', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], speedup_with_avg, width, 
                   label='(ii) Including memory copy', color='coral')
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    ax.set_title('Speedup Comparison: Effect of Memory Copy (averaged over block sizes)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_speedup_bars.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Speedup bar chart saved to: {plot_path}")
    plt.close()


def generate_analysis_summary(results, output_dir):
    """Generate a text summary of the speedup analysis."""
    if not results:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_path / "experiment_c_analysis.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Experiment C: Speedup Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        speedup_with = [r['speedup_with_memcpy'] for r in results]
        speedup_no = [r['speedup_no_memcpy'] for r in results]
        memcpy_overhead = [r.get('memcpy_overhead_pct', 0) for r in results]
        
        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Speedup (excluding memcpy):\n")
        f.write(f"  Min: {min(speedup_no):.2f}x\n")
        f.write(f"  Max: {max(speedup_no):.2f}x\n")
        f.write(f"  Avg: {sum(speedup_no)/len(speedup_no):.2f}x\n\n")
        
        f.write(f"Speedup (including memcpy):\n")
        f.write(f"  Min: {min(speedup_with):.2f}x\n")
        f.write(f"  Max: {max(speedup_with):.2f}x\n")
        f.write(f"  Avg: {sum(speedup_with)/len(speedup_with):.2f}x\n\n")
        
        f.write(f"Memory copy overhead:\n")
        f.write(f"  Min: {min(memcpy_overhead):.1f}%\n")
        f.write(f"  Max: {max(memcpy_overhead):.1f}%\n")
        f.write(f"  Avg: {sum(memcpy_overhead)/len(memcpy_overhead):.1f}%\n\n")
        
        # Best configurations
        best_with = max(results, key=lambda x: x['speedup_with_memcpy'])
        best_no = max(results, key=lambda x: x['speedup_no_memcpy'])
        
        f.write("Best Configurations:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best speedup (including memcpy): {best_with['speedup_with_memcpy']:.2f}x\n")
        f.write(f"  Matrix size: {best_with['matrix_size']}, Block size: {best_with['block_size']}\n\n")
        
        f.write(f"Best speedup (excluding memcpy): {best_no['speedup_no_memcpy']:.2f}x\n")
        f.write(f"  Matrix size: {best_no['matrix_size']}, Block size: {best_no['block_size']}\n\n")
        
        # Observations
        f.write("Key Observations:\n")
        f.write("-" * 40 + "\n")
        f.write("1. The difference between speedup (i) and (ii) shows the impact\n")
        f.write("   of memory transfer overhead on overall performance.\n\n")
        f.write("2. For small matrices, memory copy overhead is typically higher\n")
        f.write("   relative to computation time.\n\n")
        f.write("3. As matrix size increases, the compute-to-transfer ratio\n")
        f.write("   improves, making GPU acceleration more beneficial.\n")
    
    print(f"Analysis summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment C results (speedup)')
    parser.add_argument('--results-dir', default='results/experiment_c',
                        help='Directory containing output files')
    parser.add_argument('--output-dir', default='analysis_output/experiment_c',
                        help='Directory for output plots and tables')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Analysis C: Speedup Analysis (with/without memory copy)")
    print("=" * 60)
    
    # Collect results
    print(f"\nCollecting results from: {args.results_dir}")
    results = collect_results(args.results_dir)
    
    if not results:
        print("\nNo results found. Make sure experiments have completed.")
        return
    
    print(f"\nCollected {len(results)} result files")
    
    # Create tables
    print("\nGenerating speedup tables...")
    results = create_speedup_table(results, args.output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    create_speedup_plots(results, args.output_dir)
    
    # Generate summary
    print("\nGenerating analysis summary...")
    generate_analysis_summary(results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
