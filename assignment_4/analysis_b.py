#!/usr/bin/env python3
"""
Analysis B: Generate plots and tables for execution time experiments

This script parses the output files from experiment_b and generates:
- Execution time vs matrix size plots (for each block size)
- Execution time vs block size plots (for each matrix size)
- Summary tables in CSV and LaTeX format

Usage:
    python analysis_b.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
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
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Using basic CSV handling.")


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
                # Try to convert to appropriate type
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
        else:
            print(f"Warning: Could not parse {output_file.name}")
    
    return results


def create_summary_table(results, output_dir):
    """Create summary tables in CSV and LaTeX format."""
    if not results:
        print("No results to create table from.")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define columns for the table
    columns = [
        'matrix_size', 'block_size', 'cpu_time', 
        'gpu_time_with_memcpy', 'gpu_time_no_memcpy',
        'speedup_with_memcpy', 'speedup_no_memcpy'
    ]
    
    # Sort results by matrix_size, then block_size
    sorted_results = sorted(results, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    
    # Write CSV
    csv_path = output_path / "experiment_b_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in sorted_results:
            writer.writerow(row)
    print(f"CSV table saved to: {csv_path}")
    
    # Write LaTeX table
    latex_path = output_path / "experiment_b_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Execution Times for Different Matrix Sizes and Block Sizes}\n")
        f.write("\\label{tab:experiment_b}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("N & Block & CPU (s) & GPU+mem (s) & GPU only (s) & Speedup (mem) & Speedup (no mem) \\\\\n")
        f.write("\\hline\n")
        
        for row in sorted_results:
            f.write(f"{row.get('matrix_size', 'N/A')} & ")
            f.write(f"{row.get('block_size', 'N/A')} & ")
            f.write(f"{row.get('cpu_time', 'N/A'):.4f} & " if isinstance(row.get('cpu_time'), float) else "N/A & ")
            f.write(f"{row.get('gpu_time_with_memcpy', 'N/A'):.4f} & " if isinstance(row.get('gpu_time_with_memcpy'), float) else "N/A & ")
            f.write(f"{row.get('gpu_time_no_memcpy', 'N/A'):.4f} & " if isinstance(row.get('gpu_time_no_memcpy'), float) else "N/A & ")
            f.write(f"{row.get('speedup_with_memcpy', 'N/A'):.2f} & " if isinstance(row.get('speedup_with_memcpy'), float) else "N/A & ")
            f.write(f"{row.get('speedup_no_memcpy', 'N/A'):.2f}" if isinstance(row.get('speedup_no_memcpy'), float) else "N/A")
            f.write(" \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to: {latex_path}")
    
    return sorted_results


def create_plots(results, output_dir):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not results:
        print("No results to plot.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Organize data by matrix size and block size
    by_matrix_size = {}
    by_block_size = {}
    
    for r in results:
        n = r.get('matrix_size')
        b = r.get('block_size')
        
        if n not in by_matrix_size:
            by_matrix_size[n] = []
        by_matrix_size[n].append(r)
        
        if b not in by_block_size:
            by_block_size[b] = []
        by_block_size[b].append(r)
    
    # Quarter A4 page size: A4 is 8.27" x 11.69", quarter is approximately 4.13" x 2.92"
    # For nice text fitting, use a slightly adjusted size
    figsize = (4.13, 4.13)
    
    block_sizes = sorted(by_block_size.keys())
    
    # Plot 1: Execution time vs Matrix Size (CPU vs GPU time)
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    for bs in block_sizes:
        data = sorted(by_block_size[bs], key=lambda x: x.get('matrix_size', 0))
        sizes = [d['matrix_size'] for d in data]
        cpu_times = [d['cpu_time'] for d in data]
        gpu_times = [d['gpu_time_with_memcpy'] for d in data]
        
        ax1.plot(sizes, cpu_times, 'o--', label=f'CPU (block={bs})', alpha=0.7)
        ax1.plot(sizes, gpu_times, 's-', label=f'GPU (block={bs})')
    
    ax1.set_xlabel('Matrix Size (N)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_exec_time_vs_matrix_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Plot 2: Speedup vs Matrix Size
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    for bs in block_sizes:
        data = sorted(by_block_size[bs], key=lambda x: x.get('matrix_size', 0))
        sizes = [d['matrix_size'] for d in data]
        speedup_mem = [d['speedup_with_memcpy'] for d in data]
        speedup_no_mem = [d['speedup_no_memcpy'] for d in data]
        
        ax2.plot(sizes, speedup_mem, 'o--', label=f'With memcpy (block={bs})', alpha=0.7)
        ax2.plot(sizes, speedup_no_mem, 's-', label=f'Compute only (block={bs})')
    
    ax2.set_xlabel('Matrix Size (N)')
    ax2.set_ylabel('Speedup (CPU time / GPU time)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle=':', label='No speedup')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_speedup_vs_matrix_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Plot 3: GPU Time vs Block Size (grouped by matrix size)
    fig3, ax3 = plt.subplots(figsize=figsize)
    matrix_sizes = sorted(by_matrix_size.keys())
    
    for n in matrix_sizes:
        data = sorted(by_matrix_size[n], key=lambda x: x.get('block_size', 0))
        blocks = [d['block_size'] for d in data]
        gpu_times = [d['gpu_time_with_memcpy'] for d in data]
        
        ax3.plot(blocks, gpu_times, 'o-', label=f'N={n}')
    
    ax3.set_xlabel('Block Size (threads per block)')
    ax3.set_ylabel('GPU Execution Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_gpu_time_vs_block_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Plot 4: Speedup vs Block Size
    fig4, ax4 = plt.subplots(figsize=figsize)
    
    for n in matrix_sizes:
        data = sorted(by_matrix_size[n], key=lambda x: x.get('block_size', 0))
        blocks = [d['block_size'] for d in data]
        speedup = [d['speedup_with_memcpy'] for d in data]
        
        ax4.plot(blocks, speedup, 'o-', label=f'N={n}')
    
    ax4.set_xlabel('Block Size (threads per block)')
    ax4.set_ylabel('Speedup')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_speedup_vs_block_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def create_surface_plots(results, output_dir):
    """Create 3D surface plots for performance vs matrix size and block size."""
    if not HAS_MATPLOTLIB:
        print("Skipping surface plots (matplotlib not available)")
        return
    
    if not results:
        print("No results for surface plots.")
        return
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("Skipping surface plots (numpy or mplot3d not available)")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract unique values
    matrix_sizes = sorted(set(r['matrix_size'] for r in results))
    block_sizes = sorted(set(r['block_size'] for r in results))
    
    if len(matrix_sizes) < 2 or len(block_sizes) < 2:
        print("Not enough data points for surface plot")
        return
    
    # Create meshgrid
    X, Y = np.meshgrid(block_sizes, matrix_sizes)
    
    # Create Z arrays for different metrics
    Z_gpu_time = np.zeros_like(X, dtype=float)
    Z_speedup = np.zeros_like(X, dtype=float)
    Z_speedup_no_mem = np.zeros_like(X, dtype=float)
    
    for r in results:
        i = matrix_sizes.index(r['matrix_size'])
        j = block_sizes.index(r['block_size'])
        Z_gpu_time[i, j] = r.get('gpu_time_with_memcpy', 0)
        Z_speedup[i, j] = r.get('speedup_with_memcpy', 0)
        Z_speedup_no_mem[i, j] = r.get('speedup_no_memcpy', 0)
    
    # Quarter A4 page size
    figsize = (4.13, 2.92)
    
    # Surface plot 1: GPU Time
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_gpu_time, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Matrix Size (N)')
    ax1.set_zlabel('GPU Time (s)')
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_surface_gpu_time.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Surface plot saved to: {plot_path}")
    plt.close()
    
    # Surface plot 2: Speedup with memcpy
    fig2 = plt.figure(figsize=figsize)
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_speedup, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Matrix Size (N)')
    ax2.set_zlabel('Speedup')
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_surface_speedup_memcpy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Surface plot saved to: {plot_path}")
    plt.close()
    
    # Surface plot 3: Speedup including memcpy (coolwarm)
    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_speedup, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Matrix Size (N)')
    ax3.set_zlabel('Speedup')
    fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_surface_speedup_with_memcpy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Speedup surface plot saved to: {plot_path}")
    plt.close()
    
    # Surface plot 4: Speedup compute only
    fig4 = plt.figure(figsize=figsize)
    ax4 = fig4.add_subplot(111, projection='3d')
    surf4 = ax4.plot_surface(X, Y, Z_speedup_no_mem, cmap='coolwarm', alpha=0.8)
    ax4.set_xlabel('Block Size')
    ax4.set_ylabel('Matrix Size (N)')
    ax4.set_zlabel('Speedup')
    fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_b_surface_speedup_compute_only.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Speedup surface plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment B results')
    parser.add_argument('--results-dir', default='results/experiment_b',
                        help='Directory containing output files')
    parser.add_argument('--output-dir', default='analysis_output/experiment_b',
                        help='Directory for output plots and tables')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Analysis B: Execution Time vs Matrix Size & Block Size")
    print("=" * 60)
    
    # Collect results
    print(f"\nCollecting results from: {args.results_dir}")
    results = collect_results(args.results_dir)
    
    if not results:
        print("\nNo results found. Make sure experiments have completed.")
        print("Check that output files exist in the results directory.")
        return
    
    print(f"\nCollected {len(results)} result files")
    
    # Create tables
    print("\nGenerating summary tables...")
    create_summary_table(results, args.output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(results, args.output_dir)
    
    # Create surface plots
    print("\nGenerating surface plots...")
    create_surface_plots(results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
