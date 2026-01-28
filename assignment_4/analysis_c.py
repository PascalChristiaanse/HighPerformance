#!/usr/bin/env python3
"""
Analysis C: Generate speedup plots and tables

This script parses the output files from experiment_c and generates:
- Speedup comparison plots (with vs without memory copy)
- Unified Memory comparison (optional part iii)
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
    results_manual = []
    results_unified = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results_manual, results_unified
    
    for output_file in results_path.glob("run_*.out"):
        data = parse_output_file(output_file)
        if data:
            data['source_file'] = output_file.name
            # Determine if this is unified or manual based on filename or data
            mem_mgmt = data.get('memory_mgmt', 'manual')
            if 'unified' in output_file.name.lower() or mem_mgmt == 'unified':
                data['memory_mgmt'] = 'unified'
                results_unified.append(data)
                print(f"Parsed (unified): {output_file.name}")
            else:
                data['memory_mgmt'] = 'manual'
                results_manual.append(data)
                print(f"Parsed (manual): {output_file.name}")
    
    return results_manual, results_unified


def create_speedup_table(results_manual, results_unified, output_dir):
    """Create detailed speedup comparison table."""
    if not results_manual and not results_unified:
        print("No results to create table from.")
        return None, None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process manual results
    sorted_manual = sorted(results_manual, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    sorted_unified = sorted(results_unified, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    
    # Calculate additional metrics for manual results
    for r in sorted_manual:
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
    
    # Write detailed CSV for manual results
    columns_manual = [
        'matrix_size', 'block_size', 'cpu_time', 
        'gpu_time_with_memcpy', 'gpu_time_no_memcpy', 'memcpy_time',
        'speedup_with_memcpy', 'speedup_no_memcpy', 
        'speedup_difference', 'memcpy_overhead_pct'
    ]
    
    csv_path = output_path / "experiment_c_speedup_manual.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns_manual, extrasaction='ignore')
        writer.writeheader()
        for row in sorted_manual:
            writer.writerow(row)
    print(f"Manual speedup CSV saved to: {csv_path}")
    
    # Write CSV for unified memory results
    if sorted_unified:
        columns_unified = [
            'matrix_size', 'block_size', 'cpu_time', 
            'gpu_time_with_memcpy', 'gpu_time_no_memcpy', 'memcpy_time',
            'speedup_with_memcpy', 'speedup_no_memcpy'
        ]
        
        csv_path_unified = output_path / "experiment_c_speedup_unified.csv"
        with open(csv_path_unified, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns_unified, extrasaction='ignore')
            writer.writeheader()
            for row in sorted_unified:
                writer.writerow(row)
        print(f"Unified speedup CSV saved to: {csv_path_unified}")
    
    # Write combined LaTeX table
    latex_path = output_path / "experiment_c_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GPU Speedup Analysis: Manual vs Unified Memory}\n")
        f.write("\\label{tab:experiment_c}\n")
        
        # Check if we have unified results
        has_unified = len(sorted_unified) > 0
        
        if has_unified:
            f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("N & Block & \\multicolumn{2}{c|}{Manual Memory} & Unified & Memcpy \\% & Best \\\\\n")
            f.write(" &  & (i) no copy & (ii) w/ copy & (iii) & overhead & Method \\\\\n")
        else:
            f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("N & Block & Speedup (i) & Speedup (ii) & $\\Delta$ & Memcpy \\% \\\\\n")
            f.write(" &  & (no memcpy) & (with memcpy) & & overhead \\\\\n")
        f.write("\\hline\n")
        
        # Match unified results to manual by (matrix_size, block_size)
        unified_lookup = {(r['matrix_size'], r['block_size']): r for r in sorted_unified}
        
        for row in sorted_manual:
            key = (row.get('matrix_size'), row.get('block_size'))
            unified_row = unified_lookup.get(key, {})
            
            speedup_no = row.get('speedup_no_memcpy', 0)
            speedup_with = row.get('speedup_with_memcpy', 0)
            speedup_unified = unified_row.get('speedup_with_memcpy', 0)  # For unified, total time is relevant
            
            f.write(f"{row.get('matrix_size', 'N/A')} & ")
            f.write(f"{row.get('block_size', 'N/A')} & ")
            f.write(f"{speedup_no:.2f} & ")
            f.write(f"{speedup_with:.2f} & ")
            
            if has_unified:
                f.write(f"{speedup_unified:.2f} & ")
                f.write(f"{row.get('memcpy_overhead_pct', 0):.1f}\\% & ")
                # Determine best method
                if speedup_unified > speedup_with:
                    best = "Unified"
                else:
                    best = "Manual"
                f.write(f"{best}")
            else:
                f.write(f"{row.get('speedup_difference', 0):.2f} & ")
                f.write(f"{row.get('memcpy_overhead_pct', 0):.1f}\\%")
            f.write(" \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to: {latex_path}")
    
    return sorted_manual, sorted_unified


def create_speedup_plots(results_manual, results_unified, output_dir):
    """Create speedup visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not results_manual and not results_unified:
        print("No results to plot.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use manual results as primary (they have more detail)
    results = results_manual if results_manual else results_unified
    
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
    
    # Plot 2: Bar chart comparing speedups (manual vs unified if available)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for grouped bar chart
    matrix_sizes = sorted(set(r['matrix_size'] for r in results))
    x = range(len(matrix_sizes))
    
    has_unified = len(results_unified) > 0
    if has_unified:
        width = 0.25
    else:
        width = 0.35
    
    # Average speedups across block sizes
    speedup_with_avg = []
    speedup_no_avg = []
    speedup_unified_avg = []
    
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    for n in matrix_sizes:
        data_manual = [r for r in results if r['matrix_size'] == n]
        speedup_with_avg.append(sum(d['speedup_with_memcpy'] for d in data_manual) / len(data_manual) if data_manual else 0)
        speedup_no_avg.append(sum(d['speedup_no_memcpy'] for d in data_manual) / len(data_manual) if data_manual else 0)
        
        if has_unified:
            data_unified = [r for r in results_unified if r['matrix_size'] == n]
            speedup_unified_avg.append(sum(d['speedup_with_memcpy'] for d in data_unified) / len(data_unified) if data_unified else 0)
    
    if has_unified:
        bars1 = ax.bar([i - width for i in x], speedup_no_avg, width, 
                       label='(i) Compute only (no memcpy)', color='steelblue')
        bars2 = ax.bar([i for i in x], speedup_with_avg, width, 
                       label='(ii) Manual transfers (with memcpy)', color='coral')
        bars3 = ax.bar([i + width for i in x], speedup_unified_avg, width, 
                       label='(iii) Unified Memory', color='seagreen')
    else:
        bars1 = ax.bar([i - width/2 for i in x], speedup_no_avg, width, 
                       label='(i) Excluding memory copy', color='steelblue')
        bars2 = ax.bar([i + width/2 for i in x], speedup_with_avg, width, 
                       label='(ii) Including memory copy', color='coral')
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    title = 'Speedup Comparison: Effect of Memory Management' if has_unified else 'Speedup Comparison: Effect of Memory Copy'
    ax.set_title(title + ' (averaged over block sizes)', fontsize=12)
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
    
    if has_unified:
        for bar in bars3:
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
    
    # Plot 3: Unified Memory comparison (if available)
    if has_unified and results_manual:
        create_unified_comparison_plot(results_manual, results_unified, output_dir)


def create_unified_comparison_plot(results_manual, results_unified, output_dir):
    """Create dedicated plot comparing manual vs unified memory."""
    if not HAS_MATPLOTLIB:
        return
    
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create lookup for matching results
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    # Left plot: Line comparison by matrix size
    ax1 = axes[0]
    by_block = {}
    for r in results_manual:
        b = r['block_size']
        if b not in by_block:
            by_block[b] = {'sizes': [], 'manual': [], 'unified': []}
        by_block[b]['sizes'].append(r['matrix_size'])
        by_block[b]['manual'].append(r['speedup_with_memcpy'])
        key = (r['matrix_size'], r['block_size'])
        unified_r = unified_lookup.get(key, {})
        by_block[b]['unified'].append(unified_r.get('speedup_with_memcpy', 0))
    
    colors = plt.cm.tab10.colors
    for i, (bs, data) in enumerate(sorted(by_block.items())):
        sizes, manual, unified = zip(*sorted(zip(data['sizes'], data['manual'], data['unified'])))
        ax1.plot(sizes, manual, 'o-', color=colors[i], label=f'Manual B={bs}', linewidth=2)
        ax1.plot(sizes, unified, 's--', color=colors[i], label=f'Unified B={bs}', alpha=0.7)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Manual vs Unified Memory Speedup', fontsize=12)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    
    # Right plot: Speedup ratio (Unified / Manual)
    ax2 = axes[1]
    
    for i, (bs, data) in enumerate(sorted(by_block.items())):
        sizes, manual, unified = zip(*sorted(zip(data['sizes'], data['manual'], data['unified'])))
        ratio = [u/m if m > 0 else 0 for u, m in zip(unified, manual)]
        ax2.plot(sizes, ratio, 'o-', color=colors[i], label=f'Block size={bs}', linewidth=2)
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=12)
    ax2.set_ylabel('Unified / Manual Speedup Ratio', fontsize=12)
    ax2.set_title('Relative Performance: Unified vs Manual Memory', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=1, color='r', linestyle='-', alpha=0.7, label='Break-even')
    ax2.fill_between(ax2.get_xlim(), 1, 2, alpha=0.1, color='green', label='Unified better')
    ax2.fill_between(ax2.get_xlim(), 0, 1, alpha=0.1, color='red', label='Manual better')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_unified_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Unified comparison plot saved to: {plot_path}")
    plt.close()


def create_surface_plots(results_manual, results_unified, output_dir):
    """Create 3D surface plots for speedup vs matrix size and block size."""
    if not HAS_MATPLOTLIB:
        print("Skipping surface plots (matplotlib not available)")
        return
    
    if not results_manual and not results_unified:
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
    
    # Use manual results for main surface plots
    results = results_manual if results_manual else results_unified
    
    # Extract unique values
    matrix_sizes = sorted(set(r['matrix_size'] for r in results))
    block_sizes = sorted(set(r['block_size'] for r in results))
    
    if len(matrix_sizes) < 2 or len(block_sizes) < 2:
        print("Not enough data points for surface plot")
        return
    
    # Create meshgrid
    X, Y = np.meshgrid(block_sizes, matrix_sizes)
    
    # Create Z arrays
    Z_speedup_with = np.zeros_like(X, dtype=float)
    Z_speedup_no = np.zeros_like(X, dtype=float)
    Z_overhead = np.zeros_like(X, dtype=float)
    
    for r in results:
        i = matrix_sizes.index(r['matrix_size'])
        j = block_sizes.index(r['block_size'])
        Z_speedup_with[i, j] = r.get('speedup_with_memcpy', 0)
        Z_speedup_no[i, j] = r.get('speedup_no_memcpy', 0)
        Z_overhead[i, j] = r.get('memcpy_overhead_pct', 0)
    
    # Surface plot: Speedup comparison (manual memory)
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_speedup_with, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Matrix Size (N)')
    ax1.set_zlabel('Speedup')
    ax1.set_title('Speedup (with memcpy)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_speedup_no, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Matrix Size (N)')
    ax2.set_zlabel('Speedup')
    ax2.set_title('Speedup (compute only)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_overhead, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Matrix Size (N)')
    ax3.set_zlabel('Overhead (%)')
    ax3.set_title('Memory Copy Overhead')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_surface.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Surface plot saved to: {plot_path}")
    plt.close()
    
    # Additional surface plot: Unified vs Manual comparison (if available)
    if results_unified and results_manual:
        create_unified_surface_plot(results_manual, results_unified, output_dir)


def create_unified_surface_plot(results_manual, results_unified, output_dir):
    """Create surface plot comparing unified vs manual memory performance."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        return
    
    output_path = Path(output_dir)
    
    # Extract unique values
    matrix_sizes = sorted(set(r['matrix_size'] for r in results_manual))
    block_sizes = sorted(set(r['block_size'] for r in results_manual))
    
    # Create meshgrid
    X, Y = np.meshgrid(block_sizes, matrix_sizes)
    
    # Create Z arrays
    Z_manual = np.zeros_like(X, dtype=float)
    Z_unified = np.zeros_like(X, dtype=float)
    Z_ratio = np.zeros_like(X, dtype=float)
    
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    for r in results_manual:
        i = matrix_sizes.index(r['matrix_size'])
        j = block_sizes.index(r['block_size'])
        Z_manual[i, j] = r.get('speedup_with_memcpy', 0)
        
        key = (r['matrix_size'], r['block_size'])
        unified_r = unified_lookup.get(key, {})
        Z_unified[i, j] = unified_r.get('speedup_with_memcpy', 0)
        
        if Z_manual[i, j] > 0:
            Z_ratio[i, j] = Z_unified[i, j] / Z_manual[i, j]
    
    # Create figure
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_manual, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Matrix Size (N)')
    ax1.set_zlabel('Speedup')
    ax1.set_title('Manual Memory Speedup')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_unified, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Matrix Size (N)')
    ax2.set_zlabel('Speedup')
    ax2.set_title('Unified Memory Speedup')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_ratio, cmap='RdYlGn', alpha=0.8)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Matrix Size (N)')
    ax3.set_zlabel('Ratio')
    ax3.set_title('Unified/Manual Ratio (>1 = Unified better)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_unified_surface.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Unified comparison surface plot saved to: {plot_path}")
    plt.close()


def generate_analysis_summary(results_manual, results_unified, output_dir):
    """Generate a text summary of the speedup analysis."""
    if not results_manual and not results_unified:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_path / "experiment_c_analysis.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Experiment C: Speedup Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Manual memory statistics
        if results_manual:
            speedup_with = [r['speedup_with_memcpy'] for r in results_manual]
            speedup_no = [r['speedup_no_memcpy'] for r in results_manual]
            memcpy_overhead = [r.get('memcpy_overhead_pct', 0) for r in results_manual]
            
            f.write("MANUAL MEMORY MANAGEMENT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Speedup (i) - excluding memcpy:\n")
            f.write(f"  Min: {min(speedup_no):.2f}x\n")
            f.write(f"  Max: {max(speedup_no):.2f}x\n")
            f.write(f"  Avg: {sum(speedup_no)/len(speedup_no):.2f}x\n\n")
            
            f.write(f"Speedup (ii) - including memcpy:\n")
            f.write(f"  Min: {min(speedup_with):.2f}x\n")
            f.write(f"  Max: {max(speedup_with):.2f}x\n")
            f.write(f"  Avg: {sum(speedup_with)/len(speedup_with):.2f}x\n\n")
            
            f.write(f"Memory copy overhead:\n")
            f.write(f"  Min: {min(memcpy_overhead):.1f}%\n")
            f.write(f"  Max: {max(memcpy_overhead):.1f}%\n")
            f.write(f"  Avg: {sum(memcpy_overhead)/len(memcpy_overhead):.1f}%\n\n")
            
            best_with = max(results_manual, key=lambda x: x['speedup_with_memcpy'])
            best_no = max(results_manual, key=lambda x: x['speedup_no_memcpy'])
            
            f.write(f"Best config (with memcpy): {best_with['speedup_with_memcpy']:.2f}x\n")
            f.write(f"  N={best_with['matrix_size']}, Block={best_with['block_size']}\n")
            f.write(f"Best config (compute only): {best_no['speedup_no_memcpy']:.2f}x\n")
            f.write(f"  N={best_no['matrix_size']}, Block={best_no['block_size']}\n\n")
        
        # Unified memory statistics
        if results_unified:
            speedup_unified = [r['speedup_with_memcpy'] for r in results_unified]
            
            f.write("UNIFIED MEMORY (cudaMallocManaged):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Speedup (iii) - unified memory:\n")
            f.write(f"  Min: {min(speedup_unified):.2f}x\n")
            f.write(f"  Max: {max(speedup_unified):.2f}x\n")
            f.write(f"  Avg: {sum(speedup_unified)/len(speedup_unified):.2f}x\n\n")
            
            best_unified = max(results_unified, key=lambda x: x['speedup_with_memcpy'])
            f.write(f"Best config: {best_unified['speedup_with_memcpy']:.2f}x\n")
            f.write(f"  N={best_unified['matrix_size']}, Block={best_unified['block_size']}\n\n")
        
        # Comparison analysis (if both available)
        if results_manual and results_unified:
            f.write("UNIFIED vs MANUAL COMPARISON:\n")
            f.write("-" * 40 + "\n")
            
            unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
            
            unified_better = 0
            manual_better = 0
            
            for r in results_manual:
                key = (r['matrix_size'], r['block_size'])
                unified_r = unified_lookup.get(key)
                if unified_r:
                    if unified_r['speedup_with_memcpy'] > r['speedup_with_memcpy']:
                        unified_better += 1
                    else:
                        manual_better += 1
            
            f.write(f"Configurations where Unified is better: {unified_better}\n")
            f.write(f"Configurations where Manual is better: {manual_better}\n\n")
        
        # Observations
        f.write("KEY OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. The difference between speedup (i) and (ii) shows the impact\n")
        f.write("   of memory transfer overhead on overall performance.\n\n")
        f.write("2. For small matrices, memory copy overhead is typically higher\n")
        f.write("   relative to computation time.\n\n")
        f.write("3. As matrix size increases, the compute-to-transfer ratio\n")
        f.write("   improves, making GPU acceleration more beneficial.\n\n")
        
        if results_unified:
            f.write("4. Unified Memory (cudaMallocManaged) simplifies programming\n")
            f.write("   but may have different performance characteristics due to\n")
            f.write("   on-demand page migration and driver overhead.\n\n")
            f.write("5. On modern GPUs with good Unified Memory support, the\n")
            f.write("   performance difference may be minimal for compute-bound\n")
            f.write("   workloads, while Unified Memory may excel for sparse\n")
            f.write("   access patterns where not all data is needed.\n")
    
    print(f"Analysis summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment C results (speedup)')
    parser.add_argument('--results-dir', default='results/experiment_c',
                        help='Directory containing output files')
    parser.add_argument('--output-dir', default='analysis_output/experiment_c',
                        help='Directory for output plots and tables')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Analysis C: Speedup Analysis")
    print("  - (i) without memory copy")
    print("  - (ii) with memory copy")
    print("  - (iii) Unified Memory (optional)")
    print("=" * 60)
    
    # Collect results
    print(f"\nCollecting results from: {args.results_dir}")
    results_manual, results_unified = collect_results(args.results_dir)
    
    if not results_manual and not results_unified:
        print("\nNo results found. Make sure experiments have completed.")
        return
    
    print(f"\nCollected {len(results_manual)} manual memory results")
    print(f"Collected {len(results_unified)} unified memory results")
    
    # Create tables
    print("\nGenerating speedup tables...")
    results_manual, results_unified = create_speedup_table(results_manual, results_unified, args.output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    create_speedup_plots(results_manual, results_unified, args.output_dir)
    
    # Create surface plots
    print("\nGenerating surface plots...")
    create_surface_plots(results_manual, results_unified, args.output_dir)
    
    # Generate summary
    print("\nGenerating analysis summary...")
    generate_analysis_summary(results_manual, results_unified, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
