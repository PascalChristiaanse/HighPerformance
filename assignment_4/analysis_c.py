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
    """Collect all results from output files, categorized by memory type."""
    results_global = []
    results_shared = []
    results_unified = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results_global, results_shared, results_unified
    
    for output_file in results_path.glob("run_*.out"):
        data = parse_output_file(output_file)
        if data:
            data['source_file'] = output_file.name
            filename_lower = output_file.name.lower()
            
            # Determine memory type from filename or data
            mem_mode = data.get('memory_mode', '')
            mem_mgmt = data.get('memory_mgmt', '')
            
            if 'unified' in filename_lower or mem_mgmt == 'unified':
                data['memory_type'] = 'unified'
                results_unified.append(data)
                print(f"Parsed (UNIFIED): {output_file.name}")
            elif 'global' in filename_lower or mem_mode == 'global':
                data['memory_type'] = 'global'
                results_global.append(data)
                print(f"Parsed (GLOBAL): {output_file.name}")
            else:
                # Default to shared (includes old 'manual' and 'shared' files)
                data['memory_type'] = 'shared'
                results_shared.append(data)
                print(f"Parsed (SHARED): {output_file.name}")
    
    return results_global, results_shared, results_unified


def create_speedup_table(results_global, results_shared, results_unified, output_dir):
    """Create detailed speedup comparison table for all three memory types."""
    if not results_global and not results_shared and not results_unified:
        print("No results to create table from.")
        return None, None, None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort all results
    sorted_global = sorted(results_global, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    sorted_shared = sorted(results_shared, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    sorted_unified = sorted(results_unified, key=lambda x: (x.get('matrix_size', 0), x.get('block_size', 0)))
    
    # Create lookups by (matrix_size, block_size)
    global_lookup = {(r['matrix_size'], r['block_size']): r for r in sorted_global}
    shared_lookup = {(r['matrix_size'], r['block_size']): r for r in sorted_shared}
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in sorted_unified}
    
    # Get all unique (matrix_size, block_size) combinations
    all_keys = set(global_lookup.keys()) | set(shared_lookup.keys()) | set(unified_lookup.keys())
    sorted_keys = sorted(all_keys, key=lambda x: (x[0], x[1]))
    
    # Write combined CSV with all three memory types
    # For Global and Shared: show speedup both WITH and WITHOUT memcpy
    # For Unified: only show one speedup (no explicit memcpy to exclude)
    csv_path = output_path / "experiment_c_three_memory_types.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'matrix_size', 'block_size',
            'global_speedup_with_memcpy', 'global_speedup_no_memcpy',
            'shared_speedup_with_memcpy', 'shared_speedup_no_memcpy',
            'unified_speedup',
            'cpu_time', 'best_total', 'best_compute_only'
        ])
        
        for key in sorted_keys:
            n, b = key
            g = global_lookup.get(key, {})
            s = shared_lookup.get(key, {})
            u = unified_lookup.get(key, {})
            
            # Speedup with memcpy (total time including transfers)
            g_speedup_with = g.get('speedup_with_memcpy', 0)
            s_speedup_with = s.get('speedup_with_memcpy', 0)
            u_speedup = u.get('speedup_with_memcpy', 0)
            
            # Speedup without memcpy (compute-only, for Global/Shared only)
            g_speedup_no = g.get('speedup_no_memcpy', 0)
            s_speedup_no = s.get('speedup_no_memcpy', 0)
            
            # Determine best for total time (fair comparison including transfers)
            speedups_total = {'Global': g_speedup_with, 'Shared': s_speedup_with, 'Unified': u_speedup}
            best_total = max(speedups_total, key=speedups_total.get) if any(speedups_total.values()) else 'N/A'
            
            # Determine best for compute-only (Global vs Shared only, Unified not applicable)
            speedups_compute = {'Global': g_speedup_no, 'Shared': s_speedup_no}
            best_compute = max(speedups_compute, key=speedups_compute.get) if any(speedups_compute.values()) else 'N/A'
            
            cpu_time = g.get('cpu_time', s.get('cpu_time', u.get('cpu_time', 0)))
            
            writer.writerow([
                n, b,
                f"{g_speedup_with:.2f}" if g_speedup_with else 'N/A',
                f"{g_speedup_no:.2f}" if g_speedup_no else 'N/A',
                f"{s_speedup_with:.2f}" if s_speedup_with else 'N/A',
                f"{s_speedup_no:.2f}" if s_speedup_no else 'N/A',
                f"{u_speedup:.2f}" if u_speedup else 'N/A',
                f"{cpu_time:.6f}",
                best_total,
                best_compute
            ])
    print(f"Three memory types CSV saved to: {csv_path}")
    
    # Write LaTeX table comparing all three memory types
    # Table 1: With memory transfers (total time - fair comparison)
    latex_path = output_path / "experiment_c_three_memory_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GPU Speedup (Including Memory Transfers): All Three Memory Types vs CPU}\n")
        f.write("\\label{tab:three_memory_with_memcpy}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("N & Block & Global & Shared & Unified & Best \\\\\n")
        f.write(" & Size & Memory & Memory & Memory & \\\\\n")
        f.write("\\hline\n")
        
        for key in sorted_keys:
            n, b = key
            g = global_lookup.get(key, {})
            s = shared_lookup.get(key, {})
            u = unified_lookup.get(key, {})
            
            g_speedup = g.get('speedup_with_memcpy', 0)
            s_speedup = s.get('speedup_with_memcpy', 0)
            u_speedup = u.get('speedup_with_memcpy', 0)
            
            # Determine best and highlight
            speedups = {'Global': g_speedup, 'Shared': s_speedup, 'Unified': u_speedup}
            best = max(speedups, key=speedups.get) if any(speedups.values()) else 'N/A'
            
            # Format with bold for best
            g_str = f"\\textbf{{{g_speedup:.2f}x}}" if best == 'Global' and g_speedup > 0 else f"{g_speedup:.2f}x" if g_speedup else "N/A"
            s_str = f"\\textbf{{{s_speedup:.2f}x}}" if best == 'Shared' and s_speedup > 0 else f"{s_speedup:.2f}x" if s_speedup else "N/A"
            u_str = f"\\textbf{{{u_speedup:.2f}x}}" if best == 'Unified' and u_speedup > 0 else f"{u_speedup:.2f}x" if u_speedup else "N/A"
            
            f.write(f"{n} & {b} & {g_str} & {s_str} & {u_str} & {best} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{2mm}\n")
        f.write("\\footnotesize{Speedup = CPU\\_time / GPU\\_time (including memory transfers). This is the fair comparison.}\n")
        f.write("\\end{table}\n")
    print(f"Three memory types LaTeX table (with memcpy) saved to: {latex_path}")
    
    # Table 2: Without memory transfers (compute-only - Global vs Shared only)
    latex_path_compute = output_path / "experiment_c_compute_only_table.tex"
    with open(latex_path_compute, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GPU Speedup (Compute Only, No Memory Transfers): Global vs Shared Memory}\n")
        f.write("\\label{tab:compute_only}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("N & Block Size & Global Memory & Shared Memory & Best \\\\\n")
        f.write("\\hline\n")
        
        for key in sorted_keys:
            n, b = key
            g = global_lookup.get(key, {})
            s = shared_lookup.get(key, {})
            
            g_speedup = g.get('speedup_no_memcpy', 0)
            s_speedup = s.get('speedup_no_memcpy', 0)
            
            # Determine best and highlight
            speedups = {'Global': g_speedup, 'Shared': s_speedup}
            best = max(speedups, key=speedups.get) if any(speedups.values()) else 'N/A'
            
            # Format with bold for best
            g_str = f"\\textbf{{{g_speedup:.2f}x}}" if best == 'Global' and g_speedup > 0 else f"{g_speedup:.2f}x" if g_speedup else "N/A"
            s_str = f"\\textbf{{{s_speedup:.2f}x}}" if best == 'Shared' and s_speedup > 0 else f"{s_speedup:.2f}x" if s_speedup else "N/A"
            
            f.write(f"{n} & {b} & {g_str} & {s_str} & {best} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{2mm}\n")
        f.write("\\footnotesize{Speedup = CPU\\_time / GPU\\_compute\\_time (excluding memory transfers). Unified Memory not shown as it has no explicit memcpy.}\n")
        f.write("\\end{table}\n")
    print(f"Compute-only LaTeX table saved to: {latex_path_compute}")
    
    return sorted_global, sorted_shared, sorted_unified


def create_speedup_plots(results_global, results_shared, results_unified, output_dir):
    """Create speedup visualization plots comparing all three memory types."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not results_global and not results_shared and not results_unified:
        print("No results to plot.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create lookups
    global_lookup = {(r['matrix_size'], r['block_size']): r for r in results_global}
    shared_lookup = {(r['matrix_size'], r['block_size']): r for r in results_shared}
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    # Get all unique values
    all_results = results_global + results_shared + results_unified
    matrix_sizes = sorted(set(r['matrix_size'] for r in all_results))
    block_sizes = sorted(set(r['block_size'] for r in all_results))
    
    # ==========================================================================
    # Plot 1: Line plot - Speedup vs Matrix Size (for each block size)
    # ==========================================================================
    fig, axes = plt.subplots(1, len(block_sizes), figsize=(5*len(block_sizes), 5), squeeze=False)
    
    colors = {'Global': 'tab:red', 'Shared': 'tab:blue', 'Unified': 'tab:green'}
    markers = {'Global': 'o', 'Shared': 's', 'Unified': '^'}
    
    for idx, bs in enumerate(block_sizes):
        ax = axes[0, idx]
        
        for mem_type, lookup, label in [
            ('Global', global_lookup, 'Global Memory'),
            ('Shared', shared_lookup, 'Shared Memory'),
            ('Unified', unified_lookup, 'Unified Memory')
        ]:
            sizes = []
            speedups = []
            for n in matrix_sizes:
                r = lookup.get((n, bs))
                if r:
                    sizes.append(n)
                    speedups.append(r.get('speedup_with_memcpy', 0))
            
            if sizes:
                ax.plot(sizes, speedups, f'{markers[mem_type]}-', 
                       color=colors[mem_type], label=label, linewidth=2, markersize=8)
        
        ax.set_xlabel('Matrix Size (N)', fontsize=11)
        ax.set_ylabel('Speedup vs CPU', fontsize=11)
        ax.set_title(f'Block Size = {bs}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
        ax.set_xscale('log')
    
    plt.suptitle('GPU Speedup (With Memcpy): All Three Memory Types vs CPU', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_path / "experiment_c_three_memory_lines.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Three memory types line plot saved to: {plot_path}")
    plt.close()
    
    # ==========================================================================
    # Plot 1b: Line plot - Speedup WITH vs WITHOUT memcpy (Global and Shared only)
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors_with = {'Global': 'tab:red', 'Shared': 'tab:blue'}
    colors_no = {'Global': 'lightcoral', 'Shared': 'lightblue'}
    markers = {'Global': 'o', 'Shared': 's'}
    
    # Plot for a representative block size (middle one or first if only one)
    rep_block = block_sizes[len(block_sizes)//2] if block_sizes else 16
    
    # Left subplot: Global Memory (with vs without memcpy)
    ax = axes[0]
    for speedup_type, linestyle, alpha, suffix in [
        ('speedup_with_memcpy', '-', 1.0, ' (total)'),
        ('speedup_no_memcpy', '--', 0.7, ' (compute)')
    ]:
        sizes = []
        speedups = []
        for n in matrix_sizes:
            r = global_lookup.get((n, rep_block))
            if r:
                sizes.append(n)
                speedups.append(r.get(speedup_type, 0))
        if sizes:
            ax.plot(sizes, speedups, f'o{linestyle}', color='tab:red', 
                   label=f'Global{suffix}', linewidth=2, markersize=8, alpha=alpha)
    ax.set_xlabel('Matrix Size (N)', fontsize=11)
    ax.set_ylabel('Speedup vs CPU', fontsize=11)
    ax.set_title(f'Global Memory: Effect of Memory Transfers\n(Block Size = {rep_block})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    ax.set_xscale('log')
    
    # Right subplot: Shared Memory (with vs without memcpy)
    ax = axes[1]
    for speedup_type, linestyle, alpha, suffix in [
        ('speedup_with_memcpy', '-', 1.0, ' (total)'),
        ('speedup_no_memcpy', '--', 0.7, ' (compute)')
    ]:
        sizes = []
        speedups = []
        for n in matrix_sizes:
            r = shared_lookup.get((n, rep_block))
            if r:
                sizes.append(n)
                speedups.append(r.get(speedup_type, 0))
        if sizes:
            ax.plot(sizes, speedups, f's{linestyle}', color='tab:blue',
                   label=f'Shared{suffix}', linewidth=2, markersize=8, alpha=alpha)
    ax.set_xlabel('Matrix Size (N)', fontsize=11)
    ax.set_ylabel('Speedup vs CPU', fontsize=11)
    ax.set_title(f'Shared Memory: Effect of Memory Transfers\n(Block Size = {rep_block})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    ax.set_xscale('log')
    
    plt.suptitle('Memory Transfer Overhead: With vs Without cudaMemcpy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_path / "experiment_c_memcpy_overhead.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Memcpy overhead comparison plot saved to: {plot_path}")
    plt.close()
    
    # ==========================================================================
    # Plot 2: Grouped Bar Chart - Speedup comparison
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(matrix_sizes))
    width = 0.25
    
    # Average speedups across block sizes
    global_avg = []
    shared_avg = []
    unified_avg = []
    
    for n in matrix_sizes:
        g_vals = [global_lookup.get((n, b), {}).get('speedup_with_memcpy', 0) for b in block_sizes]
        s_vals = [shared_lookup.get((n, b), {}).get('speedup_with_memcpy', 0) for b in block_sizes]
        u_vals = [unified_lookup.get((n, b), {}).get('speedup_with_memcpy', 0) for b in block_sizes]
        
        global_avg.append(sum(v for v in g_vals if v) / max(len([v for v in g_vals if v]), 1))
        shared_avg.append(sum(v for v in s_vals if v) / max(len([v for v in s_vals if v]), 1))
        unified_avg.append(sum(v for v in u_vals if v) / max(len([v for v in u_vals if v]), 1))
    
    bars1 = ax.bar([i - width for i in x], global_avg, width, label='Global Memory', color='tab:red', alpha=0.8)
    bars2 = ax.bar([i for i in x], shared_avg, width, label='Shared Memory', color='tab:blue', alpha=0.8)
    bars3 = ax.bar([i + width for i in x], unified_avg, width, label='Unified Memory', color='tab:green', alpha=0.8)
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('Average Speedup vs CPU', fontsize=12)
    ax.set_title('GPU Speedup Comparison: All Three Memory Types\n(averaged over block sizes)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}x',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_c_three_memory_bars.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Three memory types bar chart saved to: {plot_path}")
    plt.close()
    
    # ==========================================================================
    # Plot 3: Heatmap-style comparison
    # ==========================================================================
    create_memory_heatmap(results_global, results_shared, results_unified, output_dir)


def create_memory_heatmap(results_global, results_shared, results_unified, output_dir):
    """Create heatmap comparing memory types."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        import numpy as np
    except ImportError:
        return
    
    output_path = Path(output_dir)
    
    # Create lookups
    global_lookup = {(r['matrix_size'], r['block_size']): r for r in results_global}
    shared_lookup = {(r['matrix_size'], r['block_size']): r for r in results_shared}
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    all_results = results_global + results_shared + results_unified
    if not all_results:
        return
    
    matrix_sizes = sorted(set(r['matrix_size'] for r in all_results))
    block_sizes = sorted(set(r['block_size'] for r in all_results))
    
    # Create data arrays for heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (lookup, title, cmap) in enumerate([
        (global_lookup, 'Global Memory', 'Reds'),
        (shared_lookup, 'Shared Memory', 'Blues'),
        (unified_lookup, 'Unified Memory', 'Greens')
    ]):
        data = np.zeros((len(matrix_sizes), len(block_sizes)))
        
        for i, n in enumerate(matrix_sizes):
            for j, b in enumerate(block_sizes):
                r = lookup.get((n, b))
                if r:
                    data[i, j] = r.get('speedup_with_memcpy', 0)
        
        ax = axes[idx]
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        ax.set_xticks(range(len(block_sizes)))
        ax.set_xticklabels(block_sizes)
        ax.set_yticks(range(len(matrix_sizes)))
        ax.set_yticklabels(matrix_sizes)
        ax.set_xlabel('Block Size')
        ax.set_ylabel('Matrix Size (N)')
        ax.set_title(title)
        
        # Add text annotations
        for i in range(len(matrix_sizes)):
            for j in range(len(block_sizes)):
                text = f'{data[i,j]:.1f}x' if data[i,j] > 0 else 'N/A'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                       color='white' if data[i,j] > data.max()*0.5 else 'black')
        
        plt.colorbar(im, ax=ax, label='Speedup')
    
    plt.suptitle('Speedup Heatmap: All Three Memory Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_path / "experiment_c_three_memory_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Three memory types heatmap saved to: {plot_path}")
    plt.close()


def create_surface_plots(results_global, results_shared, results_unified, output_dir):
    """Create 3D surface plots comparing all three memory types."""
    if not HAS_MATPLOTLIB:
        print("Skipping surface plots (matplotlib not available)")
        return
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("Skipping surface plots (numpy or mplot3d not available)")
        return
    
    all_results = results_global + results_shared + results_unified
    if not all_results:
        print("No results for surface plots.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create lookups
    global_lookup = {(r['matrix_size'], r['block_size']): r for r in results_global}
    shared_lookup = {(r['matrix_size'], r['block_size']): r for r in results_shared}
    unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
    
    # Extract unique values
    matrix_sizes = sorted(set(r['matrix_size'] for r in all_results))
    block_sizes = sorted(set(r['block_size'] for r in all_results))
    
    if len(matrix_sizes) < 2 or len(block_sizes) < 2:
        print("Not enough data points for surface plot")
        return
    
    # Create meshgrid
    X, Y = np.meshgrid(block_sizes, matrix_sizes)
    
    # Create Z arrays for each memory type
    Z_global = np.zeros_like(X, dtype=float)
    Z_shared = np.zeros_like(X, dtype=float)
    Z_unified = np.zeros_like(X, dtype=float)
    
    for i, n in enumerate(matrix_sizes):
        for j, b in enumerate(block_sizes):
            g = global_lookup.get((n, b), {})
            s = shared_lookup.get((n, b), {})
            u = unified_lookup.get((n, b), {})
            Z_global[i, j] = g.get('speedup_with_memcpy', 0)
            Z_shared[i, j] = s.get('speedup_with_memcpy', 0)
            Z_unified[i, j] = u.get('speedup_with_memcpy', 0)
    
    # Surface plot: All three memory types
    fig = plt.figure(figsize=(18, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_global, cmap='Reds', alpha=0.8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Matrix Size (N)')
    ax1.set_zlabel('Speedup')
    ax1.set_title('Global Memory')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_shared, cmap='Blues', alpha=0.8)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Matrix Size (N)')
    ax2.set_zlabel('Speedup')
    ax2.set_title('Shared Memory')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_unified, cmap='Greens', alpha=0.8)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Matrix Size (N)')
    ax3.set_zlabel('Speedup')
    ax3.set_title('Unified Memory')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.suptitle('Speedup Surface: All Three Memory Types vs CPU', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_path / "experiment_c_three_memory_surface.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Three memory types surface plot saved to: {plot_path}")
    plt.close()


def generate_analysis_summary(results_global, results_shared, results_unified, output_dir):
    """Generate a text summary of the three memory types analysis."""
    all_results = results_global + results_shared + results_unified
    if not all_results:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_path / "experiment_c_analysis.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("Experiment C: Three Memory Types Comparison Summary\n")
        f.write("=" * 65 + "\n\n")
        
        # Global memory statistics
        if results_global:
            speedup_with = [r['speedup_with_memcpy'] for r in results_global if r.get('speedup_with_memcpy')]
            speedup_no = [r['speedup_no_memcpy'] for r in results_global if r.get('speedup_no_memcpy')]
            if speedup_with:
                f.write("1. GLOBAL GPU MEMORY (VRAM):\n")
                f.write("-" * 40 + "\n")
                f.write("   Kernel: Av_Product_Global (direct VRAM access)\n")
                f.write(f"   Speedup (with memcpy): Min={min(speedup_with):.2f}x, Max={max(speedup_with):.2f}x, Avg={sum(speedup_with)/len(speedup_with):.2f}x\n")
                if speedup_no:
                    f.write(f"   Speedup (compute only): Min={min(speedup_no):.2f}x, Max={max(speedup_no):.2f}x, Avg={sum(speedup_no)/len(speedup_no):.2f}x\n")
                best = max(results_global, key=lambda x: x.get('speedup_with_memcpy', 0))
                f.write(f"   Best config: N={best['matrix_size']}, Block={best['block_size']} -> {best['speedup_with_memcpy']:.2f}x\n\n")
        
        # Shared memory statistics
        if results_shared:
            speedup_with = [r['speedup_with_memcpy'] for r in results_shared if r.get('speedup_with_memcpy')]
            speedup_no = [r['speedup_no_memcpy'] for r in results_shared if r.get('speedup_no_memcpy')]
            if speedup_with:
                f.write("2. SHARED GPU MEMORY (On-chip):\n")
                f.write("-" * 40 + "\n")
                f.write("   Kernel: Av_Product_Shared (tiled with shared memory cache)\n")
                f.write(f"   Speedup (with memcpy): Min={min(speedup_with):.2f}x, Max={max(speedup_with):.2f}x, Avg={sum(speedup_with)/len(speedup_with):.2f}x\n")
                if speedup_no:
                    f.write(f"   Speedup (compute only): Min={min(speedup_no):.2f}x, Max={max(speedup_no):.2f}x, Avg={sum(speedup_no)/len(speedup_no):.2f}x\n")
                best = max(results_shared, key=lambda x: x.get('speedup_with_memcpy', 0))
                f.write(f"   Best config: N={best['matrix_size']}, Block={best['block_size']} -> {best['speedup_with_memcpy']:.2f}x\n\n")
        
        # Unified memory statistics
        if results_unified:
            speedup = [r['speedup_with_memcpy'] for r in results_unified if r.get('speedup_with_memcpy')]
            if speedup:
                f.write("3. UNIFIED MEMORY (cudaMallocManaged):\n")
                f.write("-" * 40 + "\n")
                f.write("   Allocation: Automatic CPU-GPU migration by CUDA runtime\n")
                f.write(f"   Speedup: Min={min(speedup):.2f}x, Max={max(speedup):.2f}x, Avg={sum(speedup)/len(speedup):.2f}x\n")
                best = max(results_unified, key=lambda x: x.get('speedup_with_memcpy', 0))
                f.write(f"   Best config: N={best['matrix_size']}, Block={best['block_size']} -> {best['speedup_with_memcpy']:.2f}x\n\n")
        
        # Comparison
        f.write("COMPARISON ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        global_lookup = {(r['matrix_size'], r['block_size']): r for r in results_global}
        shared_lookup = {(r['matrix_size'], r['block_size']): r for r in results_shared}
        unified_lookup = {(r['matrix_size'], r['block_size']): r for r in results_unified}
        
        all_keys = set(global_lookup.keys()) | set(shared_lookup.keys()) | set(unified_lookup.keys())
        
        wins = {'Global': 0, 'Shared': 0, 'Unified': 0}
        for key in all_keys:
            g = global_lookup.get(key, {}).get('speedup_with_memcpy', 0)
            s = shared_lookup.get(key, {}).get('speedup_with_memcpy', 0)
            u = unified_lookup.get(key, {}).get('speedup_with_memcpy', 0)
            best = max([('Global', g), ('Shared', s), ('Unified', u)], key=lambda x: x[1])
            if best[1] > 0:
                wins[best[0]] += 1
        
        f.write(f"Best performance wins (total time including memcpy):\n")
        f.write(f"  Global Memory:  {wins['Global']} configurations\n")
        f.write(f"  Shared Memory:  {wins['Shared']} configurations\n")
        f.write(f"  Unified Memory: {wins['Unified']} configurations\n\n")
        
        # Compute-only wins (Global vs Shared only)
        wins_compute = {'Global': 0, 'Shared': 0}
        for key in all_keys:
            g = global_lookup.get(key, {}).get('speedup_no_memcpy', 0)
            s = shared_lookup.get(key, {}).get('speedup_no_memcpy', 0)
            if g > s and g > 0:
                wins_compute['Global'] += 1
            elif s > g and s > 0:
                wins_compute['Shared'] += 1
        
        f.write(f"Best compute-only performance wins (excluding memcpy):\n")
        f.write(f"  Global Memory:  {wins_compute['Global']} configurations\n")
        f.write(f"  Shared Memory:  {wins_compute['Shared']} configurations\n")
        f.write(f"  (Unified not compared - no explicit memcpy to exclude)\n\n")
        
        # Key observations
        f.write("KEY OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. GLOBAL Memory: Direct VRAM access. Simple but slow due to\n")
        f.write("   high latency global memory reads on each thread.\n\n")
        f.write("2. SHARED Memory: Fast on-chip cache (~100x faster than global).\n")
        f.write("   Tiled approach reduces global memory accesses by reusing\n")
        f.write("   data within thread blocks.\n\n")
        f.write("3. UNIFIED Memory: Automatic migration between CPU and GPU.\n")
        f.write("   Simplifies programming but adds driver overhead for page\n")
        f.write("   migration. Performance varies by access pattern.\n\n")
        f.write("4. For compute-bound workloads with regular access patterns,\n")
        f.write("   SHARED memory typically provides the best performance.\n")
    
    print(f"Analysis summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment C results (three memory types)')
    parser.add_argument('--results-dir', default='results/experiment_c',
                        help='Directory containing output files')
    parser.add_argument('--output-dir', default='analysis_output/experiment_c',
                        help='Directory for output plots and tables')
    args = parser.parse_args()
    
    print("=" * 65)
    print("Analysis C: Three Memory Types Comparison (vs CPU)")
    print("  1. Global GPU Memory (VRAM)")
    print("  2. Shared GPU Memory (on-chip cache)")
    print("  3. Unified Memory (cudaMallocManaged)")
    print("=" * 65)
    
    # Collect results
    print(f"\nCollecting results from: {args.results_dir}")
    results_global, results_shared, results_unified = collect_results(args.results_dir)
    
    total = len(results_global) + len(results_shared) + len(results_unified)
    if total == 0:
        print("\nNo results found. Make sure experiments have completed.")
        return
    
    print(f"\nCollected results:")
    print(f"  Global memory:  {len(results_global)} files")
    print(f"  Shared memory:  {len(results_shared)} files")
    print(f"  Unified memory: {len(results_unified)} files")
    
    # Create tables
    print("\nGenerating speedup tables...")
    results_global, results_shared, results_unified = create_speedup_table(
        results_global, results_shared, results_unified, args.output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    create_speedup_plots(results_global, results_shared, results_unified, args.output_dir)
    
    # Create surface plots
    print("\nGenerating surface plots...")
    create_surface_plots(results_global, results_shared, results_unified, args.output_dir)
    
    # Generate summary
    print("\nGenerating analysis summary...")
    generate_analysis_summary(results_global, results_shared, results_unified, args.output_dir)
    
    print("\n" + "=" * 65)
    print("Analysis complete!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
