#!/usr/bin/env python3
"""
Analysis D: Compare shared memory vs global memory performance

This script parses the output files from experiment_d and generates:
- Performance comparison plots (shared vs global memory)
- Speedup improvement from using shared memory
- Analysis and explanation of performance differences

This corresponds to Step 4 (question d) in the lab exercise:
"Explain the different performance results from the previous experiment."

Usage:
    python analysis_d.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
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


def create_comparison_table(results, output_dir):
    """Create memory comparison table."""
    if not results:
        print("No results to create table from.")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Organize by matrix size and block size, comparing memory modes
    comparisons = {}
    
    for r in results:
        key = (r.get('matrix_size'), r.get('block_size'))
        mem_mode = r.get('memory_mode', 'unknown')
        
        if key not in comparisons:
            comparisons[key] = {}
        comparisons[key][mem_mode] = r
    
    # Create comparison data
    comparison_rows = []
    for (n, b), modes in sorted(comparisons.items()):
        row = {
            'matrix_size': n,
            'block_size': b,
        }
        
        if 'shared' in modes:
            row['gpu_time_shared'] = modes['shared'].get('gpu_time_no_memcpy', 0)
            row['speedup_shared'] = modes['shared'].get('speedup_no_memcpy', 0)
        
        if 'global' in modes:
            row['gpu_time_global'] = modes['global'].get('gpu_time_no_memcpy', 0)
            row['speedup_global'] = modes['global'].get('speedup_no_memcpy', 0)
        
        # Calculate shared memory advantage
        if 'gpu_time_shared' in row and 'gpu_time_global' in row:
            if row['gpu_time_shared'] > 0:
                row['shared_advantage'] = row['gpu_time_global'] / row['gpu_time_shared']
            else:
                row['shared_advantage'] = 0
            row['time_saved_pct'] = ((row['gpu_time_global'] - row['gpu_time_shared']) / 
                                     row['gpu_time_global'] * 100) if row['gpu_time_global'] > 0 else 0
        
        comparison_rows.append(row)
    
    # Write CSV
    columns = [
        'matrix_size', 'block_size', 
        'gpu_time_global', 'gpu_time_shared',
        'speedup_global', 'speedup_shared',
        'shared_advantage', 'time_saved_pct'
    ]
    
    csv_path = output_path / "experiment_d_memory_comparison.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    print(f"Comparison CSV saved to: {csv_path}")
    
    # Write LaTeX table
    latex_path = output_path / "experiment_d_table.tex"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Shared Memory vs Global Memory Performance Comparison}\n")
        f.write("\\label{tab:experiment_d}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("N & Block & Global (s) & Shared (s) & Shared Speedup & Time Saved \\\\\n")
        f.write("\\hline\n")
        
        for row in comparison_rows:
            f.write(f"{row.get('matrix_size', 'N/A')} & ")
            f.write(f"{row.get('block_size', 'N/A')} & ")
            f.write(f"{row.get('gpu_time_global', 0):.6f} & ")
            f.write(f"{row.get('gpu_time_shared', 0):.6f} & ")
            f.write(f"{row.get('shared_advantage', 0):.2f}x & ")
            f.write(f"{row.get('time_saved_pct', 0):.1f}\\%")
            f.write(" \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to: {latex_path}")
    
    return comparison_rows


def create_comparison_plots(results, comparisons, output_dir):
    """Create memory comparison visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not comparisons:
        print("No comparison data to plot.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Organize by block size
    by_block = {}
    for row in comparisons:
        b = row['block_size']
        if b not in by_block:
            by_block[b] = []
        by_block[b].append(row)
    
    # Plot 1: GPU time comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for bs in sorted(by_block.keys()):
        data = sorted(by_block[bs], key=lambda x: x['matrix_size'])
        sizes = [d['matrix_size'] for d in data]
        global_times = [d.get('gpu_time_global', 0) for d in data]
        shared_times = [d.get('gpu_time_shared', 0) for d in data]
        
        ax1.plot(sizes, global_times, 'o--', label=f'Global (B={bs})', alpha=0.7)
        ax1.plot(sizes, shared_times, 's-', label=f'Shared (B={bs})', linewidth=2)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=12)
    ax1.set_ylabel('GPU Compute Time (seconds)', fontsize=12)
    ax1.set_title('GPU Execution Time: Shared vs Global Memory', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Shared memory advantage
    ax2 = axes[1]
    for bs in sorted(by_block.keys()):
        data = sorted(by_block[bs], key=lambda x: x['matrix_size'])
        sizes = [d['matrix_size'] for d in data]
        advantage = [d.get('shared_advantage', 1) for d in data]
        
        ax2.plot(sizes, advantage, 'o-', label=f'Block size={bs}', linewidth=2)
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=12)
    ax2.set_ylabel('Speedup (Global time / Shared time)', fontsize=12)
    ax2.set_title('Shared Memory Advantage over Global Memory', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='Equal performance')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_d_memory_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Memory comparison plot saved to: {plot_path}")
    plt.close()
    
    # Plot 2: Bar chart for specific matrix sizes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    matrix_sizes = sorted(set(r['matrix_size'] for r in comparisons))
    x = range(len(matrix_sizes))
    width = 0.35
    
    # Average across block sizes
    global_avg = []
    shared_avg = []
    
    for n in matrix_sizes:
        data_for_n = [r for r in comparisons if r['matrix_size'] == n]
        global_avg.append(sum(d.get('gpu_time_global', 0) for d in data_for_n) / len(data_for_n))
        shared_avg.append(sum(d.get('gpu_time_shared', 0) for d in data_for_n) / len(data_for_n))
    
    bars1 = ax.bar([i - width/2 for i in x], global_avg, width, 
                   label='Global Memory', color='coral')
    bars2 = ax.bar([i + width/2 for i in x], shared_avg, width, 
                   label='Shared Memory', color='steelblue')
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('Average GPU Time (seconds)', fontsize=12)
    ax.set_title('Memory Type Comparison (averaged over block sizes)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_path / "experiment_d_memory_bars.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Memory bar chart saved to: {plot_path}")
    plt.close()


def create_surface_plots(comparisons, output_dir):
    """Create 3D surface plots for memory comparison vs matrix size and block size."""
    if not HAS_MATPLOTLIB:
        print("Skipping surface plots (matplotlib not available)")
        return
    
    if not comparisons:
        print("No comparison data for surface plots.")
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
    matrix_sizes = sorted(set(r['matrix_size'] for r in comparisons))
    block_sizes = sorted(set(r['block_size'] for r in comparisons))
    
    if len(matrix_sizes) < 2 or len(block_sizes) < 2:
        print("Not enough data points for surface plot")
        return
    
    # Create meshgrid
    X, Y = np.meshgrid(block_sizes, matrix_sizes)
    
    # Create Z arrays
    Z_global = np.zeros_like(X, dtype=float)
    Z_shared = np.zeros_like(X, dtype=float)
    Z_advantage = np.zeros_like(X, dtype=float)
    
    for r in comparisons:
        i = matrix_sizes.index(r['matrix_size'])
        j = block_sizes.index(r['block_size'])
        Z_global[i, j] = r.get('gpu_time_global', 0)
        Z_shared[i, j] = r.get('gpu_time_shared', 0)
        Z_advantage[i, j] = r.get('shared_advantage', 1)
    
    # Surface plot: Memory type comparison
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_global, cmap='Reds', alpha=0.8)
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Matrix Size (N)')
    ax1.set_zlabel('Time (s)')
    ax1.set_title('Global Memory Time')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_shared, cmap='Blues', alpha=0.8)
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Matrix Size (N)')
    ax2.set_zlabel('Time (s)')
    ax2.set_title('Shared Memory Time')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_advantage, cmap='RdYlGn', alpha=0.8)
    ax3.set_xlabel('Block Size')
    ax3.set_ylabel('Matrix Size (N)')
    ax3.set_zlabel('Speedup')
    ax3.set_title('Shared Memory Advantage')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plot_path = output_path / "experiment_d_surface.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Surface plot saved to: {plot_path}")
    plt.close()


def generate_explanation(comparisons, output_dir):
    """Generate detailed explanation of performance differences."""
    if not comparisons:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    explanation_path = output_path / "experiment_d_explanation.txt"
    
    # Calculate statistics
    advantages = [r.get('shared_advantage', 1) for r in comparisons if r.get('shared_advantage')]
    time_saved = [r.get('time_saved_pct', 0) for r in comparisons if r.get('time_saved_pct')]
    
    with open(explanation_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Experiment D: Performance Analysis - Shared vs Global Memory\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        if advantages:
            f.write(f"Shared memory speedup over global:\n")
            f.write(f"  Minimum: {min(advantages):.2f}x\n")
            f.write(f"  Maximum: {max(advantages):.2f}x\n")
            f.write(f"  Average: {sum(advantages)/len(advantages):.2f}x\n\n")
        
        if time_saved:
            f.write(f"Time saved by using shared memory:\n")
            f.write(f"  Minimum: {min(time_saved):.1f}%\n")
            f.write(f"  Maximum: {max(time_saved):.1f}%\n")
            f.write(f"  Average: {sum(time_saved)/len(time_saved):.1f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("EXPLANATION OF PERFORMANCE DIFFERENCES\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. MEMORY HIERARCHY IN GPUs\n")
        f.write("-" * 40 + "\n")
        f.write("""
GPUs have a hierarchical memory system with varying access speeds:

  - Global Memory: Large capacity (GBs) but high latency (~400-600 cycles)
    * All threads can access
    * Main memory for GPU computation
    
  - Shared Memory: Small capacity (48-96 KB per SM) but very fast (~1-4 cycles)
    * Shared among threads in a block
    * Programmer-managed cache
    
  - Registers: Fastest but very limited
    * Private to each thread

For the NVIDIA A100 GPU (used in experiments):
  - Global memory bandwidth: ~2 TB/s
  - Shared memory bandwidth: ~19 TB/s per SM
  - L1 cache: 192 KB per SM
  - L2 cache: 80 MB total

""")
        
        f.write("2. WHY SHARED MEMORY IS FASTER FOR MATRIX-VECTOR MULTIPLICATION\n")
        f.write("-" * 40 + "\n")
        f.write("""
Matrix-vector multiplication (y = A * x) requires:
  - Each row of A to be read once
  - Vector x to be read N times (once per row)

Global Memory Approach:
  - Each thread reads a row of A from global memory
  - Each thread reads the ENTIRE vector x from global memory
  - Total global memory reads: N² (for A) + N² (for x, read N times)
  - High memory traffic and latency

Shared Memory (Tiled) Approach:
  - Matrix A is divided into tiles of size BLOCK_SIZE × BLOCK_SIZE
  - Each tile of A and corresponding portion of x loaded to shared memory
  - Threads in a block cooperatively load data (coalesced access)
  - Multiple reuse of data in shared memory
  - Reduced global memory traffic

Benefits of tiling:
  a) Data reuse: Vector x portions are loaded once per tile, used by all rows
  b) Coalesced memory access: Adjacent threads access adjacent memory
  c) Reduced memory bandwidth pressure on global memory
  d) Lower latency for repeated accesses

""")
        
        f.write("3. FACTORS AFFECTING PERFORMANCE DIFFERENCE\n")
        f.write("-" * 40 + "\n")
        f.write("""
Several factors influence how much faster shared memory is:

a) Matrix Size (N):
   - Larger matrices benefit MORE from shared memory
   - More data reuse opportunities
   - Global memory becomes the bottleneck for large N

b) Block Size:
   - Affects tile size and data reuse
   - Must balance between parallelism and shared memory capacity
   - Optimal block size depends on matrix size and GPU architecture

c) Memory Access Patterns:
   - Global memory is fastest with coalesced access
   - Shared memory has bank conflicts to consider
   - Well-designed tiling avoids bank conflicts

d) Occupancy:
   - Shared memory usage limits blocks per SM
   - Trade-off between shared memory benefit and parallelism

""")
        
        f.write("4. OBSERVATIONS FROM EXPERIMENTAL RESULTS\n")
        f.write("-" * 40 + "\n")
        
        # Find best and worst cases
        if comparisons:
            best = max(comparisons, key=lambda x: x.get('shared_advantage', 0))
            worst = min(comparisons, key=lambda x: x.get('shared_advantage', float('inf')))
            
            f.write(f"""
Best case for shared memory:
  - Matrix size: {best['matrix_size']}
  - Block size: {best['block_size']}
  - Speedup: {best.get('shared_advantage', 0):.2f}x
  - Time saved: {best.get('time_saved_pct', 0):.1f}%

Smallest improvement:
  - Matrix size: {worst['matrix_size']}
  - Block size: {worst['block_size']}
  - Speedup: {worst.get('shared_advantage', 0):.2f}x
  - Time saved: {worst.get('time_saved_pct', 0):.1f}%

""")
        
        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("""
Based on the analysis:

1. Always use shared memory for matrix operations when possible
   - The programming complexity is justified by performance gains

2. For small matrices (N < 100):
   - Shared memory benefit may be limited
   - Consider whether GPU is even beneficial (memory transfer overhead)

3. For large matrices (N > 1000):
   - Shared memory is essential for good performance
   - Consider multiple levels of tiling for very large matrices

4. Block size selection:
   - 32 threads (one warp) is a good baseline
   - Larger blocks may improve data reuse but limit occupancy
   - Experiment with 32, 64, 128 for best performance

5. Memory transfer considerations:
   - For problems where data is already on GPU, shared memory helps most
   - For one-time computations, memory transfer may dominate runtime
""")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("=" * 70 + "\n")
    
    print(f"Performance explanation saved to: {explanation_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment D results (memory comparison)')
    parser.add_argument('--results-dir', default='results/experiment_d',
                        help='Directory containing output files')
    parser.add_argument('--output-dir', default='analysis_output/experiment_d',
                        help='Directory for output plots and tables')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Analysis D: Shared vs Global Memory Performance")
    print("=" * 60)
    
    # Collect results
    print(f"\nCollecting results from: {args.results_dir}")
    results = collect_results(args.results_dir)
    
    if not results:
        print("\nNo results found. Make sure experiments have completed.")
        return
    
    print(f"\nCollected {len(results)} result files")
    
    # Create comparison table
    print("\nGenerating comparison tables...")
    comparisons = create_comparison_table(results, args.output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    create_comparison_plots(results, comparisons, args.output_dir)
    
    # Create surface plots
    print("\nGenerating surface plots...")
    create_surface_plots(comparisons, args.output_dir)
    
    # Generate explanation
    print("\nGenerating performance explanation...")
    generate_explanation(comparisons, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
