#!/usr/bin/env python3
"""
Plot Exchange Analysis Results for Exercise 1.2.11

Analyzes Exchange_Borders function to determine:
- Latency (α): Fixed overhead per communication
- Bandwidth (β): Data transfer rate
- Total data communicated

Uses communication model: t_exchange = α + data_size / β
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Configuration
RESULTS_FILE = Path("benchmark_results/exchange_analysis/exchange_analysis_results.csv")
OUTPUT_DIR = Path("benchmark_results/exchange_analysis")

def load_data(filepath: Path) -> pd.DataFrame:
    """Load and preprocess the benchmark results."""
    df = pd.read_csv(filepath)
    
    # Calculate derived quantities
    df['time_per_exchange_us'] = (df['exchange_time'] / df['exchange_count']) * 1e6  # microseconds
    df['total_data_bytes'] = df['data_per_exchange_bytes'] * df['exchange_count']
    df['total_data_mb'] = df['total_data_bytes'] / (1024 * 1024)
    
    # Average over runs
    df_avg = df.groupby(['topology', 'grid_size']).agg({
        'total_time': 'mean',
        'step_time': 'mean',
        'exchange_time': 'mean',
        'reduction_time': 'mean',
        'iterations': 'first',
        'local_nx': 'first',
        'local_ny': 'first',
        'data_per_exchange_bytes': 'first',
        'exchange_count': 'first',
        'time_per_exchange_us': ['mean', 'std'],
        'total_data_bytes': 'first',
        'total_data_mb': 'first'
    }).reset_index()
    
    # Flatten column names properly
    new_columns = []
    for col in df_avg.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first' or col[1] == 'mean':
                new_columns.append(col[0])
            else:
                new_columns.append('_'.join(col))
        else:
            new_columns.append(col)
    df_avg.columns = new_columns
    
    # Rename the std column
    if 'time_per_exchange_us_std' in df_avg.columns:
        pass  # Already named correctly
    
    return df_avg

def fit_latency_bandwidth_by_topology(df: pd.DataFrame) -> dict:
    """
    Fit per topology (across grid sizes).
    Model: time = α + (1/β) * data_size
    """
    results = {}
    
    for topology in df['topology'].unique():
        topo_data = df[df['topology'] == topology].sort_values('data_per_exchange_bytes')
        
        x = topo_data['data_per_exchange_bytes'].values
        y = topo_data['time_per_exchange_us'].values
        
        if len(x) < 2:
            continue
            
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # latency (α) = intercept in microseconds
        latency_us = max(0, intercept)  # Clamp to non-negative
        
        # bandwidth: slope = 1/β (us/byte), so β = 1/slope (bytes/us) = 1e6/slope (bytes/s)
        if slope > 0:
            bandwidth_bytes_per_s = 1e6 / slope
            bandwidth_mb_per_s = bandwidth_bytes_per_s / (1024 * 1024)
        else:
            bandwidth_bytes_per_s = float('inf')
            bandwidth_mb_per_s = float('inf')
        
        results[topology] = {
            'latency_us': latency_us,
            'latency_us_raw': intercept,  # Store raw value for diagnostics
            'bandwidth_mb_s': bandwidth_mb_per_s,
            'r_squared': r_value**2,
            'slope': slope,
            'data': topo_data
        }
    
    return results

def fit_latency_bandwidth_by_gridsize(df: pd.DataFrame) -> dict:
    """
    Fit per grid size (across topologies).
    This compares different topologies at the same grid size.
    """
    results = {}
    
    for grid_size in df['grid_size'].unique():
        grid_data = df[df['grid_size'] == grid_size].sort_values('data_per_exchange_bytes')
        
        x = grid_data['data_per_exchange_bytes'].values
        y = grid_data['time_per_exchange_us'].values
        
        if len(x) < 2:
            continue
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        latency_us = max(0, intercept)
        
        if slope > 0:
            bandwidth_bytes_per_s = 1e6 / slope
            bandwidth_mb_per_s = bandwidth_bytes_per_s / (1024 * 1024)
        else:
            bandwidth_bytes_per_s = float('inf')
            bandwidth_mb_per_s = float('inf')
        
        results[grid_size] = {
            'latency_us': latency_us,
            'latency_us_raw': intercept,
            'bandwidth_mb_s': bandwidth_mb_per_s,
            'r_squared': r_value**2,
            'slope': slope,
            'data': grid_data
        }
    
    return results

def fit_global(df: pd.DataFrame) -> dict:
    """
    Global fit across all data points.
    """
    x = df['data_per_exchange_bytes'].values
    y = df['time_per_exchange_us'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    latency_us = max(0, intercept)
    
    if slope > 0:
        bandwidth_bytes_per_s = 1e6 / slope
        bandwidth_mb_per_s = bandwidth_bytes_per_s / (1024 * 1024)
    else:
        bandwidth_bytes_per_s = float('inf')
        bandwidth_mb_per_s = float('inf')
    
    return {
        'latency_us': latency_us,
        'latency_us_raw': intercept,
        'bandwidth_mb_s': bandwidth_mb_per_s,
        'r_squared': r_value**2,
        'slope': slope
    }

def plot_exchange_time_vs_data(df: pd.DataFrame, fit_by_topo: dict, fit_by_grid: dict, 
                                fit_global: dict, output_dir: Path):
    """Simple scatterplot of exchange time vs data size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors_topo = {'4x1': 'blue', '2x2': 'red', '1x4': 'green'}
    markers_topo = {'4x1': 'o', '2x2': 's', '1x4': '^'}
    
    # Plot data points by topology
    for topology in df['topology'].unique():
        topo_data = df[df['topology'] == topology]
        color = colors_topo.get(topology, 'gray')
        marker = markers_topo.get(topology, 'o')
        
        ax.scatter(topo_data['data_per_exchange_bytes'], 
                   topo_data['time_per_exchange_us'],
                   color=color, marker=marker, s=100, 
                   label=f'{topology}', zorder=5)
    
    ax.set_xlabel('Data per Exchange (bytes)', fontsize=12)
    ax.set_ylabel('Time per Exchange (μs)', fontsize=12)
    ax.set_title('Exercise 1.2.11: Exchange Time vs Data Size', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exchange_latency_bandwidth.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'exchange_latency_bandwidth.png'}")
    plt.show()
    plt.close()

def plot_time_per_byte(df: pd.DataFrame, output_dir: Path):
    """Plot time per byte for each grid size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate time per byte (ns/byte)
    df = df.copy()
    df['time_per_byte_ns'] = (df['time_per_exchange_us'] * 1000) / df['data_per_exchange_bytes']
    
    colors_topo = {'4x1': 'blue', '2x2': 'red', '1x4': 'green'}
    markers_topo = {'4x1': 'o', '2x2': 's', '1x4': '^'}
    
    grid_sizes = sorted(df['grid_size'].unique())
    topologies = df['topology'].unique()
    x = np.arange(len(grid_sizes))
    width = 0.35
    
    # Bar plot grouped by grid size
    for i, topology in enumerate(topologies):
        topo_data = df[df['topology'] == topology].sort_values('grid_size')
        offset = (i - len(topologies)/2 + 0.5) * width
        color = colors_topo.get(topology, 'gray')
        ax.bar(x + offset, topo_data['time_per_byte_ns'], width, 
               label=topology, color=color, alpha=0.8)
    
    ax.set_xlabel('Grid Size', fontsize=12)
    ax.set_ylabel('Time per Byte (ns/byte)', fontsize=12)
    ax.set_title('Exercise 1.2.11: Communication Time per Byte by Grid Size', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_per_byte.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'time_per_byte.png'}")
    plt.show()
    plt.close()

def plot_exchange_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot exchange time breakdown by topology and grid size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    topologies = df['topology'].unique()
    grid_sizes = sorted(df['grid_size'].unique())
    x = np.arange(len(grid_sizes))
    width = 0.35
    
    # Left plot: Total exchange time
    ax1 = axes[0]
    for i, topology in enumerate(topologies):
        topo_data = df[df['topology'] == topology].sort_values('grid_size')
        offset = (i - len(topologies)/2 + 0.5) * width
        bars = ax1.bar(x + offset, topo_data['exchange_time'] * 1000, width, 
                       label=topology, alpha=0.8)
    
    ax1.set_xlabel('Grid Size', fontsize=12)
    ax1.set_ylabel('Total Exchange Time (ms)', fontsize=12)
    ax1.set_title('Total Exchange Time by Configuration', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Exchange time as percentage
    ax2 = axes[1]
    for i, topology in enumerate(topologies):
        topo_data = df[df['topology'] == topology].sort_values('grid_size')
        total_iter_time = topo_data['step_time'] + topo_data['exchange_time'] + topo_data['reduction_time']
        exchange_pct = (topo_data['exchange_time'] / total_iter_time) * 100
        offset = (i - len(topologies)/2 + 0.5) * width
        ax2.bar(x + offset, exchange_pct, width, label=topology, alpha=0.8)
    
    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_ylabel('Exchange Time (%)', fontsize=12)
    ax2.set_title('Exchange Time as % of Iteration Time', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Exercise 1.2.11: Exchange_Borders Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'exchange_breakdown.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'exchange_breakdown.png'}")
    plt.show()
    plt.close()

def generate_table(df: pd.DataFrame, fit_by_topo: dict, fit_by_grid: dict, 
                   global_fit: dict, output_dir: Path):
    """Generate LaTeX table with results."""
    
    # Main results table
    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exchange\_Borders Analysis Results (Exercise 1.2.11)}",
        r"\label{tab:exchange_analysis}",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Topology & Grid & Local Size & Data/Exch & Total Data & Exch Time & Exch \% \\",
        r"         &      &            & (bytes)   & (MB)       & (ms)      &         \\",
        r"\hline",
    ]
    
    for _, row in df.iterrows():
        total_iter_time = row['step_time'] + row['exchange_time'] + row['reduction_time']
        exch_pct = (row['exchange_time'] / total_iter_time) * 100 if total_iter_time > 0 else 0
        
        table_lines.append(
            f"{row['topology']} & {row['grid_size']}$\\times${row['grid_size']} & "
            f"{int(row['local_nx'])}$\\times${int(row['local_ny'])} & "
            f"{int(row['data_per_exchange_bytes']):,} & "
            f"{row['total_data_mb']:.2f} & "
            f"{row['exchange_time']*1000:.2f} & "
            f"{exch_pct:.1f}\\% \\\\"
        )
    
    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Estimated Latency and Bandwidth by Topology (Exercise 1.2.11)}",
        r"\label{tab:latency_bandwidth_topo}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Topology & Latency $\alpha$ ($\mu$s) & Bandwidth (MB/s) & $R^2$ \\",
        r"\hline",
    ])
    
    for topology, result in fit_by_topo.items():
        table_lines.append(
            f"{topology} & {result['latency_us_raw']:.2f} & {result['bandwidth_mb_s']:.2f} & {result['r_squared']:.4f} \\\\"
        )
    
    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Estimated Latency and Bandwidth by Grid Size (Exercise 1.2.11)}",
        r"\label{tab:latency_bandwidth_grid}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Grid Size & Latency $\alpha$ ($\mu$s) & Bandwidth (MB/s) & $R^2$ \\",
        r"\hline",
    ])
    
    for grid_size, result in sorted(fit_by_grid.items()):
        table_lines.append(
            f"{grid_size}$\\times${grid_size} & {result['latency_us_raw']:.2f} & {result['bandwidth_mb_s']:.2f} & {result['r_squared']:.4f} \\\\"
        )
    
    # Add global fit
    table_lines.append(r"\hline")
    table_lines.append(
        f"Global & {global_fit['latency_us_raw']:.2f} & {global_fit['bandwidth_mb_s']:.2f} & {global_fit['r_squared']:.4f} \\\\"
    )
    
    table_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    table_path = output_dir / 'exchange_analysis_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))
    print(f"Saved: {table_path}")
    
    # Also print to console
    print("\n" + "="*70)
    print("Exchange Analysis Results")
    print("="*70)
    print(f"{'Topology':<10} {'Grid':<12} {'Local':<12} {'Data/Exch':<12} {'Exch Time':<12} {'Exch %':<8}")
    print("-"*70)
    for _, row in df.iterrows():
        total_iter_time = row['step_time'] + row['exchange_time'] + row['reduction_time']
        exch_pct = (row['exchange_time'] / total_iter_time) * 100 if total_iter_time > 0 else 0
        print(f"{row['topology']:<10} {int(row['grid_size'])}x{int(row['grid_size']):<6} "
              f"{int(row['local_nx'])}x{int(row['local_ny']):<6} "
              f"{int(row['data_per_exchange_bytes']):<12,} "
              f"{row['exchange_time']*1000:<12.2f} "
              f"{exch_pct:<8.1f}")
    
    print("\n" + "="*70)
    print("Latency-Bandwidth Estimation (by Topology)")
    print("="*70)
    print(f"{'Topology':<10} {'Latency (μs)':<18} {'Bandwidth (MB/s)':<18} {'R²':<10}")
    print("-"*70)
    for topology, result in fit_by_topo.items():
        print(f"{topology:<10} {result['latency_us_raw']:<18.2f} {result['bandwidth_mb_s']:<18.2f} {result['r_squared']:<10.4f}")
    
    print("\n" + "="*70)
    print("Latency-Bandwidth Estimation (by Grid Size)")
    print("="*70)
    print(f"{'Grid Size':<12} {'Latency (μs)':<18} {'Bandwidth (MB/s)':<18} {'R²':<10}")
    print("-"*70)
    for grid_size, result in sorted(fit_by_grid.items()):
        print(f"{grid_size}x{grid_size:<6} {result['latency_us_raw']:<18.2f} {result['bandwidth_mb_s']:<18.2f} {result['r_squared']:<10.4f}")
    
    print("\n" + "="*70)
    print("Global Fit")
    print("="*70)
    print(f"Latency (α): {global_fit['latency_us_raw']:.2f} μs")
    print(f"Bandwidth:   {global_fit['bandwidth_mb_s']:.2f} MB/s")
    print(f"R²:          {global_fit['r_squared']:.4f}")
    
    if global_fit['latency_us_raw'] < 0:
        print("\nNote: Negative latency indicates the linear model may not fit well.")
        print("This could be due to:")
        print("  - Measurement noise dominating at small message sizes")
        print("  - Non-linear effects in the communication")
        print("  - Overhead being negligible compared to bandwidth-limited time")

def main():
    """Main function."""
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}")
        print("Run benchmark_exchange_analysis.sh first.")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data(RESULTS_FILE)
    
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nRaw data summary:")
    print(df.to_string())
    
    print("\nFitting latency-bandwidth model...")
    fit_by_topo = fit_latency_bandwidth_by_topology(df)
    fit_by_grid = fit_latency_bandwidth_by_gridsize(df)
    global_fit_result = fit_global(df)
    
    print("Generating plots...")
    plot_exchange_time_vs_data(df, fit_by_topo, fit_by_grid, global_fit_result, OUTPUT_DIR)
    plot_time_per_byte(df, OUTPUT_DIR)
    plot_exchange_breakdown(df, OUTPUT_DIR)
    
    print("Generating table...")
    generate_table(df, fit_by_topo, fit_by_grid, global_fit_result, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
