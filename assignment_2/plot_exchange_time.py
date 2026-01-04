#!/usr/bin/env python3
"""
Plot Exchange Time Analysis Results

Visualizes the benchmark data from Exercise 1.2.10:
- Exchange time vs grid size, grouped by number of processes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_FILE = Path("benchmark_results/exchange_time/exchange_time_results.csv")
OUTPUT_DIR = Path("benchmark_results/exchange_time")

def load_data(filepath: Path) -> pd.DataFrame:
    """Load and preprocess the benchmark results."""
    df = pd.read_csv(filepath)
    
    # Average over runs
    df_avg = df.groupby(['grid_size', 'num_procs']).agg({
        'total_time': 'mean',
        'step_time': 'mean',
        'exchange_time': 'mean',
        'reduction_time': 'mean',
        'exchange_ratio': 'mean',
        'points_per_proc': 'first'
    }).reset_index()
    
    return df_avg

def plot_exchange_time_vs_gridsize(df: pd.DataFrame, output_dir: Path):
    """Plot exchange time vs grid size, grouped by number of processes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    proc_counts = sorted(df['num_procs'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(proc_counts)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    # Calculate exchange percentage of total iteration time
    df['exchange_pct'] = (df['exchange_time'] / (df['step_time'] + df['exchange_time'] + df['reduction_time'])) * 100
    
    # Left plot: Exchange time in milliseconds
    ax1 = axes[0]
    for i, nprocs in enumerate(proc_counts):
        data = df[df['num_procs'] == nprocs].sort_values('grid_size')
        ax1.plot(data['grid_size'], data['exchange_time'] * 1000,  # Convert to ms
                marker=markers[i % len(markers)],
                color=colors[i], 
                label=f'{nprocs} processes', 
                linewidth=2, 
                markersize=10)
    
    ax1.set_xlabel('Grid Size (N×N)', fontsize=14)
    ax1.set_ylabel('Exchange Time (ms)', fontsize=14)
    ax1.set_title('MPI Boundary Exchange Time vs Grid Size', fontsize=14)
    ax1.legend(title='Processes', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xticks(sorted(df['grid_size'].unique()))
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Right plot: Exchange time as percentage of total iteration time
    ax2 = axes[1]
    for i, nprocs in enumerate(proc_counts):
        data = df[df['num_procs'] == nprocs].sort_values('grid_size')
        ax2.plot(data['grid_size'], data['exchange_pct'],
                marker=markers[i % len(markers)],
                color=colors[i], 
                label=f'{nprocs} processes', 
                linewidth=2, 
                markersize=10)
    
    ax2.set_xlabel('Grid Size (N×N)', fontsize=14)
    ax2.set_ylabel('Exchange Time (% of iteration)', fontsize=14)
    ax2.set_title('Exchange Time as % of Total Iteration Time', fontsize=14)
    ax2.legend(title='Processes', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(sorted(df['grid_size'].unique()))
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Add horizontal line at 50% to show crossover point
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (crossover)')
    
    plt.suptitle('Exercise 1.2.10: Exchange Time Analysis\n(grouped by number of processes)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'exchange_time_vs_gridsize.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'exchange_time_vs_gridsize.png'}")
    plt.show()
    plt.close()

def main():
    """Main function to generate all plots."""
    # Check if results file exists
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}")
        print("Please run the benchmark script first: ./benchmark_exchange_time.sh")
        return
    
    # Load data
    print(f"Loading data from {RESULTS_FILE}...")
    df = load_data(RESULTS_FILE)
    print(f"Loaded {len(df)} data points")
    print(f"Grid sizes: {sorted(df['grid_size'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plot
    print("\nGenerating plot...")
    plot_exchange_time_vs_gridsize(df, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
