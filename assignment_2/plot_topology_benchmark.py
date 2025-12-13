#!/usr/bin/env python3
"""
Plot benchmark results: grid size vs time per iteration, grouped by topology.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the benchmark results
df = pd.read_csv('benchmark_topology_results.csv')

# Calculate average time_per_iter_ms for each topology and grid_size
# (averaging across the different iteration counts: 50, 100, 200, 500)
avg_df = df.groupby(['topology', 'grid_size'])['time_per_iter_ms'].mean().reset_index()

# Define colors for each topology
colors = {
    '4x1': '#1f77b4',  # blue
    '2x2': '#ff7f0e',  # orange
    '1x4': '#2ca02c',  # green
}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each topology
for topology in avg_df['topology'].unique():
    topo_data = avg_df[avg_df['topology'] == topology].sort_values('grid_size')
    
    # Scatter plot
    ax.scatter(topo_data['grid_size'], topo_data['time_per_iter_ms'], 
               color=colors.get(topology, 'gray'), label=topology, s=100, zorder=3)
    
    # Connect with lines
    ax.plot(topo_data['grid_size'], topo_data['time_per_iter_ms'], 
            color=colors.get(topology, 'gray'), linestyle='-', linewidth=2, zorder=2)

# Customize the plot
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('Time per Iteration (ms)', fontsize=12)
ax.set_title('MPI Topology Comparison: Time per Iteration vs Grid Size\n(Average across 50, 100, 200, 500 iterations)', fontsize=14)
ax.legend(title='Topology', fontsize=10)
ax.grid(True, alpha=0.3)
# ax.set_yscale('log')
# Set x-axis ticks to the actual grid sizes
ax.set_xticks(avg_df['grid_size'].unique())
ax.set_xscale('log', base=2)
plt.tight_layout()
plt.savefig('topology_benchmark_plot.png', dpi=150)
plt.show()

print("Plot saved to 'topology_benchmark_plot.png'")

# Create pivot table for LaTeX output
pivot_df = avg_df.pivot(index='grid_size', columns='topology', values='time_per_iter_ms')
pivot_df.index.name = 'Grid Size'

# Print as formatted table
print("\n" + "="*60)
print("Average Time per Iteration (ms)")
print("="*60)
print(pivot_df.to_string(float_format='%.4f'))

# Generate LaTeX table
print("\n" + "="*60)
print("LaTeX Table")
print("="*60)
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Average time per iteration (ms) for different MPI topologies and grid sizes}
\label{tab:topology_benchmark}
\begin{tabular}{r|ccc}
\toprule
Grid Size & 4$\times$1 & 2$\times$2 & 1$\times$4 \\
\midrule
"""

for grid_size in sorted(pivot_df.index):
    row = pivot_df.loc[grid_size]
    latex_table += f"{grid_size}$\\times${grid_size} & "
    latex_table += f"{row.get('4x1', 'N/A'):.4f} & " if pd.notna(row.get('4x1')) else "N/A & "
    latex_table += f"{row.get('2x2', 'N/A'):.4f} & " if pd.notna(row.get('2x2')) else "N/A & "
    latex_table += f"{row.get('1x4', 'N/A'):.4f}" if pd.notna(row.get('1x4')) else "N/A"
    latex_table += r" \\" + "\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_table)

# Save LaTeX to file
with open('topology_benchmark_table.tex', 'w') as f:
    f.write(latex_table)
print("\nLaTeX table saved to 'topology_benchmark_table.tex'")
