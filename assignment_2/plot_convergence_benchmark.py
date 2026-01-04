#!/usr/bin/env python3
"""
Plot benchmark results: grid size vs iterations to convergence.
Analyzes how the number of iterations needed to reach convergence depends on problem size.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the benchmark results
df = pd.read_csv('benchmark_convergence_results.csv')

# Filter out failed/timeout runs
df_valid = df[~df['iterations'].isin(['TIMEOUT', 'FAILED'])].copy()
df_valid['iterations'] = pd.to_numeric(df_valid['iterations'])
df_valid['time_s'] = pd.to_numeric(df_valid['time_s'])

# Calculate average iterations and time for each grid size
avg_df = df_valid.groupby('grid_size').agg({
    'iterations': ['mean', 'std', 'count'],
    'time_s': ['mean', 'std'],
    'converged': 'first'
}).reset_index()

# Flatten column names
avg_df.columns = ['grid_size', 'iterations_mean', 'iterations_std', 'num_runs', 
                  'time_mean', 'time_std', 'converged']

# Calculate N (total grid points = grid_size^2)
avg_df['N'] = avg_df['grid_size'] ** 2

print("="*60)
print("Convergence vs Grid Size Analysis")
print("="*60)
print(f"\nData points: {len(avg_df)}")
print(avg_df.to_string(index=False))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Iterations vs Grid Size
ax1.errorbar(avg_df['grid_size'], avg_df['iterations_mean'], 
             yerr=avg_df['iterations_std'], fmt='o-', capsize=5, 
             markersize=10, linewidth=2, color='#1f77b4')
ax1.set_xlabel('Grid Size (N×N)', fontsize=12)
ax1.set_ylabel('Iterations to Convergence', fontsize=12)
ax1.set_title('Iterations to Convergence vs Grid Size', fontsize=14)
# ax1.set_xscale('log', base=2)
# ax1.set_yscale('log', base=2)
ax1.set_yticks(avg_df['iterations_mean'].values)
ax1.set_yticklabels([f'{int(y)}' for y in avg_df['iterations_mean'].values])
ax1.grid(True, alpha=0.3, which='both')
ax1.grid(True, alpha=0.5, which='major')

# Add trend line (linear fit: iterations = a + b * grid_size)
if len(avg_df) >= 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(avg_df['grid_size'], avg_df['iterations_mean'])
    
    x_fit = np.linspace(avg_df['grid_size'].min(), avg_df['grid_size'].max(), 100)
    y_fit = intercept + slope * x_fit
    ax1.plot(x_fit, y_fit, '--', color='red', linewidth=2, 
             label=f'Fit: iter = {intercept:.0f} + {slope:.2f}·N (R²={r_value**2:.3f})')
    ax1.legend(fontsize=10)

# Plot 2: Time vs Grid Size
ax2.errorbar(avg_df['grid_size'], avg_df['time_mean'], 
             yerr=avg_df['time_std'], fmt='s-', capsize=5, 
             markersize=10, linewidth=2, color='#ff7f0e')
ax2.set_xlabel('Grid Size (N×N)', fontsize=12)
ax2.set_ylabel('Time to Convergence (ms)', fontsize=12)
ax2.set_title('Time to Convergence vs Grid Size', fontsize=14)
ax2.grid(True, alpha=0.3)
# ax2.set_xscale('log', base=2)
# ax2.set_yscale('log', base=2)

# Add trend line for time (power law fit)
if len(avg_df) >= 2:
    log_x = np.log(avg_df['grid_size'])
    log_t = np.log(avg_df['time_mean'])
    slope_t, intercept_t, r_value_t, p_value_t, std_err_t = stats.linregress(log_x, log_t)
    
    t_fit = np.exp(intercept_t) * x_fit ** slope_t
    ax2.plot(x_fit, t_fit, '--', color='red', linewidth=2, 
             label=f'Fit: time ∝ N^{slope_t:.2f} (R²={r_value_t**2:.3f})')
    ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('convergence_benchmark_plot.png', dpi=150)
plt.show()

print("\nPlot saved to 'convergence_benchmark_plot.png'")

# Print scaling analysis
print("\n" + "="*60)
print("Scaling Analysis")
print("="*60)

if len(avg_df) >= 2:
    print(f"\nIteration scaling: iterations ∝ N^{slope:.2f}")
    print(f"  - If exponent ≈ 1: iterations scale linearly with grid size (optimal SOR)")
    print(f"  - If exponent ≈ 2: iterations scale quadratically (simple iterative methods)")
    print(f"\nTime scaling: time ∝ N^{slope_t:.2f}")
    print(f"  - Total work = iterations × work_per_iteration")
    print(f"  - Work per iteration ∝ N² (grid points)")
    print(f"  - Expected total time ∝ N^(2+{slope:.2f}) = N^{2+slope:.2f}")

# Calculate ratios relative to smallest grid
print("\n" + "="*60)
print("Iteration Count Ratios")
print("="*60)

baseline_size = avg_df['grid_size'].min()
baseline_iter = avg_df[avg_df['grid_size'] == baseline_size]['iterations_mean'].values[0]

print(f"\nRelative to {baseline_size}×{baseline_size} ({baseline_iter:.0f} iterations):")
for _, row in avg_df.iterrows():
    size = row['grid_size']
    iterations = row['iterations_mean']
    ratio = iterations / baseline_iter
    size_ratio = size / baseline_size
    print(f"  {size}×{size}: {iterations:.0f} iterations (ratio: {ratio:.2f}x, size ratio: {size_ratio:.1f}x)")

# Generate formatted table
print("\n" + "="*60)
print("Formatted Table")
print("="*60)
print("\nGrid Size | Iterations | Time (s) | Iter Ratio | Size Ratio")
print("----------|------------|----------|------------|------------")
for _, row in avg_df.iterrows():
    size = row['grid_size']
    iterations = row['iterations_mean']
    time = row['time_mean']
    ratio = iterations / baseline_iter
    size_ratio = size / baseline_size
    print(f"{size:>9} | {iterations:>10.0f} | {time:>8.3f} | {ratio:>10.2f} | {size_ratio:>10.1f}")

# Generate LaTeX table
print("\n" + "="*60)
print("LaTeX Table")
print("="*60)

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Iterations to convergence for different grid sizes using SOR solver ($\omega = 1.95$)}
\label{tab:convergence_benchmark}
\begin{tabular}{r|rrr|rr}
\toprule
Grid Size & Iterations & Std Dev & Time (s) & Iter Ratio & Size Ratio \\
\midrule
"""

for _, row in avg_df.iterrows():
    size = int(row['grid_size'])
    iterations = row['iterations_mean']
    iter_std = row['iterations_std'] if pd.notna(row['iterations_std']) else 0
    time = row['time_mean']
    ratio = iterations / baseline_iter
    size_ratio = size / baseline_size
    
    latex_table += f"{size}$\\times${size} & {iterations:.0f} & {iter_std:.1f} & {time:.3f} & {ratio:.2f} & {size_ratio:.1f}"
    latex_table += r" \\" + "\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex_table)

# Save LaTeX to file
with open('convergence_benchmark_table.tex', 'w') as f:
    f.write(latex_table)
print("\nLaTeX table saved to 'convergence_benchmark_table.tex'")

# Additional: Expected vs Actual scaling
print("\n" + "="*60)
print("Expected Scaling Behavior")
print("="*60)
print("""
For iterative methods solving the Poisson equation:

1. Gauss-Seidel: iterations ∝ N² (where N is grid size in one dimension)
   - Condition number κ ∝ N², convergence rate ∝ 1 - O(1/κ)

2. SOR with optimal ω: iterations ∝ N
   - Optimal ω ≈ 2 - O(1/N)
   - Convergence rate improved to ∝ 1 - O(1/√κ) = 1 - O(1/N)

3. Conjugate Gradient: iterations ∝ N  
   - Convergence rate ∝ (√κ - 1)/(√κ + 1) = 1 - O(1/N)

Your measured exponent: """ + f"{slope:.2f}" + """
""")
