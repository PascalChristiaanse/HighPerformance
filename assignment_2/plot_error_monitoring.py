#!/usr/bin/env python3
"""
Plot error (residual) vs iteration number for the 800x800 grid.
Analyzes convergence behavior of the iterative solver.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Configuration
INPUT_CSV = 'error_monitoring/error_vs_iteration_800.csv'
GRID_SIZE = 800

# Check if file exists
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found!")
    print("Please run benchmark_error_monitoring.sh first.")
    exit(1)

# Read the benchmark results
df = pd.read_csv(INPUT_CSV)

print("="*60)
print(f"Error vs Iteration Analysis for {GRID_SIZE}×{GRID_SIZE} Grid")
print("="*60)
print(f"\nTotal iterations: {len(df)}")
print(f"Initial residual: {df['residual'].iloc[0]:.6e}")
print(f"Final residual: {df['residual'].iloc[-1]:.6e}")

# Calculate convergence rate
# For iterative methods: residual(n) ≈ residual(0) * ρ^n
# where ρ is the spectral radius (convergence rate)
# log(residual) = log(residual(0)) + n * log(ρ)

# Filter out zero or negative residuals for log fitting
df_positive = df[df['residual'] > 0].copy()

if len(df_positive) > 10:
    # Fit linear model to log(residual) vs iteration
    log_residual = np.log10(df_positive['residual'])
    iterations = df_positive['iteration']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, log_residual)
    
    # Convergence rate ρ = 10^slope
    rho = 10 ** slope
    
    print(f"\nConvergence Analysis:")
    print(f"  Convergence rate (ρ): {rho:.6f}")
    print(f"  Log10 slope: {slope:.6f}")
    print(f"  R² of fit: {r_value**2:.4f}")
    print(f"  Residual reduction per iteration: {(1-rho)*100:.2f}%")
    
    # Estimate iterations to reduce by factor of 10
    iters_per_decade = -1.0 / slope if slope != 0 else float('inf')
    print(f"  Iterations to reduce error by 10x: {iters_per_decade:.1f}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Residual vs Iteration (log scale)
ax1 = axes[0, 0]
ax1.semilogy(df['iteration'], df['residual'], 'b-', linewidth=1, alpha=0.8)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Residual (log scale)', fontsize=12)
ax1.set_title(f'Residual vs Iteration ({GRID_SIZE}×{GRID_SIZE} grid)', fontsize=14)
ax1.grid(True, alpha=0.3, which='both')

# Add trend line
if len(df_positive) > 10:
    x_fit = np.linspace(0, df['iteration'].max(), 100)
    y_fit = 10 ** (intercept + slope * x_fit)
    ax1.semilogy(x_fit, y_fit, 'r--', linewidth=2, 
                 label=f'Fit: ρ = {rho:.4f} (R² = {r_value**2:.3f})')
    ax1.legend(fontsize=10)

# Plot 2: Residual vs Iteration (linear scale, early iterations)
ax2 = axes[0, 1]
early_iters = min(100, len(df))
ax2.plot(df['iteration'][:early_iters], df['residual'][:early_iters], 'b-', linewidth=1.5)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_title(f'Early Convergence (first {early_iters} iterations)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: Convergence rate per iteration
ax3 = axes[1, 0]
if len(df) > 1:
    # Calculate iteration-by-iteration convergence rate
    residuals = df['residual'].values
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        conv_rate = residuals[1:] / residuals[:-1]
        conv_rate = np.where(np.isfinite(conv_rate) & (conv_rate > 0), conv_rate, np.nan)
    
    ax3.plot(df['iteration'][1:], conv_rate, 'g-', linewidth=0.5, alpha=0.5)
    
    # Add moving average
    window = min(50, len(conv_rate) // 10)
    if window > 1:
        conv_rate_smooth = pd.Series(conv_rate).rolling(window=window, center=True).mean()
        ax3.plot(df['iteration'][1:], conv_rate_smooth, 'r-', linewidth=2, 
                 label=f'Moving avg (window={window})')
    
    ax3.axhline(y=rho, color='k', linestyle='--', linewidth=2, label=f'Average ρ = {rho:.4f}')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Convergence Rate (ρ = r_{n+1}/r_n)', fontsize=12)
    ax3.set_title('Per-Iteration Convergence Rate', fontsize=14)
    ax3.set_ylim([0.9, 1.01])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

# Plot 4: Time breakdown per iteration
ax4 = axes[1, 1]
if 'step_time' in df.columns and 'exchange_time' in df.columns:
    # Use moving average for smoother visualization
    window = min(50, len(df) // 10)
    if window > 1:
        step_smooth = df['step_time'].rolling(window=window, center=True).mean() * 1000
        exchange_smooth = df['exchange_time'].rolling(window=window, center=True).mean() * 1000
        reduction_smooth = df['reduction_time'].rolling(window=window, center=True).mean() * 1000
        
        ax4.plot(df['iteration'], step_smooth, 'b-', linewidth=1.5, label='Compute', alpha=0.8)
        ax4.plot(df['iteration'], exchange_smooth, 'r-', linewidth=1.5, label='Exchange', alpha=0.8)
        ax4.plot(df['iteration'], reduction_smooth, 'g-', linewidth=1.5, label='Reduction', alpha=0.8)
    else:
        ax4.plot(df['iteration'], df['step_time'] * 1000, 'b-', linewidth=0.5, label='Compute', alpha=0.5)
        ax4.plot(df['iteration'], df['exchange_time'] * 1000, 'r-', linewidth=0.5, label='Exchange', alpha=0.5)
        ax4.plot(df['iteration'], df['reduction_time'] * 1000, 'g-', linewidth=0.5, label='Reduction', alpha=0.5)
    
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Time (ms)', fontsize=12)
    ax4.set_title('Time Breakdown per Iteration', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('error_monitoring_plot.png', dpi=150)
plt.show()

print("\nPlot saved to 'error_monitoring_plot.png'")

# Generate summary statistics table
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

# Calculate statistics at different iteration checkpoints
checkpoints = [10, 50, 100, 500, 1000, 5000, len(df)-1]
checkpoints = [c for c in checkpoints if c < len(df)]

print("\nIteration | Residual      | Reduction from Start")
print("----------|---------------|---------------------")
initial_residual = df['residual'].iloc[0]
for cp in checkpoints:
    residual = df['residual'].iloc[cp]
    reduction = initial_residual / residual if residual > 0 else float('inf')
    print(f"{cp:>9} | {residual:>13.6e} | {reduction:>19.2e}x")

# Generate LaTeX table
print("\n" + "="*60)
print("LaTeX Table")
print("="*60)

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Convergence behavior for """ + f"{GRID_SIZE}$\\times${GRID_SIZE}" + r""" grid using SOR solver ($\omega = 1.95$)}
\label{tab:error_monitoring}
\begin{tabular}{r|cc}
\toprule
Iteration & Residual & Reduction Factor \\
\midrule
"""

for cp in checkpoints:
    residual = df['residual'].iloc[cp]
    reduction = initial_residual / residual if residual > 0 else float('inf')
    latex_table += f"{cp} & {residual:.4e} & {reduction:.2e}$\\times$"
    latex_table += r" \\" + "\n"

latex_table += r"""\midrule
\multicolumn{3}{l}{Convergence rate $\rho$ = """ + f"{rho:.6f}" + r"""} \\
\multicolumn{3}{l}{Iterations per decade = """ + f"{iters_per_decade:.1f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}"""

print(latex_table)

# Save LaTeX to file
with open('error_monitoring_table.tex', 'w') as f:
    f.write(latex_table)
print("\nLaTeX table saved to 'error_monitoring_table.tex'")

# Additional analysis: theoretical vs actual convergence rate
print("\n" + "="*60)
print("Convergence Rate Analysis")
print("="*60)
print(f"""
For the SOR method with optimal omega, the theoretical convergence rate is:

  ρ_optimal ≈ 1 - 2π/N  for an N×N grid

For N = {GRID_SIZE}:
  ρ_theoretical ≈ 1 - 2π/{GRID_SIZE} ≈ {1 - 2*np.pi/GRID_SIZE:.6f}

Measured convergence rate:
  ρ_measured = {rho:.6f}

The actual omega used (ω = 1.95) may not be exactly optimal for this grid size.
The optimal omega for an N×N grid is approximately:

  ω_optimal ≈ 2 / (1 + sin(π/N)) ≈ {2 / (1 + np.sin(np.pi/GRID_SIZE)):.4f}
""")

# Save raw data summary to CSV
summary_data = {
    'metric': ['grid_size', 'total_iterations', 'initial_residual', 'final_residual', 
               'convergence_rate', 'iters_per_decade', 'r_squared'],
    'value': [GRID_SIZE, len(df), df['residual'].iloc[0], df['residual'].iloc[-1],
              rho, iters_per_decade, r_value**2]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('error_monitoring/summary_800.csv', index=False)
print("\nSummary saved to 'error_monitoring/summary_800.csv'")
