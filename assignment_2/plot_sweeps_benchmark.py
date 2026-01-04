#!/usr/bin/env python3
"""
Plot benchmark results: sweeps per exchange vs iterations/time.
Analyzes the effect of multiple sweeps between border exchanges on convergence.

KEY FINDING: Doing multiple sweeps without boundary exchange HURTS convergence
because the boundary values become stale, causing local divergence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the benchmark results
df = pd.read_csv('benchmark_sweeps_results.csv')

# Filter out failed/timeout runs
df_valid = df[~df['iterations'].isin(['TIMEOUT', 'FAILED'])].copy()
df_valid['iterations'] = pd.to_numeric(df_valid['iterations'])
df_valid['total_sweep_pairs'] = pd.to_numeric(df_valid['total_sweep_pairs'])
df_valid['time_s'] = pd.to_numeric(df_valid['time_s'])
df_valid['time_per_iter_ms'] = pd.to_numeric(df_valid['time_per_iter_ms'])
df_valid['time_per_sweep_ms'] = pd.to_numeric(df_valid['time_per_sweep_ms'])

# Calculate averages for each sweeps_per_exchange value
avg_df = df_valid.groupby('sweeps_per_exchange').agg({
    'iterations': ['mean', 'std'],
    'total_sweep_pairs': ['mean', 'std'],
    'time_s': ['mean', 'std'],
    'time_per_iter_ms': ['mean', 'std'],
    'time_per_sweep_ms': ['mean', 'std'],
    'converged': 'first'
}).reset_index()

# Flatten column names
avg_df.columns = ['sweeps', 'iterations_mean', 'iterations_std', 
                  'total_sweeps_mean', 'total_sweeps_std',
                  'time_mean', 'time_std',
                  'time_per_iter_mean', 'time_per_iter_std',
                  'time_per_sweep_mean', 'time_per_sweep_std',
                  'converged']

# Number of boundary exchanges = iterations
avg_df['exchanges'] = avg_df['iterations_mean']

# Calculate expected iterations if sweeps didn't hurt convergence
baseline_iters = avg_df[avg_df['sweeps'] == 1]['iterations_mean'].values[0]
avg_df['expected_iters'] = baseline_iters / avg_df['sweeps']
avg_df['convergence_penalty'] = avg_df['iterations_mean'] / avg_df['expected_iters']

print("="*60)
print("Sweeps per Exchange Analysis (SOR Solver)")
print("="*60)
print(f"\nData points: {len(avg_df)}")
print("\nAverages:")
print(avg_df.to_string(index=False))

print("\n" + "="*60)
print("Key Finding: Multiple sweeps without exchange HURTS convergence!")
print("="*60)
print(f"Baseline (sweeps=1): {baseline_iters:.0f} iterations")
print()
for _, row in avg_df.iterrows():
    expected = row['expected_iters']
    actual = row['iterations_mean']
    penalty = row['convergence_penalty']
    print(f"Sweeps={int(row['sweeps']):2d}: expected {expected:5.0f} iters, got {actual:5.0f} "
          f"({penalty:.1f}x penalty), total work: {row['total_sweeps_mean']:.0f} sweep-pairs, "
          f"{row['time_per_sweep_mean']:.4f} ms/sweep")

# Create figure with multiple subplots (3x2 grid)
fig, axes = plt.subplots(1, 2, figsize=(8.27, 8.27/1.5))
# fig.suptitle('Effect of Multiple Sweeps per Exchange on SOR Convergence\n(Key Finding: Stale boundaries HURT convergence)', 
            #  fontsize=14, fontweight='bold')

# Plot 1: Iterations - Actual vs Expected
# ax1 = axes[0]
# ax1.plot(avg_df['sweeps'], avg_df['iterations_mean'], 'o-', 
#          markersize=10, linewidth=2, color='#1f77b4', label='Actual iterations')
# ax1.plot(avg_df['sweeps'], avg_df['expected_iters'], 's--', 
#          markersize=8, linewidth=2, color='#ff7f0e', label='Expected (if no penalty)')
# ax1.fill_between(avg_df['sweeps'], avg_df['expected_iters'], avg_df['iterations_mean'],
#                  alpha=0.3, color='red', label='Convergence penalty')
# ax1.set_xlabel('Sweeps per Exchange', fontsize=12)
# ax1.set_ylabel('Iterations to Convergence', fontsize=12)
# ax1.set_title('Iterations: Actual vs Expected', fontsize=14)
# ax1.grid(True, alpha=0.3)
# ax1.set_xticks(avg_df['sweeps'])
# ax1.legend(fontsize=9)

# # Plot 2: Convergence Penalty Factor
# ax2 = axes[0, 1]
# ax2.bar(avg_df['sweeps'], avg_df['convergence_penalty'], color='#d62728', alpha=0.7)
# ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='No penalty (1.0x)')
# ax2.set_xlabel('Sweeps per Exchange', fontsize=12)
# ax2.set_ylabel('Convergence Penalty Factor', fontsize=12)
# ax2.set_title('How Much Slower Due to Stale Boundaries', fontsize=14)
# ax2.grid(True, alpha=0.3, axis='y')
# ax2.set_xticks(avg_df['sweeps'])
# for i, (s, p) in enumerate(zip(avg_df['sweeps'], avg_df['convergence_penalty'])):
#     ax2.text(s, p + 0.3, f'{p:.1f}x', ha='center', fontsize=10, fontweight='bold')

# # Plot 3: Total Computational Work
# ax3 = axes[0, 2]
# ax3.bar(avg_df['sweeps'], avg_df['total_sweeps_mean'], color='#9467bd', alpha=0.7)
# # Add baseline reference
# baseline_work = avg_df[avg_df['sweeps'] == 1]['total_sweeps_mean'].values[0]
# ax3.axhline(y=baseline_work, color='green', linestyle='--', linewidth=2, 
#             label=f'Baseline: {baseline_work:.0f}')
# ax3.set_xlabel('Sweeps per Exchange', fontsize=12)
# ax3.set_ylabel('Total Sweep-Pairs', fontsize=12)
# ax3.set_title('Total Computational Work', fontsize=14)
# ax3.grid(True, alpha=0.3, axis='y')
# ax3.set_xticks(avg_df['sweeps'])
# ax3.legend(fontsize=10)

# Plot 4: Total Time vs Sweeps
ax4 = axes[0]
ax4.errorbar(avg_df['sweeps'], avg_df['time_mean'], 
             yerr=avg_df['time_std'], fmt='s-', capsize=5, 
             markersize=10, linewidth=2, color='#ff7f0e')
ax4.set_xlabel('Sweeps per Exchange', fontsize=12)
ax4.set_ylabel('Total Time (s)', fontsize=12)
ax4.set_title('Total Runtime', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(avg_df['sweeps'])

# Plot 5: Time per Sweep-Pair (should be constant or slightly decreasing)
ax5 = axes[1]
ax5.errorbar(avg_df['sweeps'], avg_df['time_per_sweep_mean'], 
             yerr=avg_df['time_per_sweep_std'], fmt='^-', capsize=5,
             markersize=10, linewidth=2, color='#2ca02c')
ax5.set_xlabel('Sweeps per Exchange', fontsize=12)
ax5.set_ylabel('Time per Sweep-Pair (ms)', fontsize=12)
ax5.set_title('Computational Efficiency', fontsize=14)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(avg_df['sweeps'])
# Show mean line
mean_time = avg_df['time_per_sweep_mean'].mean()
ax5.axhline(y=mean_time, color='r', linestyle='--', linewidth=2, 
            label=f'Mean: {mean_time:.4f} ms')
ax5.legend(fontsize=10)

# # Plot 6: Efficiency - Exchanges saved vs Work increase
# ax6 = axes[1, 2]
# baseline_exchanges = avg_df[avg_df['sweeps'] == 1]['exchanges'].values[0]
# baseline_work = avg_df[avg_df['sweeps'] == 1]['total_sweeps_mean'].values[0]

# # Calculate ratios relative to baseline
# exchanges_saved_pct = (1 - avg_df['exchanges'] / baseline_exchanges) * 100
# work_increase_pct = (avg_df['total_sweeps_mean'] / baseline_work - 1) * 100

# x = np.arange(len(avg_df))
# width = 0.35
# ax6.bar(x - width/2, -exchanges_saved_pct, width, label='Exchanges saved (%)', color='#2ca02c', alpha=0.7)
# ax6.bar(x + width/2, work_increase_pct, width, label='Extra work (%)', color='#d62728', alpha=0.7)
# ax6.axhline(y=0, color='k', linewidth=1)
# ax6.set_xlabel('Sweeps per Exchange', fontsize=12)
# ax6.set_ylabel('Change from Baseline (%)', fontsize=12)
# ax6.set_title('Trade-off: Communication vs Computation', fontsize=14)
# ax6.set_xticks(x)
# ax6.set_xticklabels(avg_df['sweeps'].astype(int))
# ax6.legend(fontsize=10, loc='upper left')
# ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('sweeps_benchmark_plot.png', dpi=150)
plt.show()

print("\nPlot saved to 'sweeps_benchmark_plot.png'")

# Print analysis
print("\n" + "="*60)
print("Analysis")
print("="*60)

baseline_iters = avg_df[avg_df['sweeps'] == 1]['iterations_mean'].values[0]
baseline_time = avg_df[avg_df['sweeps'] == 1]['time_mean'].values[0]
baseline_total_sweeps = avg_df[avg_df['sweeps'] == 1]['total_sweeps_mean'].values[0]

print("\nEffect of increasing sweeps per exchange:")
print("-" * 50)

for _, row in avg_df.iterrows():
    sweeps = int(row['sweeps'])
    iters = row['iterations_mean']
    time = row['time_mean']
    total_sweeps = row['total_sweeps_mean']
    time_per_sweep = row['time_per_sweep_mean']
    
    sweep_ratio = total_sweeps / baseline_total_sweeps
    time_ratio = time / baseline_time
    
    print(f"Sweeps={sweeps:2d}: {iters:5.0f} iters, {total_sweeps:5.0f} sweep-pairs ({sweep_ratio:5.2f}x work), "
          f"{time:6.3f}s ({time_ratio:5.2f}x time), {time_per_sweep:.4f} ms/sweep")

# Find optimal sweeps value
optimal_idx = avg_df['time_mean'].idxmin()
optimal_sweeps = avg_df.loc[optimal_idx, 'sweeps']
optimal_time = avg_df.loc[optimal_idx, 'time_mean']
optimal_iters = avg_df.loc[optimal_idx, 'iterations_mean']

print(f"\nOptimal configuration:")
print(f"  Sweeps per exchange: {optimal_sweeps}")
print(f"  Time: {optimal_time:.3f} s")
print(f"  Iterations: {optimal_iters:.0f}")
print(f"  Speedup vs baseline: {baseline_time/optimal_time:.2f}x")

# Generate LaTeX table
print("\n" + "="*60)
print("LaTeX Table")
print("="*60)

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Effect of sweeps per exchange on SOR solver convergence}
\label{tab:sweeps_benchmark}
\begin{tabular}{r|rrr|rr}
\toprule
Sweeps & Iterations & Total Sweep-Pairs & Time (s) & ms/Sweep & Exchanges \\
\midrule
"""

for _, row in avg_df.iterrows():
    sweeps = int(row['sweeps'])
    iters = row['iterations_mean']
    total_sweeps = row['total_sweeps_mean']
    time = row['time_mean']
    time_per_sweep = row['time_per_sweep_mean']
    exchanges = row['exchanges']
    
    latex_table += f"{sweeps} & {iters:.0f} & {total_sweeps:.0f} & {time:.3f} & {time_per_sweep:.4f} & {exchanges:.0f}"
    latex_table += r" \\" + "\n"

latex_table += r"""\midrule
\multicolumn{6}{l}{Optimal: """ + f"{int(optimal_sweeps)} sweeps, {optimal_time:.3f}s, {baseline_time/optimal_time:.2f}$\\times$ speedup" + r"""} \\
\bottomrule
\end{tabular}
\end{table}"""

print(latex_table)

# Save LaTeX to file
with open('sweeps_benchmark_table.tex', 'w') as f:
    f.write(latex_table)
print("\nLaTeX table saved to 'sweeps_benchmark_table.tex'")

# Additional analysis
print("\n" + "="*60)
print("Discussion")
print("="*60)
print("""
By performing multiple sweeps between border exchanges:

1. COMMUNICATION REDUCTION:
   - Fewer MPI exchanges are performed per iteration
   - Each exchange has latency overhead that is amortized over more work
   
2. CONVERGENCE IMPACT:
   - More sweeps = more "stale" boundary data
   - Can increase the number of iterations needed to converge
   - The numerical algorithm is modified (not mathematically equivalent)

3. TRADE-OFF:
   - Optimal value balances communication savings vs extra iterations
   - Depends on: network latency, problem size, processor count

4. EXPECTED BEHAVIOR:
   - Time per iteration should DECREASE (fewer exchanges)
   - Total iterations may INCREASE (stale boundaries slow convergence)
   - Total time has an optimal point (minimum)
""")
