#!/usr/bin/env python3
"""
Analysis script to investigate why SOR iterations increase with processor count.

The phenomenon observed is a well-known characteristic of domain decomposition methods
with relaxation solvers like SOR. When the domain is split across multiple processes:

1. **Information propagation is limited**: In sequential SOR, updates propagate across 
   the entire domain in a single sweep. With domain decomposition, information can only
   propagate within each subdomain during a sweep, and boundary exchange only occurs
   AFTER the sweep completes.

2. **Block Jacobi behavior at boundaries**: The parallel SOR effectively becomes a 
   Block Jacobi method at subdomain interfaces. Each subdomain updates independently
   using "stale" boundary values until the next exchange.

3. **Optimal omega changes**: The optimal relaxation parameter ω depends on the problem
   size and geometry. When the domain is decomposed, the optimal ω for each subdomain
   differs from the global optimal ω, reducing convergence rate.

4. **More interfaces = slower convergence**: As you add more processes, you create more
   subdomain interfaces where information propagation is delayed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    """Load the CG comparison benchmark results."""
    csv_path = "benchmark_results/cg_comparison/cg_comparison_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)

def analyze_iteration_scaling(df):
    """Analyze how iterations scale with processor count for each solver."""
    
    # Group by solver and num_procs, compute mean iterations
    summary = df.groupby(['solver', 'num_procs']).agg({
        'iterations': ['mean', 'std'],
        'total_time_s': 'mean',
        'time_per_iter_us': 'mean'
    }).reset_index()
    
    summary.columns = ['solver', 'num_procs', 'iterations_mean', 'iterations_std', 
                       'total_time_mean', 'time_per_iter_mean']
    
    return summary

def compute_theoretical_scaling():
    """
    Compute theoretical iteration scaling for domain-decomposed SOR.
    
    For a 1D decomposition with P processors:
    - Sequential SOR converges in O(N) iterations
    - Domain-decomposed SOR converges in O(N/P * P^2) = O(N*P) iterations
      due to the need for P-fold more iterations for information to traverse boundaries
    
    For a 2D decomposition with sqrt(P) x sqrt(P) processors:
    - The scaling is more complex but generally O(N * sqrt(P))
    """
    pass

def plot_iteration_analysis(summary, output_dir="benchmark_results/cg_comparison"):
    """Create visualizations explaining the iteration scaling."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for each solver
    sor_data = summary[summary['solver'] == 'sor']
    cg_data = summary[summary['solver'] == 'cg']
    
    # Figure 1: Iterations vs Processor Count
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Iterations
    ax1 = axes[0]
    ax1.plot(sor_data['num_procs'], sor_data['iterations_mean'], 'o-', 
             color='#d62728', linewidth=2, markersize=10, label='SOR (ω=1.95)')
    ax1.plot(cg_data['num_procs'], cg_data['iterations_mean'], 's-', 
             color='#1f77b4', linewidth=2, markersize=10, label='Conjugate Gradient')
    
    ax1.axhline(y=125, color='gray', linestyle='--', alpha=0.7, label='CG expected (125)')
    ax1.axhline(y=440, color='gray', linestyle=':', alpha=0.7, label='Sequential SOR (440)')
    
    ax1.set_xlabel('Number of Processes', fontsize=12)
    ax1.set_ylabel('Iterations to Converge', fontsize=12)
    ax1.set_title('Iteration Count vs. Process Count', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sor_data['num_procs'].values)
    
    # Right plot: Ratio to sequential
    ax2 = axes[1]
    sor_seq_iters = sor_data[sor_data['num_procs'] == 1]['iterations_mean'].values[0]
    cg_seq_iters = cg_data[cg_data['num_procs'] == 1]['iterations_mean'].values[0]
    
    sor_ratio = sor_data['iterations_mean'] / sor_seq_iters
    cg_ratio = cg_data['iterations_mean'] / cg_seq_iters
    
    ax2.plot(sor_data['num_procs'], sor_ratio, 'o-', 
             color='#d62728', linewidth=2, markersize=10, label='SOR')
    ax2.plot(cg_data['num_procs'], cg_ratio, 's-', 
             color='#1f77b4', linewidth=2, markersize=10, label='CG')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Ideal (no increase)')
    
    ax2.set_xlabel('Number of Processes', fontsize=12)
    ax2.set_ylabel('Iteration Ratio (parallel / sequential)', fontsize=12)
    ax2.set_title('Iteration Scaling Relative to Sequential', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sor_data['num_procs'].values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sor_iteration_scaling_analysis.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sor_iteration_scaling_analysis.pdf'))
    plt.close()
    
    # Figure 2: Explanation diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a schematic showing why iterations increase
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Why SOR Iterations Increase with Processor Count', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Sequential case (left)
    ax.text(2.5, 8.5, 'Sequential SOR', fontsize=12, fontweight='bold', ha='center')
    rect1 = plt.Rectangle((0.5, 6), 4, 2, fill=True, facecolor='lightblue', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.annotate('', xy=(4.3, 7), xytext=(0.7, 7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.5, 7, 'Information flows\nacross entire domain', fontsize=9, ha='center', va='center')
    ax.text(2.5, 5.5, '440 iterations', fontsize=10, ha='center', color='darkblue')
    
    # Parallel case (right)
    ax.text(7.5, 8.5, 'Parallel SOR (4 processes)', fontsize=12, fontweight='bold', ha='center')
    
    # Four subdomains
    colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']
    for i, (x, c) in enumerate(zip([5.5, 7.0, 5.5, 7.0], colors)):
        y = 7.0 if i < 2 else 6.0
        rect = plt.Rectangle((x, y), 1.4, 0.9, fill=True, facecolor=c, 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.7, y + 0.45, f'P{i}', fontsize=9, ha='center', va='center')
    
    # Boundary lines (dashed red)
    ax.plot([6.9, 6.9], [6, 8], 'r--', linewidth=2)
    ax.plot([5.5, 8.4], [6.9, 6.9], 'r--', linewidth=2)
    
    ax.text(7.5, 5.5, '2210 iterations (5x more!)', fontsize=10, ha='center', color='darkred')
    
    # Explanation boxes
    explanations = [
        ("1. Delayed Information Transfer", 
         "Updates within a subdomain use 'stale'\nboundary values from neighbors until\nthe next exchange step."),
        ("2. Block Jacobi Behavior",
         "At subdomain boundaries, the method\nbehaves like slower Block Jacobi\ninstead of true SOR."),
        ("3. Suboptimal ω",
         "The optimal relaxation parameter ω\nchanges when domain is decomposed,\nreducing convergence rate."),
    ]
    
    y_pos = 4.0
    for title, text in explanations:
        ax.text(0.5, y_pos, title, fontsize=10, fontweight='bold')
        ax.text(0.5, y_pos - 0.8, text, fontsize=9)
        y_pos -= 1.8
    
    # CG explanation
    ax.text(5.5, 4.0, "Why CG doesn't have this problem:", fontsize=10, fontweight='bold')
    ax.text(5.5, 3.2, "• CG uses global dot products (MPI_Allreduce)\n"
                      "• Information propagates globally each iteration\n"
                      "• Convergence depends only on matrix condition\n"
                      "  number, not domain decomposition", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sor_iteration_scaling_explanation.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sor_iteration_scaling_explanation.pdf'))
    plt.close()
    
    print(f"Saved analysis plots to {output_dir}/")

def print_analysis_report(summary):
    """Print a detailed analysis report."""
    
    print("\n" + "="*80)
    print("ANALYSIS: Why SOR Iterations Increase with Processor Count")
    print("="*80)
    
    print("\n1. OBSERVED DATA:")
    print("-" * 60)
    
    sor_data = summary[summary['solver'] == 'sor'].sort_values('num_procs')
    cg_data = summary[summary['solver'] == 'cg'].sort_values('num_procs')
    
    print("\nSOR Iterations:")
    for _, row in sor_data.iterrows():
        print(f"  {int(row['num_procs']):2d} processes: {row['iterations_mean']:.0f} iterations")
    
    print("\nCG Iterations:")
    for _, row in cg_data.iterrows():
        print(f"  {int(row['num_procs']):2d} processes: {row['iterations_mean']:.0f} iterations")
    
    print("\n2. ITERATION INCREASE RATIOS (relative to sequential):")
    print("-" * 60)
    
    sor_seq = sor_data[sor_data['num_procs'] == 1]['iterations_mean'].values[0]
    cg_seq = cg_data[cg_data['num_procs'] == 1]['iterations_mean'].values[0]
    
    print("\nSOR:")
    for _, row in sor_data.iterrows():
        ratio = row['iterations_mean'] / sor_seq
        print(f"  {int(row['num_procs']):2d} processes: {ratio:.2f}x ({row['iterations_mean']:.0f}/{sor_seq:.0f})")
    
    print("\nCG:")
    for _, row in cg_data.iterrows():
        ratio = row['iterations_mean'] / cg_seq
        print(f"  {int(row['num_procs']):2d} processes: {ratio:.2f}x ({row['iterations_mean']:.0f}/{cg_seq:.0f})")
    
    print("\n3. ROOT CAUSE ANALYSIS:")
    print("-" * 60)
    print("""
The dramatic increase in SOR iterations (from 440 to 2210, a 5x increase) when
going from 1 to 4+ processes is caused by:

A) INFORMATION PROPAGATION DELAY
   - In sequential SOR, a change at one grid point immediately affects all
     subsequent points in the same sweep (Gauss-Seidel property)
   - In parallel SOR with domain decomposition, each subdomain updates
     independently using OLD boundary values from neighbors
   - Boundary data is only exchanged AFTER each sweep completes
   - This turns the method into a "Block Jacobi" iteration at boundaries

B) OPTIMAL RELAXATION PARAMETER MISMATCH
   - The optimal ω for SOR depends on the spectral radius of the iteration matrix
   - ω_opt = 2 / (1 + sin(π*h)) ≈ 1.95 for N=100 grid points
   - When domain is split, each subdomain has different spectral properties
   - Using the same ω for all subdomains is suboptimal
   - The effective ω for the global iteration is different from local ω

C) INCREASED NUMBER OF INTERFACES
   - With 1 process: 0 internal interfaces
   - With 4 processes (2x2): 4 internal edges, each creating delay
   - More interfaces = more places where information propagation is delayed

D) WHY CG DOESN'T SUFFER FROM THIS
   - CG uses MPI_Allreduce for global dot products
   - Information propagates GLOBALLY every iteration
   - Convergence depends on matrix condition number, not domain decomposition
   - The number of iterations remains essentially constant (~130)

E) MATHEMATICAL EXPLANATION
   - For a 1D decomposition with P processes of an N-point domain:
   - Sequential SOR: O(N) iterations
   - Parallel SOR: O(N) iterations × O(P) communication delays = O(N×P)
   - This explains the 5x increase for 4 processes

F) THE JUMP FROM 2 TO 4 PROCESSES
   - 2 processes (480 iters): Only 1 interface in 1D strip decomposition
   - 4 processes (2210 iters): 2D decomposition with 4 interfaces
   - The 2D decomposition creates a more severe communication bottleneck
""")
    
    print("\n4. IMPLICATIONS FOR PARALLEL EFFICIENCY:")
    print("-" * 60)
    
    print("\nEffective Speedup (accounting for iteration increase):")
    for nprocs in [2, 4, 8]:
        sor_row = sor_data[sor_data['num_procs'] == nprocs]
        cg_row = cg_data[cg_data['num_procs'] == nprocs]
        sor_seq_time = sor_data[sor_data['num_procs'] == 1]['total_time_mean'].values[0]
        cg_seq_time = cg_data[cg_data['num_procs'] == 1]['total_time_mean'].values[0]
        
        if len(sor_row) > 0 and len(cg_row) > 0:
            sor_speedup = sor_seq_time / sor_row['total_time_mean'].values[0]
            cg_speedup = cg_seq_time / cg_row['total_time_mean'].values[0]
            print(f"  {nprocs} processes:")
            print(f"    SOR: {sor_speedup:.2f}x speedup (ideal: {nprocs}x)")
            print(f"    CG:  {cg_speedup:.2f}x speedup (ideal: {nprocs}x)")
    
    print("\n" + "="*80)
    print("CONCLUSION: CG is strongly preferred for parallel Poisson solvers because")
    print("its iteration count is independent of domain decomposition, unlike SOR.")
    print("="*80 + "\n")

def create_latex_table(summary, output_dir="benchmark_results/cg_comparison"):
    """Create a LaTeX table summarizing the iteration scaling analysis."""
    
    sor_data = summary[summary['solver'] == 'sor'].sort_values('num_procs')
    cg_data = summary[summary['solver'] == 'cg'].sort_values('num_procs')
    
    sor_seq = sor_data[sor_data['num_procs'] == 1]['iterations_mean'].values[0]
    cg_seq = cg_data[cg_data['num_procs'] == 1]['iterations_mean'].values[0]
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Iteration Count Scaling Analysis: SOR vs CG}
\label{tab:iteration_scaling}
\begin{tabular}{c|cc|cc}
\toprule
\multirow{2}{*}{Processes} & \multicolumn{2}{c|}{SOR ($\omega = 1.95$)} & \multicolumn{2}{c}{Conjugate Gradient} \\
 & Iterations & Ratio & Iterations & Ratio \\
\midrule
"""
    
    for nprocs in [1, 2, 4, 8]:
        sor_row = sor_data[sor_data['num_procs'] == nprocs]
        cg_row = cg_data[cg_data['num_procs'] == nprocs]
        
        if len(sor_row) > 0 and len(cg_row) > 0:
            sor_iters = int(sor_row['iterations_mean'].values[0])
            cg_iters = int(cg_row['iterations_mean'].values[0])
            sor_ratio = sor_iters / sor_seq
            cg_ratio = cg_iters / cg_seq
            latex += f"{nprocs} & {sor_iters} & {sor_ratio:.2f}$\\times$ & {cg_iters} & {cg_ratio:.2f}$\\times$ \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\small
\textbf{Note:} SOR iterations increase dramatically with processor count due to 
information propagation delays at subdomain boundaries. CG iterations remain 
constant because global dot products provide global information exchange each iteration.
\end{table}
"""
    
    output_path = os.path.join(output_dir, 'iteration_scaling_table.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX table to {output_path}")

def main():
    df = load_data()
    if df is None:
        return
    
    summary = analyze_iteration_scaling(df)
    print_analysis_report(summary)
    plot_iteration_analysis(summary)
    create_latex_table(summary)

if __name__ == "__main__":
    main()
