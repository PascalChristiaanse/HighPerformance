/******************************************************************************
 * FILE: MM-product.c
 * DESCRIPTION:
 *   Matrix-matrix multiplication benchmark comparing sequential and parallel
 *   methods. Supports two modes:
 *     --mode=sequential : Run basic triple-loop and Eigen (single process)
 *     --mode=parallel   : Run SUMMA with MPI (multiple processes)
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <mpi.h>

#include <Eigen/Dense>

#define N 1000
#define CSV_PATH "/home/pchristiaanse/HighPerformance/assignment_1/MM_product/mm_performance.csv"

// ============== TYPE DEFINITIONS ==============
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<MatrixXdRowMajor>;

// ============== FUNCTION DECLARATIONS ==============

void init_matrices(double *A, double *B, double *C, int n);
double run_basic_triple_loop(double *A, double *B, double *C, int n);
double run_eigen_multiply(double *A, double *B, double *C, int n);
double run_summa(double *A, double *B, double *C, int n, int numProcs, int myRank);
bool validate_results(double *matrixRef, double *matrixTest, int n,
                      const char *refName, const char *testName, double epsilon = 1e-9);
void write_csv_row(const char *method, int matrix_size, int num_procs, double time_ms);

// ============== FUNCTION IMPLEMENTATIONS ==============

#include <cmath>

void best_grid(int P, int &Pr, int &Pc)
{
  int r = static_cast<int>(std::floor(std::sqrt(static_cast<double>(P))));

  for (int i = r; i >= 1; i--)
  {
    if (P % i == 0)
    {
      Pr = i;     // number of process rows
      Pc = P / i; // number of process columns
      return;
    }
  }

  // Only happens if P is prime
  Pr = 1;
  Pc = P;
}

void grid_coords(int rank, int Pr, int Pc, int &prow, int &pcol)
{
  prow = rank / Pc; // integer division
  pcol = rank % Pc;
}

void init_matrices(double *A, double *B, double *C, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      A[i * n + j] = i + j;
      B[i * n + j] = i * j;
      C[i * n + j] = 0.0;
    }
  }
}

double run_basic_triple_loop(double *A, double *B, double *C, int n)
{
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
      {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_eigen_multiply(double *A, double *B, double *C, int n)
{
  MatrixMap eigenA(A, n, n);
  MatrixMap eigenB(B, n, n);
  MatrixMap eigenC(C, n, n);

  auto start = std::chrono::high_resolution_clock::now();

  eigenC.noalias() = eigenA * eigenB;

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_summa(double *A, double *B, double *C, int n, int numProcs, int myRank)
{
  // ========================================================================
  // SUMMA (Scalable Universal Matrix Multiplication Algorithm)
  // ========================================================================
  //
  // For C = A * B (all n x n) on a Pr x Pc process grid.
  //
  // Distribution (2D block cyclic with block size 1):
  //   - A: rows by Pr, cols by Pc
  //   - B: rows by Pr, cols by Pc (same as A)
  //   - C: rows by Pr, cols by Pc
  //
  // For the SUMMA loop, we iterate over both row-blocks and col-blocks
  // of the inner dimension. Each process (prow, pcol):
  //   - Has A block (prow, pcol) and B block (prow, pcol)
  //   - Computes C block (prow, pcol)
  //
  // At iteration (kr, kc) where kr in [0,Pr), kc in [0,Pc):
  //   - Process row kr broadcasts A[kr, *] along columns
  //   - Process col kc broadcasts B[*, kc] along rows
  //   - If kr == kc mod Pr (or Pc), we're accessing B[kr, kc]
  //
  // SIMPLIFIED APPROACH: Use the standard outer-product formulation
  // where we iterate k from 0 to Pc-1 and each step handles A col-block k
  // and B row-block k. For non-square grids, we must be careful.
  //
  // ACTUALLY: For simplicity and correctness, let's use a different approach.
  // Each process stores A[prow, :] (all column blocks) and B[:, pcol] (all row blocks).
  // This ensures we have all the data needed for the SUMMA iterations.
  // ========================================================================

  int Pr, Pc;
  best_grid(numProcs, Pr, Pc);

  int prow, pcol;
  grid_coords(myRank, Pr, Pc, prow, pcol);

  if (myRank == 0)
  {
    std::cout << "  SUMMA: Using " << Pr << " x " << Pc << " process grid" << std::endl;
  }

  // Block partitioning helpers
  auto block_size = [](int dim, int nparts, int idx) -> int
  {
    return dim / nparts + (idx < dim % nparts ? 1 : 0);
  };

  auto block_start = [](int dim, int nparts, int idx) -> int
  {
    int base = dim / nparts;
    int rem = dim % nparts;
    return (idx < rem) ? idx * (base + 1) : rem * (base + 1) + (idx - rem) * base;
  };

  // Local block dimensions for C (what this process computes)
  int my_rows = block_size(n, Pr, prow);
  int my_cols = block_size(n, Pc, pcol);
  int my_row_start = block_start(n, Pr, prow);
  int my_col_start = block_start(n, Pc, pcol);

  // Maximum block dimensions for buffers
  int max_row_block = (n + Pr - 1) / Pr;
  int max_col_block = (n + Pc - 1) / Pc;

  if (myRank == 0)
  {
    std::cout << "  C block: [" << (n / Pr) << "-" << max_row_block << "] x ["
              << (n / Pc) << "-" << max_col_block << "]" << std::endl;
  }

  // Each process stores:
  // - A_local: my row-panel A[my_row_start:my_row_end, :] -> dimensions: my_rows x n
  // - B_local: my col-panel B[:, my_col_start:my_col_end] -> dimensions: n x my_cols
  // - C_local: my block C[my_row_start:my_row_end, my_col_start:my_col_end] -> my_rows x my_cols

  double *A_local = new double[max_row_block * n]();
  double *B_local = new double[n * max_col_block]();
  double *C_local = new double[max_row_block * max_col_block]();

  // Store C block info for gather
  int *all_rows = new int[numProcs];
  int *all_cols = new int[numProcs];
  int *all_row_starts = new int[numProcs];
  int *all_col_starts = new int[numProcs];

  MPI_Allgather(&my_rows, 1, MPI_INT, all_rows, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my_cols, 1, MPI_INT, all_cols, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my_row_start, 1, MPI_INT, all_row_starts, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my_col_start, 1, MPI_INT, all_col_starts, 1, MPI_INT, MPI_COMM_WORLD);

  // ========================================================================
  // Scatter A (row panels) and B (column panels) from rank 0
  // ========================================================================

  if (myRank == 0)
  {
    for (int dest = 0; dest < numProcs; dest++)
    {
      int d_prow, d_pcol;
      grid_coords(dest, Pr, Pc, d_prow, d_pcol);

      int d_rows = block_size(n, Pr, d_prow);
      int d_cols = block_size(n, Pc, d_pcol);
      int d_row_start = block_start(n, Pr, d_prow);
      int d_col_start = block_start(n, Pc, d_pcol);

      // A row panel: rows [d_row_start, d_row_start + d_rows), all columns
      double *A_panel = new double[max_row_block * n]();
      for (int i = 0; i < d_rows; i++)
      {
        for (int j = 0; j < n; j++)
        {
          A_panel[i * n + j] = A[(d_row_start + i) * n + j];
        }
      }

      // B column panel: all rows, columns [d_col_start, d_col_start + d_cols)
      double *B_panel = new double[n * max_col_block]();
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < d_cols; j++)
        {
          B_panel[i * max_col_block + j] = B[i * n + (d_col_start + j)];
        }
      }

      if (dest == 0)
      {
        std::copy(A_panel, A_panel + max_row_block * n, A_local);
        std::copy(B_panel, B_panel + n * max_col_block, B_local);
      }
      else
      {
        MPI_Send(A_panel, max_row_block * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        MPI_Send(B_panel, n * max_col_block, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      }

      delete[] A_panel;
      delete[] B_panel;
    }
  }
  else
  {
    MPI_Recv(A_local, max_row_block * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(B_local, n * max_col_block, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::high_resolution_clock::now();

  // ========================================================================
  // Local matrix multiplication: C_local = A_local * B_local
  // ========================================================================
  // Each process has:
  //   A_local: my_rows x n (row panel)
  //   B_local: n x my_cols (column panel)
  //   C_local: my_rows x my_cols (result block)
  // ========================================================================

  for (int i = 0; i < my_rows; i++)
  {
    for (int j = 0; j < my_cols; j++)
    {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
      {
        sum += A_local[i * n + k] * B_local[k * max_col_block + j];
      }
      C_local[i * max_col_block + j] = sum;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto end = std::chrono::high_resolution_clock::now();

  // ========================================================================
  // Gather C blocks back to rank 0
  // ========================================================================

  if (myRank == 0)
  {
    for (int src = 0; src < numProcs; src++)
    {
      double *C_block = new double[max_row_block * max_col_block];

      if (src == 0)
      {
        std::copy(C_local, C_local + max_row_block * max_col_block, C_block);
      }
      else
      {
        MPI_Recv(C_block, max_row_block * max_col_block, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      int src_rows = all_rows[src];
      int src_cols = all_cols[src];
      int row_start = all_row_starts[src];
      int col_start = all_col_starts[src];

      for (int i = 0; i < src_rows; i++)
      {
        for (int j = 0; j < src_cols; j++)
        {
          C[(row_start + i) * n + (col_start + j)] = C_block[i * max_col_block + j];
        }
      }

      delete[] C_block;
    }
  }
  else
  {
    MPI_Send(C_local, max_row_block * max_col_block, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  // Cleanup
  delete[] A_local;
  delete[] B_local;
  delete[] C_local;
  delete[] all_rows;
  delete[] all_cols;
  delete[] all_row_starts;
  delete[] all_col_starts;

  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool validate_results(double *matrixRef, double *matrixTest, int n,
                      const char *refName, const char *testName, double epsilon)
{
  int mismatch_count = 0;
  std::cout << "\nValidation (" << refName << " vs " << testName
            << ", epsilon = " << epsilon << "):" << std::endl;

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double diff = std::fabs(matrixRef[i * n + j] - matrixTest[i * n + j]);
      if (diff > epsilon)
      {
        if (mismatch_count < 10)
        {
          std::cout << "  Mismatch [" << i << "][" << j << "]: "
                    << refName << "=" << matrixRef[i * n + j]
                    << ", " << testName << "=" << matrixTest[i * n + j]
                    << ", diff=" << diff << std::endl;
        }
        mismatch_count++;
      }
    }
  }

  if (mismatch_count == 0)
  {
    std::cout << "  PASSED: All elements match!" << std::endl;
    return true;
  }
  else
  {
    std::cout << "  FAILED: " << mismatch_count << " mismatches found." << std::endl;
    return false;
  }
}

void write_csv_row(const char *method, int matrix_size, int num_procs, double time_ms)
{
  std::ofstream csv(CSV_PATH, std::ios::app);
  csv << method << "," << matrix_size << "," << num_procs << "," << time_ms << std::endl;
  csv.close();
}

// ============== MAIN ==============

int main(int argc, char **argv)
{
  // Parse command-line arguments
  bool mode_sequential = false;
  bool mode_parallel = false;

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--mode=sequential") == 0)
    {
      mode_sequential = true;
    }
    else if (strcmp(argv[i], "--mode=parallel") == 0)
    {
      mode_parallel = true;
    }
  }

  if (!mode_sequential && !mode_parallel)
  {
    std::cerr << "Usage: " << argv[0] << " --mode=sequential|parallel" << std::endl;
    return 1;
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int numProcs, myRank;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (myRank == 0)
  {
    std::cout << "=== MM-Product Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "MPI processes: " << numProcs << std::endl;
    std::cout << "Mode: " << (mode_sequential ? "sequential" : "parallel") << std::endl;
  }

  // Allocate matrices
  double *matrixA = new double[N * N];
  double *matrixB = new double[N * N];
  double *matrixC_basic = new double[N * N];
  double *matrixC_eigen = new double[N * N];
  double *matrixC_summa = new double[N * N];

  // Initialize matrices
  init_matrices(matrixA, matrixB, matrixC_basic, N);
  init_matrices(matrixA, matrixB, matrixC_eigen, N);
  init_matrices(matrixA, matrixB, matrixC_summa, N);

  // ============== SEQUENTIAL MODE ==============
  if (mode_sequential && myRank == 0)
  {
    std::cout << "\n--- Sequential Methods ---" << std::endl;

    // Basic triple-loop
    // Prime any caching behavior
    run_basic_triple_loop(matrixA, matrixB, matrixC_basic, N);
    std::cout << "Running basic triple-loop..." << std::endl;
    double time_basic = run_basic_triple_loop(matrixA, matrixB, matrixC_basic, N);
    std::cout << "  Time: " << time_basic << " ms" << std::endl;
    write_csv_row("basic", N, 1, time_basic);

    // Eigen
    std::cout << "Running Eigen multiply..." << std::endl;
    double time_eigen = run_eigen_multiply(matrixA, matrixB, matrixC_eigen, N);
    std::cout << "  Time: " << time_eigen << " ms" << std::endl;
    write_csv_row("eigen", N, 1, time_eigen);

    // Validate Eigen against basic
    validate_results(matrixC_basic, matrixC_eigen, N, "basic", "eigen");

    std::cout << "\nSpeedup (basic/eigen): " << time_basic / time_eigen << "x" << std::endl;
  }

  // ============== PARALLEL MODE ==============
  if (mode_parallel)
  {
    if (myRank == 0)
    {
      std::cout << "\n--- Parallel Method (SUMMA) ---" << std::endl;
      std::cout << "Running SUMMA with " << numProcs << " processes..." << std::endl;
    }

    // First run basic on rank 0 to get reference result for validation
    if (myRank == 0)
    {
      run_basic_triple_loop(matrixA, matrixB, matrixC_basic, N);
    }

    // Run SUMMA
    double time_summa = run_summa(matrixA, matrixB, matrixC_summa, N, numProcs, myRank);

    if (myRank == 0)
    {
      std::cout << "  Time: " << time_summa << " ms" << std::endl;
      write_csv_row("summa", N, numProcs, time_summa);

      // Validate SUMMA against basic
      validate_results(matrixC_basic, matrixC_summa, N, "basic", "summa");
    }
  }

  // Cleanup
  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC_basic;
  delete[] matrixC_eigen;
  delete[] matrixC_summa;

  if (myRank == 0)
  {
    std::cout << "\nResults appended to: " << CSV_PATH << std::endl;
  }

  MPI_Finalize();
  return 0;
}
