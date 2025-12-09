# Lab Exercise 1.1: Parallelizing a Poisson Solver with MPI

## Overview

The goal of this lab is to become familiar with the message-passing paradigm (MPI) and strategies for achieving high performance in specific applications. After an introductory exercise, you will work through three different applications or case studies, each introducing additional important aspects of message passing and high-performance computing.

In this document, we describe the steps to create an MPI program. Rather than starting from scratch, we will take an existing sequential program and transform it step by step into a realistic message-passing program.

---

## 1.1 First Exercise

### 1.1.1 Introduction

This first exercise focuses on the numerical solution of **Poisson's equation**. This second-order elliptic differential equation appears frequently in many areas of physics. Although the equation itself is relatively simple, there are numerous methods for solving the system of linear equations that emerges after discretization.

We consider a two-dimensional computational domain: the unit square $[0,1] \times [0,1]$. Dirichlet-type boundary conditions are imposed at the edges of the domain.

To keep the path to generalizations open and to create a more realistic problem, we impose conditions other than Poisson's equation at some interior grid points. These can be interpreted as auxiliary boundary conditions or as constraints imposed by the physics of the problem. At any grid point, one of two possible relations applies:

1. Points where the Poisson equation holds
2. Points where another equation or condition is imposed (e.g., a fixed value)

Grid points with simple conditions (like a constant value) could be removed entirely from the system of equations, but we retain them for generality.

**Why this generalization?** In realistic problems, the domain of interest is typically not a simple square but a more general shape. Some direct solvers (such as those using Fast Fourier Transforms) are difficult to apply in these cases. Iterative methods, on the other hand, are almost always applicable—they work regardless of whether the equation at a grid point comes from Poisson's equation or some other condition.

#### Communication Patterns

The types of communication in this exercise are relatively simple:

1. **Local communication**: In each iteration step, every grid point needs the values of its neighboring grid points to evaluate the discrete Poisson equation (sparse matrix-vector multiplication).

2. **Global communication**: Some algorithms (e.g., Conjugate Gradient) require global dot products. Each process can only evaluate its local contribution, so all processes must participate in a global reduction operation to obtain the complete result.

3. **Stopping criterion**: The global residue after a certain number of iterations may be needed to decide whether to stop. All processes must stop in the same phase of the calculation.

#### Important Considerations for Stopping

In a message-passing system, it is **not** a good idea to let a process stop once its local residue is small enough:

- The residue depends on grid points not owned by the process
- Grid points along subdomain boundaries need values from neighboring processes
- Those neighboring values may still change in subsequent iterations

Similarly, monitoring the maximum change within a single iteration for each subdomain separately is problematic. Even if one subdomain converges, other processes may not have reached convergence yet. We must wait until the last process finds a converged solution. Having idle processes wastes computing power.

**Conclusion**: A simultaneous stop of all processes is necessary.

---

### 1.1.2 The Poisson Problem and Parallelism

The central problem in this exercise is Poisson's equation and strategies for solving it on a parallel computer. We limit ourselves to the 2-dimensional case:

$$\nabla^2 \phi(x, y) = S(x, y), \quad 0 \leq x, y \leq 1$$

Here $S(x, y)$ is called the **source function**. Boundary conditions at the edges of the unit square are required.

In practical situations, not all points may satisfy Poisson's equation. At some interior points, $\phi$ might be fixed at a prescribed value:

$$\phi(x, y) = \phi_0(x, y)$$

or in discretized form:

$$\phi_{i,j} = (\phi_0)_{i,j}$$

We will not mention these alternative conditions explicitly in subsequent equations but assume they apply at certain grid points.

#### Discretization

Using straightforward finite differencing, the partial differential equation becomes a system of equations:

$$-\phi_{i-1,j} - \phi_{i,j-1} + 4\phi_{i,j} - \phi_{i,j+1} - \phi_{i+1,j} = h^2 S_{i,j}, \quad i,j = 1, \ldots, N-1$$

where the grid spacing is $h = 1/N$, so that $x_i = i \cdot h$ and $y_j = j \cdot h$. The $(N-1)^2$ unknowns can be solved from these $(N-1)^2$ equations.

**Note**: Border points $\phi_{i,j}$ with either $i$ or $j$ equal to 0 or $N$ are **not** unknowns—they must be fixed or expressed in terms of interior values.

#### Iterative Solvers

There are several strategies for solving systems of linear equations like the one above. In this exercise, we focus on simple iterative techniques and their parallel implementation, where domain decomposition arises naturally.

The basic idea in many iterative solvers is to rewrite the equation in "solved form." In the $(n+1)$-th iteration, a preliminary value at grid point $(i,j)$ is:

$$\tilde{\phi}_{i,j}^{(n+1)} = \frac{\phi_{i-1,j}^{(n)} + \phi_{i,j-1}^{(n)} + \phi_{i,j+1}^{(n)} + \phi_{i+1,j}^{(n)} + h^2 S_{i,j}}{4}$$

This preliminary value is combined with the previous value:

$$\phi_{i,j}^{(n+1)} = \omega \tilde{\phi}_{i,j}^{(n+1)} + (1 - \omega) \phi_{i,j}^{(n)}$$

#### Iteration Methods

| Method | Description |
|--------|-------------|
| **Jacobi** ($\omega = 1$) | Iteration counter increases only after preliminary values for *all* grid points are calculated |
| **Gauss-Seidel** ($\omega = 1$) | Iteration counter increases each time a preliminary value is obtained |
| **SOR** (Successive Over-Relaxation) | Gauss-Seidel with appropriate $\omega$ value |

For the **Jacobi algorithm**, the order in which grid points are visited is irrelevant. For **Gauss-Seidel**, the sequence matters significantly.

#### Parallelization Challenges

Consider a Gauss-Seidel scheme where points are visited such that subsequent points are always neighbors. This algorithm is **essentially sequential** and cannot be parallelized effectively. However, Gauss-Seidel converges faster than Jacobi.

Fortunately, when looping over lattice points in the "normal way" (row after row, left to right), subsequent points are not always neighbors—sometimes a new row begins. This enables parallelization techniques, known in the literature as the **"wavefront"** or **"hyperplane"** approach.

#### Red-Black Ordering

An elegant alternative is to split the grid into **"red"** and **"black"** points like a checkerboard:

1. First, update all grid points of one color
2. Then, update all grid points of the other color

This **red-black Gauss-Seidel** scheme updates values only twice per sweep and is much easier to parallelize than standard Gauss-Seidel—only slightly more difficult than Jacobi.

With $N$ grid points in each direction, red-black Gauss-Seidel updates $N^2/2$ points per color phase. All these updates can be performed in parallel.

---

### Parallel Algorithm Outline

The structure of any parallel iterative algorithm is approximately:

1. **Initialize**: Within each subdomain, set an initial guess $\phi^{(0)}(x,y)$ or read data from a file

2. **Communicate**: Exchange information at boundaries between adjacent subdomains

3. **Compute**: Perform one or more iterations for all interior points. Points may be updated when all required information is available

4. **Check**: Test whether the result has converged sufficiently. If not, return to step 2

5. **Finalize**: Save the solution

#### Questions Worth Investigating

- Is it worthwhile to perform multiple iterations between communication steps? Convergence may be slower in iteration count but faster in wall-clock time.
- How much data should be transferred? Just border values, or multiple layers?
- What is the optimal domain partitioning? Square subdomains, horizontal strips, or vertical strips?
- How does performance scale with problem size or number of processes?

---

### 1.1.3 Description of the Sequential Code: `SEQ_Poisson.c`

The starting point is `SEQ_Poisson.c`, which solves Poisson's equation on a rectangular domain using the red-black Gauss-Seidel scheme. This program can be downloaded from Brightspace.

#### Key Routines

1. **`Setup_Grid`**: Reads input (grid dimensions, precision goal, maximum iterations), allocates memory, and initializes data structures. The example input file `input.dat` specifies 3 interior points with prescribed values.

2. **`Do_Step`**: Updates all red or black points once, monitoring the maximum change for use as a stopping criterion.

3. **`Solve`**: Repeats iterations until a stopping criterion is satisfied. Calls `Do_Step` for each color. The **parity** of a grid point $(x_i, y_j)$ is defined as $(i + j) \mod 2$.

4. **`Write_Grid`**: Writes the resulting field to `output.dat`.

5. **`Clean_Up`**: Frees allocated memory.

6. **`main`**: Calls the routines above. Timer routines (`start_timer`, `stop_timer`, `print_timer`, `resume_timer`) measure execution time between program locations.

---

### 1.1.4 Building a Parallel Program Using MPI

The sequential code `SEQ_Poisson.c` serves as our starting point. Creating a parallel version requires modifications at several points, similar to the MPI introduction exercises. You will build a working parallel code step by step while learning additional MPI library functionalities.

---

## Step 1: Basic MPI Initialization

**Goal**: Execute clones of the original code on multiple processes.

Add calls to `MPI_Init` and `MPI_Finalize` in the main program—this is the minimum code required to use MPI. Rename the program to `MPI_Poisson.c`.

Compile and run on multiple processes using `srun`. **Question**: How do you verify that the program executed on multiple processes?

### I/O Considerations

The first challenge on a parallel computer relates to I/O:
- The sequential program reads from a file, generates output, and writes to the screen
- Not all processors may be able to perform all these operations

**Simple approach**: Let each process perform I/O exactly as in the sequential program. This may fail on loosely-coupled systems where processes have different file systems, or where simultaneous writes cause conflicts.

**Robust approach**: Designate one process to handle all I/O and broadcast data to others.

---

## Step 2: Identify Process Output

**Goal**: Determine which process produces each line of output.

Use `MPI_Comm_rank` to get each process's rank. Declare a global variable `proc_rank` and modify print statements:

```c
// Change this:
printf("Number of iterations %i\n", count);

// To this:
printf("(%i) Number of iterations %i\n", proc_rank, count);
```

---

## Step 3: MPI-Based Timing

**Goal**: Measure execution time for each process using MPI timing routines.

Replace the four timing routines with the following (available as `mptimers.c` on Brightspace):

```c
void start_timer()
{
    if (!timer_on)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        ticks = clock();
        wtime = MPI_Wtime();
        timer_on = 1;
    }
}

void resume_timer()
{
    if (!timer_on)
    {
        ticks = clock() - ticks;
        wtime = MPI_Wtime() - wtime;
        timer_on = 1;
    }
}

void stop_timer()
{
    if (timer_on)
    {
        ticks = clock() - ticks;
        wtime = MPI_Wtime() - wtime;
        timer_on = 0;
    }
}

void print_timer()
{
    if (timer_on)
    {
        stop_timer();
        printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
               proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
        resume_timer();
    }
    else
        printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
               proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}
```

Add this global variable declaration:

```c
double wtime; /* wallclock time */
```

Run with multiple processes and verify timing works correctly. The `MPI_Barrier` ensures all processes start timing simultaneously.

---

## Step 4: Verify Identical Results

**Goal**: Confirm that all process clones produce identical results.

Check both terminal output and output files. If working correctly, all processes should report the same iteration count—but this doesn't guarantee identical results.

### Separate Output Files

To avoid file conflicts, have each process write to a different file:

```c
// Change this:
if ((f = fopen("output.dat", "w")) == NULL)
    Debug("Write_Grid fopen failed", 1);

// To this:
char filename[40];
sprintf(filename, "output%i.dat", proc_rank);
if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid fopen failed", 1);
```

Verify files are identical using:
```bash
diff output0.dat output1.dat
```

---

## Step 5: Broadcast Input Data

**Goal**: Have only rank 0 read input, then broadcast to all processes.

Modify `Setup_Grid` to use `MPI_Bcast`:

```c
if (proc_rank == 0)  /* only process 0 reads input */
{
    f = fopen("input.dat", "r");
    if (f == NULL)
        Debug("Error opening input.dat", 1);
    fscanf(f, "nx %i\n", &gridsize[X_DIR]);
    fscanf(f, "ny %i\n", &gridsize[Y_DIR]);
    fscanf(f, "precision goal %lf\n", &precision_goal);
    fscanf(f, "max iterations %i\n", &max_iter);
}

MPI_Bcast(gridsize, 2, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

Also modify the source point reading at the end of `Setup_Grid`:

```c
do
{
    if (proc_rank == 0)
        s = fscanf(f, "source %lf %lf %lf\n", &source_x, &source_y, &source_val);
    
    MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (s == 3)
    {
        MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        x = gridsize[X_DIR] * source_x;
        y = gridsize[Y_DIR] * source_y;
        x += 1;
        y += 1;
        phi[x][y] = source_val;
        source[x][y] = 1;
    }
}
while (s == 3);

if (proc_rank == 0)
    fclose(f);
```

---

## Step 6: Setup the Process Grid

**Goal**: Make each process recognize its role in a 2D process grid.

Create a new routine `Setup_Proc_Grid` and call it immediately after `MPI_Init`:

```c
Setup_Proc_Grid(argc, argv);
```

### Global Variables (Process-Specific)

```c
int proc_rank;                                    /* rank of current process */
int proc_coord[2];                                /* coordinates in process grid */
int proc_top, proc_right, proc_bottom, proc_left; /* neighbor ranks */
```

### Global Variables (Same for All Processes)

```c
int P;                  /* total number of processes */
int P_grid[2];          /* process grid dimensions */
MPI_Comm grid_comm;     /* grid communicator */
MPI_Status status;
```

### Implementation

```c
void Setup_Proc_Grid(int argc, char **argv)
{
    int wrap_around[2];
    int reorder;
    
    Debug("My_MPI_Init", 0);
    
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    
    /* Get grid dimensions from command line */
    if (argc > 2)
    {
        P_grid[X_DIR] = atoi(argv[1]);
        P_grid[Y_DIR] = atoi(argv[2]);
        if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
            Debug("ERROR: Process grid dimensions do not match P", 1);
    }
    else
        Debug("ERROR: Wrong parameter input", 1);
    
    /* Create 2D Cartesian topology */
    wrap_around[X_DIR] = 0;
    wrap_around[Y_DIR] = 0;  /* no periodic boundaries */
    reorder = 1;              /* allow rank reordering */
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm);
    
    /* Get rank and coordinates in new communicator */
    MPI_Comm_rank(grid_comm, &proc_rank);
    MPI_Cart_coords(grid_comm, proc_rank, 2, proc_coord);
    
    printf("(%i) (x,y)=(%i,%i)\n", proc_rank, proc_coord[X_DIR], proc_coord[Y_DIR]);
    
    /* Find neighbor ranks */
    MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_bottom, &proc_top);
    MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);
    
    if (DEBUG)
        printf("(%i) top %i, right %i, bottom %i, left %i\n",
               proc_rank, proc_top, proc_right, proc_bottom, proc_left);
}
```

### Key MPI Functions

| Function | Purpose |
|----------|---------|
| `MPI_Cart_create` | Creates a Cartesian process topology |
| `MPI_Cart_coords` | Gets process coordinates in the grid |
| `MPI_Cart_shift` | Finds neighbor ranks in a given direction |

**Note**: If a neighbor doesn't exist (at grid boundaries), the rank is set to `MPI_PROC_NULL`. Communication with this "null process" returns immediately—no special boundary handling needed!

**Important**: After this step, use `grid_comm` instead of `MPI_COMM_WORLD` in all subsequent MPI calls.

---

## Step 7: Distribute Work Among Processes

**Goal**: Each process works on its portion of the domain.

Modify `Setup_Grid` so each process:
1. Determines its subdomain size based on problem size and grid position
2. Allocates appropriate memory
3. Initializes only its portion of the field

### Calculate Local Grid Dimensions

```c
/* Calculate local grid offset (top-left corner in global coordinates) */
offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];

upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / P_grid[X_DIR];
upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / P_grid[Y_DIR];

/* Calculate local grid dimensions */
dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];
dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];

/* Add ghost layers for neighboring processes */
dim[X_DIR] += 2;
dim[Y_DIR] += 2;
```

**Note**: `offset` is global; `upper_offset` is local to `Setup_Grid`.

### Handle Fixed Points

Each process must check whether a fixed point lies within its subdomain:

```c
x = x - offset[X_DIR];
y = y - offset[Y_DIR];

if (x > 0 && x < dim[X_DIR] - 1 && y > 0 && y < dim[Y_DIR] - 1)
{
    /* Point is in this process's domain */
    phi[x][y] = source_val;
    source[x][y] = 1;
}
```

**Test**: Run with 2, 3, or 4 processes. Verify processes handle different portions. With 3 processes, some may perform no iterations—can you explain why?

---

## Step 8: Exchange Border Data

**Goal**: Neighboring processes exchange ghost point values.

Create `Exchange_Borders` to exchange data in all four directions using `MPI_Sendrecv`.

### Define MPI Datatypes for Borders

```c
void Setup_MPI_Datatypes()
{
    Debug("Setup_MPI_Datatypes", 0);
    
    /* Datatype for vertical exchange (Y direction) */
    MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR], MPI_DOUBLE, &border_type[Y_DIR]);
    MPI_Type_commit(&border_type[Y_DIR]);
    
    /* Datatype for horizontal exchange (X direction) */
    MPI_Type_vector(dim[Y_DIR] - 2, 1, 1, MPI_DOUBLE, &border_type[X_DIR]);
    MPI_Type_commit(&border_type[X_DIR]);
}
```

Add to main program and declare globally:
```c
MPI_Datatype border_type[2];
```

### Implement Border Exchange

```c
void Exchange_Borders()
{
    Debug("Exchange_Borders", 0);
    
    /* Exchange in Y direction (top-bottom) */
    MPI_Sendrecv(&phi[1][dim[Y_DIR]-2], 1, border_type[Y_DIR], proc_top, 0,
                 &phi[1][0], 1, border_type[Y_DIR], proc_bottom, 0,
                 grid_comm, &status);
    
    MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_bottom, 0,
                 &phi[1][dim[Y_DIR]-1], 1, border_type[Y_DIR], proc_top, 0,
                 grid_comm, &status);
    
    /* Exchange in X direction (left-right) */
    MPI_Sendrecv(&phi[dim[X_DIR]-2][1], 1, border_type[X_DIR], proc_right, 0,
                 &phi[0][1], 1, border_type[X_DIR], proc_left, 0,
                 grid_comm, &status);
    
    MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
                 &phi[dim[X_DIR]-1][1], 1, border_type[X_DIR], proc_right, 0,
                 grid_comm, &status);
}
```

Call `Exchange_Borders` after each red and black iteration phase.

**Warning**: The program will likely hang because processes don't coordinate their stopping. If running interactively, use `Ctrl-C` to kill. On DelftBlue, set a time limit: `--time 00:00:60`.

---

## Step 9: Global Convergence Criterion

**Goal**: Ensure all processes stop simultaneously based on global convergence.

The problem: each process may reach local convergence at different times, but they must all stop together.

**Solution**: Calculate a global error from all local errors using `MPI_Allreduce`.

In `Solve`, add:

```c
double delta;        /* local error */
double global_delta; /* global error */

/* After computing local delta: */
MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, grid_comm);

/* Use global_delta in the convergence check */
```

Replace `delta` with `global_delta` wherever the stopping criterion is evaluated.

**Verify**: The number of iterations should match the sequential version.

---

## Step 10: Final Corrections

### 10.1 Fix Output File Indices

The current code writes local indices. Change to global indices:

```c
/* In Write_Grid, when writing coordinates: */
fprintf(f, "%f %f %f\n", 
        (x + offset[X_DIR]) * h, 
        (y + offset[Y_DIR]) * h, 
        phi[x][y]);
```

Ghost points should not be written, so each point appears in exactly one file.

### 10.2 Fix Red-Black Parity

Results may differ slightly from the sequential version because processes may have different "starting colors" at their local `[1,1]` position.

**Solution**: Account for global parity. Modify the condition in `Do_Step`:

```c
// Change this:
if ((x + y) % 2 == parity && source[x][y] != 1)

// To this:
if ((x + y + offset[X_DIR] + offset[Y_DIR]) % 2 == parity && source[x][y] != 1)
```

This ensures the global parity is used, producing results identical to the sequential program.

---

## Summary

You have now built a complete parallel Poisson solver by:

1. Adding basic MPI initialization
2. Identifying process output by rank
3. Implementing MPI-based timing
4. Separating output files per process
5. Broadcasting input from rank 0
6. Creating a 2D Cartesian process topology
7. Distributing the domain among processes
8. Exchanging ghost point data between neighbors
9. Implementing a global convergence criterion
10. Correcting output indices and red-black parity

The resulting program correctly solves Poisson's equation in parallel using domain decomposition and the red-black Gauss-Seidel method.
