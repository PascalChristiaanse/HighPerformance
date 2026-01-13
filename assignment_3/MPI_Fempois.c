/*
 * MPI_Fempois.c
 * 2D Poisson equation solver with MPI and FEM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include "mpi.h"

#define DEBUG 0

#define TYPE_GHOST 1
#define TYPE_SOURCE 2

#define MAXCOL 20
#define MAX_CONVERGENCE_HISTORY 100000

typedef struct
{
  int type;
  double x, y;
}
Vertex;

typedef int Element[3];

typedef struct
{
  int Ncol;
  int *col;
  double *val;
}
Matrixrow;

/* global variables */
double precision_goal;		/* precision_goal of solution */
int max_iter;			/* maximum number of iterations alowed */
int P;				/* total number of processes */
int P_grid[2];			/* processgrid dimensions */
MPI_Comm grid_comm;		/* grid COMMUNICATOR */
MPI_Status status;

/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
double wtime;			/* wallclock time */
int timer_on = 0;		/* is timer running? */

/* local process related variables */
int proc_rank;			/* rank of current process */
int proc_coord[2];		/* coordinates of current procces in processgrid */
int N_neighb;			/* Number of neighbouring processes */
int *proc_neighb;		/* ranks of neighbouring processes */
MPI_Datatype *send_type;	/* MPI Datatypes for sending */
MPI_Datatype *recv_type;	/* MPI Datatypes for receiving */

/* local grid related variables */
Vertex *vert;			/* vertices */
double *phi;			/* vertex values */
int N_vert;			/* number of vertices */
Matrixrow *A;			/* matrix A */

/* CLI parameters */
int cli_precision_set = 0;      /* flag: precision set via CLI */
int cli_maxiter_set = 0;        /* flag: max_iter set via CLI */
double cli_precision;           /* CLI precision value */
int cli_maxiter;                /* CLI max_iter value */
char telemetry_file[256] = "";  /* telemetry output file */
int output_convergence = 0;     /* flag: output convergence history */

/* Telemetry variables */
typedef struct {
  double time_total;            /* total wall time */
  double time_setup;            /* setup time (grid + MPI datatypes) */
  double time_solve;            /* solve time */
  double time_comm_neighbors;   /* point-to-point communication time */
  double time_comm_global;      /* global communication time (Allreduce) */
  double time_compute;          /* pure computation time */
  double time_io;               /* I/O time */
  int iterations;               /* number of CG iterations */
  int N_vert_local;             /* local vertex count */
  int N_vert_owned;             /* owned (non-ghost) vertex count */
  int N_vert_global;            /* global vertex count */
  int N_neighb;                 /* number of neighbors */
  int comm_calls;               /* number of Exchange_Borders calls */
  int global_comm_calls;        /* number of MPI_Allreduce calls */
  double final_residual;        /* final residual norm */
  double *convergence_history;  /* residual at each iteration */
  int convergence_history_size; /* size of convergence history */
  /* Communication volume tracking */
  int total_send_count;         /* total elements sent per Exchange_Borders */
  int total_recv_count;         /* total elements received per Exchange_Borders */
  long long bytes_sent_total;   /* total bytes sent during solve */
  long long bytes_recv_total;   /* total bytes received during solve */
} Telemetry;

Telemetry telemetry;
double comm_time_accumulator = 0.0;        /* accumulates neighbor communication time */
double global_comm_time_accumulator = 0.0; /* accumulates global communication time */

/* Arrays to track send/recv counts per neighbor */
int *send_counts = NULL;
int *recv_counts = NULL;

void Setup_Proc_Grid();
void Setup_Grid();
void Build_ElMatrix(Element el);
void Sort_MPI_Datatypes();
void Setup_MPI_Datatypes(FILE *f);
void Exchange_Borders(double *vect);
void Solve();
void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();
void parse_arguments(int argc, char **argv);
void init_telemetry();
void write_telemetry();
void print_usage(const char *prog_name);
void resume_timer();
void stop_timer();
void print_timer();

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
    ticks = clock()-ticks;
    wtime = MPI_Wtime()-wtime;
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
    printf("(%i) Elapsed Wtime: %14.6f s (%5.1f%% CPU)\n",
	   proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
    resume_timer();
  }
  else
    printf("(%i) Elapsed Wtime: %14.6f s (%5.1f%% CPU)\n",
	   proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}

void print_usage(const char *prog_name)
{
  if (proc_rank == 0)
  {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -p, --precision <value>   Set precision goal (default: from input.dat)\n");
    printf("  -m, --max-iter <value>    Set max iterations (default: from input.dat)\n");
    printf("  -t, --telemetry <file>    Output telemetry to CSV file\n");
    printf("  -c, --convergence         Include convergence history in telemetry\n");
    printf("  -h, --help                Show this help message\n");
  }
}

void parse_arguments(int argc, char **argv)
{
  static struct option long_options[] = {
    {"precision",   required_argument, 0, 'p'},
    {"max-iter",    required_argument, 0, 'm'},
    {"telemetry",   required_argument, 0, 't'},
    {"convergence", no_argument,       0, 'c'},
    {"help",        no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };

  int opt;
  int option_index = 0;

  /* Reset getopt for MPI compatibility */
  optind = 1;

  while ((opt = getopt_long(argc, argv, "p:m:t:ch", long_options, &option_index)) != -1)
  {
    switch (opt)
    {
      case 'p':
        cli_precision = atof(optarg);
        cli_precision_set = 1;
        break;
      case 'm':
        cli_maxiter = atoi(optarg);
        cli_maxiter_set = 1;
        break;
      case 't':
        strncpy(telemetry_file, optarg, sizeof(telemetry_file) - 1);
        telemetry_file[sizeof(telemetry_file) - 1] = '\0';
        break;
      case 'c':
        output_convergence = 1;
        break;
      case 'h':
        print_usage(argv[0]);
        MPI_Finalize();
        exit(0);
      default:
        print_usage(argv[0]);
        MPI_Finalize();
        exit(1);
    }
  }
}

void init_telemetry()
{
  memset(&telemetry, 0, sizeof(Telemetry));
  comm_time_accumulator = 0.0;

  if (output_convergence)
  {
    telemetry.convergence_history = malloc(MAX_CONVERGENCE_HISTORY * sizeof(double));
    if (telemetry.convergence_history == NULL)
      Debug("init_telemetry: malloc(convergence_history) failed", 1);
  }
  else
  {
    telemetry.convergence_history = NULL;
  }
  telemetry.convergence_history_size = 0;
}

void write_telemetry()
{
  FILE *f;
  int i;
  int global_vert;
  int min_vert, max_vert;
  double max_time_solve, min_time_solve;
  double max_time_comm_neighbors, max_time_comm_global, max_time_compute;
  double min_time_comm_neighbors, min_time_comm_global, min_time_compute;
  double avg_time_comm_neighbors, avg_time_comm_global, avg_time_compute;
  char conv_filename[270];
  /* Communication volume stats */
  int min_send_count, max_send_count, total_send_count;
  int min_recv_count, max_recv_count, total_recv_count;
  int min_neighb, max_neighb;
  long long total_bytes_sent, total_bytes_recv;

  /* Gather global statistics */
  MPI_Reduce(&telemetry.N_vert_owned, &global_vert, 1, MPI_INT, MPI_SUM, 0, grid_comm);
  MPI_Reduce(&telemetry.N_vert_owned, &min_vert, 1, MPI_INT, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.N_vert_owned, &max_vert, 1, MPI_INT, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.time_solve, &max_time_solve, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.time_solve, &min_time_solve, 1, MPI_DOUBLE, MPI_MIN, 0, grid_comm);
  
  /* Neighbor communication stats */
  MPI_Reduce(&telemetry.time_comm_neighbors, &max_time_comm_neighbors, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.time_comm_neighbors, &min_time_comm_neighbors, 1, MPI_DOUBLE, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.time_comm_neighbors, &avg_time_comm_neighbors, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm);
  avg_time_comm_neighbors /= P;

  /* Communication volume stats */
  MPI_Reduce(&telemetry.total_send_count, &min_send_count, 1, MPI_INT, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.total_send_count, &max_send_count, 1, MPI_INT, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.total_send_count, &total_send_count, 1, MPI_INT, MPI_SUM, 0, grid_comm);
  MPI_Reduce(&telemetry.total_recv_count, &min_recv_count, 1, MPI_INT, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.total_recv_count, &max_recv_count, 1, MPI_INT, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.total_recv_count, &total_recv_count, 1, MPI_INT, MPI_SUM, 0, grid_comm);
  MPI_Reduce(&telemetry.N_neighb, &min_neighb, 1, MPI_INT, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.N_neighb, &max_neighb, 1, MPI_INT, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.bytes_sent_total, &total_bytes_sent, 1, MPI_LONG_LONG, MPI_SUM, 0, grid_comm);
  MPI_Reduce(&telemetry.bytes_recv_total, &total_bytes_recv, 1, MPI_LONG_LONG, MPI_SUM, 0, grid_comm);

  /* Global communication stats */
  MPI_Reduce(&telemetry.time_comm_global, &max_time_comm_global, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.time_comm_global, &min_time_comm_global, 1, MPI_DOUBLE, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.time_comm_global, &avg_time_comm_global, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm);
  avg_time_comm_global /= P;

  /* Compute time stats */
  MPI_Reduce(&telemetry.time_compute, &max_time_compute, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
  MPI_Reduce(&telemetry.time_compute, &min_time_compute, 1, MPI_DOUBLE, MPI_MIN, 0, grid_comm);
  MPI_Reduce(&telemetry.time_compute, &avg_time_compute, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm);
  avg_time_compute /= P;

  telemetry.N_vert_global = global_vert;

  if (proc_rank == 0 && strlen(telemetry_file) > 0)
  {
    /* Calculate idle time: difference between max solve time and each process's active time */
    /* Idle time = max_time_solve - (compute + comm_neighbors + comm_global) for slowest process */
    double time_idle = max_time_solve - (avg_time_compute + avg_time_comm_neighbors + avg_time_comm_global);
    if (time_idle < 0) time_idle = 0;

    /* Write summary CSV */
    f = fopen(telemetry_file, "w");
    if (f == NULL)
    {
      printf("Warning: Could not open telemetry file %s\n", telemetry_file);
      return;
    }

    fprintf(f, "# MPI_Fempois Telemetry Summary\n");
    fprintf(f, "metric,value\n");
    fprintf(f, "num_processes,%d\n", P);
    fprintf(f, "precision_goal,%.10e\n", precision_goal);
    fprintf(f, "max_iterations,%d\n", max_iter);
    fprintf(f, "iterations,%d\n", telemetry.iterations);
    fprintf(f, "final_residual,%.10e\n", telemetry.final_residual);
    fprintf(f, "converged,%d\n", telemetry.final_residual <= precision_goal ? 1 : 0);
    fprintf(f, "global_vertices,%d\n", global_vert);
    fprintf(f, "min_vertices_per_proc,%d\n", min_vert);
    fprintf(f, "max_vertices_per_proc,%d\n", max_vert);
    fprintf(f, "load_imbalance,%.4f\n", (double)max_vert / (double)min_vert);
    fprintf(f, "time_total,%.6f\n", telemetry.time_total);
    fprintf(f, "time_setup,%.6f\n", telemetry.time_setup);
    fprintf(f, "time_solve,%.6f\n", max_time_solve);
    fprintf(f, "time_solve_min,%.6f\n", min_time_solve);
    fprintf(f, "time_compute_max,%.6f\n", max_time_compute);
    fprintf(f, "time_compute_min,%.6f\n", min_time_compute);
    fprintf(f, "time_compute_avg,%.6f\n", avg_time_compute);
    fprintf(f, "time_comm_neighbors_max,%.6f\n", max_time_comm_neighbors);
    fprintf(f, "time_comm_neighbors_min,%.6f\n", min_time_comm_neighbors);
    fprintf(f, "time_comm_neighbors_avg,%.6f\n", avg_time_comm_neighbors);
    fprintf(f, "time_comm_global_max,%.6f\n", max_time_comm_global);
    fprintf(f, "time_comm_global_min,%.6f\n", min_time_comm_global);
    fprintf(f, "time_comm_global_avg,%.6f\n", avg_time_comm_global);
    fprintf(f, "time_idle_est,%.6f\n", time_idle);
    fprintf(f, "time_io,%.6f\n", telemetry.time_io);
    fprintf(f, "comm_neighbor_calls,%d\n", telemetry.comm_calls);
    fprintf(f, "comm_global_calls,%d\n", telemetry.global_comm_calls);
    fprintf(f, "comm_neighbors_fraction,%.4f\n", avg_time_comm_neighbors / max_time_solve);
    fprintf(f, "comm_global_fraction,%.4f\n", avg_time_comm_global / max_time_solve);
    fprintf(f, "compute_fraction,%.4f\n", avg_time_compute / max_time_solve);
    /* Communication volume metrics */
    fprintf(f, "min_neighbors,%d\n", min_neighb);
    fprintf(f, "max_neighbors,%d\n", max_neighb);
    fprintf(f, "send_count_per_iter_min,%d\n", min_send_count);
    fprintf(f, "send_count_per_iter_max,%d\n", max_send_count);
    fprintf(f, "send_count_per_iter_total,%d\n", total_send_count);
    fprintf(f, "recv_count_per_iter_min,%d\n", min_recv_count);
    fprintf(f, "recv_count_per_iter_max,%d\n", max_recv_count);
    fprintf(f, "recv_count_per_iter_total,%d\n", total_recv_count);
    fprintf(f, "bytes_sent_total,%lld\n", total_bytes_sent);
    fprintf(f, "bytes_recv_total,%lld\n", total_bytes_recv);
    fprintf(f, "bytes_per_iter_sent,%lld\n", total_bytes_sent / telemetry.comm_calls);
    fprintf(f, "bytes_per_iter_recv,%lld\n", total_bytes_recv / telemetry.comm_calls);

    fclose(f);

    /* Write convergence history if requested */
    if (output_convergence && telemetry.convergence_history != NULL)
    {
      snprintf(conv_filename, sizeof(conv_filename), "%s.convergence", telemetry_file);
      f = fopen(conv_filename, "w");
      if (f != NULL)
      {
        fprintf(f, "iteration,residual\n");
        for (i = 0; i < telemetry.convergence_history_size; i++)
        {
          fprintf(f, "%d,%.10e\n", i + 1, telemetry.convergence_history[i]);
        }
        fclose(f);
      }
    }

    printf("Telemetry written to %s\n", telemetry_file);
  }

  /* Clean up convergence history */
  if (telemetry.convergence_history != NULL)
  {
    free(telemetry.convergence_history);
    telemetry.convergence_history = NULL;
  }
}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("(%i) %s\n", proc_rank, mesg);
  if (terminate)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
}

void Setup_Proc_Grid()
{
  FILE *f = NULL;
  char filename[25];
  int i;
  int N_nodes = 0, N_edges = 0;
  int *index, *edges, reorder;

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  Debug("My_MPI_Init", 0);

  /* Retrieve the number of processes and current process rank */
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  /* Create process topology (Graph) */
  if (proc_rank == 0)
  {
    sprintf(filename, "mapping%i.dat", P);
    if ((f = fopen(filename, "r")) == NULL)
      Debug("My_MPI_Init : Can't open mapping inputfile", 1);

    /* after reading N_nodes, a line is skipped */
    fscanf(f, "N_proc : %i\n%*[^\n]\n", &N_nodes);
    if (N_nodes != P)
      Debug("My_MPI_Init : Mismatch of number of processes in mapping inputfile", 1);
  }
  else
    N_nodes = P;

  if ((index = malloc(N_nodes * sizeof(int))) == NULL)
      Debug("My_MPI_Init : malloc(index) failed", 1);

  if (proc_rank == 0)
  {
    for (i = 0; i < N_nodes; i++)
      fscanf(f, "%i\n", &index[i]);
  }

  MPI_Bcast(index, N_nodes, MPI_INT, 0, MPI_COMM_WORLD);

  N_edges = index[N_nodes - 1];
  if (N_edges>0)
  {
    if ((edges = malloc(N_edges * sizeof(int))) == NULL)
      Debug("My_MPI_Init : malloc(edges) failed", 1);
  }
  else
    edges = index; /* this is actually nonsense,
                      but 'edges' needs to be a non-null pointer */

  if (proc_rank == 0)
  {
    fscanf(f, "%*[^\n]\n");		/* skip a line of the file */
    for (i = 0; i < N_edges; i++)
      fscanf(f, "%i\n", &edges[i]);

    fclose(f);
  }

  MPI_Bcast(edges, N_edges, MPI_INT, 0, MPI_COMM_WORLD);

  reorder = 1;
  MPI_Graph_create(MPI_COMM_WORLD, N_nodes, index, edges, reorder, &grid_comm);

  /* Retrieve new rank of this process */
  MPI_Comm_rank(grid_comm, &proc_rank);

  if (N_edges>0)
    free(edges);
  free(index);
}

void Setup_Grid()
{
  int i, j, v;
  Element element;
  int N_elm;
  char filename[25];
  FILE *f;

  Debug("Setup_Grid", 0);

  /* read general parameters (precision/max_iter) */
  if (proc_rank==0)
  {
    if ((f = fopen("input.dat", "r")) == NULL)
      Debug("Setup_Grid : Can't open input.dat", 1);
    fscanf(f, "precision goal: %lf\n", &precision_goal);
    fscanf(f, "max iterations: %i", &max_iter);
    fclose(f);
  }
  MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
  MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);

  /* Apply CLI overrides if set */
  if (cli_precision_set)
    precision_goal = cli_precision;
  if (cli_maxiter_set)
    max_iter = cli_maxiter;

  /* read process specific data */
  sprintf(filename, "input%i-%i.dat", P, proc_rank);
  if ((f = fopen(filename, "r")) == NULL)
    Debug("Setup_Grid : Can't open data inputfile", 1);
  fscanf(f, "N_vert: %i\n%*[^\n]\n", &N_vert);

  /* allocate memory for phi and A */
  if ((vert = malloc(N_vert * sizeof(Vertex))) == NULL)
    Debug("Setup_Grid : malloc(vert) failed", 1);
  if ((phi = malloc(N_vert * sizeof(double))) == NULL)
    Debug("Setup_Grid : malloc(phi) failed", 1);

  if ((A = malloc(N_vert * sizeof(*A))) == NULL)
    Debug("Setup_Grid : malloc(*A) failed", 1);
  for (i=0; i<N_vert; i++)
  {
    if ((A[i].col=malloc(MAXCOL*sizeof(int)))==NULL)
      Debug("Setup_Grid : malloc(A.col) failed", 1);
    if ((A[i].val=malloc(MAXCOL*sizeof(double)))==NULL)
      Debug("Setup_Grid : malloc(A.val) failed", 1);
  }

  /* init matrix rows of A */
  for (i = 0; i < N_vert; i++)
      A[i].Ncol = 0;

  /* Read all values */
  for (i = 0; i < N_vert; i++)
  {
    fscanf(f, "%i", &v);
    fscanf(f, "%lf %lf %i %lf\n", &vert[v].x, &vert[v].y,
	   &vert[v].type, &phi[v]);
  }

  /* build matrix from elements */
  fscanf(f, "N_elm: %i\n%*[^\n]\n", &N_elm);
  for (i = 0; i < N_elm; i++)
  {
    fscanf(f, "%*i");  /* we are not interested in the element-id */
    for (j = 0; j < 3; j++)
    {
      fscanf(f, "%i", &v);
      element[j] = v;
    }
    fscanf(f, "\n");
    Build_ElMatrix(element);
  }

  Setup_MPI_Datatypes(f);

  fclose(f);
}

void Add_To_Matrix(int i, int j, double a)
{
  int k;
  k=0;
  
  while ( (k<A[i].Ncol) && (A[i].col[k]!=j) )
    k++;
  if (k<A[i].Ncol)
    A[i].val[k]+=a;
  else
  {
    if (A[i].Ncol>=MAXCOL)
      Debug("Add_To_Matrix : MAXCOL exceeded", 1);
    A[i].val[A[i].Ncol]=a;
    A[i].col[A[i].Ncol]=j;
    A[i].Ncol++;
  }
}

void Build_ElMatrix(Element el)
{
  int i, j;
  double e[3][2];
  double s[3][3];
  double det;

  e[0][0] = vert[el[1]].y - vert[el[2]].y;	/* y1-y2 */
  e[1][0] = vert[el[2]].y - vert[el[0]].y;	/* y2-y0 */
  e[2][0] = vert[el[0]].y - vert[el[1]].y;	/* y0-y1 */
  e[0][1] = vert[el[2]].x - vert[el[1]].x;	/* x2-x1 */
  e[1][1] = vert[el[0]].x - vert[el[2]].x;	/* x0-x2 */
  e[2][1] = vert[el[1]].x - vert[el[0]].x;	/* x1-x0 */

  det = e[2][0] * e[0][1] - e[2][1] * e[0][0];
  if (det == 0.0)
    Debug("One of the elements has a zero surface", 1);

  det = fabs(2 * det);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      s[i][j] = (e[i][0] * e[j][0] + e[i][1] * e[j][1]) / det;

  for (i = 0; i < 3; i++)
    if (!((vert[el[i]].type & TYPE_GHOST) |
	  (vert[el[i]].type & TYPE_SOURCE)))
      for (j = 0; j < 3; j++)
        Add_To_Matrix(el[i],el[j],s[i][j]);
}

void Sort_MPI_Datatypes()
{
  int i, j;
  MPI_Datatype data2;
  int proc2, count2;

  for (i=0;i<N_neighb-1;i++)
    for (j=i+1;j<N_neighb;j++)
      if (proc_neighb[j]<proc_neighb[i])
      {
        proc2 = proc_neighb[i];
        proc_neighb[i] = proc_neighb[j]; 
        proc_neighb[j] = proc2;
        data2 = send_type[i];
        send_type[i] = send_type[j];
        send_type[j] = data2;
        data2 = recv_type[i];
        recv_type[i] = recv_type[j];
        recv_type[j] = data2;
        /* Also sort the count arrays */
        if (send_counts != NULL)
        {
          count2 = send_counts[i];
          send_counts[i] = send_counts[j];
          send_counts[j] = count2;
        }
        if (recv_counts != NULL)
        {
          count2 = recv_counts[i];
          recv_counts[i] = recv_counts[j];
          recv_counts[j] = count2;
        }
      }
}

void Setup_MPI_Datatypes(FILE * f)
{
  int i, s;
  int count;
  int *indices;
  int *blocklens;
  int recv_count_temp;

  Debug("Setup_MPI_Datatypes", 0);

  fscanf(f, "Neighbours: %i\n", &N_neighb);

  /* allocate memory */

  if (N_neighb>0)
  {
    if ((proc_neighb = malloc(N_neighb * sizeof(int))) == NULL)
        Debug("Setup_MPI_Datatypes: malloc(proc_neighb) failed", 1);
    if ((send_type = malloc(N_neighb * sizeof(MPI_Datatype))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(send_type) failed", 1);
    if ((recv_type = malloc(N_neighb * sizeof(MPI_Datatype))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(recv_type) failed", 1);
    /* Allocate arrays for tracking send/recv counts */
    if ((send_counts = malloc(N_neighb * sizeof(int))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(send_counts) failed", 1);
    if ((recv_counts = malloc(N_neighb * sizeof(int))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(recv_counts) failed", 1);
  }
  else
  {
    proc_neighb = NULL;
    send_type = NULL;
    recv_type = NULL;
    send_counts = NULL;
    recv_counts = NULL;
  }

  if ((indices = malloc(N_vert * sizeof(int))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(indices) failed", 1);
  if ((blocklens = malloc(N_vert * sizeof(int))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(blocklens) failed", 1);

  for (i = 0; i < N_vert; i++)
    blocklens[i] = 1;

  /* Initialize totals for telemetry */
  telemetry.total_send_count = 0;
  telemetry.total_recv_count = 0;

  /* read vertices per neighbour */
  for (i = 0; i < N_neighb; i++)
  {
    fscanf(f, "from %i :", &proc_neighb[i]);
    s = 1;
    count = 0;
    while (s == 1)
    {
      s = fscanf(f, "%i", &indices[count]);
      if ((s == 1) && !(vert[indices[count]].type & TYPE_SOURCE))
	count++;
    }
    fscanf(f, "\n");
    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &recv_type[i]);
    MPI_Type_commit(&recv_type[i]);
    recv_counts[i] = count;  /* Store recv count for this neighbor */
    telemetry.total_recv_count += count;

    fscanf(f, "to %i :", &proc_neighb[i]);
    s = 1;
    count = 0;
    while (s == 1)
    {
      s = fscanf(f, "%i", &indices[count]);
      if ((s == 1) && !(vert[indices[count]].type & TYPE_SOURCE))
	count++;
    }
    fscanf(f, "\n");
    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &send_type[i]);
    MPI_Type_commit(&send_type[i]);
    send_counts[i] = count;  /* Store send count for this neighbor */
    telemetry.total_send_count += count;
  }

  Sort_MPI_Datatypes();

  free(blocklens);
  free(indices);

  /* Record neighbor count for telemetry */
  telemetry.N_neighb = N_neighb;
}





void Exchange_Borders(double *vect)
{
    int i;
    int tag = 0;
    double start_time, end_time;

    start_time = MPI_Wtime();

    // Exchange data with each neighboring process
    for (i = 0; i < N_neighb; i++)
    {
        MPI_Sendrecv(vect, 1, send_type[i], proc_neighb[i], tag,
                     vect, 1, recv_type[i], proc_neighb[i], tag,
                     grid_comm, &status);
    }

    end_time = MPI_Wtime();
    comm_time_accumulator += (end_time - start_time);
    telemetry.comm_calls++;
    
    /* Accumulate bytes sent/received */
    telemetry.bytes_sent_total += (long long)telemetry.total_send_count * sizeof(double);
    telemetry.bytes_recv_total += (long long)telemetry.total_recv_count * sizeof(double);
}




void Solve()
{
  int count = 0;
  int i, j;
  double *r, *p, *q;
  double a, b, r1, r2 = 1;
  double sub;
  double solve_start, solve_end;
  double compute_time_accumulator = 0.0;
  double t_start, t_end;

  Debug("Solve", 0);

  /* Reset communication accumulators for this solve */
  comm_time_accumulator = 0.0;
  global_comm_time_accumulator = 0.0;
  telemetry.comm_calls = 0;
  telemetry.global_comm_calls = 0;

  solve_start = MPI_Wtime();

  if ((r = malloc(N_vert * sizeof(double))) == NULL)
      Debug("Solve : malloc(r) failed", 1);
  if ((p = malloc(N_vert * sizeof(double))) == NULL)
      Debug("Solve : malloc(p) failed", 1);
  if ((q = malloc(N_vert * sizeof(double))) == NULL)
      Debug("Solve : malloc(q) failed", 1);

  /* Implementation of the CG algorithm : */

  Exchange_Borders(phi);

  /* r = b-Ax */
  t_start = MPI_Wtime();
  for (i = 0; i < N_vert; i++)
  {
    r[i] = 0.0;
    for (j = 0; j < A[i].Ncol; j++)
      r[i] -= A[i].val[j] * phi[A[i].col[j]];
  }
  t_end = MPI_Wtime();
  compute_time_accumulator += (t_end - t_start);

  r1 = 2 * precision_goal;
  while ((count < max_iter) && (r1 > precision_goal))
  {
    /* r1 = r' * r */
    t_start = MPI_Wtime();
    sub = 0.0;
    for (i = 0; i < N_vert; i++)
      if (!(vert[i].type & TYPE_GHOST))
	sub += r[i] * r[i];
    t_end = MPI_Wtime();
    compute_time_accumulator += (t_end - t_start);

    t_start = MPI_Wtime();
    MPI_Allreduce(&sub, &r1, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
    t_end = MPI_Wtime();
    global_comm_time_accumulator += (t_end - t_start);
    telemetry.global_comm_calls++;

    /* Record convergence history */
    if (output_convergence && telemetry.convergence_history != NULL 
        && count < MAX_CONVERGENCE_HISTORY)
    {
      telemetry.convergence_history[count] = r1;
      telemetry.convergence_history_size = count + 1;
    }

    t_start = MPI_Wtime();
    if (count == 0)
    {
      /* p = r */
      for (i = 0; i < N_vert; i++)
	p[i] = r[i];
    }
    else
    {
      b = r1 / r2;

      /* p = r + b*p */
      for (i = 0; i < N_vert; i++)
	p[i] = r[i] + b * p[i];
    }
    t_end = MPI_Wtime();
    compute_time_accumulator += (t_end - t_start);

    Exchange_Borders(p);

    /* q = A * p */
    t_start = MPI_Wtime();
    for (i = 0; i < N_vert; i++)
    {
      q[i] = 0;
      for (j = 0; j < A[i].Ncol; j++)
        q[i] += A[i].val[j] * p[A[i].col[j]];
    }
    t_end = MPI_Wtime();
    compute_time_accumulator += (t_end - t_start);

    /* a = r1 / (p' * q) */
    t_start = MPI_Wtime();
    sub = 0.0;
    for (i = 0; i < N_vert; i++)
      if (!(vert[i].type & TYPE_GHOST))
	sub += p[i] * q[i];
    t_end = MPI_Wtime();
    compute_time_accumulator += (t_end - t_start);

    t_start = MPI_Wtime();
    MPI_Allreduce(&sub, &a, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
    t_end = MPI_Wtime();
    global_comm_time_accumulator += (t_end - t_start);
    telemetry.global_comm_calls++;

    t_start = MPI_Wtime();
    a = r1 / a;

    /* x = x + a*p */
    for (i = 0; i < N_vert; i++)
      phi[i] += a * p[i];

    /* r = r - a*q */
    for (i = 0; i < N_vert; i++)
      r[i] -= a * q[i];
    t_end = MPI_Wtime();
    compute_time_accumulator += (t_end - t_start);

    r2 = r1;

    count++;
  }

  solve_end = MPI_Wtime();

  free(q);
  free(p);
  free(r);

  /* Record telemetry */
  telemetry.iterations = count;
  telemetry.final_residual = r1;
  telemetry.time_solve = solve_end - solve_start;
  telemetry.time_comm_neighbors = comm_time_accumulator;
  telemetry.time_comm_global = global_comm_time_accumulator;
  telemetry.time_compute = compute_time_accumulator;

  if (proc_rank == 0)
    printf("Number of iterations : %i\n", count);
}

void Write_Grid()
{
  int i;
  char filename[25];
  FILE *f;
  double io_start, io_end;

  Debug("Write_Grid", 0);

  io_start = MPI_Wtime();

  sprintf(filename, "output%i-%i.dat", P, proc_rank);
  if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid : Can't open data outputfile", 1);

  for (i = 0; i < N_vert; i++)
    if (!(vert[i].type & TYPE_GHOST))
      fprintf(f, "%f %f %f\n", vert[i].x, vert[i].y, phi[i]);

  fclose(f);

  io_end = MPI_Wtime();
  telemetry.time_io = io_end - io_start;
}

void Clean_Up()
{
  int i;
  Debug("Clean_Up", 0);

  if (N_neighb>0)
  {
    free(recv_type);
    free(send_type);
    free(proc_neighb);
    if (send_counts != NULL) free(send_counts);
    if (recv_counts != NULL) free(recv_counts);
  }

  for (i=0;i<N_vert;i++)
  {
    free(A[i].col);
    free(A[i].val);
  }
  free(A);
  free(vert);
  free(phi);
}

int main(int argc, char **argv)
{
  double time_start, time_setup_end, time_total_end;
  int i, owned_count;

  MPI_Init(&argc, &argv);

  /* Get initial rank for argument parsing */
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  /* Parse command line arguments */
  parse_arguments(argc, argv);

  /* Initialize telemetry */
  init_telemetry();

  time_start = MPI_Wtime();
  start_timer();

  Setup_Proc_Grid();

  Setup_Grid();

  time_setup_end = MPI_Wtime();
  telemetry.time_setup = time_setup_end - time_start;

  /* Count local vertices for telemetry */
  telemetry.N_vert_local = N_vert;
  owned_count = 0;
  for (i = 0; i < N_vert; i++)
    if (!(vert[i].type & TYPE_GHOST))
      owned_count++;
  telemetry.N_vert_owned = owned_count;

  Solve();

  Write_Grid();

  time_total_end = MPI_Wtime();
  telemetry.time_total = time_total_end - time_start;

  /* Write telemetry before cleanup */
  write_telemetry();

  Clean_Up();

  print_timer();

  Debug("MPI_Finalize", 0);

  MPI_Finalize();

  return 0;
}
