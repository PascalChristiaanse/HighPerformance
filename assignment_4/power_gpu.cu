// Power Method GPU Implementation
// Computes dominant eigenvalue using power iteration
// Usage: ./power_gpu --size N --blocksize B --max_iteration M --use_shared [0|1]

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

// Default parameters (can be overridden by command line)
int GlobalSize = 5000;         // Matrix dimension N x N
int BlockSize = 32;            // Number of threads per block
const float EPS = 0.000005f;   // Convergence tolerance
int max_iteration = 100;       // Maximum iteration steps
int UseSharedMemory = 1;       // 1 = use shared memory, 0 = use global memory
int UseUnifiedMemory = 0;      // 1 = use unified memory (cudaMallocManaged), 0 = manual transfers

// Host arrays
float* h_MatA = NULL;
float* h_VecV = NULL;
float* h_VecW = NULL;
float* h_NormW = NULL;

// Device arrays
float* d_MatA = NULL;
float* d_VecV = NULL;
float* d_VecW = NULL;
float* d_NormW = NULL;
float* d_Lamda = NULL;

// Function declarations
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
void Arguments(int, char**);
void checkCardVersion(void);

// ============================================================================
// CUDA Kernels
// ============================================================================

// Matrix-Vector Product using Shared Memory (tiled approach)
// Each thread computes one element of the result vector W[row] = sum(A[row,j] * V[j])
__global__ void Av_Product_Shared(float* g_MatA, float* g_VecV, float* g_VecW, int N, int blockSize)
{
    extern __shared__ float sharedMem[];
    float* Vs = sharedMem;  // Tile of vector V (blockSize elements)

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + blockSize - 1) / blockSize;

    for (int tile = 0; tile < numTiles; tile++)
    {
        // Cooperatively load tile of vector V into shared memory
        int vIndex = tile * blockSize + tx;
        if (vIndex < N)
            Vs[tx] = g_VecV[vIndex];
        else
            Vs[tx] = 0.0f;

        __syncthreads();

        // Each thread computes partial dot product for its row
        if (row < N)
        {
            int tileStart = tile * blockSize;
            int tileEnd = min(tileStart + blockSize, N);
            for (int j = tileStart; j < tileEnd; j++)
            {
                // Read matrix element from global memory (coalesced access)
                // Read vector element from shared memory (reused across threads in block)
                sum += g_MatA[row * N + j] * Vs[j - tileStart];
            }
        }

        __syncthreads();
    }

    if (row < N)
        g_VecW[row] = sum;
}

// Matrix-Vector Product using Global Memory only
__global__ void Av_Product_Global(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N)
    {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
        {
            sum += g_MatA[row * N + j] * g_VecV[j];
        }
        g_VecW[row] = sum;
    }
}

// Compute squared norm of W (partial reduction per block)
__global__ void FindNormW(float* g_VecW, float* g_NormW, int N)
{
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and compute square
    if (globalid < N)
        sdata[tid] = g_VecW[globalid] * g_VecW[globalid];
    else
        sdata[tid] = 0.0f;
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Thread 0 of each block writes result atomically
    if (tid == 0)
        atomicAdd(g_NormW, sdata[0]);
}

// Normalize W into V: V[i] = W[i] / normW
__global__ void NormalizeW(float* g_VecV, float* g_VecW, float normW, int N)
{
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalid < N)
        g_VecV[globalid] = g_VecW[globalid] / normW;
}

// Compute lambda = V dot W (parallel reduction)
__global__ void ComputeLamda(float* g_VecV, float* g_VecW, float* g_Lamda, int N)
{
    extern __shared__ float sdataVW[];
    
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalid < N)
        sdataVW[tid] = g_VecV[globalid] * g_VecW[globalid];
    else
        sdataVW[tid] = 0.0f;
    
    __syncthreads();
    
    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdataVW[tid] += sdataVW[tid + s];
        __syncthreads();
    }
    
    if (tid == 0)
        atomicAdd(g_Lamda, sdataVW[0]);
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

void CPU_AvProduct()
{
    int N = GlobalSize;
    for (int i = 0; i < N; i++)
    {
        h_VecW[i] = 0.0f;
        for (int j = 0; j < N; j++)
            h_VecW[i] += h_MatA[i * N + j] * h_VecV[j];
    }
}

void CPU_NormalizeW()
{
    int N = GlobalSize;
    float normW = 0.0f;
    for (int i = 0; i < N; i++)
        normW += h_VecW[i] * h_VecW[i];
    
    normW = sqrtf(normW);
    for (int i = 0; i < N; i++)
        h_VecV[i] = h_VecW[i] / normW;
}

float CPU_ComputeLamda()
{
    int N = GlobalSize;
    float lamda = 0.0f;
    for (int i = 0; i < N; i++)
        lamda += h_VecV[i] * h_VecW[i];
    return lamda;
}

float RunCPUPowerMethod()
{
    float oldLamda = 0.0f;
    float lamda = 0.0f;
    int iterations = 0;
    
    CPU_AvProduct();
    
    for (int i = 0; i < max_iteration; i++)
    {
        CPU_NormalizeW();
        CPU_AvProduct();
        lamda = CPU_ComputeLamda();
        iterations = i + 1;
        
        if (fabsf(oldLamda - lamda) < EPS)
            break;
        oldLamda = lamda;
    }
    
    printf("CPU converged after %d iterations, lambda = %f\n", iterations, lamda);
    return lamda;
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv)
{
    struct timespec t_start, t_end, t_mem_start, t_mem_end;
    double cpu_runtime, gpu_runtime, gpu_runtime_no_memcpy, memcpy_time;
    
    Arguments(argc, argv);
    
    int N = GlobalSize;
    printf("Matrix size: %d x %d\n", N, N);
    printf("Block size: %d threads\n", BlockSize);
    printf("Max iterations: %d\n", max_iteration);
    printf("Kernel mode: %s\n", UseSharedMemory ? "Shared" : "Global");
    printf("Memory management: %s\n", UseUnifiedMemory ? "Unified (cudaMallocManaged)" : "Manual transfers");
    
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t scalar_size = sizeof(float);
    
    // Allocate host memory
    h_NormW = (float*)malloc(scalar_size);
    h_MatA = (float*)malloc(mat_size);
    h_VecV = (float*)malloc(vec_size);
    h_VecW = (float*)malloc(vec_size);
    
    if (!h_MatA || !h_VecV || !h_VecW || !h_NormW)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(1);
    }
    
    // Initialize matrix and vector
    srand(42);  // Fixed seed for reproducibility
    UploadArray(h_MatA, N);
    InitOne(h_VecV, N);
    
    // ========================================================================
    // CPU Power Method
    // ========================================================================
    printf("\n=== CPU Power Method ===\n");
    clock_gettime(CLOCK_REALTIME, &t_start);
    float cpu_lamda = RunCPUPowerMethod();
    clock_gettime(CLOCK_REALTIME, &t_end);
    cpu_runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9 * (t_end.tv_nsec - t_start.tv_nsec);
    printf("CPU runtime: %f seconds\n", cpu_runtime);
    
    // ========================================================================
    // GPU Power Method
    // ========================================================================
    printf("\n=== GPU Power Method ===\n");
    checkCardVersion();
    
    // Re-initialize vector for GPU
    InitOne(h_VecV, N);
    
    // Kernel configuration
    int threadsPerBlock = BlockSize;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    // Shared memory for Av_Product_Shared: only need vector tile (blockSize floats)
    int sharedMemSizeAv = threadsPerBlock * sizeof(float);
    
    // Start total GPU timing (including memory transfers)
    clock_gettime(CLOCK_REALTIME, &t_start);
    
    // Allocate device memory (unified or manual)
    clock_gettime(CLOCK_REALTIME, &t_mem_start);
    
    if (UseUnifiedMemory)
    {
        // Unified Memory: allocate managed memory accessible by both CPU and GPU
        cudaMallocManaged((void**)&d_MatA, mat_size);
        cudaMallocManaged((void**)&d_VecV, vec_size);
        cudaMallocManaged((void**)&d_VecW, vec_size);
        cudaMallocManaged((void**)&d_NormW, scalar_size);
        cudaMallocManaged((void**)&d_Lamda, scalar_size);
        
        // Copy data directly to unified memory (replaces host arrays)
        memcpy(d_MatA, h_MatA, mat_size);
        memcpy(d_VecV, h_VecV, vec_size);
        
        // Prefetch to GPU for better initial performance
        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(d_MatA, mat_size, device);
        cudaMemPrefetchAsync(d_VecV, vec_size, device);
        cudaDeviceSynchronize();
    }
    else
    {
        // Manual memory management
        cudaMalloc((void**)&d_MatA, mat_size);
        cudaMalloc((void**)&d_VecV, vec_size);
        cudaMalloc((void**)&d_VecW, vec_size);
        cudaMalloc((void**)&d_NormW, scalar_size);
        cudaMalloc((void**)&d_Lamda, scalar_size);
        
        // Copy data to device
        cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
    }
    
    clock_gettime(CLOCK_REALTIME, &t_mem_end);
    memcpy_time = (t_mem_end.tv_sec - t_mem_start.tv_sec) + 1e-9 * (t_mem_end.tv_nsec - t_mem_start.tv_nsec);
    
    // Start compute-only timing
    struct timespec t_compute_start, t_compute_end;
    clock_gettime(CLOCK_REALTIME, &t_compute_start);
    
    float oldLamda = 0.0f;
    float lamda = 0.0f;
    float normW = 0.0f;
    int gpu_iterations = 0;
    
    // Initial Av product
    if (UseSharedMemory)
        Av_Product_Shared<<<blocksPerGrid, threadsPerBlock, sharedMemSizeAv>>>(d_MatA, d_VecV, d_VecW, N, BlockSize);
    else
        Av_Product_Global<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
    cudaDeviceSynchronize();
    
    // Power iteration loop
    for (int i = 0; i < max_iteration; i++)
    {
        // Step 1: Find norm of W
        cudaMemset(d_NormW, 0, scalar_size);
        FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW, N);
        cudaDeviceSynchronize();
        
        // Copy norm to host and compute sqrt
        if (UseUnifiedMemory)
            normW = sqrtf(*d_NormW);  // Direct access with unified memory
        else
        {
            cudaMemcpy(h_NormW, d_NormW, scalar_size, cudaMemcpyDeviceToHost);
            normW = sqrtf(*h_NormW);
        }
        
        // Step 2: Normalize W into V
        NormalizeW<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW, normW, N);
        cudaDeviceSynchronize();
        
        // Step 3: Compute W = A * V
        if (UseSharedMemory)
            Av_Product_Shared<<<blocksPerGrid, threadsPerBlock, sharedMemSizeAv>>>(d_MatA, d_VecV, d_VecW, N, BlockSize);
        else
            Av_Product_Global<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
        cudaDeviceSynchronize();
        
        // Step 4: Compute lambda = V . W
        cudaMemset(d_Lamda, 0, scalar_size);
        ComputeLamda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecV, d_VecW, d_Lamda, N);
        cudaDeviceSynchronize();
        
        // Copy lambda to host
        if (UseUnifiedMemory)
            lamda = *d_Lamda;  // Direct access with unified memory
        else
            cudaMemcpy(&lamda, d_Lamda, scalar_size, cudaMemcpyDeviceToHost);
        
        gpu_iterations = i + 1;
        
        // Check convergence
        if (fabsf(oldLamda - lamda) < EPS)
            break;
        oldLamda = lamda;
    }
    
    clock_gettime(CLOCK_REALTIME, &t_compute_end);
    gpu_runtime_no_memcpy = (t_compute_end.tv_sec - t_compute_start.tv_sec) + 
                            1e-9 * (t_compute_end.tv_nsec - t_compute_start.tv_nsec);
    
    clock_gettime(CLOCK_REALTIME, &t_end);
    gpu_runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9 * (t_end.tv_nsec - t_start.tv_nsec);
    
    printf("GPU converged after %d iterations, lambda = %f\n", gpu_iterations, lamda);
    printf("GPU runtime (with memcpy): %f seconds\n", gpu_runtime);
    printf("GPU runtime (compute only): %f seconds\n", gpu_runtime_no_memcpy);
    printf("Memory transfer time: %f seconds\n", memcpy_time);
    
    // ========================================================================
    // Results Summary and CSV Output
    // ========================================================================
    float speedup_with_memcpy = cpu_runtime / gpu_runtime;
    float speedup_no_memcpy = cpu_runtime / gpu_runtime_no_memcpy;
    
    printf("\n=== Performance Summary ===\n");
    printf("Speedup (with memcpy): %.2fx\n", speedup_with_memcpy);
    printf("Speedup (compute only): %.2fx\n", speedup_no_memcpy);
    printf("Lambda difference (CPU-GPU): %e\n", fabsf(cpu_lamda - lamda));
    
    // CSV output for automated parsing
    const char* mem_mgmt = UseUnifiedMemory ? "unified" : "manual";
    printf("\n=== CSV_OUTPUT ===\n");
    printf("matrix_size,block_size,memory_mode,memory_mgmt,cpu_time,gpu_time_with_memcpy,gpu_time_no_memcpy,memcpy_time,speedup_with_memcpy,speedup_no_memcpy,cpu_lambda,gpu_lambda,cpu_iterations,gpu_iterations\n");
    printf("%d,%d,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n",
           N, BlockSize, UseSharedMemory ? "shared" : "global", mem_mgmt,
           cpu_runtime, gpu_runtime, gpu_runtime_no_memcpy, memcpy_time,
           speedup_with_memcpy, speedup_no_memcpy,
           cpu_lamda, lamda, max_iteration, gpu_iterations);
    
    Cleanup();
    return 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

void Cleanup(void)
{
    if (d_MatA) cudaFree(d_MatA);
    if (d_VecV) cudaFree(d_VecV);
    if (d_VecW) cudaFree(d_VecW);
    if (d_NormW) cudaFree(d_NormW);
    if (d_Lamda) cudaFree(d_Lamda);
    
    if (h_MatA) free(h_MatA);
    if (h_VecV) free(h_VecV);
    if (h_VecW) free(h_VecW);
    if (h_NormW) free(h_NormW);
}

void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0.0f;
    data[0] = 1.0f;
}

void UploadArray(float* data, int n)
{
    int total = n * n;
    for (int i = 0; i < total; i++)
        data[i] = (float)(rand() % 101);
}

void Arguments(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0 ||
            strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n") == 0)
        {
            if (i + 1 < argc)
                GlobalSize = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0 ||
                 strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--b") == 0)
        {
            if (i + 1 < argc)
                BlockSize = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0 ||
                 strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--m") == 0)
        {
            if (i + 1 < argc)
                max_iteration = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--use_shared") == 0 || strcmp(argv[i], "-use_shared") == 0 ||
                 strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--s") == 0)
        {
            if (i + 1 < argc)
                UseSharedMemory = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--use_unified") == 0 || strcmp(argv[i], "-use_unified") == 0 ||
                 strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--u") == 0)
        {
            if (i + 1 < argc)
                UseUnifiedMemory = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --size, -n N           Matrix size (NxN), default: 5000\n");
            printf("  --blocksize, -b B      Threads per block, default: 32\n");
            printf("  --max_iteration, -m M  Max iterations, default: 100\n");
            printf("  --use_shared, -s [0|1] Use shared memory (1) or global (0), default: 1\n");
            printf("  --use_unified, -u [0|1] Use unified memory (1) or manual transfers (0), default: 0\n");
            printf("  --help, -h             Show this help\n");
            exit(0);
        }
    }
}

void checkCardVersion()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    if (prop.major < 2)
    {
        fprintf(stderr, "Error: Need compute capability 2.0 or higher\n");
        exit(1);
    }
}
