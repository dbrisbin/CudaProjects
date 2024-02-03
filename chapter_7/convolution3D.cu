#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define TILE_WIDTH 8
#define FILTER_RADIUS 3
#define IN_TILE_DIM 10
#define OUT_TILE_DIM ((IN_TILE_DIM)-2 * FILTER_RADIUS)
#define TILE_DIM 8

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

enum KernelToUse
{
    kBasic = 0,
    kFInConstantMemory = 1,
    kTiledWithFInConstantMemory = 2,
    kTiledWithCachingAndFInConstantMemory = 3,
    kNumFilters = 4
};

/// @brief Compute linearized index for 3 dimensional array.
/// @param z z index in array
/// @param y z index in array
/// @param x z index in array
/// @param width width of the array
/// @param height height of the array
/// @return linearized index
__device__ __host__ inline int linearized3DIndex(const int z, const int y, const int x,
                                                 const int width, const int height)
{
    return (z * height + y) * width + x;
}

/// @brief Naive implementation of 3D convolution of N of size height x width with kernel F of
/// radius filter_radius.
/// @param N Matrix to perform convolution on
/// @param F Convolution kernel
/// @param[out] P Result matrix
/// @param filter_radius radius of convolution kernel
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @return
__global__ void conv3DBasicKernel(float* N, float* F, float* P, int filter_radius, int width,
                                  int height, int depth)
{
    const int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    float p_value = 0.0f;
    const int f_width = 2 * filter_radius + 1;
    if (out_z >= 0 && out_z < depth && out_y >= 0 && out_y < height && out_x >= 0 && out_x < width)
    {
        for (int f_z = 0; f_z < f_width; ++f_z)
        {
            for (int f_y = 0; f_y < f_width; ++f_y)
            {
                for (int f_x = 0; f_x < f_width; ++f_x)
                {
                    const int n_z = out_z + f_z - filter_radius;
                    const int n_y = out_y + f_y - filter_radius;
                    const int n_x = out_x + f_x - filter_radius;
                    if (n_z >= 0 && n_z < depth && n_y >= 0 && n_y < height && n_x >= 0 &&
                        n_x < width)
                    {
                        p_value += F[linearized3DIndex(f_z, f_y, f_x, f_width, f_width)] *
                                   N[linearized3DIndex(n_z, n_y, n_x, width, height)];
                    }
                }
            }
        }
        P[linearized3DIndex(out_z, out_y, out_x, width, height)] = p_value;
    }
}

/// @brief Naive implementation of 3D convolution of N of size height x width with kernel F_c of
/// radius filter_radius stored in constant memory.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @return
__global__ void conv3DConstantMemFilterKernel(float* N, float* P, int width, int height, int depth)
{
    const int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    float p_value = 0.0f;
    if (out_z >= 0 && out_z < depth && out_y >= 0 && out_y < height && out_x >= 0 && out_x < width)
    {
        for (int f_z = 0; f_z < FILTER_RADIUS * 2 + 1; ++f_z)
        {
            for (int f_y = 0; f_y < FILTER_RADIUS * 2 + 1; ++f_y)
            {
                for (int f_x = 0; f_x < FILTER_RADIUS * 2 + 1; ++f_x)
                {
                    const int n_z = out_z + f_z - FILTER_RADIUS;
                    const int n_y = out_y + f_y - FILTER_RADIUS;
                    const int n_x = out_x + f_x - FILTER_RADIUS;
                    if (n_z >= 0 && n_z < depth && n_y >= 0 && n_y < height && n_x >= 0 &&
                        n_x < width)
                    {
                        p_value +=
                            F_c[f_z][f_y][f_x] * N[linearized3DIndex(n_z, n_y, n_x, width, height)];
                    }
                }
            }
        }
        P[linearized3DIndex(out_z, out_y, out_x, width, height)] = p_value;
    }
}

/// @brief Tiled implementation of 3D convolution of N of size depth x height x width with kernel
/// F_c of radius filter_radius stored in constant memory.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @return
__global__ void conv3DTiledConstantMemFilterKernel(float* N, float* P, int width, int height,
                                                   int depth)
{
    const int out_z = (int)blockIdx.z * OUT_TILE_DIM + (int)threadIdx.z - FILTER_RADIUS;
    const int out_y = (int)blockIdx.y * OUT_TILE_DIM + (int)threadIdx.y - FILTER_RADIUS;
    const int out_x = (int)blockIdx.x * OUT_TILE_DIM + (int)threadIdx.x - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    bool thread_computes_entry_in_N =
        out_z >= 0 && out_z < depth && out_y >= 0 && out_y < height && out_x >= 0 && out_x < width;
    if (thread_computes_entry_in_N)
    {
        float n_val = N[linearized3DIndex(out_z, out_y, out_x, width, height)];
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = n_val;
    }
    else
    {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float p_value = 0.0f;
    const int tile_z = (int)threadIdx.z - FILTER_RADIUS;
    const int tile_y = (int)threadIdx.y - FILTER_RADIUS;
    const int tile_x = (int)threadIdx.x - FILTER_RADIUS;

    bool thread_computes_out_tile_entry = tile_z >= 0 && tile_z < OUT_TILE_DIM && tile_y >= 0 &&
                                          tile_y < OUT_TILE_DIM && tile_x >= 0 &&
                                          tile_x < OUT_TILE_DIM;

    if (thread_computes_out_tile_entry && thread_computes_entry_in_N)
    {
        for (int f_z = 0; f_z < 2 * FILTER_RADIUS + 1; ++f_z)
        {
            for (int f_y = 0; f_y < 2 * FILTER_RADIUS + 1; ++f_y)
            {
                for (int f_x = 0; f_x < 2 * FILTER_RADIUS + 1; ++f_x)
                {
                    p_value += F_c[f_z][f_y][f_x] * N_s[tile_z + f_z][tile_y + f_y][tile_x + f_x];
                }
            }
        }
        P[linearized3DIndex(out_z, out_y, out_x, width, height)] = p_value;
    }
}

/// @brief Tiled implementation of 3D convolution of N of size depth x height x width with kernel
/// F_c of radius filter_radius stored in constant memory which exploits caching of values.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @return
__global__ void conv3DTiledCachingConstantMemFilterKernel(float* N, float* P, int width, int height,
                                                          int depth)
{
    const int out_z = blockIdx.z * TILE_DIM + threadIdx.z;
    const int out_y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int out_x = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float N_s[TILE_DIM][TILE_DIM][TILE_DIM];
    bool out_is_in_P = out_z < depth && out_y < height && out_x < width;
    if (out_is_in_P)
    {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] =
            N[linearized3DIndex(out_z, out_y, out_x, width, height)];
    }
    else
    {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float p_value = 0.0f;
    if (out_is_in_P)
    {
        for (int f_z = 0; f_z < 2 * FILTER_RADIUS + 1; ++f_z)
        {
            for (int f_y = 0; f_y < 2 * FILTER_RADIUS + 1; ++f_y)
            {
                for (int f_x = 0; f_x < 2 * FILTER_RADIUS + 1; ++f_x)
                {
                    bool thread_is_in_tile = ((int)threadIdx.z - FILTER_RADIUS + f_z >= 0 &&
                                              (int)threadIdx.z - FILTER_RADIUS + f_z < TILE_DIM &&
                                              (int)threadIdx.y - FILTER_RADIUS + f_y >= 0 &&
                                              (int)threadIdx.y - FILTER_RADIUS + f_y < TILE_DIM &&
                                              (int)threadIdx.x - FILTER_RADIUS + f_x >= 0 &&
                                              (int)threadIdx.x - FILTER_RADIUS + f_x < TILE_DIM);
                    bool thread_is_in_N =
                        (out_z - FILTER_RADIUS + f_z >= 0 && out_z - FILTER_RADIUS + f_z < depth &&
                         out_y - FILTER_RADIUS + f_y >= 0 && out_y - FILTER_RADIUS + f_y < height &&
                         out_x - FILTER_RADIUS + f_x >= 0 && out_x - FILTER_RADIUS + f_x < width);
                    if (thread_is_in_tile)
                    {
                        p_value += F_c[f_z][f_y][f_x] * N_s[threadIdx.z + f_z - FILTER_RADIUS]
                                                           [threadIdx.y + f_y - FILTER_RADIUS]
                                                           [threadIdx.x + f_x - FILTER_RADIUS];
                    }
                    else if (thread_is_in_N)
                    {
                        p_value += F_c[f_z][f_y][f_x] *
                                   N[linearized3DIndex(out_z - FILTER_RADIUS + f_z,
                                                       out_y - FILTER_RADIUS + f_y,
                                                       out_x - FILTER_RADIUS + f_x, width, height)];
                    }
                }
            }
        }
        P[linearized3DIndex(out_z, out_y, out_x, width, height)] = p_value;
    }
}

/// @brief Prepare and run a 3D convolution of N of size depth x height x width with kernel F of
/// radius filter_radius on a GPU.
/// @param N_h Matrix to perform convolution on (stored on CPU host)
/// @param F_h Convolution kernel (stored on CPU host)
/// @param[out] P_h Result matrix (stored on CPU host)
/// @param filter_radius radius of convolution kernel
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @param kernel_to_use kernel to use for computing the convolution
/// @param iters number of iterations to run for timing. default: 1
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float convolutionDriver(float* N_h, float* F_h, float* P_h, int filter_radius, int width,
                        int height, int depth, KernelToUse kernel_to_use, int iters)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

    dim3 dimGrid(ceil((float)width / dimBlock.x), ceil((float)height / dimBlock.y),
                 ceil((float)depth / dimBlock.z));
    float *N_d, *F_d, *P_d;

    int filter_width = 2 * filter_radius + 1;

    cudaMalloc((void**)&N_d, width * height * depth * sizeof(float));
    cudaMalloc((void**)&F_d, filter_width * filter_width * filter_width * sizeof(float));
    cudaMalloc((void**)&P_d, width * height * depth * sizeof(float));

    cudaMemcpy(N_d, N_h, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, filter_width * filter_width * filter_width * sizeof(float),
               cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    switch (kernel_to_use)
    {
        case (KernelToUse::kBasic):
            for (int iter = 0; iter < iters; ++iter)
            {
                conv3DBasicKernel<<<dimGrid, dimBlock>>>(N_d, F_d, P_d, filter_radius, width,
                                                         height, depth);
            }
            break;
        case (KernelToUse::kFInConstantMemory):
            if (filter_radius != FILTER_RADIUS)
            {
                printf(
                    "Please change the defined size of the constant "
                    "filter "
                    "to use a kernel with the filter in constant "
                    "memory!\n");
                return -1.0f;
            }
            cudaMemcpyToSymbol(F_c, F_h,
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) *
                                   (2 * FILTER_RADIUS + 1) * sizeof(float));
            for (int iter = 0; iter < iters; ++iter)
            {
                conv3DConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height,
                                                                     depth);
            }
            break;
        case (KernelToUse::kTiledWithFInConstantMemory):
            if (filter_radius != FILTER_RADIUS)
            {
                printf(
                    "Please change the defined size of the constant "
                    "filter "
                    "to use a kernel with the filter in constant "
                    "memory!\n");
                return -1.0f;
            }
            cudaMemcpyToSymbol(F_c, F_h,
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) *
                                   (2 * FILTER_RADIUS + 1) * sizeof(float));
            dimBlock = dim3(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
            dimGrid = dim3(ceil((float)width / OUT_TILE_DIM), ceil((float)height / OUT_TILE_DIM),
                           ceil((float)depth / OUT_TILE_DIM));
            for (int iter = 0; iter < iters; ++iter)
            {
                conv3DTiledConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height,
                                                                          depth);
            }
            break;
        case (KernelToUse::kTiledWithCachingAndFInConstantMemory):
            if (filter_radius != FILTER_RADIUS)
            {
                printf(
                    "Please change the defined size of the constant "
                    "filter "
                    "to use a kernel with the filter in constant "
                    "memory!\n");
                return -1.0f;
            }
            cudaMemcpyToSymbol(F_c, F_h,
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) *
                                   (2 * FILTER_RADIUS + 1) * sizeof(float));
            dimBlock = dim3(TILE_DIM, TILE_DIM, TILE_DIM);
            dimGrid = dim3(ceil((float)width / dimBlock.x), ceil((float)height / dimBlock.y),
                           ceil((float)depth / dimBlock.z));
            for (int iter = 0; iter < iters; ++iter)
            {
                conv3DTiledCachingConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width,
                                                                                 height, depth);
            }
            break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err =
        cudaMemcpy(P_h, P_d, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(N_d);
    cudaFree(F_d);
    cudaFree(P_d);

    return time;
}

struct matrixComparisonResult
{
    bool success;
    int index_of_first_mismatch;
};

/// @brief Compare two matrices for equality within eps.
/// @param A first matrix to compare
/// @param B second matric to compare
/// @param width width of matrices
/// @param height height of matrices
/// @param depth depth of matrices
/// @param eps epsilon to use for floating point equality comparisons
/// @return True if the matrices are element-wise equal within eps, false otherwise.
matrixComparisonResult matricesAreEqual(const float* A, const float* B, const int width,
                                        const int height, const int depth, const float eps = 0.0001)
{
    try
    {
        for (int z = 0; z < depth; ++z)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    int linear_idx = linearized3DIndex(z, y, x, width, height);
                    if (abs(A[linear_idx] - B[linear_idx]) > eps)
                    {
                        return {false, linear_idx};
                    }
                }
            }
        }
    }
    catch (...)
    {
        return {false, -1};
    }

    return {true, 0};
}

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix to print
/// @param height height of matrix to print
/// @param depth depth of matrix to print
void printMatrix(const float* mat, const int width, const int height, const int depth)
{
    for (int z = 0; z < depth; ++z)
    {
        printf("Layer %d:\n", z);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                printf("%.0f ", mat[linearized3DIndex(z, y, x, width, height)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Please provide an input file.\n");
        return 1;
    }

    cudaDeviceProp* properties = new cudaDeviceProp;
    cudaError_t err = cudaGetDeviceProperties(properties, 0);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // First line of file should be the number of cols in N, number of rows in N, number of
    // layers in N, and the radius of the convolution kernel F, respectively. Second line should
    // be 1 or 0 indicating whether or not to print the matrices. Third line should be the
    // integer representation of a KernelToUse. Remaining lines should be values for the
    // matrices, N then F, then P_expected.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int height, width, depth, filter_radius;
    int print_matrices;
    KernelToUse kernel_to_use;

    // Read dimensions and parameters
    fscanf(file_ptr, "%d %d %d %d", &width, &height, &depth, &filter_radius);
    fscanf(file_ptr, "%d", &print_matrices);
    fscanf(file_ptr, "%d", (int*)&kernel_to_use);

    if (kernel_to_use >= KernelToUse::kNumFilters)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int filter_width = 2 * filter_radius + 1;
    float *N, *F, *P, *P_expected;

    // allocate memory for matrices
    N = (float*)malloc(depth * height * width * sizeof(float));
    F = (float*)malloc(filter_width * filter_width * filter_width * sizeof(float));
    P = (float*)malloc(depth * height * width * sizeof(float));
    P_expected = (float*)malloc(depth * height * width * sizeof(float));

    // Read N.
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                fscanf(file_ptr, "%f", &N[linearized3DIndex(z, y, x, width, height)]);
            }
        }
    }
    if (print_matrices != 0)
    {
        printf("Matrix N:\n");
        printMatrix(N, width, height, depth);
    }

    // Read filter
    for (int z = 0; z < filter_width; ++z)
    {
        for (int y = 0; y < filter_width; ++y)
        {
            for (int x = 0; x < filter_width; ++x)
            {
                fscanf(file_ptr, "%f", &F[linearized3DIndex(z, y, x, filter_width, filter_width)]);
            }
        }
    }
    if (print_matrices != 0)
    {
        printf("Convolution kernel F:\n");
        printMatrix(F, filter_width, filter_width, filter_width);
    }

    // Optionally read P_expected
    int scanf_result{0};
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                scanf_result =
                    fscanf(file_ptr, "%f", &P_expected[linearized3DIndex(z, y, x, width, height)]);
                if (scanf_result == EOF)
                    break;
            }
            if (scanf_result == EOF)
                break;
        }
        if (scanf_result == EOF)
            break;
    }

    fclose(file_ptr);
    int iters = 100;

    // compute the convolution.
    printf("Computing Convolution:\n");
    printf("Took %.1f msec for %d iterations.\n",
           convolutionDriver(N, F, P, filter_radius, width, height, depth, kernel_to_use, iters),
           iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printMatrix(P, width, height, depth);
    }

    // P_expected was not read. Fall back to naive approach for GT.
    if (scanf_result == EOF)
    {
        convolutionDriver(N, F, P_expected, filter_radius, width, height, depth,
                          KernelToUse::kBasic, 1);
    }

    matrixComparisonResult matrix_comparison_result =
        matricesAreEqual(P, P_expected, width, height, depth);
    if (!matrix_comparison_result.success)
    {
        printf("\nMatrices do not match!\n");
        printf("First mismatch occurs at index: %d\n",
               matrix_comparison_result.index_of_first_mismatch);
        if (print_matrices)
        {
            printf("Actual:\n");
            printMatrix(P, width, height, depth);

            printf("\nExpected:\n");
            printMatrix(P_expected, width, height, depth);
        }
    }
    else
    {
        printf("\nResult is equal to expected!\n");
    }
}