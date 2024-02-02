#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define TILE_WIDTH 4
#define FILTER_RADIUS 1
#define IN_TILE_DIM 4
#define OUT_TILE_DIM ((IN_TILE_DIM)-2 * FILTER_RADIUS)
#define TILE_DIM IN_TILE_DIM

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

enum KernelToUse
{
    kBasic = 0,
    kFInConstantMemory = 1,
    kTiledWithFInConstantMemory = 2,
    kTiledWithCachingAndFInConstantMemory = 3,
    kNumFilters = 4
};

/// @brief Naive implementation of 2D convolution of N of size height x width with kernel F of
/// radius filter_radius.
/// @param N Matrix to perform convolution on
/// @param F Convolution kernel
/// @param[out] P Result matrix
/// @param filter_radius radius of convolution kernel
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @return
__global__ void conv2DBasicKernel(float* N, float* F, float* P, int filter_radius, int width,
                                  int height)
{
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float p_value = 0.0f;
    const int f_width = 2 * filter_radius + 1;
    for (int f_row = -filter_radius; f_row <= filter_radius; ++f_row)
    {
        for (int f_col = -filter_radius; f_col <= filter_radius; ++f_col)
        {
            const int n_row = out_row + f_row;
            const int n_col = out_col + f_col;
            if (n_row >= 0 && n_row < height && n_col >= 0 && n_col < width)
            {
                p_value += F[(f_row + filter_radius) * f_width + f_col + filter_radius] *
                           N[n_row * width + n_col];
            }
        }
    }
    P[out_row * width + out_col] = p_value;
}

/// @brief Naive implementation of 2D convolution of N of size height x width with kernel F_c of
/// radius filter_radius stored in constant memory.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @return
__global__ void conv2DConstantMemFilterKernel(float* N, float* P, int width, int height)
{
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float p_value = 0.0f;
    for (int f_row = -FILTER_RADIUS; f_row <= FILTER_RADIUS; ++f_row)
    {
        for (int f_col = -FILTER_RADIUS; f_col <= FILTER_RADIUS; ++f_col)
        {
            const int n_row = out_row + f_row;
            const int n_col = out_col + f_col;
            if (n_row >= 0 && n_row < height && n_col >= 0 && n_col < width)
            {
                p_value +=
                    F_c[f_row + FILTER_RADIUS][f_col + FILTER_RADIUS] * N[n_row * width + n_col];
            }
        }
    }
    P[out_row * width + out_col] = p_value;
}

/// @brief Tiled implementation of 2D convolution of N of size height x width with kernel F_c of
/// radius filter_radius stored in constant memory.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @return
__global__ void conv2DTiledConstantMemFilterKernel(float* N, float* P, int width, int height)
{
    const int out_col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    const int out_row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (out_col < 0 || out_col >= width || out_row < 0 || out_row >= height)
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = N[out_row * width + out_col];
    }
    __syncthreads();

    float p_value = 0.0f;
    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;
    bool thread_computes_out_tile_entry =
        tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 && tile_row < OUT_TILE_DIM;
    bool thread_computes_entry_in_N =
        out_col >= 0 && out_col < width && out_row >= 0 && out_row < height;
    if (thread_computes_out_tile_entry && thread_computes_entry_in_N)
    {
        for (int f_row = 0; f_row < 2 * FILTER_RADIUS + 1; ++f_row)
        {
            for (int f_col = 0; f_col < 2 * FILTER_RADIUS + 1; ++f_col)
            {
                p_value += F_c[f_row][f_col] * N_s[tile_row + f_row][tile_col + f_col];
            }
        }
        P[out_row * width + out_col] = p_value;
    }
}

/// @brief Tiled implementation of 2D convolution of N of size height x width with kernel F_c of
/// radius filter_radius stored in constant memory which exploits caching of values.
/// @param N Matrix to perform convolution on
/// @param[out] P Result matrix
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @return
__global__ void conv2DTiledCachingConstantMemFilterKernel(float* N, float* P, int width, int height)
{
    const int out_col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int out_row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (out_col >= width || out_row >= height)
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = N[out_row * width + out_col];
    }
    __syncthreads();

    float p_value = 0.0f;
    if (out_col < width && out_row < height)
    {
        for (int f_row = 0; f_row < 2 * FILTER_RADIUS + 1; ++f_row)
        {
            for (int f_col = 0; f_col < 2 * FILTER_RADIUS + 1; ++f_col)
            {
                bool thread_is_in_tile = ((int)threadIdx.x - FILTER_RADIUS + f_col >= 0 &&
                                          (int)threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM &&
                                          (int)threadIdx.y - FILTER_RADIUS + f_row >= 0 &&
                                          (int)threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM);
                bool thread_is_in_N = (out_row - FILTER_RADIUS + f_row >= 0 &&
                                       out_row - FILTER_RADIUS + f_row < height &&
                                       out_col - FILTER_RADIUS + f_col >= 0 &&
                                       out_col - FILTER_RADIUS + f_col < width);
                if (thread_is_in_tile)
                {
                    p_value += F_c[f_row][f_col] * N_s[threadIdx.y + f_row - FILTER_RADIUS]
                                                      [threadIdx.x + f_col - FILTER_RADIUS];
                }
                else if (thread_is_in_N)
                {
                    p_value += F_c[f_row][f_col] * N[(out_row - FILTER_RADIUS + f_row) * width +
                                                     out_col - FILTER_RADIUS + f_col];
                }
            }
        }
        P[out_row * width + out_col] = p_value;
    }
}

/// @brief Prepare and run a 2D convolution of N of size height x width with kernel F of radius
/// filter_radius on a GPU.
/// @param N_h Matrix to perform convolution on (stored on CPU host)
/// @param F_h Convolution kernel (stored on CPU host)
/// @param[out] P_h Result matrix (stored on CPU host)
/// @param filter_radius radius of convolution kernel
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param iters number of iterations to run for timing. default: 1.
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float convolutionDriver(float* N_h, float* F_h, float* P_h, int filter_radius, int width,
                        int height, KernelToUse kernel_to_use, int iters)
{

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int gridDim{(int)(max(ceil((float)width / dimBlock.x), ceil((float)height / dimBlock.y)))};
    dim3 dimGrid(gridDim, gridDim, 1);
    float *N_d, *F_d, *P_d;

    int filter_width = 2 * filter_radius + 1;
    cudaMalloc((void**)&N_d, width * height * sizeof(float));
    cudaMalloc((void**)&F_d, filter_width * filter_width * sizeof(float));
    cudaMalloc((void**)&P_d, width * height * sizeof(float));

    cudaMemcpy(N_d, N_h, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice);

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
                conv2DBasicKernel<<<dimGrid, dimBlock>>>(N_d, F_d, P_d, filter_radius, width,
                                                         height);
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
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));
            for (int iter = 0; iter < iters; ++iter)
            {
                conv2DConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
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
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));
            dimBlock = dim3(IN_TILE_DIM, IN_TILE_DIM, 1);
            dimGrid = dim3((float)height / OUT_TILE_DIM, (float)width / OUT_TILE_DIM, 1);
            for (int iter = 0; iter < iters; ++iter)
            {
                conv2DTiledConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
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
                               (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));
            dimBlock = dim3(TILE_DIM, TILE_DIM, 1);
            dimGrid = dim3((float)height / dimBlock.x, (float)width / dimBlock.y, 1);
            for (int iter = 0; iter < iters; ++iter)
            {
                conv2DTiledCachingConstantMemFilterKernel<<<dimGrid, dimBlock>>>(N_d, P_d, width,
                                                                                 height);
            }
            break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err = cudaMemcpy(P_h, P_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(N_d);
    cudaFree(F_d);
    cudaFree(P_d);

    return time;
}

/// @brief Compare two matrices for equality within eps.
/// @param A first matrix to compare
/// @param B second matric to compare
/// @param height height of matrices
/// @param width width of matrices
/// @return True if the matrices are element-wise equal within eps, false otherwise.
bool matricesAreEqual(const float* A, const float* B, const int height, const int width,
                      const float eps = 0.0001)
{
    try
    {
        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int linear_idx = row * width + col;
                if (abs(A[linear_idx] - B[linear_idx]) > eps)
                    return false;
            }
        }
    }
    catch (...)
    {
        return false;
    }

    return true;
}

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix
/// @param height height of matrix to print
void printMatrix(const float* mat, const int height, const int width)
{
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            printf("%.0f ", mat[row * width + col]);
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

    // First line of file should be the number of rows in N, number of cols in N, and the radius of
    // the convolution kernel F, respectively.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Remaining lines should be values for the matrices, N then F, then P_expected.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int height, width, filter_radius;
    int print_matrices;
    KernelToUse kernel_to_use;

    fscanf(file_ptr, "%d %d %d", &height, &width, &filter_radius);
    fscanf(file_ptr, "%d", &print_matrices);
    fscanf(file_ptr, "%d", (int*)&kernel_to_use);

    if (kernel_to_use >= KernelToUse::kNumFilters)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int filter_width = 2 * filter_radius + 1;
    float *N, *F, *P, *P_expected;

    N = (float*)malloc(height * width * sizeof(float));
    F = (float*)malloc(filter_width * filter_width * sizeof(float));
    P = (float*)malloc(height * width * sizeof(float));
    P_expected = (float*)malloc(height * width * sizeof(float));

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            fscanf(file_ptr, "%f", &N[row * width + col]);
        }
    }
    if (print_matrices != 0)
    {
        printf("Matrix N:\n");
        printMatrix(N, height, width);
    }

    for (int row = 0; row < filter_width; ++row)
    {
        for (int col = 0; col < filter_width; ++col)
        {
            fscanf(file_ptr, "%f", &F[row * filter_width + col]);
        }
    }
    if (print_matrices != 0)
    {
        printf("Convolution kernel F:\n");
        printMatrix(F, filter_width, filter_width);
    }

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            fscanf(file_ptr, "%f", &P_expected[row * width + col]);
        }
    }

    fclose(file_ptr);
    int iters = 100;

    printf("Computing Convolution:\n");
    printf("Took %.1f msec for %d iterations.\n",
           convolutionDriver(N, F, P, filter_radius, width, height, kernel_to_use, iters), iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printMatrix(P, height, width);
    }

    bool is_correct = matricesAreEqual(P, P_expected, height, width);
    if (!is_correct)
    {
        printf("\nMatrices do not match!\n");
        if (print_matrices)
        {
            printf("Actual:\n");
            printMatrix(P, height, width);

            printf("\nExpected:\n");
            printMatrix(P_expected, height, width);
        }
    }
    else
    {
        printf("\nResult is equal to expected!\n");
    }
}