#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/// @brief Computes one row of matrix product of two square matrices.
/// @param M_d LHS of matrix product.
/// @param N_d RHS of matrix product.
/// @param[out] P_d matrix product result.
/// @param width width/height of matrices.
__global__ void matMulComputeRow(float *M, float *N, float *P, int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width)
    {
        for (int col = 0; col < width; ++col)
        {
            float dot_product = 0;
            for (int idx = 0; idx < width; ++idx)
            {
                dot_product += M[row * width + idx] * N[idx * width + col];
            }
            P[row * width + col] = dot_product;
        }
    }
}

/// @brief Computes one column of matrix product of two square matrices.
/// @param M_d LHS of matrix product.
/// @param N_d RHS of matrix product.
/// @param[out] P_d matrix product result.
/// @param width width/height of matrices.
__global__ void matMulComputeCol(float *M, float *N, float *P, int width)
{
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width)
    {
        for (int row = 0; row < width; ++row)
        {
            float dot_product = 0;
            for (int idx = 0; idx < width; ++idx)
            {
                dot_product += M[row * width + idx] * N[idx * width + col];
            }
            P[row * width + col] = dot_product;
        }
    }
}

/// @brief Computes one entry of matrix product of two square matrices.
/// @param M_d LHS of matrix product.
/// @param N_d RHS of matrix product.
/// @param[out] P_d matrix product result.
/// @param width width/height of matrices.
__global__ void matMulSingleElementKernel(float *M_d, float *N_d, float *P_d, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        float dot_product = 0;
        for (int k = 0; k < width; ++k)
        {
            dot_product = dot_product + M_d[row * width + k] * N_d[k * width + col];
        }
        P_d[row * width + col] = dot_product;
    }
}

/// @brief Computes matrix product of two square matrices.
/// @param M_h LHS of matrix product.
/// @param N_h RHS of matrix product.
/// @param[out] P_h matrix product result.
/// @param width width/height of matrices
/// @param kernel kernel function to use.
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float matMul(float *M_h, float *N_h, float *P_h, int width, void (*kernel)(float *, float *, float *, int), int iters, dim3 dimBlock, dim3 dimGrid)
{
    int required_size = width * width * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, required_size);
    cudaMalloc((void **)&N_d, required_size);
    cudaMalloc((void **)&P_d, required_size);

    cudaMemcpy(M_d, M_h, required_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, required_size, cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < iters; ++i)
    {
        (*kernel)<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err = cudaMemcpy(P_h, P_d, required_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return time;
}

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width/height of matrix
void printSquareMatrix(float *mat, int width)
{
    for (int row = 0; row < width; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            printf("%.1f ", mat[row * width + col]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Please provide an input file.\n");
        return 1;
    }

    bool print_device_properties = true;
    cudaDeviceProp *properties = new cudaDeviceProp;
    cudaError_t err = cudaGetDeviceProperties(properties, 0);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    else if (print_device_properties)
    {

        printf("name: %s\n", properties->name);
        printf("totalGlobalMem: %lu\n", properties->totalGlobalMem);
        printf("sharedMemPerBlock: %lu\n", properties->sharedMemPerBlock);
        printf("regsPerBlock: %d\n", properties->regsPerBlock);
        printf("warpSize: %d\n", properties->warpSize);
        printf("memPitch: %lu\n", properties->memPitch);
        printf("maxThreadsPerBlock: %d\n", properties->maxThreadsPerBlock);
        printf("maxThreadsDim: (%d, %d, %d)\n", properties->maxThreadsDim[0], properties->maxThreadsDim[1], properties->maxThreadsDim[2]);
        printf("maxGridSize: (%d, %d, %d)\n", properties->maxGridSize[0], properties->maxGridSize[1], properties->maxGridSize[2]);
        printf("clockRate: %d\n", properties->clockRate);
        printf("totalConstMem: %lu\n", properties->totalConstMem);
        printf("major: %d\n", properties->major);
        printf("minor: %d\n", properties->minor);
        printf("textureAlignment: %lu\n", properties->textureAlignment);
        printf("deviceOverlap: %d\n", properties->deviceOverlap);
    }

    // First line of file should be the width/height of the square matrices.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Remaining lines should be values for the matrices.
    FILE *file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int width;
    int print_matrices;

    fscanf(file_ptr, "%d", &width);
    fscanf(file_ptr, "%d", &print_matrices);

    float *M, *N, *P;

    int required_size = width * width * sizeof(float);
    M = (float *)malloc(required_size);
    N = (float *)malloc(required_size);
    P = (float *)malloc(required_size);

    for (int row = 0; row < width; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            fscanf(file_ptr, "%f", &M[row * width + col]);
        }
    }

    if (print_matrices != 0)
    {
        printf("Matrix M:\n");
        printSquareMatrix(M, width);
    }

    for (int row = 0; row < width; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            fscanf(file_ptr, "%f", &N[row * width + col]);
        }
    }
    fclose(file_ptr);
    if (print_matrices != 0)
    {
        printf("Matrix N:\n");
        printSquareMatrix(N, width);
    }

    int iters = 100;

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil((float)width / dimBlock.x), ceil((float)width / dimBlock.y), 1);
    printf("Computing Matrix Multiplication (matMulSingleElementKernel):\n");
    printf("Took %.1f msec for %d iterations.\n", matMul(M, N, P, width, matMulSingleElementKernel, iters, dimBlock, dimGrid), iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printSquareMatrix(P, width);
    }

    dimBlock = dim3(32, 1, 1);
    dimGrid = dim3(ceil((float)width / dimBlock.x), 1, 1);
    printf("\nComputing Matrix Multiplication (matMulComputeRow):\n");
    printf("Took %.1f msec for %d iterations.\n", matMul(M, N, P, width, matMulComputeRow, iters, dimBlock, dimGrid), iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printSquareMatrix(P, width);
    }

    dimBlock = dim3(1, 32, 1);
    dimGrid = dim3(1, ceil((float)width / dimBlock.y), 1);
    printf("\nComputing Matrix Multiplication (matMulComputeCol):\n");
    printf("Took %.1f msec for %d iterations.\n", matMul(M, N, P, width, matMulComputeCol, iters, dimBlock, dimGrid), iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printSquareMatrix(P, width);
    }
}