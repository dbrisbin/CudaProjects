#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/// @brief Computes one entry of product of a matrix with a vector.
/// @param M_d LHS of product.
/// @param v_d RHS of product.
/// @param[out] r_d product result.
/// @param width width/height of matrix/length of vector.
__global__ void matXVecMulSingleElementKernel(float *M_d, float *v_d, float *r_d, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width)
    {
        float dot_product = 0;
        for (int k = 0; k < width; ++k)
        {
            dot_product = dot_product + M_d[row * width + k] * v_d[k];
        }
        r_d[row] = dot_product;
    }
}

/// @brief Computes product of a matrix with a vector.
/// @param M_h LHS of product.
/// @param v_h RHS of product.
/// @param[out] r_h product result.
/// @param width width/height of matrices/length of vector
/// @param kernel kernel function to use.
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float matXVecMul(float *M_h, float *v_h, float *r_h, int width, void (*kernel)(float *, float *, float *, int), int iters)
{
    float *M_d, *v_d, *r_d;

    cudaMalloc((void **)&M_d, width * width * sizeof(float));
    cudaMalloc((void **)&v_d, width * sizeof(float));
    cudaMalloc((void **)&r_d, width * sizeof(float));

    cudaMemcpy(M_d, M_h, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(1, 32, 1);
    dim3 dimGrid(1, ceil((float)width / dimBlock.y), 1);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < iters; ++i)
    {
        (*kernel)<<<dimGrid, dimBlock>>>(M_d, v_d, r_d, width);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err = cudaMemcpy(r_h, r_d, width * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(M_d);
    cudaFree(v_d);
    cudaFree(r_d);

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

void printVector(float *vec, int length)
{
    printf("[");
    for (int idx = 0; idx < length; ++idx)
    {
        printf("%.1f ", vec[idx]);
    }
    printf("]^T\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Please provide an input file.\n");
        return 1;
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

    float *M, *v, *p;

    M = (float *)malloc(width * width * sizeof(float));
    v = (float *)malloc(width * sizeof(float));
    p = (float *)malloc(width * sizeof(float));

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

    for (int elt = 0; elt < width; ++elt)
    {
        fscanf(file_ptr, "%f", &v[elt]);
    }
    fclose(file_ptr);
    if (print_matrices != 0)
    {
        printf("Vector v:\n");
        printVector(v, width);
    }

    int iters = 1000;

    printf("Computing Matrix X Vector Multiplication (matXVecMulSingleElementKernel):\n");
    printf("Took %.1f msec for %d iterations.\n", matXVecMul(M, v, p, width, matXVecMulSingleElementKernel, iters), iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printVector(p, width);
    }
}