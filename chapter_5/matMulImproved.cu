#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define TILE_WIDTH 2

/// @brief Compute general matrix multiplication using a tiling approach
/// @param M LHS matrix multiplication operand of size i x j.
/// @param N RHS matrix multiplication operand of size j x k.
/// @param[out] P Output matrix storing result of LHS * RHS of size i x k.
/// @param i # of rows in M and P.
/// @param j # of cols in M and # of rows in N.
/// @param k # of cols in N and P.
__global__ void matMulSingleElementKernel(float* M, float* N, float* P, int i,
                                          int j, int k) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float dot_product = 0;

    int num_phases =
        max(ceil(i / (float)TILE_WIDTH), ceil(k / (float)TILE_WIDTH));
    num_phases = max(num_phases, (int)ceil(j / (float)TILE_WIDTH));
    for (int phase = 0; phase < num_phases; ++phase) {
        // Collaborative loading of M and N tiles into shared memory.
        if (row < i && (phase * TILE_WIDTH + tx) < j) {
            Mds[ty][tx] = M[row * j + phase * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        if ((phase * TILE_WIDTH + ty) < j && col < k) {
            Nds[ty][tx] = N[(phase * TILE_WIDTH + ty) * k + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int dot_product_idx = 0; dot_product_idx < TILE_WIDTH;
             ++dot_product_idx) {
            dot_product += Mds[ty][dot_product_idx] * Nds[dot_product_idx][tx];
        }
        __syncthreads();
    }

    if (row < i && col < k) {
        P[row * k + col] = dot_product;
    }
}

/// @brief Computes matrix product of two matrices repeatedly for a specified
/// number of times.
/// @param M LHS matrix multiplication operand of size i x j.
/// @param N RHS matrix multiplication operand of size j x k.
/// @param[out] P Output matrix storing result of LHS * RHS of size i x k.
/// @param i # of rows in M and P.
/// @param j # of cols in M and # of rows in N.
/// @param k # of cols in N and P.
/// @param kernel kernel function to use.

/// @param iters number of iterations to run for timing. default: 1.
/// @return Time (in msec) taken to process the kernel (excludes memory
/// management)
float matMul(float* M_h, float* N_h, float* P_h, int i, int j, int k,
             void (*kernel)(float*, float*, float*, int, int, int),
             int iters = 1) {

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int gridDim{static_cast<int>(
        max(max(ceil((float)i / dimBlock.x), ceil((float)j / dimBlock.x)),
            ceil((float)k / dimBlock.x)))};
    dim3 dimGrid(gridDim, gridDim, 1);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void**)&M_d, i * j * sizeof(float));
    cudaMalloc((void**)&N_d, j * k * sizeof(float));
    cudaMalloc((void**)&P_d, i * k * sizeof(float));

    cudaMemcpy(M_d, M_h, i * j * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, j * k * sizeof(float), cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int iter = 0; iter < iters; ++iter) {
        (*kernel)<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, i, j, k);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err =
        cudaMemcpy(P_h, P_d, i * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__,
               __LINE__);
    }
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return time;
}

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix
/// @param height height of matrix to print
void printMatrix(float* mat, int height, int width) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            printf("%.0f ", mat[row * width + col]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Please provide an input file.\n");
        return 1;
    }

    cudaDeviceProp* properties = new cudaDeviceProp;
    cudaError_t err = cudaGetDeviceProperties(properties, 0);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__,
               __LINE__);
    }

    // First line of file should be the number of rows in M, number of cols in M,
    // and number of cols in N, respectively of the matrices.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Remaining lines should be values for the matrices.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL) {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int i, j, k;
    int print_matrices;

    fscanf(file_ptr, "%d %d %d", &i, &j, &k);
    fscanf(file_ptr, "%d", &print_matrices);

    float *M, *N, *P;

    M = (float*)malloc(i * j * sizeof(float));
    N = (float*)malloc(j * k * sizeof(float));
    P = (float*)malloc(i * k * sizeof(float));

    for (int row = 0; row < i; ++row) {
        for (int col = 0; col < j; ++col) {
            fscanf(file_ptr, "%f", &M[row * j + col]);
        }
    }

    if (print_matrices != 0) {
        printf("Matrix M:\n");
        printMatrix(M, i, j);
    }

    for (int row = 0; row < j; ++row) {
        for (int col = 0; col < k; ++col) {
            fscanf(file_ptr, "%f", &N[row * k + col]);
        }
    }
    fclose(file_ptr);
    if (print_matrices != 0) {
        printf("Matrix N:\n");
        printMatrix(N, j, k);
    }

    int iters = 100;

    printf("Computing Matrix Multiplication (matMulSingleElementKernel):\n");
    printf("Took %.1f msec for %d iterations.\n",
           matMul(M, N, P, i, j, k, matMulSingleElementKernel, iters), iters);
    if (print_matrices != 0) {
        printf("Result:\n");
        printMatrix(P, i, k);
    }
}