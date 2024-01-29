#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 20

__global__ void BlockTranspose(int *A_elements, int A_width, int A_height)
{
    __shared__ int blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int base_idx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    base_idx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[base_idx];
    __syncthreads();
    A_elements[base_idx] = blockA[threadIdx.x][threadIdx.y];
}

/// @brief generate a matrix for block transpose.
/// @param A pointer to beginning of matrix
/// @param A_width matrix width
/// @param A_height matrix height
void generateMatrix(int *A, int A_width, int A_height)
{
    int curr_val = 1;
    int step = 13;
    int maximum = BLOCK_WIDTH; // ensure columns are all same value (makes validating easy)

    for (int row = 0; row < A_height; ++row)
    {
        for (int col = 0; col < A_width; ++col)
        {
            A[A_width * row + col] = curr_val;
            curr_val = (curr_val + step) % maximum;
        }
    }
}

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix
/// @param height height of matrix to print
void printMatrix(int *mat, int width, int height)
{
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            printf("%d ", mat[row * width + col]);
        }
        printf("\n");
    }
}

bool validateResult(int *mat, int width, int height)
{
    for (int row = 0; row < height; ++row)
    {
        for (int col = 1; col < width; ++col)
        {
            if (mat[row * width + col] != mat[row * width])
            {
                return false;
            }
        }
    }
    return true;
}
int main()
{
    int A_width = BLOCK_WIDTH * 20;
    int A_height = BLOCK_WIDTH * 3;
    int *A = (int *)malloc(A_width * A_height * sizeof(int));
    generateMatrix(A, A_width, A_height);
    int *A_d;

    cudaMalloc((void **)&A_d, A_width * A_height * sizeof(int));
    cudaMemcpy(A_d, A, A_width * A_height * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid_dim(A_width / block_dim.x, A_height / block_dim.y);
    BlockTranspose<<<grid_dim, block_dim>>>(A_d, A_width, A_height);
    cudaMemcpy(A, A_d, A_width * A_height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(A_d);

    bool success = validateResult(A, A_width, A_height);
    free(A);
    if (success)
    {
        printf("Success!\n");
        return 0;
    }
    else
    {
        printf("FAILED!\n");
        return 1;
    }
}