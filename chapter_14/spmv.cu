/// @file spmv.cu
/// @brief Implementation of kernels declared in spmv.h.

#include <cuda_runtime.h>
#include "spmv.h"
#include "spmv_utils.h"

__global__ void CooSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const int* __restrict__ row_indices, const float* __restrict__ vec,
                              float* result, const int num_nonzeros)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nonzeros)
    {
        const int row = row_indices[tid];
        const int col = col_indices[tid];
        atomicAdd(&result[row], values[tid] * vec[col]);
    }
}

__global__ void CsrSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const int* __restrict__ row_pointers, const float* __restrict__ vec,
                              float* result, const int num_rows)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows)
    {
        float sum = 0.0f;
        for (int i = row_pointers[row]; i < row_pointers[row + 1]; ++i)
        {
            sum += values[i] * vec[col_indices[i]];
        }
        result[row] = sum;
    }
}

__global__ void EllSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const float* __restrict__ vec, float* result, const int num_rows,
                              const int* nnz_per_row)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows)
    {
        float sum = 0.0f;
        for (unsigned int t = 0; t < nnz_per_row[row]; ++t)
        {
            const int i = t * num_rows + row;
            sum += values[i] * vec[col_indices[i]];
        }
        result[row] = sum;
    }
}