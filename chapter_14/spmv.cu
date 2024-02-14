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
                              const int* __restrict__ nnz_per_row, const float* __restrict__ vec,
                              float* result, const int num_rows)
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

__host__ void CooSpmvCPU(const float* values, const int* row_indices, const int* col_indices,
                         const float* vec, float* result, const int nnz, const int rows)
{
    // Set result to zero
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0.0f;
    }

    for (int i = 0; i < nnz; ++i)
    {
        const int row = row_indices[i];
        const int col = col_indices[i];
        result[row] += values[i] * vec[col];
    }
}

__global__ void JdsSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const int* __restrict__ row_permutation,
                              const int* __restrict__ iter_ptrs, const float* __restrict__ vec,
                              float* result, const int rows, const int max_nnz_per_row)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_perm = row_permutation[row];
    for (int i = 0; i < max_nnz_per_row; ++i)
    {
        const int num_active_rows = iter_ptrs[i + 1] - iter_ptrs[i];
        if (row < num_active_rows)
        {
            const int col = col_indices[iter_ptrs[i] + row];
            result[row_perm] += values[iter_ptrs[i] + row] * vec[col];
        }
    }
}