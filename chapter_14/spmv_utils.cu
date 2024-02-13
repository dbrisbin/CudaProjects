/// @file spmv_utils.cu
/// @brief Implementation of utility functions declared in spmv_utils.h.

#include <cuda_runtime.h>
#include <algorithm>
#include "spmv_utils.h"
#include "types/constants.h"

__host__ void UncompressedToCOO(const float* matrix, const int rows, const int cols, float* values,
                                int* row_indices, int* col_indices)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                *values++ = matrix[i * cols + j];
                *row_indices++ = i;
                *col_indices++ = j;
            }
        }
    }
}

__host__ void DecompressCOO(const float* values, const int* row_indices, const int* col_indices,
                            const int num_nnz, float* matrix, const int cols)
{
    for (int i = 0; i < num_nnz; ++i)
    {
        matrix[row_indices[i] * cols + col_indices[i]] = values[i];
    }
}

__host__ __device__ void UncompressedToCSR(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* row_pointers)
{
    int nnz{0};
    *row_pointers++ = nnz;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                *values++ = matrix[i * cols + j];
                *col_indices++ = j;
                ++nnz;
            }
        }
        *row_pointers++ = nnz;
    }
}
__host__ __device__ void DecompressCSR(const float* values, const int* col_indices,
                                       const int* row_pointers, const int rows, const int cols,
                                       float* matrix)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = row_pointers[i]; j < row_pointers[i + 1]; ++j)
        {
            matrix[i * cols + col_indices[j]] = values[j];
        }
    }
}

__host__ __device__ void UncompressedToELL(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* nnz_per_row,
                                           const int num_nnz)
{
    int* values_ = new int[num_nnz];
    int* col_indices_ = new int[num_nnz];
    int* row_ptr = new int[rows];
    int accumulated_num_nnz{0};
    int max_nnz_per_row{0};

    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                values_[accumulated_num_nnz] = matrix[i * cols + j];
                col_indices_[accumulated_num_nnz++] = j;
                ++nnz;
            }
        }
        nnz_per_row[i] = nnz;
        row_ptr[i] = accumulated_num_nnz;
        max_nnz_per_row = max_nnz_per_row > nnz ? max_nnz_per_row : nnz;
    }

    for (int j = 0; j < max_nnz_per_row; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            if (j < nnz_per_row[i])
            {
                *values++ = values_[i * max_nnz_per_row + j];
                *col_indices++ = col_indices_[i * max_nnz_per_row + j];
            }
            else
            {
                *values++ = 0;
                *col_indices++ = 0;
            }
        }
    }
}

__host__ void DecompressELL(const float* values, const int* col_indices, const int* nnz_per_row,
                            const int rows, const int cols, float* matrix)
{
    const int max_nnz_per_row{*std::max_element(nnz_per_row, nnz_per_row + rows)};
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < nnz_per_row[i]; ++j)
        {
            matrix[i * cols + col_indices[i * max_nnz_per_row + j]] =
                values[i * max_nnz_per_row + j];
        }
    }
}

__global__ void ResetArray(unsigned int* data, unsigned int length, unsigned int val)
{
    unsigned int idx = CFACTOR * blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < min(CFACTOR * blockDim.x, length); i += blockDim.x)
    {
        data[i] = val;
    }
}
