/// @file spmv.h
/// @brief Declaration of available kernels for computing the sparse matrix-vector multiplication.

#ifndef CHAPTER_14_SPMV_H
#define CHAPTER_14_SPMV_H

#include <cuda_runtime.h>

/// @brief Performs a sparse matrix-vector multiplication using the COO format.
/// @param values values of the non-zero elements of the matrix
/// @param col_indices column indices of the non-zero elements of the matrix
/// @param row_indices row indices of the non-zero elements of the matrix
/// @param vec input vector
/// @param[out] result output vector
/// @param num_nonzeros number of non-zero elements of the matrix
/// @return
__global__ void CooSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const int* __restrict__ row_indices, const float* __restrict__ vec,
                              float* result, const int num_nonzeros);

/// @brief Performs a sparse matrix-vector multiplication using the CSR format.
/// @param values values of the non-zero elements of the matrix
/// @param col_indices column indices of the non-zero elements of the matrix
/// @param row_pointers row pointers of the matrix
/// @param vec input vector
/// @param[out] result output vector
/// @param num_rows number of rows of the matrix
__global__ void CsrSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const int* __restrict__ row_pointers, const float* __restrict__ vec,
                              float* result, const int num_rows);

__global__ void EllSpmvKernel(const float* __restrict__ values, const int* __restrict__ col_indices,
                              const float* __restrict__ vec, float* result, const int num_rows,
                              const int num_cols, const int max_nonzeros_per_row);

#endif  // CHAPTER_14_SPMV_H