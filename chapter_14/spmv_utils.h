/// @file spmv_utils.h
/// @brief Declaration of utility functions used by kernels and driver code.

#ifndef CHAPTER_14_SPMV_UTILS_H
#define CHAPTER_14_SPMV_UTILS_H

#include <cuda_runtime.h>

extern __device__ int block_counter;

/// @brief Convert a dense matrix to COO format.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @param[out] values values of the non-zero elements stored in COO format
/// @param[out] row_indices values of the row indices of the non-zero elements stored in COO format
/// @param[out] col_indices values of the column indices of the non-zero elements stored in COO
/// format
__host__ __device__ void UncompressedToCOO(const float* matrix, const int rows, const int cols,
                                           float* values, int* row_indices, int* col_indices);

/// @brief Decompress a COO matrix into a dense matrix.
/// @param values values of the non-zero elements stored in COO format
/// @param row_indices values of the row indices of the non-zero elements stored in COO format
/// @param col_indices values of the column indices of the non-zero elements stored in COO format
/// @param num_nnz number of non-zero elements
/// @param[out] matrix destination dense matrix
/// @param cols number of columns in the dense matrix
__host__ __device__ void DecompressCOO(const float* values, const int* row_indices,
                                       const int* col_indices, const int num_nnz, float* matrix,
                                       const int cols);

/// @brief Convert a dense matrix to CSR format.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @param[out] values values of the non-zero elements stored in CSR format
/// @param[out] col_indices values of the column indices of the non-zero elements stored in CSR
/// format
/// @param[out] row_pointers map rows to start positions in other arrays (values and col_indices)
__host__ __device__ void UncompressedToCSR(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* row_pointers);

/// @brief Decompress a CSR matrix into a dense matrix.
/// @param values values of the non-zero elements stored in CSR format
/// @param col_indices values of the column indices of the non-zero elements stored in CSR format
/// @param row_pointers map rows to start positions in other arrays (values and col_indices)
/// @param rows number of rows in the dense matrix
/// @param cols number of columns in the dense matrix
/// @param[out] matrix destination dense matrix
__host__ __device__ void DecompressCSR(const float* values, const int* col_indices,
                                       const int* row_pointers, const int rows, const int cols,
                                       float* matrix);

/// @brief Convert a dense matrix to ELL format.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @param[out] values values of the non-zero elements stored in ELL format
/// @param[out] col_indices values of the column of the non-zero elements stored in ELL format
/// @param[out] nnz_per_row number of non-zero elements in a row
__host__ __device__ void UncompressedToELL(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* nnz_per_row);

/// @brief Decompress an ELL matrix into a dense matrix.
/// @param values values of the non-zero elements stored in ELL format
/// @param col_indices values of the column of the non-zero elements stored in ELL format
/// @param nnz_per_row number of non-zero elements in a row
/// @param rows number of rows in the dense matrix
/// @param cols number of columns in the dense matrix
/// @param[out] matrix destination dense matrix
__host__ void DecompressELL(const float* values, const int* col_indices, const int* nnz_per_row,
                            const int rows, const int cols, float* matrix);

__global__ void ResetArray(unsigned int* data, unsigned int length, unsigned int val);

#endif  // CHAPTER_14_SPMV_UTILS_H