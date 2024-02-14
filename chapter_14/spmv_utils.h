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

/// @brief Get the maximum number of non-zero elements in a row of a dense matrix.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @return maximum number of non-zero elements in a row
__host__ int MaxNnzPerRow(const float* matrix, const int rows, const int cols);

/// @brief Convert a dense matrix to ELL format.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @param[out] values values of the non-zero elements stored in ELL format
/// @param[out] col_indices values of the column of the non-zero elements stored in ELL format
/// @param[out] nnz_per_row number of non-zero elements in a row
__host__ __device__ void UncompressedToELL(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* nnz_per_row,
                                           const int max_nnz_per_row);

/// @brief Decompress an ELL matrix into a dense matrix.
/// @param values values of the non-zero elements stored in ELL format
/// @param col_indices values of the column of the non-zero elements stored in ELL format
/// @param nnz_per_row number of non-zero elements in a row
/// @param rows number of rows in the dense matrix
/// @param cols number of columns in the dense matrix
/// @param[out] matrix destination dense matrix
__host__ void DecompressELL(const float* values, const int* col_indices, const int* nnz_per_row,
                            const int rows, const int cols, float* matrix);

/// @brief Convert a dense matrix to ELL-COO Hybrid format.
/// @param matrix matrix stored in dense format
/// @param rows number of rows in the matrix
/// @param cols number of cols in the matrix
/// @param[out] ell_values values of the non-zero elements stored in ELL format
/// @param[out] ell_col_indices values of the column of the non-zero elements stored in ELL format
/// @param[out] ell_nnz_per_row number of non-zero elements in a row in ELL format
/// @param[out] coo_values values of the non-zero elements stored in COO format
/// @param[out] coo_row_indices values of the row indices of the non-zero elements stored in COO
/// @param[out] coo_col_indices values of the column indices of the non-zero elements stored in COO
/// @param[out] max_nnz_per_row_ell maximum number of non-zero elements in a row in ELL format
/// @param[out] num_nnz_coo number of non-zero elements in COO format
__host__ void UncompressedToELLCOO(const float* matrix, const int rows, const int cols,
                                   float* ell_values, int* ell_col_indices, int* ell_nnz_per_row,
                                   float* coo_values, int* coo_row_indices, int* coo_col_indices,
                                   int* max_nnz_per_row_ell, int* num_nnz_coo);
/// @brief Reset an array to a given value.
/// @param data array to reset
/// @param length length of the array
/// @param val value to set
__global__ void ResetArray(unsigned int* data, unsigned int length, unsigned int val);

/// @brief Convert a COO matrix to CSR format.
/// @param values values of the non-zero elements stored in COO format
/// @param row_indices values of the row indices of the non-zero elements stored in COO format
/// @param col_indices values of the column indices of the non-zero elements stored in COO format
/// @param num_nnz number of non-zero elements
/// @param[out] csr_values values of the non-zero elements stored in CSR format
/// @param[out] csr_col_indices values of the column indices of the non-zero elements stored in CSR
/// @param[out] csr_row_pointers map rows to start positions in other arrays (values and
/// col_indices)
/// @param rows number of rows in the dense matrix
__global__ void ConvertCOOToCSR(const float* values, const int* row_indices, const int* col_indices,
                                const int num_nnz, float* csr_values, int* csr_col_indices,
                                int* csr_row_pointers, const int rows);

/// @brief Compute a histogram over an array of data.
/// @param data array to compute the histogram over
/// @param length length of the array
/// @param[out] hist histogram of the data
__device__ void basicParallelHistogram(const int* data, const int length, int* hist);

/// @brief Compute the exclusive sacn of an array of data using the Kogge-Stone algorithm.
/// @param data array to compute the exclusive scan over
/// @param result array to store the result of the exclusive scan
/// @param length length of the array
__device__ void KoggeStoneExclusiveScan(const int* data, int* result, int length);

#endif  // CHAPTER_14_SPMV_UTILS_H