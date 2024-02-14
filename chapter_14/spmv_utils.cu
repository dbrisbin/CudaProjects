/// @file spmv_utils.cu
/// @brief Implementation of utility functions declared in spmv_utils.h.

#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
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
    // Fill matrix with zeros
    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = 0;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = row_pointers[i]; j < row_pointers[i + 1]; ++j)
        {
            matrix[i * cols + col_indices[j]] = values[j];
        }
    }
}

__host__ int MaxNnzPerRow(const float* matrix, const int rows, const int cols)
{
    int max_nnz_per_row{0};
    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                ++nnz;
            }
        }
        max_nnz_per_row = max_nnz_per_row > nnz ? max_nnz_per_row : nnz;
    }
    return max_nnz_per_row;
}

__host__ __device__ void UncompressedToELL(const float* matrix, const int rows, const int cols,
                                           float* values, int* col_indices, int* nnz_per_row,
                                           const int max_nnz_per_row)
{
    float* values_ = new float[max_nnz_per_row * rows];
    int* col_indices_ = new int[max_nnz_per_row * rows];

    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                values_[i * max_nnz_per_row + nnz] = matrix[i * cols + j];
                col_indices_[i * max_nnz_per_row + nnz] = j;
                ++nnz;
            }
        }
        nnz_per_row[i] = nnz;
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

    delete[] col_indices_;
    delete[] values_;
}

__host__ void UncompressedToELLCOO(const float* matrix, const int rows, const int cols,
                                   float* ell_values, int* ell_col_indices, int* ell_nnz_per_row,
                                   float* coo_values, int* coo_row_indices, int* coo_col_indices,
                                   int* max_nnz_per_row_ell, int* num_nnz_coo)
{
    *num_nnz_coo = 0;
    int* nnz_per_row = new int[rows];
    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                ++nnz;
            }
        }
        nnz_per_row[i] = nnz;
    }

    // compute the mean nnz per row
    float mean_nnz_per_row{std::accumulate(nnz_per_row, nnz_per_row + rows, 0) /
                           static_cast<float>(rows)};

    // Compute the stdev of nnz data
    float stdev_nnz_per_row{std::accumulate(
        nnz_per_row, nnz_per_row + rows, 0.0f, [mean_nnz_per_row](float acc, int nnz) -> float {
            return acc + (nnz - mean_nnz_per_row) * (nnz - mean_nnz_per_row);
        })};
    stdev_nnz_per_row = sqrt(stdev_nnz_per_row / static_cast<float>(rows));

    // Place up to mean + stdev elements from each row in ELL format. Store the rest in COO.
    int max_nnz_per_row{static_cast<int>(mean_nnz_per_row + stdev_nnz_per_row)};
    float* values_ = new float[max_nnz_per_row * rows];
    int* col_indices_ = new int[max_nnz_per_row * rows];
    int num_nnz = mean_nnz_per_row * rows;

    // Fill values and col_indices arrays with zeros
    for (int i = 0; i < max_nnz_per_row * rows; ++i)
    {
        values_[i] = 0;
        col_indices_[i] = 0;
    }
    // Fill CoO values and indices arrays with zeros
    for (int i = 0; i < num_nnz; ++i)
    {
        coo_values[i] = 0.0;
        coo_row_indices[i] = 0;
        coo_col_indices[i] = 0;
    }
    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                if (nnz < max_nnz_per_row)
                {
                    values_[i * max_nnz_per_row + nnz] = matrix[i * cols + j];
                    col_indices_[i * max_nnz_per_row + nnz] = j;
                }
                else
                {
                    *coo_values++ = matrix[i * cols + j];
                    *coo_row_indices++ = i;
                    *coo_col_indices++ = j;
                    ++(*num_nnz_coo);
                }
                ++nnz;
            }
        }
        ell_nnz_per_row[i] = std::min(nnz, max_nnz_per_row);
    }

    for (int j = 0; j < max_nnz_per_row; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            if (j < nnz_per_row[i])
            {
                *ell_values++ = values_[i * max_nnz_per_row + j];
                *ell_col_indices++ = col_indices_[i * max_nnz_per_row + j];
            }
            else
            {
                *ell_values++ = 0;
                *ell_col_indices++ = 0;
            }
        }
    }

    delete[] col_indices_;
    delete[] values_;
    *max_nnz_per_row_ell = max_nnz_per_row;
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
__device__ void basicParallelHistogram(const int* data, const int length, int* hist)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        atomicAdd(&hist[data[i]], 1);
    }
}

__device__ void KoggeStoneExclusiveScan(const int* data, int* result, int length)
{
    __shared__ int XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length && tx != 0)
    {
        XY[tx] = data[i - 1];
    }
    else
    {
        XY[tx] = 0;
    }
    unsigned int temp;
    for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        if (tx >= stride)
        {
            temp = XY[tx] + XY[tx - stride];
        }
        __syncthreads();
        if (tx >= stride)
        {
            XY[tx] = temp;
        }
    }
    if (i < length)
    {
        result[i] = XY[tx];
    }
}

__global__ void ConvertCOOToCSR(const float* values, const int* row_indices, const int* col_indices,
                                const int num_nnz, float* csr_values, int* csr_col_indices,
                                int* csr_row_pointers, const int rows)
{
    // Part 1: Apply histogram to count the number of non-zero elements in each row
    __shared__ int hist[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SECTION_SIZE)
    {
        hist[i] = 0;
    }
    __syncthreads();
    basicParallelHistogram(row_indices, num_nnz, hist);
    __syncthreads();

    // Part 2: Apply scan to calculate the row pointers
    KoggeStoneExclusiveScan(hist, csr_row_pointers, rows);
    if (i == 0)
    {
        csr_row_pointers[rows] = num_nnz;
    }
    __syncthreads();

    // Part 3: Move the nonzero elements to the correct position in CSR format.
    // Reuse hist to track the next position to write the non-zero elements.
    if (i < SECTION_SIZE)
    {
        hist[i] = 0;
    }
    __syncthreads();
    for (unsigned int j = i; j < num_nnz; j += blockDim.x)
    {
        int row = row_indices[j];
        int dest = csr_row_pointers[row] + atomicAdd(&hist[row], 1);
        csr_values[dest] = values[j];
        csr_col_indices[dest] = col_indices[j];
    }
}

__host__ void UncompressedToJDS(const float* matrix, const int rows, const int cols, float* values,
                                int* col_indices, int* row_permutation, int* iter_ptr,
                                const int max_nnz_per_row)
{
    float* values_ = new float[max_nnz_per_row * rows];
    int* col_indices_ = new int[max_nnz_per_row * rows];
    std::pair<int, int>* nnz_per_row = new std::pair<int, int>[rows];

    for (int i = 0; i < rows; ++i)
    {
        int nnz{0};
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(matrix[i * cols + j]) > FLOAT_EPS)
            {
                values_[i * max_nnz_per_row + nnz] = matrix[i * cols + j];
                col_indices_[i * max_nnz_per_row + nnz] = j;
                ++nnz;
            }
        }
        nnz_per_row[i] = std::make_pair(nnz, i);
    }

    std::sort(nnz_per_row, nnz_per_row + rows, std::greater<std::pair<int, int>>());

    for (int i = 0; i < rows; ++i)
    {
        row_permutation[i] = nnz_per_row[i].second;
    }

    *iter_ptr++ = 0;
    int num_elts{0};
    for (int j = 0; j < max_nnz_per_row; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            int row = row_permutation[i];
            if (j < nnz_per_row[i].first)
            {
                *values++ = values_[row * max_nnz_per_row + j];
                *col_indices++ = col_indices_[row * max_nnz_per_row + j];
                ++num_elts;
            }
            else
            {
                break;
            }
        }
        *iter_ptr++ = num_elts;
    }
}