/// @file spmv_driver.cu
/// @brief Definition of driver function declared in spmv_driver.h.

#include <stdio.h>
#include <algorithm>
#include "spmv_utils.h"

/// @brief Compare two dense matrices.
/// @param mat1 Pointer to the first matrix.
/// @param mat2 Pointer to the second matrix.
/// @param m Number of rows in the matrix.
/// @param n Number of columns in the matrix.
/// @return True if the matrices are equal, false otherwise.
bool CompareMatrices(const float* mat1, const float* mat2, const int m, const int n)
{
    const int length = m * n;
    for (int i = 0; i < length; i++)
    {
        if (std::fabs(mat1[i] - mat2[i]) > 1)
        {
            // Print the indices of the mismatched elements
            printf("Mismatch at index %d: %.1f %.1f\n", i, mat1[i], mat2[i]);
            return false;
        }
    }
    return true;
}

/// @brief Print a dense matrix.
/// @param mat Pointer to the matrix.
/// @param m Number of rows in the matrix.
/// @param n Number of columns in the matrix.
void PrintMatrix(const float* mat, const int m, const int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.1f ", mat[i * n + j]);
        }
        printf("\n");
    }
}

/// @brief Convert a dense matrix to COO format and then to CSR format.
/// @param mat_h Host pointer to the dense matrix.
/// @param m Number of rows in the matrix.
/// @param n Number of columns in the matrix.
/// @return True if the original and converted matrices are equal, false otherwise.
bool ConvertAndCompare(const float* mat_h, const int m, const int n)
{
    float *values_coo_d{}, *values_csr_d{}, *values_coo_h{}, *values_csr_h{};
    int *col_indices_coo_d{}, *col_indices_csr_d{}, *row_indices_coo_d{}, *row_pointers_csr_d{},
        *col_indices_coo_h{}, *col_indices_csr_h{}, *row_indices_coo_h{}, *row_pointers_csr_h{};
    const int length = m * n;
    const int number_of_nnz_elts =
        std::count_if(mat_h, mat_h + length, [](float i) { return std::fabs(i) > 1e-3; });

    dim3 dim_block{}, dim_grid{};

    // Convert to COO format
    values_coo_h = new float[number_of_nnz_elts];
    row_indices_coo_h = new int[number_of_nnz_elts];
    col_indices_coo_h = new int[number_of_nnz_elts];

    UncompressedToCOO(mat_h, m, n, values_coo_h, row_indices_coo_h, col_indices_coo_h);

    // Allocate device memory
    cudaMalloc((void**)&values_coo_d, number_of_nnz_elts * sizeof(float));
    cudaMalloc((void**)&col_indices_coo_d, number_of_nnz_elts * sizeof(int));
    cudaMalloc((void**)&row_indices_coo_d, number_of_nnz_elts * sizeof(int));

    // Copy data to device
    cudaMemcpy(values_coo_d, values_coo_h, number_of_nnz_elts * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_coo_d, col_indices_coo_h, number_of_nnz_elts * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(row_indices_coo_d, row_indices_coo_h, number_of_nnz_elts * sizeof(int),
               cudaMemcpyHostToDevice);

    // delete host memory
    delete[] col_indices_coo_h;
    delete[] row_indices_coo_h;
    delete[] values_coo_h;

    // Convert to CSR format
    cudaMalloc((void**)&values_csr_d, number_of_nnz_elts * sizeof(float));
    cudaMalloc((void**)&col_indices_csr_d, number_of_nnz_elts * sizeof(int));
    cudaMalloc((void**)&row_pointers_csr_d, (m + 1) * sizeof(int));

    dim_block = dim3(1024, 1, 1);
    dim_grid = dim3(1, 1, 1);
    ConvertCOOToCSR<<<dim_grid, dim_block>>>(values_coo_d, row_indices_coo_d, col_indices_coo_d,
                                             number_of_nnz_elts, values_csr_d, col_indices_csr_d,
                                             row_pointers_csr_d, m);

    // Allocate host memory
    values_csr_h = new float[number_of_nnz_elts];
    col_indices_csr_h = new int[number_of_nnz_elts];
    row_pointers_csr_h = new int[m + 1];

    // Copy data to host
    cudaMemcpy(values_csr_h, values_csr_d, number_of_nnz_elts * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices_csr_h, col_indices_csr_d, number_of_nnz_elts * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(row_pointers_csr_h, row_pointers_csr_d, (m + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Decompress CSR to dense matrix
    float* mat_converted_h = new float[length];
    DecompressCSR(values_csr_h, col_indices_csr_h, row_pointers_csr_h, m, n, mat_converted_h);

    delete[] values_csr_h;
    delete[] col_indices_csr_h;
    delete[] row_pointers_csr_h;

    // Compare the original and converted matrices
    bool is_equal = CompareMatrices(mat_h, mat_converted_h, m, n);
    // Print matrices if they are not equal
    if (!is_equal)
    {
        printf("Original matrix:\n");
        PrintMatrix(mat_h, m, n);
        printf("Converted matrix:\n");
        PrintMatrix(mat_converted_h, m, n);
    }

    delete[] mat_converted_h;

    // delete device memory
    cudaFree(values_csr_d);
    cudaFree(values_coo_d);
    cudaFree(col_indices_csr_d);
    cudaFree(col_indices_coo_d);
    cudaFree(row_pointers_csr_d);
    cudaFree(row_indices_coo_d);

    return is_equal;
}

/// @brief Main function to test the SPMV kernel.
/// @param argc Number of command-line arguments.
/// @param argv Command-line arguments.
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: <input file>.\n");
        return 1;
    }

    // First line of file should contain the dimensions of the dense matrix, subsequent lines should
    // contain the data.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int m{}, n{};

    int scanf_result = fscanf(file_ptr, "%d %d", &m, &n);

    float* A{};

    A = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            scanf_result = fscanf(file_ptr, "%f", &A[i * n + j]);
        }
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    // Convert to COO and then to CSR format
    if (ConvertAndCompare(A, m, n))
    {
        printf("Conversion successful.\n");
    }
    else
    {
        printf("Conversion failed.\n");
    }
    return 0;
}