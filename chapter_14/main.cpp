/// @file main.cpp
/// @brief Main function to call the kernel driver to call the appropriate kernel, as determined by
/// input argument, on an input file.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include "spmv_driver.h"
#include "spmv_utils.h"
#include "types/constants.h"

/// @brief Perform a dense matrix-vector multiplication on the CPU.
/// @param matrix the dense matrix
/// @param vec the vector
/// @param result the result of the multiplication
/// @param rows number of rows in the matrix
/// @param cols number of columns in the matrix
void SpMVCPU(const float* matrix, const float* vec, float* result, const int rows, const int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[i * cols + j] * vec[j];
        }
    }
}

/// @brief Check if two vectors are equal.
/// @param actual first vector to compare
/// @param expected second vector to compare
/// @param length length of the vectors
/// @return true if the vectors are equal, false otherwise
bool VectorsAreEqual(const float* actual, const float* expected, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        if (std::fabs(actual[i] - expected[i]) > sqrt(FLOAT_EPS))
        {
            return false;
        }
    }
    return true;
}

/// @brief Print the contents of a vector.
/// @param vec the vector to print
/// @param length number of elements in the vector
void PrintVector(const float* vec, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n",
               static_cast<int>(SpmvKernel::kNumKernels) - 1);
        return 1;
    }

    // First line of file should contain the dimensions of the dense matrix, subsequent lines should
    // contain the data to perform SpMV.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    SpmvKernel kernel_to_use{atoi(argv[2])};
    if (kernel_to_use >= SpmvKernel::kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int m{}, n{};

    int scanf_result = fscanf(file_ptr, "%d %d", &m, &n);

    float* A{};
    float* vec{};
    float* result{};
    float* expected{};

    A = (float*)malloc(m * n * sizeof(float));
    vec = (float*)malloc(n * sizeof(float));
    result = (float*)malloc(m * sizeof(float));
    expected = (float*)malloc(m * sizeof(float));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            scanf_result = fscanf(file_ptr, "%f", &A[i * n + j]);
        }
    }

    for (int i = 0; i < n; ++i)
    {
        scanf_result = fscanf(file_ptr, "%f", &vec[i]);
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    int iters{1};

    const auto time_to_compute = SpMVDriver(A, vec, result, m, n, iters, kernel_to_use);
    printf("Took %.2f msec to compute %d iterations.", time_to_compute, iters);

    SpMVCPU(A, vec, expected, m, n);
    if (!VectorsAreEqual(result, expected, m))
    {
        printf("\nResults are not equal!\n");
        printf("Expected:\n");
        PrintVector(expected, m);

        printf("\nActual:\n");
        PrintVector(result, m);
    }
    else
    {
        printf("\nResult is correct!\n");
        PrintVector(result, m);
    }

    free(result);
    free(vec);
    free(A);

    return 0;
}