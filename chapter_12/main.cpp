/// @file main.cpp
/// @brief Main function to call the kernel driver to call the appropriate kernel, as determined by
/// input argument, on an input file.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include "merge_driver.h"
#include "merge_utils.h"
#include "types/constants.h"

/// @brief Compare two arrays for equality. Does not check that lengths are equal.
/// @param actual first array to check for equality
/// @param expected second array to check for equality
/// @param length length up to which to check for equality
/// @return true if the arrays are equal up to length length, false otherwise
bool ArraysAreEqual(const std::pair<int, int>* actual, const std::pair<int, int>* expected,
                    const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        if (actual[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

/// @brief Print a histogram to standard output.
/// @param arr array to print
/// @param length length of array to print
void PrintArr(const std::pair<int, int>* arr, const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        printf("{%d %d} ", arr[i].first, arr[i].second);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n",
               static_cast<int>(MergeKernel::kNumKernels) - 1);
        return 1;
    }

    // First line of file should contain the length of the data, subsequent lines should contain
    // the data to compute merge of.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    MergeKernel kernel_to_use{atoi(argv[2])};
    if (kernel_to_use >= MergeKernel::kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int m{};
    int n{};

    int scanf_result = fscanf(file_ptr, "%d %d", &m, &n);

    std::pair<int, int>* A{};
    std::pair<int, int>* B{};
    std::pair<int, int>* C{};
    std::pair<int, int>* C_expected{};

    A = (std::pair<int, int>*)malloc(m * sizeof(std::pair<int, int>));
    B = (std::pair<int, int>*)malloc(n * sizeof(std::pair<int, int>));
    C = (std::pair<int, int>*)malloc((m + n) * sizeof(std::pair<int, int>));
    C_expected = (std::pair<int, int>*)malloc((m + n) * sizeof(std::pair<int, int>));

    for (int i = 0; i < m; ++i)
    {
        scanf_result = fscanf(file_ptr, "%d %d", &(A[i].first), &(A[i].second));
    }
    for (int i = 0; i < n; ++i)
    {
        scanf_result = fscanf(file_ptr, "%d %d", &(B[i].first), &(B[i].second));
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    int iters{1};
    const auto time_to_compute = MergeDriver(A, m, B, n, C, kernel_to_use, iters);
    printf("Took %.2f msec to compute %d iterations.", time_to_compute, iters);

    MergeSequential(A, m, B, n, C_expected);
    if (!ArraysAreEqual(C, C_expected, m + n))
    {
        printf("\nResults are not equal!\n");
        printf("Expected:\n");
        if (m + n < 100)
        {
            PrintArr(C_expected, m + n);
        }
        else
        {
            PrintArr(C_expected, 100);
            printf("...\n");
        }

        printf("\nActual:\n");
        if (m + n < 100)
        {
            PrintArr(C, m + n);
        }
        else
        {
            PrintArr(C, 100);
            printf("...\n");
        }
    }
    else
    {
        printf("\nResult is correct!\n");
        if (m + n < 100)
        {
            PrintArr(C, m + n);
        }
        else
        {
            PrintArr(C, 100);
            printf("...\n");
        }
    }

    return 0;
}