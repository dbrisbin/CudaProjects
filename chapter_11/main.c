#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "parallel_scan.h"
#include "parallel_scan_driver.h"
#include "types/constants.h"

bool ArraysAreEqual(const ParallelScanDataType* actual, const ParallelScanDataType* expected,
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

/// @brief Print a histogram to standard output
/// @param arr array to print
/// @param length length of array to print
void PrintArr(const ParallelScanDataType* arr, const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        printf("%" FP_PRECISION FORMAT_TYPE " ", arr[i]);
    }
    printf("\n");
}

/// @brief Compute the parallel scan of data using addition operation
/// @param data data on which to compute the parallel scan
/// @param[out] result array in which to store the result of parallel scan
/// @param length length of the data array
void ComputeParallelScanCPU(const ParallelScanDataType* data, ParallelScanDataType* result,
                            const unsigned int length)
{
    if (length == 0)
    {
        return;
    }

    result[0] = data[0];
    for (unsigned int i = 1; i < length; ++i)
    {
        result[i] = ParallelScanOperation(result[i - 1], data[i]);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n", kNumKernels - 1);
        return 1;
    }

    // First line of file should contain the length of the data, subsequent lines should contain
    // the data to compute parallel scan on.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    enum parallelScanKernelToUse kernel_to_use = atoi(argv[2]);
    if (kernel_to_use >= kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    unsigned int length;
    int scanf_result = fscanf(file_ptr, "%u", &length);
    ParallelScanDataType* data;
    ParallelScanDataType* result;
    ParallelScanDataType* expected;
    data = (ParallelScanDataType*)malloc(length * sizeof(ParallelScanDataType));
    result = (ParallelScanDataType*)malloc(length * sizeof(ParallelScanDataType));
    expected = (ParallelScanDataType*)malloc(length * sizeof(ParallelScanDataType));

    for (unsigned int i = 0; i < length; ++i)
    {
        scanf_result = fscanf(file_ptr, "%" FORMAT_TYPE, &data[i]);
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    int iters = 1;
    // compute the parallel scan.
    float time_to_compute_parallel_scan =
        ParallelScanDriver(data, result, length, kernel_to_use, iters);

    printf("Took %.1f msec for %d iterations.\n", time_to_compute_parallel_scan, iters);
    ComputeParallelScanCPU(data, expected, length);
    if (!ArraysAreEqual(result, expected, length))
    {
        printf("\nResults are not equal!\n");
        printf("Expected:\n");
        PrintArr(expected, length);

        printf("\nActual:\n");
    }
    else
    {
        printf("\nResult is correct!\n");
    }
    PrintArr(result, length);
    return 0;
}