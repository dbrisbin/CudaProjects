#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "reduction.h"
#include "reductionDriver.h"
#include "types/constants.h"

/// @brief Compute the reduction of data using addition operation
/// @param data data on which to compute the reduction
/// @param length length of the data array
/// @return the result of reduction
ReductionDataType computeReductionCPU(const ReductionDataType* data, const int length)
{
    ReductionDataType result = reductionIdentity();
    for (int i = 0; i < length; ++i)
    {
        result = reductionOperation(result, data[i]);
    }
    return result;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n", kNumKernels - 1);
        return 1;
    }

    // First line of file should contain the length of the data, subsequent lines should contain
    // the data to compute reduction on.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    enum reductionKernelToUse kernel_to_use = atoi(argv[2]);
    if (kernel_to_use >= kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int length;
    int scanf_result = fscanf(file_ptr, "%d", &length);
    ReductionDataType* data;
    data = (ReductionDataType*)malloc(length * sizeof(ReductionDataType));
    for (int i = 0; i < length; ++i)
    {
        scanf_result = fscanf(file_ptr, "%" FORMAT_TYPE, &data[i]);
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }
    ReductionDataType result = -1;

    int iters = 1;
    // compute the reduction.
    float time_to_compute_reduction = reductionDriver(data, length, &result, kernel_to_use, iters);

    printf("Took %.1f msec for %d iterations.\n", time_to_compute_reduction, iters);
    ReductionDataType expected_result = computeReductionCPU(data, length);
    if (result != expected_result)
    {
        printf("\nResults are not equal!\n");
        printf("Expected result:\n%" FORMAT_TYPE "\n", expected_result);

        printf("\nActual result:\n%" FORMAT_TYPE "\n", result);
    }
    else
    {
        printf("\nResult is correct!\n%" FORMAT_TYPE "\n", result);
    }
    return 0;
}