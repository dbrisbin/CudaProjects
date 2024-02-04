#include <stdio.h>
#include <stdlib.h>
#include "matrixUtils.h"
#include "stencilDriver.h"
#include "types/constants.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Please provide an input file.\n");
        return 1;
    }

    // First line of file should be the number of cols in N, number of rows in N, number of
    // layers in N, and the radius of the convolution kernel F, respectively. Second line should
    // be 1 or 0 indicating whether or not to print the matrices. Third line should be the
    // integer representation of a KernelToUse. Remaining lines should be values for the
    // matrices, N then F, then P_expected.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int height, width, depth;
    int print_matrices;
    enum StencilKernelToUse kernel_to_use;

    int scanf_result = 0;
    // Read dimensions and parameters
    scanf_result = fscanf(file_ptr, "%d %d %d", &width, &height, &depth);
    scanf_result = fscanf(file_ptr, "%d", &print_matrices);
    scanf_result = fscanf(file_ptr, "%d", (int*)&kernel_to_use);

    if (kernel_to_use >= kNumFilters)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    float *N, *P, *P_expected;
    float c[NUM_STENCIL_POINTS];

    // allocate memory for matrices
    N = (float*)malloc(depth * height * width * sizeof(float));
    P = (float*)malloc(depth * height * width * sizeof(float));
    P_expected = (float*)malloc(depth * height * width * sizeof(float));

    // Read N.
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                scanf_result =
                    fscanf(file_ptr, "%f", &N[linearized3DIndex(z, y, x, width, height)]);
            }
        }
    }
    if (print_matrices != 0)
    {
        printf("Matrix N:\n");
        printMatrix3D(N, width, height, depth);
    }

    // Read c.
    for (int x = 0; x < NUM_STENCIL_POINTS; ++x)
    {
        scanf_result = fscanf(file_ptr, "%f", &c[x]);
    }
    if (print_matrices != 0)
    {
        printf("c:\n");
        printMatrix2D(c, NUM_STENCIL_POINTS, 1);
    }

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }
    // Optionally read P_expected
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                scanf_result =
                    fscanf(file_ptr, "%f", &P_expected[linearized3DIndex(z, y, x, width, height)]);
                if (scanf_result == EOF)
                    break;
            }
            if (scanf_result == EOF)
                break;
        }
        if (scanf_result == EOF)
            break;
    }

    fclose(file_ptr);

    printf("Computing Stencil:\n");
    int iters = 1000;

    // compute the stencil.
    float time_to_compute_stencil =
        stencilDriver(N, P, c, width, height, depth, kernel_to_use, iters);

    printf("Took %.1f msec for %d iterations.\n", time_to_compute_stencil, iters);
    if (print_matrices != 0)
    {
        printf("Result:\n");
        printMatrix3D(P, width, height, depth);
    }

    // P_expected was not read. Fall back to naive approach for GT.
    if (scanf_result == EOF)
    {
        stencilDriver(N, P_expected, c, width, height, depth, kBasic, 1);
    }

    struct matrixComparisonResult matrix_comparison_result =
        matricesAreEqual3D(P, P_expected, width, height, depth);
    if (!matrix_comparison_result.success)
    {
        printf("\nMatrices do not match!\n");
        printf("First mismatch occurs at index: %d\n",
               matrix_comparison_result.index_of_first_mismatch);
        if (print_matrices)
        {
            printf("Actual:\n");
            printMatrix3D(P, width, height, depth);

            printf("\nExpected:\n");
            printMatrix3D(P_expected, width, height, depth);
        }
    }
    else
    {
        printf("\nResult is equal to expected!\n");
    }
}