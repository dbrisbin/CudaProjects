#include "matrixUtils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "types/constants.h"

struct matrixComparisonResult matricesAreEqual2D(const float* A, const float* B, const int width,
                                                 const int height)
{
    struct matrixComparisonResult result;
    result.index_of_first_mismatch = 0;
    result.success = true;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int linear_idx = linearized2DIndex(y, x, width);
            if (fabs(A[linear_idx] - B[linear_idx]) > EPS_FOR_MATRIX_ELEMENT_EQUALITY)
            {
                result.success = false;
                result.index_of_first_mismatch = linear_idx;
                return result;
            }
        }
    }

    return result;
}

void printMatrix2D(const float* mat, const int width, const int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            printf("%.0f ", mat[linearized2DIndex(y, x, width)]);
        }
        printf("\n");
    }
}

struct matrixComparisonResult matricesAreEqual3D(const float* A, const float* B, const int width,
                                                 const int height, const int depth)
{
    struct matrixComparisonResult result;
    result.index_of_first_mismatch = 0;
    result.success = true;

    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int linear_idx = linearized3DIndex(z, y, x, width, height);
                if (fabs(A[linear_idx] - B[linear_idx]) > EPS_FOR_MATRIX_ELEMENT_EQUALITY)
                {
                    result.success = false;
                    result.index_of_first_mismatch = linear_idx;
                    return result;
                }
            }
        }
    }

    return result;
}

void printMatrix3D(const float* mat, const int width, const int height, const int depth)
{
    for (int z = 0; z < depth; ++z)
    {
        printf("Layer %d:\n", z);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                printf("%.0f ", mat[linearized3DIndex(z, y, x, width, height)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
