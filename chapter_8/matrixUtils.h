#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <cuda_runtime.h>
#include "types/types.h"

/// @brief Compute linearized index for 2 dimensional array.
/// @param y z index in array
/// @param x z index in array
/// @param width width of the array
/// @return linearized index
__device__ __host__ static inline int linearized2DIndex(const int y, const int x, const int width)
{
    return y * width + x;
}

// @brief Compute linearized index for 3 dimensional array.
/// @param z z index in array
/// @param y z index in array
/// @param x z index in array
/// @param width width of the array
/// @param height height of the array
/// @return linearized index
__device__ __host__ static inline int linearized3DIndex(const int z, const int y, const int x,
                                                        const int width, const int height)
{
    return (z * height + y) * width + x;
}

/// @brief Compare two matrices for equality within EPS.
/// @param A first matrix to compare
/// @param B second matric to compare
/// @param width width of matrices
/// @param height height of matrices
/// @pre matrices must be same dimension. If not, result is undefined.
/// @return True if the matrices are element-wise equal within eps, false otherwise.
struct matrixComparisonResult matricesAreEqual2D(const float* A, const float* B, const int width,
                                                 const int height);

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix to print
/// @param height height of matrix to print
void printMatrix2D(const float* mat, const int width, const int height);

/// @brief Compare two matrices for equality within EPS.
/// @param A first matrix to compare
/// @param B second matric to compare
/// @param width width of matrices
/// @param height height of matrices
/// @param depth depth of matrices
/// @pre matrices must be same dimension. If not, result is undefined.
/// @return True if the matrices are element-wise equal within eps, false otherwise.
struct matrixComparisonResult matricesAreEqual3D(const float* A, const float* B, const int width,
                                                 const int height, const int depth);

/// @brief Prints a matrix to standard output.
/// @param mat matrix to print
/// @param width width of matrix to print
/// @param height height of matrix to print
/// @param depth depth of matrix to print
void printMatrix3D(const float* mat, const int width, const int height, const int depth);

#endif  // MATRIX_UTILS_H