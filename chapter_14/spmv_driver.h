/// @file spmv_driver.h
/// @brief Declaration of driver function to setup and call the appropriate kernel to compute the
/// sparse matrix vector product on GPU.

#ifndef CHAPTER_14_SPMV_DRIVER_H
#define CHAPTER_14_SPMV_DRIVER_H

#include <utility>
#include "types/constants.h"

/// @brief Driver function to call the appropriate kernel to perform SpMV on GPU.
/// @param mat_h matrix in dense format stored in host memory for SpMV
/// @param vec_h vector stored in host memory for SpMV
/// @param result location to store the result of SpMV
/// @param m number of rows in the matrix
/// @param n number of columns in the matrix
/// @param iters number of iterations to run the SpMV kernel
/// @param kernel_to_use kernel to use for SpMV
/// @return time taken (in ms) to run the SpMV kernel iters times
float SpMVDriver(const float* mat_h, const float* vec_h, float* result, const int m, const int n,
                 const int iters, const SpmvKernel kernel_to_use);

#endif  // CHAPTER_14_SPMV_DRIVER_H