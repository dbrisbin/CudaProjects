/// @file sort_utils.h
/// @brief Declaration of utility functions used by kernels and driver code.

#ifndef CHAPTER_13_SORT_UTILS_H
#define CHAPTER_13_SORT_UTILS_H

#include <cuda_runtime.h>

extern __device__ int block_counter;

/// @brief Compute inclusive scan using a streaming kernel.
/// @param[out] data data to compute scan on
/// @param[out] result result of inclusive scan
/// @param length length of data
__device__ void ExclusiveScan(unsigned int* data, unsigned int* output, const unsigned int length,
                              unsigned int* flags, unsigned int* scan_value,
                              const unsigned int iter);

__global__ void InclusiveScan(unsigned int* data, unsigned int* output, const unsigned int length,
                              unsigned int* flags, unsigned int* scan_value,
                              const unsigned int iter);

__global__ void ResetArray(unsigned int* data, unsigned int length, unsigned int val);

#endif  // CHAPTER_13_SORT_UTILS_H