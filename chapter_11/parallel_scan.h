#ifndef CHAPTER_11_PARALLEL_SCAN_H
#define CHAPTER_11_PARALLEL_SCAN_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/types.h"

__device__ __host__ ParallelScanDataType ParallelScanOperation(const ParallelScanDataType lhs,
                                                               const ParallelScanDataType rhs);

/// @brief Compute parallel scan using the Kogge-Stone algorithm.
/// @param data data on which to compute parallel scan
/// @param[out] result result of parallel scan
/// @param length length of data
__global__ void KoggeStoneKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                 unsigned int length);

/// @brief Compute parallel scan using the Kogge-Stone algorithm with a modification to use
/// double-buffering.
/// @param data data on which to compute parallel scan
/// @param[out] result result of parallel scan
/// @param length length of data
__global__ void KoggeStoneDoubleBufferingKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length);

/// @brief Compute parallel scan using the Brent-Kung algorithm.
/// @param data data on which to compute parallel scan
/// @param[out] result result of parallel scan
/// @param length length of data
__global__ void BrentKungKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                unsigned int length);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_11_PARALLEL_SCAN_H