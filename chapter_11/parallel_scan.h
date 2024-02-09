#ifndef CHAPTER_11_PARALLEL_SCAN_H
#define CHAPTER_11_PARALLEL_SCAN_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/types.h"

__device__ int block_counter = 0;

/// @brief Compute the binary associative operation on two arguments.
/// @param lhs left hand side of the operation
/// @param rhs right hand side of the operation
/// @return result of applying the operation
__device__ __host__ ParallelScanDataType ParallelScanOperation(const ParallelScanDataType lhs,
                                                               const ParallelScanDataType rhs);

/// @brief Return the identity element for the operation in ParallelScanOperation().
/// @return an identity element
__device__ __host__ ParallelScanDataType ParallelScanIdentity();

/// @brief Compute inclusive scan using the Kogge-Stone algorithm.
/// @param data data on which to compute inclusive scan
/// @param[out] result result of inclusive scan
/// @param length length of data
__global__ void KoggeStoneInclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                          unsigned int length);

/// @brief Compute exclusive scan using the Kogge-Stone algorithm.
/// @param data data on which to compute exclusive scan
/// @param[out] result result of exclusive scan
/// @param length length of data
__global__ void KoggeStoneExclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                          unsigned int length);

/// @brief Compute inclusive scan using the Kogge-Stone algorithm with a modification to use
/// double-buffering.
/// @param data data on which to compute inclusive scan
/// @param[out] result result of inclusive scan
/// @param length length of data
__global__ void KoggeStoneDoubleBufferingInclusiveKernel(ParallelScanDataType* data,
                                                         ParallelScanDataType* result,
                                                         unsigned int length);

/// @brief Compute exclusive scan using the Kogge-Stone algorithm with a modification to use
/// double-buffering.
/// @param data data on which to compute exclusive scan
/// @param[out] result result of exclusive scan
/// @param length length of data
__global__ void KoggeStoneDoubleBufferingExclusiveKernel(ParallelScanDataType* data,
                                                         ParallelScanDataType* result,
                                                         unsigned int length);

/// @brief Compute inclusive scan using the Brent-Kung algorithm.
/// @param data data on which to compute inclusive scan
/// @param[out] result result of inclusive scan
/// @param length length of data
__global__ void BrentKungInclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                         unsigned int length);

/// @brief Compute exclusive scan using the Brent-Kung algorithm.
/// @param data data on which to compute exclusive scan
/// @param[out] result result of exclusive scan
/// @param length length of data
__global__ void BrentKungExclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                         unsigned int length);

/// @brief Compute inclusive scan using a thread coarsening algorithm.
/// @param data data on which to compute inclusive scan
/// @param[out] result result of inclusive scan
/// @param length length of data
__global__ void ThreadCoarseningInclusiveKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length);

/// @brief Compute exclusive scan using a thread coarsening algorithm.
/// @param data data on which to compute exclusive scan
/// @param[out] result result of exclusive scan
/// @param length length of data
__global__ void ThreadCoarseningExclusiveKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length);

/// @brief Compute inclusive scan using a thread coarsening algorithm.
/// @param data data on which to compute inclusive scan
/// @param[out] result result of inclusive scan on a thread block's subarray
/// @param[out] end_vals the final value in a thread block's subarray
/// @param length length of data
__global__ void ThreadCoarseningSegmentedScanKernelPhase1(ParallelScanDataType* data,
                                                          ParallelScanDataType* result,
                                                          ParallelScanDataType* end_vals,
                                                          unsigned int length);

/// @brief Compute inclusive scan using a thread coarsening algorithm.
/// @param[out] data data holding partial inclusive scan and to hold the output
/// @param[out] end_vals_scanned result of inclusive scan on the final values in each thread block's
/// subarray
/// @param length length of data
__global__ void ThreadCoarseningSegmentedScanKernelPhase3(ParallelScanDataType* data,
                                                          ParallelScanDataType* end_vals_scanned,
                                                          unsigned int length);

/// @brief Compute inclusive scan using a streaming kernel.
/// @param[out] data data to compute scan on
/// @param[out] result result of inclusive scan
/// @param length length of data
__global__ void StreamingKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                unsigned int length);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_11_PARALLEL_SCAN_H