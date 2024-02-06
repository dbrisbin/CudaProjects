#ifndef CHAPTER_10_REDUCTION_H
#define CHAPTER_10_REDUCTION_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/types.h"

__device__ __host__ ReductionDataType reductionOperation(const ReductionDataType a,
                                                         const ReductionDataType b);

__device__ __host__ ReductionDataType reductionIdentity();

/// @brief Compute reduction using a basic kernel.
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void basicReduction(ReductionDataType* data, int length, ReductionDataType* result);

/// @brief Compute reduction using an arrangement which encourages coalesced data accesses.
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void coalescingReduction(ReductionDataType* data, int length, ReductionDataType* result);

/// @brief Compute reduction using an arrangement which encourages coalesced data accesses. Modified
/// for problem 10.4
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void coalescingModifiedReduction(ReductionDataType* data, int length,
                                            ReductionDataType* result);

/// @brief Compute reduction using shared memory and coalesced data accesses.
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void sharedMemoryReduction(ReductionDataType* data, int length,
                                      ReductionDataType* result);

/// @brief Compute reduction on greater than 2048 elements.
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void segmentedReduction(ReductionDataType* data, int length, ReductionDataType* result);

/// @brief Compute reduction using thread coarsening.
/// @param data data on which to compute reduction
/// @param length length of data
/// @param[out] result result of reduction
__global__ void coarseningReduction(ReductionDataType* data, int length, ReductionDataType* result);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_10_REDUCTION_H