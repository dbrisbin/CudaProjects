#ifndef CHAPTER_9_PARALLEL_HISTOGRAM_H
#define CHAPTER_9_PARALLEL_HISTOGRAM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>

__global__ void basicParallelHistogram(int* data, int length, int* hist);

__global__ void privatizedParallelHistogram(int* data, int length, int* hist);

__global__ void privatizedWithSharedMemoryParallelHistogram(int* data, int length, int* hist);

__global__ void coarseningParallelHistogram(int* data, int length, int* hist);

__global__ void coarseningWithCoalescedAccessParallelHistogram(int* data, int length, int* hist);

__global__ void aggregatedParallelHistogram(int* data, int length, int* hist);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_9_PARALLEL_HISTOGRAM_H