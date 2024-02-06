#ifndef CHAPTER_10_REDUCTION_H
#define CHAPTER_10_REDUCTION_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/types.h"

__global__ void basicReduction(ReductionDataType* data, int length, ReductionDataType* result);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_10_REDUCTION_H