#ifndef CHAPTER_10_REDUCTION_DRIVER_H
#define CHAPTER_10_REDUCTION_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/constants.h"
#include "types/types.h"

/// @brief Call the correct kernel to compute the reduction of data.
/// @param data_h data stored on host to compute reduction on
/// @param length length of data
/// @param[out] result_h result stored on host to store result of reduction
/// @param kernel_to_use kernel to use to compute reduction
/// @param iters number of iterations to run reduction to compute runtime
/// @return total runtime to compute reduction iters times
float reductionDriver(const ReductionDataType* data_h, const int length,
                      ReductionDataType* result_h, const enum reductionKernelToUse kernel_to_use,
                      const int iters);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_10_REDUCTION_DRIVER_H