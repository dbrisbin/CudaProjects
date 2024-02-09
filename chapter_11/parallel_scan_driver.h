#ifndef CHAPTER_11_PARALLEL_SCAN_DRIVER_H
#define CHAPTER_11_PARALLEL_SCAN_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include "types/constants.h"
#include "types/types.h"

/// @brief Call the correct kernel to compute the parallel scan of data.
/// @param data_h data stored on host to compute parallel scan on
/// @param[out] result_h result stored on host to store result of parallel scan
/// @param length length of data and result
/// @param kernel_to_use kernel to use to compute parallel scan
/// @param iters number of iterations to run parallel scan to compute runtime
/// @param inclusive_scan boolean indicating whether to run inclusive scan or exclusive scan
/// @return total runtime to compute parallel scan iters times. -1.0 if invalid parameters are
/// provided.
float ParallelScanDriver(const ParallelScanDataType* data_h, ParallelScanDataType* result_h,
                         const unsigned int length,
                         const enum parallelScanKernelToUse kernel_to_use, const int iters,
                         const bool inclusive_scan);

#ifdef __cplusplus
}
#endif

#endif  // CHAPTER_11_PARALLEL_SCAN_DRIVER_H