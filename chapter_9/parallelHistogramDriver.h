#ifndef STENCIL_DRIVER_H
#define STENCIL_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "types/constants.h"

/// @brief Compute the frequency histogram for the input phrase.
/// @param phrase_h the phrase to generate histogram for
/// @param length length of the phrase
/// @param hist_h histogram to store result in
/// @param kernel_to_use kernel to use for computing the histogram
/// @param iters number of iterations to run for timing.
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float parallelHistogramDriver(int* data_h, int length, int* hist_h,
                              enum parallelHistogramKernelToUse kernel_to_use, int iters);

#ifdef __cplusplus
}
#endif

#endif  // STENCIL_DRIVER_H