#ifndef STENCIL_DRIVER_H
#define STENCIL_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stencil3d.h"
#include "types/constants.h"

/// @brief Prepare and run a 3D convolution of N of size depth x height x width with kernel F of
/// radius filter_radius on a GPU.
/// @param N_h Matrix to perform convolution on (stored on CPU host)
/// @param[out] P_h Result matrix (stored on CPU host)
/// @param width width of input and output matrices
/// @param height height of input and output matrices
/// @param depth depth of input and output matrices
/// @param kernel_to_use kernel to use for computing the convolution
/// @param iters number of iterations to run for timing. default: 1
/// @return Time (in msec) taken to process the kernel (excludes memory management)
float stencilDriver(float* N_h, float* P_h, float* c_h, int width, int height, int depth,
                    enum StencilKernelToUse kernel_to_use, int iters);

#ifdef __cplusplus
}
#endif

#endif  // STENCIL_DRIVER_H