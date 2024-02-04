#ifndef STENCIL_3D_H
#define STENCIL_3D_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>

void copyArrayToConstantMemory(float* Arr, int size);

__global__ void basicStencil(float* N, float* P, const int width, const int height,
                             const int depth);

__global__ void tilingStencil(float* N, float* P, const int width, const int height,
                              const int depth);

__global__ void threadCoarseningStencil(float* N, float* P, const int width, const int height,
                                        const int depth);

__global__ void registerTilingStencil(float* N, float* P, const int width, const int height,
                                      const int depth);

#ifdef __cplusplus
}
#endif

#endif  // STENCIL_3D_H