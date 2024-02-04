#include <cuda_runtime.h>
#include <stdio.h>
#include "stencil3d.h"
#include "stencilDriver.h"
#include "types/constants.h"

extern "C" float stencilDriver(float* N_h, float* P_h, float* c_h, int width, int height, int depth,
                               enum StencilKernelToUse kernel_to_use, int iters)
{
    float *N_d, *P_d;
    dim3 dimBlock;
    dim3 dimGrid;
    int matrix_memory_req = width * height * depth * sizeof(float);
    cudaMalloc((void**)&N_d, matrix_memory_req);
    cudaMalloc((void**)&P_d, matrix_memory_req);

    cudaMemcpy(N_d, N_h, matrix_memory_req, cudaMemcpyHostToDevice);
    copyArrayToConstantMemory(c_h, NUM_STENCIL_POINTS * sizeof(float));

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int iter = 0; iter < iters; ++iter)
    {
        switch (kernel_to_use)
        {
            case kBasic:
                dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
                dimGrid = dim3(ceil((float)width / dimBlock.x), ceil((float)height / dimBlock.y),
                               ceil((float)depth / dimBlock.z));
                basicStencil<<<dimGrid, dimBlock>>>(N_d, P_d, width, height, depth);
                break;
            case kTiling:
                dimBlock = dim3(IN_TILE_DIM_3D, IN_TILE_DIM_3D, IN_TILE_DIM_3D);
                dimGrid = dim3(ceil((float)width / OUT_TILE_DIM_3D),
                               ceil((float)height / OUT_TILE_DIM_3D),
                               ceil((float)depth / OUT_TILE_DIM_3D));
                tilingStencil<<<dimGrid, dimBlock>>>(N_d, P_d, width, height, depth);
                break;
            case kThreadCoarsening:
                dimBlock = dim3(IN_TILE_DIM_2D, IN_TILE_DIM_2D, 1);
                dimGrid = dim3(ceil((float)width / OUT_TILE_DIM_2D),
                               ceil((float)height / OUT_TILE_DIM_2D),
                               ceil((float)depth / OUT_TILE_DIM_2D));
                threadCoarseningStencil<<<dimGrid, dimBlock,
                                          IN_TILE_DIM_2D * IN_TILE_DIM_2D * 3 * sizeof(float)>>>(
                    N_d, P_d, width, height, depth);
                break;
            case kRegisterTiling:
                dimBlock = dim3(IN_TILE_DIM_2D, IN_TILE_DIM_2D, 1);
                dimGrid = dim3(ceil((float)width / OUT_TILE_DIM_2D),
                               ceil((float)height / OUT_TILE_DIM_2D),
                               ceil((float)depth / OUT_TILE_DIM_2D));
                registerTilingStencil<<<dimGrid, dimBlock,
                                        IN_TILE_DIM_2D * IN_TILE_DIM_2D * 3 * sizeof(float)>>>(
                    N_d, P_d, width, height, depth);
                break;
            case kNumFilters:
            default:
                break;
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaError_t err = cudaMemcpy(P_h, P_d, matrix_memory_req, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(N_d);
    cudaFree(P_d);

    return time;
}
