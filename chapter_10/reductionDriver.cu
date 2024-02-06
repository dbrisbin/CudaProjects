#include <stdio.h>
#include "reduction.h"
#include "reductionDriver.h"
#include "types/constants.h"

extern "C" float reductionDriver(const ReductionDataType* data_h, const int length,
                                 ReductionDataType* result_h,
                                 const enum reductionKernelToUse kernel_to_use, const int iters)
{
    ReductionDataType* data_d;
    ReductionDataType* result_d;
    dim3 dimBlock, dimGrid;

    cudaMalloc((void**)&data_d, length * sizeof(ReductionDataType));
    cudaMalloc((void**)&result_d, sizeof(ReductionDataType));

    float time;
    float total_time = 0.0f;
    cudaEvent_t start, stop;

    for (int iter = 0; iter < iters; ++iter)
    {
        cudaMemcpy(data_d, data_h, length * sizeof(ReductionDataType), cudaMemcpyHostToDevice);

        switch (kernel_to_use)
        {
            case kBasic:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(1, 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                basicReduction<<<dimGrid, dimBlock>>>(data_d, length, result_d);
                break;
            case kCoalescing:
            case kSharedMemory:
            case kSegmented:
            case kCoarsening:
            case kNumKernels:
            default:
                break;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    cudaError_t err =
        cudaMemcpy(result_h, result_d, sizeof(ReductionDataType), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaFree(result_d);
    cudaFree(data_d);

    return total_time;
}
