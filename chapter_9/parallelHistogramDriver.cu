#include <cuda_runtime.h>
#include <stdio.h>
#include "parallelHistogram.h"
#include "parallelHistogramDriver.h"
#include "types/constants.h"

extern "C" float parallelHistogramDriver(int* data_h, int length, int* hist_h,
                                         enum parallelHistogramKernelToUse kernel_to_use, int iters)
{
    int *data_d, *hist_d;
    dim3 dimBlock, dimGrid;

    int hist_d_size = NUM_BINS * sizeof(int);
    if (kernel_to_use == kPrivatized)
    {
        hist_d_size *= ceil((float)length / NUM_BINS);
    }
    cudaMalloc((void**)&data_d, length * sizeof(int));
    cudaMalloc((void**)&hist_d, hist_d_size);

    cudaMemcpy(data_d, data_h, length * sizeof(int), cudaMemcpyHostToDevice);

    float time;
    float total_time = 0.0f;
    cudaEvent_t start, stop;

    for (int iter = 0; iter < iters; ++iter)
    {
        cudaMemset(hist_d, 0, hist_d_size);
        switch (kernel_to_use)
        {
            case kBasic:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil((float)length / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                basicParallelHistogram<<<dimGrid, dimBlock>>>(data_d, length, hist_d);
                break;
            case kPrivatized:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil((float)length / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                privatizedParallelHistogram<<<dimGrid, dimBlock>>>(data_d, length, hist_d);
                break;
            case kPrivatizedWithSharedMemory:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil((float)length / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                privatizedWithSharedMemoryParallelHistogram<<<dimGrid, dimBlock>>>(data_d, length,
                                                                                   hist_d);
                break;
            case kCoarsening:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil(((float)length / CFACTOR) / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                coarseningParallelHistogram<<<dimGrid, dimBlock>>>(data_d, length, hist_d);
                break;
            case kCoarseningWithCoalescedAccess:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil(((float)length / CFACTOR) / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                coarseningWithCoalescedAccessParallelHistogram<<<dimGrid, dimBlock>>>(
                    data_d, length, hist_d);
                break;
            case kAggregated:
                dimBlock = dim3(TILE_WIDTH, 1, 1);
                dimGrid = dim3(ceil(((float)length / CFACTOR) / dimBlock.x), 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                aggregatedParallelHistogram<<<dimGrid, dimBlock>>>(data_d, length, hist_d);
                break;
            case kNumKernels:
            default:
                break;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    cudaError_t err = cudaMemcpy(hist_h, hist_d, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(hist_d);
    cudaFree(data_d);

    return total_time;
}
