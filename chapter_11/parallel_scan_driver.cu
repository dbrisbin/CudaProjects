#include <stdio.h>
#include "parallel_scan.h"
#include "parallel_scan_driver.h"
#include "types/constants.h"

extern "C" float ParallelScanDriver(const ParallelScanDataType* data_h,
                                    ParallelScanDataType* result_h, const unsigned int length,
                                    const enum parallelScanKernelToUse kernel_to_use,
                                    const int iters)
{
    ParallelScanDataType* data_d;
    ParallelScanDataType* result_d;
    dim3 dimBlock, dimGrid;

    cudaMalloc((void**)&data_d, length * sizeof(ParallelScanDataType));
    cudaMalloc((void**)&result_d, length * sizeof(ParallelScanDataType));

    cudaMemcpy(data_d, data_h, length * sizeof(ParallelScanDataType), cudaMemcpyHostToDevice);

    float time;
    float total_time = 0.0f;
    cudaEvent_t start, stop;

    for (int iter = 0; iter < iters; ++iter)
    {

        switch (kernel_to_use)
        {
            case kKoggeStone:
                dimBlock = dim3(SECTION_SIZE, 1, 1);
                dimGrid = dim3(1, 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                KoggeStoneKernel<<<dimGrid, dimBlock>>>(data_d, result_d, length);
                break;
            case kKoggeStoneDoubleBuffering:
                dimBlock = dim3(SECTION_SIZE, 1, 1);
                dimGrid = dim3(1, 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                KoggeStoneDoubleBufferingKernel<<<dimGrid, dimBlock>>>(data_d, result_d, length);
                break;
            case kBrentKung:
                dimBlock = dim3(SECTION_SIZE, 1, 1);
                dimGrid = dim3(1, 1, 1);
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                BrentKungKernel<<<dimGrid, dimBlock>>>(data_d, result_d, length);
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

    cudaError_t err = cudaMemcpy(result_h, result_d, length * sizeof(ParallelScanDataType),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaFree(result_d);
    cudaFree(data_d);

    return total_time;
}
