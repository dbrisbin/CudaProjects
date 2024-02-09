#include <stdio.h>
#include "parallel_scan.h"
#include "parallel_scan_driver.h"
#include "types/constants.h"

extern "C" float ParallelScanDriver(const ParallelScanDataType* data_h,
                                    ParallelScanDataType* result_h, const unsigned int length,
                                    const enum parallelScanKernelToUse kernel_to_use,
                                    const int iters, bool inclusive_scan)
{
    ParallelScanDataType* data_d;
    ParallelScanDataType* result_d;
    ParallelScanDataType* end_vals_d;
    dim3 dim_block, dim_grid;

    cudaMalloc((void**)&data_d, length * sizeof(ParallelScanDataType));
    cudaMalloc((void**)&result_d, length * sizeof(ParallelScanDataType));
    cudaMalloc((void**)&end_vals_d, length / SECTION_SIZE / CFACTOR * sizeof(ParallelScanDataType));
    cudaMemcpy(data_d, data_h, length * sizeof(ParallelScanDataType), cudaMemcpyHostToDevice);

    float time;
    float total_time = 0.0f;
    cudaEvent_t start, stop;

    for (int iter = 0; iter < iters; ++iter)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        if (inclusive_scan)
        {
            switch (kernel_to_use)
            {
                case kKoggeStoneInclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    KoggeStoneInclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d, length);
                    break;
                case kKoggeStoneDoubleBufferingInclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    KoggeStoneDoubleBufferingInclusiveKernel<<<dim_grid, dim_block>>>(
                        data_d, result_d, length);
                    break;
                case kBrentKungInclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    BrentKungInclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d, length);
                    break;
                case kCoarseningInclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    ThreadCoarseningInclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d,
                                                                             length);
                    break;
                case kCoarseningSegmented:
                    // Phase 1:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(length / SECTION_SIZE / CFACTOR, 1, 1);
                    ThreadCoarseningSegmentedScanKernelPhase1<<<dim_grid, dim_block>>>(
                        data_d, result_d, end_vals_d, length);

                    // Phase 2:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    ThreadCoarseningInclusiveKernel<<<dim_grid, dim_block>>>(
                        end_vals_d, end_vals_d, length / SECTION_SIZE / CFACTOR);

                    // Phase 3:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(length / SECTION_SIZE / CFACTOR, 1, 1);
                    ThreadCoarseningSegmentedScanKernelPhase3<<<dim_grid, dim_block>>>(
                        result_d, end_vals_d, length);
                    break;
                case kKoggeStoneExclusive:
                case kKoggeStoneDoubleBufferingExclusive:
                case kBrentKungExclusive:
                case kCoarseningExclusive:
                    printf(
                        "Exclusive scan kernel selected to compute inclusive scan! Try again!\n");
                    return -1.0;
                case kNumKernels:
                default:
                    printf("Invalid kernel selected! Try again!\n");
                    return -1.0;
            }
        }
        else
        {
            switch (kernel_to_use)
            {
                case kKoggeStoneExclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    KoggeStoneExclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d, length);
                    break;
                case kKoggeStoneDoubleBufferingExclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    KoggeStoneDoubleBufferingExclusiveKernel<<<dim_grid, dim_block>>>(
                        data_d, result_d, length);
                    break;
                case kBrentKungExclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    BrentKungExclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d, length);
                    break;
                case kCoarseningExclusive:
                    dim_block = dim3(SECTION_SIZE, 1, 1);
                    dim_grid = dim3(1, 1, 1);
                    ThreadCoarseningExclusiveKernel<<<dim_grid, dim_block>>>(data_d, result_d,
                                                                             length);
                    break;
                case kCoarseningSegmented:
                case kCoarseningInclusive:
                case kKoggeStoneInclusive:
                case kKoggeStoneDoubleBufferingInclusive:
                case kBrentKungInclusive:
                    printf(
                        "Inclusive scan kernel selected to compute exclusive scan! Try again!\n");
                    return -1.0;
                case kNumKernels:
                default:
                    printf("Invalid kernel selected! Try again!\n");
                    return -1.0;
            }
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
    cudaFree(end_vals_d);

    return total_time;
}
