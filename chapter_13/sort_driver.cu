/// @file sort_driver.cu
/// @brief Definition of driver function declared in sort_driver.h.

#include <stdio.h>
#include "sort.h"
#include "sort_driver.h"
#include "sort_utils.h"
#include "types/constants.h"

template <typename Cont>
void PrintArr(const Cont& arr, const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

float SortDriver(const unsigned int* data_h, unsigned int* result_h, const int length,
                 const SortKernel kernel_to_use, const int iters)
{
    unsigned int* result_d{};
    unsigned int* data_d{};
    unsigned int* bits_in_d{};
    unsigned int* bits_out_d{};
    unsigned int* end_vals_d;
    unsigned int* flags_d;

    dim3 dim_block{}, dim_grid{};

    cudaMalloc((void**)&result_d, length * sizeof(unsigned int));
    cudaMalloc((void**)&data_d, length * sizeof(unsigned int));

    cudaMalloc((void**)&bits_in_d, length * sizeof(unsigned int));
    cudaMalloc((void**)&bits_out_d, length * sizeof(unsigned int));
    cudaMalloc((void**)&end_vals_d,
               ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR) * sizeof(unsigned int));
    cudaMalloc((void**)&flags_d,
               ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR) * sizeof(unsigned int));

    // cudaMemcpy(result_d, data_h, length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data_h, length * sizeof(unsigned int), cudaMemcpyHostToDevice);

    float time{};
    float total_time{};
    cudaEvent_t start, stop;

    for (int iter = 0; iter < iters; ++iter)
    {

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        switch (kernel_to_use)
        {
            case SortKernel::kRadix:
                printf("Kernel is broken! Results may be incorrect!\n");
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(length) / SECTION_SIZE), 1, 1);

                for (unsigned int iter{0}; iter < 32U; ++iter)
                {
                    ResetArray<<<dim_grid, dim_block>>>(
                        flags_d, ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR), 0);
                    ResetArray<<<dim_grid, dim_block>>>(bits_in_d, length, 0);
                    ResetArray<<<dim_grid, dim_block>>>(bits_out_d, length, 0);
                    ResetArray<<<dim_grid, dim_block>>>(
                        end_vals_d, ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR), 0);

                    RadixSortIter<<<dim_grid, dim_block>>>(data_d, result_d, bits_in_d, bits_out_d,
                                                           length, iter, flags_d, end_vals_d);

                    cudaMemcpy(data_d, result_d, length * sizeof(unsigned int),
                               cudaMemcpyDeviceToDevice);
                }
                break;
            case SortKernel::kRadixSplit:
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(length) / SECTION_SIZE), 1, 1);

                for (unsigned int iter{0}; iter < 32U; ++iter)
                {
                    ResetArray<<<dim_grid, dim_block>>>(
                        flags_d, ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR), 0);
                    ResetArray<<<dim_grid, dim_block>>>(bits_in_d, length, 0);
                    ResetArray<<<dim_grid, dim_block>>>(bits_out_d, length, 0);
                    ResetArray<<<dim_grid, dim_block>>>(
                        end_vals_d, ceil(static_cast<double>(length) / SECTION_SIZE / CFACTOR), 0);

                    RadixSortIterPhase1<<<dim_grid, dim_block>>>(data_d, bits_in_d, length, iter);
                    InclusiveScan<<<dim_grid, dim_block>>>(bits_in_d, bits_out_d, length, flags_d,
                                                           end_vals_d, iter);
                    RadixSortIterPhase2<<<dim_grid, dim_block>>>(data_d, result_d, bits_out_d,
                                                                 length, iter);
                    cudaMemcpy(data_d, result_d, length * sizeof(unsigned int),
                               cudaMemcpyDeviceToDevice);
                }
                break;
            case SortKernel::kNumKernels:
            default:
                printf("Invalid kernel selected! Try again!\n");
                return -1.0;
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    cudaError_t err = cudaMemcpy(result_h, result_d, length * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaFree(flags_d);
    cudaFree(end_vals_d);
    cudaFree(bits_out_d);
    cudaFree(bits_in_d);
    cudaFree(data_d);
    cudaFree(result_d);

    return total_time;
}
