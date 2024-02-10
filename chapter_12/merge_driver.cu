/// @file merge_driver.cu
/// @brief Definition of driver function declared in merge_driver.h.

#include <stdio.h>
#include "merge.h"
#include "merge_driver.h"
#include "types/constants.h"

float MergeDriver(const std::pair<int, int>* A_h, const int m, const std::pair<int, int>* B_h,
                  const int n, std::pair<int, int>* C_h, const MergeKernel kernel_to_use,
                  const int iters)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared mem per block: %luB", prop.sharedMemPerBlock);

    std::pair<int, int>* A_d;
    std::pair<int, int>* B_d;
    std::pair<int, int>* C_d;

    dim3 dim_block, dim_grid;
    int shared_mem_size{};

    auto length = m + n;

    cudaMalloc((void**)&A_d, m * sizeof(std::pair<int, int>));
    cudaMalloc((void**)&B_d, n * sizeof(std::pair<int, int>));
    cudaMalloc((void**)&C_d, length * sizeof(std::pair<int, int>));

    cudaMemcpy(A_d, A_h, m * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);

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
            case MergeKernel::kBasic:
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(
                    ceil(static_cast<double>(length) / NUM_ELTS_PER_THREAD / SECTION_SIZE), 1, 1);
                BasicKernel<<<dim_grid, dim_block>>>(A_d, m, B_d, n, C_d);
                break;
            case MergeKernel::kTiled:
                dim_block = dim3(BLOCKSIZE_FOR_TILED, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(length) / OUTPUT_ELTS_PER_BLOCK), 1, 1);
                shared_mem_size =
                    2 * NUM_ELTS_PER_TILE * sizeof(std::pair<int, int>) + 4 * sizeof(int);
                printf("\nUsed: %dB\n", shared_mem_size);
                TiledKernel<<<dim_grid, dim_block, shared_mem_size>>>(A_d, m, B_d, n, C_d,
                                                                      NUM_ELTS_PER_TILE);
                break;
            case MergeKernel::kModifiedTiled:
                dim_block = dim3(BLOCKSIZE_FOR_TILED, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(length) / OUTPUT_ELTS_PER_BLOCK), 1, 1);
                shared_mem_size =
                    2 * NUM_ELTS_PER_TILE * sizeof(std::pair<int, int>) + 4 * sizeof(int);
                printf("\nUsed: %dB\n", shared_mem_size);
                TiledKernel<<<dim_grid, dim_block, shared_mem_size>>>(A_d, m, B_d, n, C_d,
                                                                      NUM_ELTS_PER_TILE);
                break;
            case MergeKernel::kCircularBuffer:
                dim_block = dim3(BLOCKSIZE_FOR_TILED, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(length) / OUTPUT_ELTS_PER_BLOCK), 1, 1);
                shared_mem_size =
                    2 * NUM_ELTS_PER_TILE * sizeof(std::pair<int, int>) + 4 * sizeof(int);
                printf("\nUsed: %dB\n", shared_mem_size);
                CircularBufferKernel<<<dim_grid, dim_block, shared_mem_size>>>(A_d, m, B_d, n, C_d,
                                                                               NUM_ELTS_PER_TILE);
                break;
            case MergeKernel::kNumKernels:
            default:
                printf("Invalid kernel selected! Try again!\n");
                return -1.0;
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    cudaError_t err =
        cudaMemcpy(C_h, C_d, length * sizeof(std::pair<int, int>), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return total_time;
}
