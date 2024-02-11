/// @file sort_utils.cu
/// @brief Implementation of utility functions declared in sort_utils.h.

#include <cuda_runtime.h>
#include "sort_utils.h"
#include "types/constants.h"

__device__ int block_counter = 0;

__device__ void ExclusiveScan(unsigned int* data, unsigned int* output, const unsigned int length,
                              unsigned int* flags, unsigned int* scan_value,
                              const unsigned int iter)
{
    __shared__ unsigned int XY[CFACTOR * SECTION_SIZE];
    __shared__ unsigned int dyn_block_id_s;
    unsigned int tx = threadIdx.x;
    if (tx == 0)
    {
        dyn_block_id_s = atomicAdd(&block_counter, 1);
    }
    __syncthreads();
    unsigned int dyn_block_id = dyn_block_id_s - iter * gridDim.x;
    unsigned int i = CFACTOR * dyn_block_id * blockDim.x + tx;
    // load data into shared memory.
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            XY[tx + offset] = data[i + offset];
        }
        else
        {
            XY[tx + offset] = 0;
        }
    }
    __syncthreads();
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] += XY[curr_idx - 1];
    }
    // Kogge-Stone on final elements of each threads' local scan.
    unsigned int temp = 0;
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        if (curr_idx >= stride)
        {
            temp = XY[curr_idx - stride] + XY[curr_idx];
        }
        __syncthreads();
        if (curr_idx >= stride)
        {
            XY[curr_idx] = temp;
        }
    }
    __syncthreads();
    // Add the preceding final element to all non-final elements
    if (tx != 0)
    {
        for (unsigned int i_scan = 0; i_scan < CFACTOR - 1; ++i_scan)
        {
            unsigned int curr_idx = i_scan + tx * CFACTOR;
            XY[curr_idx] += XY[tx * CFACTOR - 1];
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            output[i + offset] = XY[tx + offset];
        }
    }
    __syncthreads();

    __shared__ unsigned int previous_sum;
    if (threadIdx.x == 0)
    {
        if (dyn_block_id != 0)
        {
            while (atomicAdd(&flags[dyn_block_id - 1], 0) == 0)
            {}
            previous_sum = scan_value[dyn_block_id - 1];
        }
        else
        {
            previous_sum = 0;
        }
        scan_value[dyn_block_id] =
            previous_sum + output[min(length - 1, CFACTOR * (dyn_block_id + 1) * blockDim.x - 1)];
        atomicAdd(&flags[dyn_block_id], 1);
    }
    __syncthreads();

    if (dyn_block_id != 0)
    {
        for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
        {
            if (i + offset < length)
            {
                output[i + offset] += previous_sum;
            }
        }
    }
}

__global__ void ResetArray(unsigned int* data, unsigned int length, unsigned int val)
{
    unsigned int idx = CFACTOR * blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < min(CFACTOR * blockDim.x, length); i += blockDim.x)
    {
        data[i] = val;
    }
}

__global__ void InclusiveScan(unsigned int* data, unsigned int* output, const unsigned int length,
                              unsigned int* flags, unsigned int* scan_value,
                              const unsigned int iter)
{
    __shared__ unsigned int XY[CFACTOR * SECTION_SIZE];
    __shared__ unsigned int dyn_block_id_s;
    unsigned int tx = threadIdx.x;
    if (tx == 0)
    {
        dyn_block_id_s = atomicAdd(&block_counter, 1);
    }
    __syncthreads();
    unsigned int dyn_block_id = dyn_block_id_s - iter * gridDim.x;
    unsigned int i = CFACTOR * dyn_block_id * blockDim.x + tx;
    // load data into shared memory.
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            XY[tx + offset] = data[i + offset];
        }
        else
        {
            XY[tx + offset] = 0;
        }
    }
    __syncthreads();
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] += XY[curr_idx - 1];
    }
    // Kogge-Stone on final elements of each threads' local scan.
    unsigned int temp = 0;
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        if (curr_idx >= stride)
        {
            temp = XY[curr_idx - stride] + XY[curr_idx];
        }
        __syncthreads();
        if (curr_idx >= stride)
        {
            XY[curr_idx] = temp;
        }
    }
    __syncthreads();
    // Add the preceding final element to all non-final elements
    if (tx != 0)
    {
        for (unsigned int i_scan = 0; i_scan < CFACTOR - 1; ++i_scan)
        {
            unsigned int curr_idx = i_scan + tx * CFACTOR;
            XY[curr_idx] += XY[tx * CFACTOR - 1];
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            output[i + offset] = XY[tx + offset];
        }
    }
    __syncthreads();

    __shared__ unsigned int previous_sum;
    if (threadIdx.x == 0)
    {
        if (dyn_block_id != 0)
        {
            while (atomicAdd(&flags[dyn_block_id - 1], 0) == 0)
            {}
            previous_sum = scan_value[dyn_block_id - 1];
        }
        else
        {
            previous_sum = 0;
        }
        scan_value[dyn_block_id] =
            previous_sum + output[min(length - 1, CFACTOR * (dyn_block_id + 1) * blockDim.x - 1)];
        atomicAdd(&flags[dyn_block_id], 1);
    }
    __syncthreads();

    if (dyn_block_id != 0)
    {
        for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
        {
            if (i + offset < length)
            {
                output[i + offset] += previous_sum;
            }
        }
    }
}