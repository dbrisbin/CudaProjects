/// @file sort.cu
/// @brief Implementation of kernels declared in sort.h.

#include <cuda_runtime.h>
#include "sort.h"
#include "sort_utils.h"

__device__ int num_finished_setting_bits = 0;
__device__ int num_finished_scanning = 0;

__global__ void RadixSortIter(unsigned int* data, unsigned int* output, unsigned int* bits_in,
                              unsigned int* bits_out, const unsigned int length,
                              const unsigned int iter, unsigned int* flags,
                              unsigned int* scan_values)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key{};
    unsigned int bit{};
    if (i < length)
    {
        key = data[i];
        bit = (key >> iter) & 1;
        bits_in[i] = bit;
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(&num_finished_setting_bits, 1);
        while (atomicAdd(&num_finished_setting_bits, 0) != gridDim.x * (iter + 1))
        {}
    }
    __syncthreads();
    ExclusiveScan(bits_in, bits_out, length, flags, scan_values, iter);
    if (threadIdx.x == 0)
    {
        atomicAdd(&num_finished_scanning, 1);
        while (atomicAdd(&num_finished_scanning, 0) != gridDim.x * (iter + 1))
        {}
    }
    __syncthreads();
    if (i < length)
    {
        unsigned int dest_idx{bit == 0 ? (i - bits_out[i])
                                       : (length - bits_out[length - 1] + bits_out[i] - 1)};
        output[dest_idx] = key;
    }
}

__global__ void RadixSortIterPhase1(const unsigned int* data, unsigned int* bits,
                                    const unsigned int length, const unsigned int iter)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key{};
    unsigned int bit{};
    if (i < length)
    {
        key = data[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }
}

__global__ void RadixSortIterPhase2(const unsigned int* data, unsigned int* output,
                                    const unsigned int* bits, const unsigned int length,
                                    const unsigned int iter)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key{};
    unsigned int bit{};
    if (i < length)
    {
        key = data[i];
        bit = (key >> iter) & 1;
        unsigned int dest_idx{bit == 0 ? (i - bits[i]) : (length - bits[length - 1] + bits[i] - 1)};
        output[dest_idx] = key;
    }
}