#include "parallelHistogram.h"
#include "types/constants.h"

__global__ void basicParallelHistogram(int* data, int length, int* hist)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        atomicAdd(&hist[data[i] / VALS_PER_BIN], 1);
    }
}

__global__ void privatizedParallelHistogram(int* data, int length, int* hist)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        atomicAdd(&hist[blockIdx.x * NUM_BINS + data[i] / VALS_PER_BIN], 1);
    }
    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        {
            unsigned int bin_val = hist[blockIdx.x * NUM_BINS + bin];
            if (bin_val > 0)
            {
                atomicAdd(&hist[bin], bin_val);
            }
        }
    }
}

__global__ void privatizedWithSharedMemoryParallelHistogram(int* data, int length, int* hist)
{
    __shared__ int hist_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        hist_s[bin] = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        atomicAdd(&hist_s[data[i] / VALS_PER_BIN], 1);
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int bin_val = hist_s[bin];
        if (bin_val > 0)
        {
            atomicAdd(&hist[bin], bin_val);
        }
    }
}

__global__ void coarseningParallelHistogram(int* data, int length, int* hist)
{
    __shared__ int hist_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        hist_s[bin] = 0;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid * CFACTOR; i < min(length, (tid + 1) * CFACTOR); ++i)
    {
        atomicAdd(&hist_s[data[i] / VALS_PER_BIN], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int bin_val = hist_s[bin];
        if (bin_val > 0)
        {
            atomicAdd(&hist[bin], bin_val);
        }
    }
}

__global__ void coarseningWithCoalescedAccessParallelHistogram(int* data, int length, int* hist)
{
    __shared__ int hist_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        hist_s[bin] = 0;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&hist_s[data[i] / VALS_PER_BIN], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int bin_val = hist_s[bin];
        if (bin_val > 0)
        {
            atomicAdd(&hist[bin], bin_val);
        }
    }
}

__global__ void aggregatedParallelHistogram(int* data, int length, int* hist)
{
    __shared__ int hist_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        hist_s[bin] = 0;
    }
    __syncthreads();

    unsigned int accumulator = 0;
    int prev_bin = -1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
        int bin = data[i] / VALS_PER_BIN;
        if (bin == prev_bin)
        {
            accumulator++;
        }
        else
        {
            if (accumulator > 0)
            {
                atomicAdd(&hist_s[prev_bin], accumulator);
            }
            prev_bin = bin;
            accumulator = 1;
        }
    }
    if (accumulator > 0)
    {
        atomicAdd(&hist_s[prev_bin], accumulator);
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int bin_val = hist_s[bin];
        if (bin_val > 0)
        {
            atomicAdd(&hist[bin], bin_val);
        }
    }
}
