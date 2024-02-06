#include "reduction.h"
#include "types/constants.h"

// currently supported:
// atomicAdd and addition
// atomicMin and min operation
// atomicMax and max operation
// atomicAnd and bitwise and (&)
// atomicOr  and bitwise or (|)
// atomicXor and xor (^)
// Note: Product is NOT supported.
__device__ __host__ ReductionDataType reductionOperation(const ReductionDataType a,
                                                         const ReductionDataType b)
{
    return min(a, b);
}

__device__ __host__ ReductionDataType reductionIdentity() { return INT_MAX; }

namespace
{

__device__ void atomicOp(ReductionDataType* loc, const ReductionDataType new_val)
{
    atomicMin(loc, new_val);
}

}  // namespace

__global__ void basicReduction(ReductionDataType* data, int length, ReductionDataType* result)
{
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0 && (i + stride) < length)
        {
            data[i] = reductionOperation(data[i], data[i + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *result = data[0];
    }
}

__global__ void coalescingReduction(ReductionDataType* data, int length, ReductionDataType* result)
{
    unsigned int tx = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (tx < stride && (tx + stride) < length)
        {
            data[tx] = reductionOperation(data[tx], data[tx + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *result = data[0];
    }
}

__global__ void coalescingModifiedReduction(ReductionDataType* data, int length,
                                            ReductionDataType* result)
{
    unsigned int tx = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            data[length - stride + tx] =
                reductionOperation(data[length - stride + tx], data[length - 2 * stride + tx]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *result = data[length - 1];
    }
}

__global__ void sharedMemoryReduction(ReductionDataType* data, int length,
                                      ReductionDataType* result)
{
    __shared__ ReductionDataType data_s[TILE_WIDTH];
    unsigned int tx = threadIdx.x;
    if (tx + blockDim.x < length)
    {
        data_s[tx] = reductionOperation(data[tx], data[tx + blockDim.x]);
    }
    else if (tx < length)
    {
        data_s[tx] = data[tx];
    }
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tx < stride && tx + stride < length)
        {
            data_s[tx] = reductionOperation(data_s[tx], data_s[tx + stride]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *result = data_s[0];
    }
}

__global__ void segmentedReduction(ReductionDataType* data, int length, ReductionDataType* result)
{
    __shared__ ReductionDataType data_s[TILE_WIDTH];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int tx = threadIdx.x;
    unsigned int i = tx + segment;

    if (i + blockDim.x < length)
    {
        data_s[tx] = reductionOperation(data[i], data[i + blockDim.x]);
    }
    else if (i < length)
    {
        data_s[tx] = data[i];
    }
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tx < stride && tx + stride < length)
        {
            data_s[tx] = reductionOperation(data_s[tx], data_s[tx + stride]);
        }
    }
    __syncthreads();
    if (tx == 0)
    {
        atomicOp(result, data_s[0]);
    }
}

__global__ void coarseningReduction(ReductionDataType* data, int length, ReductionDataType* result)
{
    __shared__ ReductionDataType data_s[TILE_WIDTH];
    unsigned int segment = CFACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int tx = threadIdx.x;
    unsigned int i = tx + segment;

    ReductionDataType cum_op = 0;
    if (i < length)
    {
        cum_op = data[i];
    }

    for (unsigned int tile = 1; tile < CFACTOR * 2; ++tile)
    {
        if ((i + tile * blockDim.x) < length)
        {
            cum_op = reductionOperation(cum_op, data[i + tile * blockDim.x]);
        }
    }
    if (tx < length)
    {
        data_s[tx] = cum_op;
    }
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tx < stride && tx + stride < length)
        {
            data_s[tx] = reductionOperation(data_s[tx], data_s[tx + stride]);
        }
    }
    __syncthreads();
    if (tx == 0)
    {
        atomicOp(result, data_s[0]);
    }
}