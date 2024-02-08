#include "parallel_scan.h"
#include "types/constants.h"

__device__ __host__ ParallelScanDataType ParallelScanOperation(const ParallelScanDataType lhs,
                                                               const ParallelScanDataType rhs)
{
    return lhs + rhs;
}

__global__ void KoggeStoneKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                 unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length)
    {
        XY[tx] = data[i];
    }
    unsigned int temp;
    for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        if (tx >= stride)
        {
            temp = ParallelScanOperation(XY[tx], XY[tx - stride]);
        }
        __syncthreads();
        if (tx >= stride)
        {
            XY[tx] = temp;
        }
    }
    if (i < length)
    {
        result[i] = XY[tx];
    }
}

__global__ void KoggeStoneDoubleBufferingKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length)
{
    __shared__ ParallelScanDataType XY_odd[SECTION_SIZE];
    __shared__ ParallelScanDataType XY_even[SECTION_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length)
    {
        XY_odd[tx] = data[i];
        XY_even[tx] = data[i];
    }
    unsigned int iter = 0;
    for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        if (tx >= stride)
        {
            if (iter % 2 == 0)
            {
                XY_even[tx] = ParallelScanOperation(XY_odd[tx], XY_odd[tx - stride]);
            }
            else
            {
                XY_odd[tx] = ParallelScanOperation(XY_even[tx], XY_even[tx - stride]);
            }
            ++iter;
        }
        else if (iter % 2 == 0)
        {
            XY_even[tx] = XY_odd[tx];
        }
        else
        {
            XY_odd[tx] = XY_even[tx];
        }
    }

    if (i < length)
    {

        if (iter % 2 == 0)
        {
            result[i] = XY_odd[tx];
        }
        else
        {
            result[i] = XY_even[tx];
        }
    }
}

__global__ void BrentKungKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length)
    {
        XY[tx] = data[i];
    }
    if (i + blockDim.x < length)
    {
        XY[tx + blockDim.x] = data[i + blockDim.x];
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = (tx + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
        {
            XY[index] = ParallelScanOperation(XY[index], XY[index - stride]);
        }
    }
    for (unsigned int stride = SECTION_SIZE / 4; stride >= 1; stride /= 2)
    {
        __syncthreads();
        unsigned int index = (tx + 1) * 2 * stride - 1;
        if (index + stride < SECTION_SIZE)
        {
            XY[index + stride] = ParallelScanOperation(XY[index + stride], XY[index]);
        }
    }
    __syncthreads();
    if (i < length)
    {
        result[i] = XY[tx];
    }
    if (i + blockDim.x < length)
    {
        result[i + blockDim.x] = XY[tx + blockDim.x];
    }
}
