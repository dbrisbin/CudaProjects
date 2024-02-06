#include "reduction.h"
#include "types/constants.h"

__global__ void basicReduction(ReductionDataType* data, int length, ReductionDataType* result)
{
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
        {
            data[i] += data[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *result = data[0];
    }
}