#include "parallel_scan.h"
#include "types/constants.h"

__device__ __host__ ParallelScanDataType ParallelScanOperation(const ParallelScanDataType lhs,
                                                               const ParallelScanDataType rhs)
{
    return lhs + rhs;
}

__device__ __host__ ParallelScanDataType ParallelScanIdentity() { return 0; }

__global__ void ResetArray(int* data, unsigned int length, int val)
{
    unsigned int idx = CFACTOR * blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < min(CFACTOR * blockDim.x, length); i += blockDim.x)
    {
        data[i] = val;
    }
}

__global__ void KoggeStoneInclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                          unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length)
    {
        XY[tx] = data[i];
    }
    else
    {
        XY[tx] = ParallelScanIdentity();
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

__global__ void KoggeStoneExclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                          unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length && tx != 0)
    {
        XY[tx] = data[i - 1];
    }
    else
    {
        XY[tx] = ParallelScanIdentity();
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

__global__ void KoggeStoneDoubleBufferingInclusiveKernel(ParallelScanDataType* data,
                                                         ParallelScanDataType* result,
                                                         unsigned int length)
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
    else
    {
        XY_odd[tx] = ParallelScanIdentity();
        XY_even[tx] = ParallelScanIdentity();
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

__global__ void KoggeStoneDoubleBufferingExclusiveKernel(ParallelScanDataType* data,
                                                         ParallelScanDataType* result,
                                                         unsigned int length)
{
    __shared__ ParallelScanDataType XY_odd[SECTION_SIZE];
    __shared__ ParallelScanDataType XY_even[SECTION_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length && tx != 0)
    {
        XY_odd[tx] = data[i - 1];
        XY_even[tx] = data[i - 1];
    }
    else
    {
        XY_odd[tx] = ParallelScanIdentity();
        XY_even[tx] = ParallelScanIdentity();
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

__global__ void BrentKungInclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                         unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length)
    {
        XY[tx] = data[i];
    }
    else
    {
        XY[tx] = ParallelScanIdentity();
    }
    if (i + blockDim.x < length)
    {
        XY[tx + blockDim.x] = data[i + blockDim.x];
    }
    else
    {
        XY[tx + blockDim.x] = ParallelScanIdentity();
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

__global__ void BrentKungExclusiveKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                         unsigned int length)
{
    __shared__ ParallelScanDataType XY[SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;
    if (i < length && tx != 0)
    {
        XY[tx] = data[i - 1];
    }
    else
    {
        XY[tx] = ParallelScanIdentity();
    }
    if (i + blockDim.x < length && tx != 0)
    {
        XY[tx + blockDim.x] = data[i + blockDim.x - 1];
    }
    else if (tx + blockDim.x < SECTION_SIZE)
    {
        XY[tx + blockDim.x] = ParallelScanIdentity();
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

__global__ void ThreadCoarseningInclusiveKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length)
{
    __shared__ ParallelScanDataType XY[CFACTOR * SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = CFACTOR * blockIdx.x * blockDim.x + tx;
    // load data into shared memory.
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            XY[tx + offset] = data[i + offset];
        }
        else
        {
            XY[tx + offset] = ParallelScanIdentity();
        }
    }
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] = ParallelScanOperation(XY[curr_idx - 1], XY[curr_idx]);
    }
    // Kogge-Stone on final elements of each threads' local scan.
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        ParallelScanDataType temp;
        if (curr_idx >= stride)
        {
            temp = ParallelScanOperation(XY[curr_idx - stride], XY[curr_idx]);
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
            XY[curr_idx] = ParallelScanOperation(XY[tx * CFACTOR - 1], XY[curr_idx]);
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            result[i + offset] = XY[tx + offset];
        }
    }
}

__global__ void ThreadCoarseningExclusiveKernel(ParallelScanDataType* data,
                                                ParallelScanDataType* result, unsigned int length)
{
    __shared__ ParallelScanDataType XY[CFACTOR * SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = CFACTOR * blockIdx.x * blockDim.x + tx;
    // load data into shared memory.
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length && (tx + offset) != 0)
        {
            XY[tx + offset] = data[i + offset - 1];
        }
        else
        {
            XY[tx + offset] = ParallelScanIdentity();
        }
    }
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] = ParallelScanOperation(XY[curr_idx - 1], XY[curr_idx]);
    }
    // Kogge-Stone on final elements of each threads' local scan.
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        ParallelScanDataType temp;
        if (curr_idx >= stride)
        {
            temp = ParallelScanOperation(XY[curr_idx - stride], XY[curr_idx]);
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
            XY[curr_idx] = ParallelScanOperation(XY[tx * CFACTOR - 1], XY[curr_idx]);
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            result[i + offset] = XY[tx + offset];
        }
    }
}

__global__ void ThreadCoarseningSegmentedScanKernelPhase1(ParallelScanDataType* data,
                                                          ParallelScanDataType* result,
                                                          ParallelScanDataType* end_vals,
                                                          unsigned int length)
{
    __shared__ ParallelScanDataType XY[CFACTOR * SECTION_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = CFACTOR * blockIdx.x * blockDim.x + tx;
    // load data into shared memory.
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            XY[tx + offset] = data[i + offset];
        }
        else
        {
            XY[tx + offset] = ParallelScanIdentity();
        }
    }
    __syncthreads();
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] = ParallelScanOperation(XY[curr_idx - 1], XY[curr_idx]);
    }
    // Kogge-Stone on final elements of each threads' local scan.
    ParallelScanDataType temp = ParallelScanIdentity();
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        if (curr_idx >= stride)
        {
            temp = ParallelScanOperation(XY[curr_idx - stride], XY[curr_idx]);
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
            XY[curr_idx] = ParallelScanOperation(XY[tx * CFACTOR - 1], XY[curr_idx]);
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            result[i + offset] = XY[tx + offset];
        }
    }
    __syncthreads();

    if (tx == 0)
    {
        end_vals[blockIdx.x] = result[min(length - 1, CFACTOR * (blockIdx.x + 1) * blockDim.x - 1)];
    }
}

__global__ void ThreadCoarseningSegmentedScanKernelPhase3(ParallelScanDataType* data,
                                                          ParallelScanDataType* end_vals_scanned,
                                                          unsigned int length)
{
    __shared__ ParallelScanDataType increment_val;
    if (blockIdx.x > 0)
    {
        if (threadIdx.x == 0)
        {
            increment_val = end_vals_scanned[blockIdx.x - 1];
        }
        __syncthreads();
        unsigned int i = CFACTOR * blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
        {
            if (i + offset < length)
            {
                data[i + offset] = ParallelScanOperation(increment_val, data[i + offset]);
            }
        }
    }
}

__global__ void StreamingKernel(ParallelScanDataType* data, ParallelScanDataType* result,
                                int* flags, ParallelScanDataType* scan_value, unsigned int length)
{
    __shared__ ParallelScanDataType XY[CFACTOR * SECTION_SIZE];
    __shared__ unsigned int dyn_block_id_s;
    unsigned int tx = threadIdx.x;
    if (tx == 0)
    {
        dyn_block_id_s = atomicAdd(&block_counter, 1);
    }
    __syncthreads();
    unsigned int dyn_block_id = dyn_block_id_s;
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
            XY[tx + offset] = ParallelScanIdentity();
        }
    }
    __syncthreads();
    // inclusive scan on elements thread is responsible for.
    for (unsigned int i_scan = 1; i_scan < CFACTOR; ++i_scan)
    {
        unsigned int curr_idx = i_scan + tx * CFACTOR;
        XY[curr_idx] = ParallelScanOperation(XY[curr_idx - 1], XY[curr_idx]);
    }
    // Kogge-Stone on final elements of each threads' local scan.
    ParallelScanDataType temp = ParallelScanIdentity();
    for (unsigned int stride = CFACTOR; stride < CFACTOR * SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int curr_idx = (tx + 1) * CFACTOR - 1;
        if (curr_idx >= stride)
        {
            temp = ParallelScanOperation(XY[curr_idx - stride], XY[curr_idx]);
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
            XY[curr_idx] = ParallelScanOperation(XY[tx * CFACTOR - 1], XY[curr_idx]);
        }
    }
    __syncthreads();
    // Write to result
    for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
    {
        if (i + offset < length)
        {
            result[i + offset] = XY[tx + offset];
        }
    }
    __syncthreads();

    __shared__ ParallelScanDataType previous_sum;
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
            previous_sum = ParallelScanIdentity();
        }
        scan_value[dyn_block_id] =
            previous_sum + result[min(length - 1, CFACTOR * (dyn_block_id + 1) * blockDim.x - 1)];
        atomicAdd(&flags[dyn_block_id], 1);
    }
    __syncthreads();

    if (dyn_block_id != 0)
    {
        for (unsigned int offset = 0; offset < CFACTOR * SECTION_SIZE; offset += blockDim.x)
        {
            if (i + offset < length)
            {
                result[i + offset] = ParallelScanOperation(previous_sum, result[i + offset]);
            }
        }
    }
}