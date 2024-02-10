#include <cuda_runtime.h>
#include <utility>
#include "merge.h"
#include "merge_utils.h"

__global__ void basicKernel(const std::pair<int, int>* A, const int m, const std::pair<int, int>* B,
                            const int n, std::pair<int, int>* C)
{
    const auto length = m + n;
    const auto elts_per_thread =
        static_cast<int>(ceil(length / (static_cast<double>(blockDim.x) * gridDim.x)));
    const auto tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto k_curr = tid * elts_per_thread;
    if (k_curr < length)
    {
        const auto k_next = min(k_curr + elts_per_thread, length);
        const auto i_curr = CoRank(k_curr, A, m, B, n);
        const auto i_next = CoRank(k_next, A, m, B, n);
        const auto j_curr = k_curr - i_curr;
        const auto j_next = k_next - i_next;
        MergeSequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
    }
}

__global__ void tiledKernel(const std::pair<int, int>* A, const int m, const std::pair<int, int>* B,
                            const int n, std::pair<int, int>* C, const int tile_size)
{
    extern __shared__ std::pair<int, int> AB_s[];
    __shared__ int A_start, A_end, B_start, B_end;
    auto* A_s = AB_s;
    auto* B_s = &AB_s[tile_size];

    auto C_start = static_cast<int>(blockIdx.x * ceil(static_cast<double>(m + n) / gridDim.x));
    auto C_end = min(
        static_cast<int>((blockIdx.x + 1) * ceil(static_cast<double>(m + n) / gridDim.x)), m + n);

    if (threadIdx.x == 0)
    {
        A_start = CoRank(C_start, A, m, B, n);
        A_end = CoRank(C_end, A, m, B, n);
        B_start = C_start - A_start;
        B_end = C_end - A_end;
    }
    __syncthreads();

    auto A_len = A_end - A_start;
    auto B_len = B_end - B_start;
    auto C_len = C_end - C_start;

    int A_consumed{0};
    int B_consumed{0};
    int C_completed{0};

    auto num_iters = static_cast<int>(ceil(static_cast<double>(C_len) / tile_size));
    for (auto iter = 0; iter < num_iters; ++iter)
    {
        for (auto offset = 0; offset < tile_size; offset += blockDim.x)
        {
            if (threadIdx.x + offset < A_len - A_consumed)
            {
                A_s[threadIdx.x + offset].first =
                    A[A_start + A_consumed + threadIdx.x + offset].first;
                A_s[threadIdx.x + offset].second =
                    A[A_start + A_consumed + threadIdx.x + offset].second;
            }
            if (threadIdx.x + offset < B_len - B_consumed)
            {
                B_s[threadIdx.x + offset].first =
                    B[B_start + B_consumed + threadIdx.x + offset].first;
                B_s[threadIdx.x + offset].second =
                    B[B_start + B_consumed + threadIdx.x + offset].second;
            }
        }
        __syncthreads();

        auto k_curr = min(threadIdx.x * (tile_size / blockDim.x), C_len - C_completed);
        auto k_next = min((threadIdx.x + 1) * (tile_size / blockDim.x), C_len - C_completed);

        auto i_curr = CoRank(k_curr, A_s, min(tile_size, A_len - A_consumed), B_s,
                             min(tile_size, B_len - B_consumed));
        auto i_next = CoRank(k_next, A_s, min(tile_size, A_len - A_consumed), B_s,
                             min(tile_size, B_len - B_consumed));
        auto j_curr = k_curr - i_curr;
        auto j_next = k_next - i_next;

        MergeSequential(&A_s[i_curr], i_next - i_curr, &B_s[j_curr], j_next - j_curr,
                        &C[C_start + C_completed + k_curr]);
        C_completed += tile_size;
        A_consumed += CoRank(tile_size, A_s, tile_size, B_s, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}