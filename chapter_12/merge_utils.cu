/// @file merge_utils.cu
/// @brief Implementation of utility functions declared in merge_utils.h.

#include <cuda_runtime.h>
#include <utility>
#include "merge_utils.h"

__device__ __host__ void MergeSequential(const std::pair<int, int>* A, const int m,
                                         const std::pair<int, int>* B, const int n,
                                         std::pair<int, int>* C)
{
    int i{};
    int j{};
    int k{};

    while ((i < m) && (j < n))
    {
        if (A[i].first <= B[j].first)
        {
            C[k].first = A[i].first;
            C[k++].second = A[i++].second;
        }
        else
        {
            C[k].first = B[j].first;
            C[k++].second = B[j++].second;
        }
    }
    if (i < m)
    {
        while (i < m)
        {
            C[k].first = A[i].first;
            C[k++].second = A[i++].second;
        }
    }
    else
    {
        while (j < n)
        {
            C[k].first = B[j].first;
            C[k++].second = B[j++].second;
        }
    }
}

__device__ int CoRank(const int k, const std::pair<int, int>* A, const int m,
                      const std::pair<int, int>* B, const int n)
{
    auto i = min(k, m);
    auto j = k - i;
    auto i_low = max(0, k - n);
    auto j_low = max(0, k - m);

    decltype(i) delta{};
    bool finished{false};
    while (!finished)
    {
        if (i > 0 && j < n && A[i - 1].first > B[j].first)
        {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if (i < m && j > 0 && A[i].first <= B[j - 1].first)
        {
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
        }
        else
        {
            finished = true;
        }
    }
    return i;
}

__device__ void MergeSequentialCircular(const std::pair<int, int>* A, const int m,
                                        const std::pair<int, int>* B, const int n,
                                        std::pair<int, int>* C, const int A_section_start,
                                        const int B_section_start, const int tile_size)
{
    int i{};
    int j{};
    int k{};

    while ((i < m) && (j < n))
    {
        const int i_circ = (i + A_section_start) % tile_size;
        const int j_circ = (j + A_section_start) % tile_size;
        if (A[i_circ].first <= B[j_circ].first)
        {
            C[k].first = A[i_circ].first;
            C[k++].second = A[i_circ].second;
            i++;
        }
        else
        {
            C[k].first = B[j_circ].first;
            C[k++].second = B[j_circ].second;
            j++;
        }
    }
    if (i < m)
    {
        while (i < m)
        {
            const int i_circ = (i + A_section_start) % tile_size;
            C[k].first = A[i_circ].first;
            C[k++].second = A[i_circ].second;
            i++;
        }
    }
    else
    {
        while (j < n)
        {
            const int j_circ = (j + A_section_start) % tile_size;
            C[k].first = B[j_circ].first;
            C[k++].second = B[j_circ].second;
            j++;
        }
    }
}

__device__ int CoRankCircular(const int k, const std::pair<int, int>* A, const int m,
                              const std::pair<int, int>* B, const int n, const int A_section_start,
                              const int B_section_start, const int tile_size)
{
    auto i = min(k, m);
    auto j = k - i;
    auto i_low = max(0, k - n);
    auto j_low = max(0, k - m);

    decltype(i) delta{};
    bool finished{false};
    while (!finished)
    {
        const int i_circ = (i + A_section_start) % tile_size;
        const int i_circ_minus_1 = (i_circ - 1) % tile_size;
        const int j_circ = (j + A_section_start) % tile_size;
        const int j_circ_minus_1 = (j_circ - 1) % tile_size;
        if (i > 0 && j < n && A[i_circ_minus_1].first > B[j_circ].first)
        {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if (i < m && j > 0 && A[i_circ].first <= B[j_circ_minus_1].first)
        {
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
        }
        else
        {
            finished = true;
        }
    }
    return i;
}
