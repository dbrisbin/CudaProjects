#include <cuda_runtime.h>
#include <utility>
#include "merge_utils.h"

__device__ __host__ void MergeSequential(const std::pair<int, int>* A, const int m,
                                         const std::pair<int, int>* B, const int n,
                                         std::pair<int, int>* C)
{
    int i_A{};
    int i_B{};
    int i_C{};

    while ((i_A < m) && (i_B < n))
    {
        if (A[i_A].first <= B[i_B].first)
        {
            C[i_C].first = A[i_A].first;
            C[i_C++].second = A[i_A++].second;
        }
        else
        {
            C[i_C].first = B[i_B].first;
            C[i_C++].second = B[i_B++].second;
        }
    }
    if (i_A < m)
    {
        while (i_A < m)
        {
            C[i_C].first = A[i_A].first;
            C[i_C++].second = A[i_A++].second;
        }
    }
    else
    {
        while (i_B < n)
        {
            C[i_C].first = B[i_B].first;
            C[i_C++].second = B[i_B++].second;
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