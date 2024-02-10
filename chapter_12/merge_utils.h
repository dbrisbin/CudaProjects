#ifndef CHAPTER_12_MERGE_UTILS_H
#define CHAPTER_12_MERGE_UTILS_H

#include <cuda_runtime.h>
#include <utility>

/// @brief Merge two sorted arrays into a single larger sorted array.
/// @param A first array to merge
/// @param m length of A
/// @param B second array to merge
/// @param n length of B
/// @param[out] C store result of the merge of length m + n
__device__ __host__ void MergeSequential(const std::pair<int, int>* A, const int m,
                                         const std::pair<int, int>* B, const int n,
                                         std::pair<int, int>* C);

/// @brief Compute the co-rank i for array A given A, B, their lengths, and rank k of C.
/// @param k rank of C to find co-rank values for
/// @param A first input array to find co-rank i in
/// @param m length of A
/// @param B second input array (can compute its co-rank as j = k - i)
/// @param n length of B
/// @return co-rank of A for rank k of C
__device__ int CoRank(const int k, const std::pair<int, int>* A, const int m,
                      const std::pair<int, int>* B, const int n);

#endif  // CHAPTER_12_MERGE_UTILS_H