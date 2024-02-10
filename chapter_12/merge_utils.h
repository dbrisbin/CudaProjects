/// @file merge_utils.h
/// @brief Declaration of utility functions used by kernels and driver code.

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

/// @brief Merge two sorted arrays stored in circular buffers into a single larger sorted array.
/// @param A first circular buffer to merge
/// @param m number of elements to merge from A
/// @param B second circular buffer to merge
/// @param n number of elements to merge from B
/// @param[out] C store result of the merge of length m + n
/// @param A_section_start start idx of the circular buffer A
/// @param B_section_start start idx of the circular buffer B
/// @param tile_size capacity of A and B
__device__ void MergeSequentialCircular(const std::pair<int, int>* A, const int m,
                                        const std::pair<int, int>* B, const int n,
                                        std::pair<int, int>* C, const int A_section_start,
                                        const int B_section_start, const int tile_size);

/// @brief Compute the co-rank i for circular buffer A given A, B, their lengths, rank k of
/// array C, the start positions in the buffers for A and B, and the capacities of the buffers.
/// @param k rank of C to find co-rank values for
/// @param A first input circular buffer to find co-rank i in
/// @param m size of A
/// @param B second input circular buffer (can compute its co-rank as j = k - i)
/// @param n size of B
/// @param A_section_start start idx of the circular buffer A
/// @param B_section_start start idx of the circular buffer B
/// @param tile_size capacity of A and B
/// @return co-rank of A for rank k of C
__device__ int CoRankCircular(const int k, const std::pair<int, int>* A, const int m,
                              const std::pair<int, int>* B, const int n, const int A_section_start,
                              const int B_section_start, const int tile_size);

#endif  // CHAPTER_12_MERGE_UTILS_H