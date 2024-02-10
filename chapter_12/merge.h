/// @file merge.h
/// @brief Declaration of available kernels for computing a merge of sorted arrays.

#ifndef CHAPTER_12_MERGE_H
#define CHAPTER_12_MERGE_H

#include <cuda_runtime.h>
#include <utility>

/// @brief Perform a merge of two sorted arrays using CoRank() and MergeSequential().
/// @param A first sorted array to merge
/// @param m length of A
/// @param B second sorted array to merge
/// @param n length of B
/// @param[out] C result of merging
/// @pre A and B are sorted
__global__ void BasicKernel(const std::pair<int, int>* A, const int m, const std::pair<int, int>* B,
                            const int n, std::pair<int, int>* C);

/// @brief Perform a merge of two sorted arrays using by loading parts of A and B into shared
/// memory.
/// @param A first sorted array to merge
/// @param m length of A
/// @param B second sorted array to merge
/// @param n length of B
/// @param[out] C result of merging
/// @param tile_size number of elements to load into shared memory per iteration
/// @pre A and B are sorted
__global__ void TiledKernel(const std::pair<int, int>* A, const int m, const std::pair<int, int>* B,
                            const int n, std::pair<int, int>* C, const int tile_size);

/// @brief Perform a merge of two sorted arrays using by loading parts of A and B into shared
/// memory. The modified version only loads the necessary parts of A and B into shared memory.
/// @param A first sorted array to merge
/// @param m length of A
/// @param B second sorted array to merge
/// @param n length of B
/// @param[out] C result of merging
/// @param tile_size number of elements to load into shared memory per iteration
/// @pre A and B are sorted
__global__ void ModifiedTiledKernel(const std::pair<int, int>* A, const int m,
                                    const std::pair<int, int>* B, const int n,
                                    std::pair<int, int>* C, const int tile_size);

/// @brief Perform a merge of two sorted arrays using by loading parts of A and B into shared
/// memory as circular buffers.
/// @param A first sorted array to merge
/// @param m length of A
/// @param B second sorted array to merge
/// @param n length of B
/// @param[out] C result of merging
/// @param tile_size number of elements to load into shared memory per iteration
/// @pre A and B are sorted
__global__ void CircularBufferKernel(const std::pair<int, int>* A, const int m,
                                     const std::pair<int, int>* B, const int n,
                                     std::pair<int, int>* C, const int tile_size);

#endif  // CHAPTER_12_MERGE_H