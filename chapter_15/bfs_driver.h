/// @file bfs_driver.h
/// @brief Declaration of driver function to setup and call the appropriate kernel to compute the
/// sparse matrix vector product on GPU.

#ifndef CHAPTER_15_BFS_DRIVER_H
#define CHAPTER_15_BFS_DRIVER_H

#include "types/adjacency_matrix.h"
#include "types/constants.h"

/// @brief Driver function to setup and call the appropriate kernel to run BFS on the GPU.
/// @param graph graph to perform BFS on
/// @param result result of the BFS
/// @param n number of nodes in the graph
/// @param iters number of iterations to run the kernel
/// @param kernel_to_use kernel to use for BFS
/// @return time taken to run the kernel iters times
float BfsDriver(AdjacencyMatrix& adj_matrix, int* result, const int iters,
                const BfsKernel kernel_to_use);

#endif  // CHAPTER_15_BFS_DRIVER_H