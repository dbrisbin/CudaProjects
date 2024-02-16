/// @file bfs_driver.h
/// @brief Declaration of driver functions to setup and call the appropriate kernel to compute the
/// sparse matrix vector product on GPU.

#ifndef CHAPTER_15_BFS_DRIVER_H
#define CHAPTER_15_BFS_DRIVER_H

#include "types/adjacency_matrix.h"
#include "types/constants.h"

/// @brief Driver function to setup and call the Edge-centric kernel to run BFS on the GPU.
/// @param adj_matrix graph to perform BFS on
/// @param[out] result_h result_h of the BFS
/// @param iters number of iterations to run the kernel
/// @return time taken to run the kernel iters times
float EdgeCentricDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters);

/// @brief Driver function to setup and call the vertex-centric push kernel to run BFS on the GPU.
/// @param adj_matrix graph to perform BFS on
/// @param[out] result_h result_h of the BFS
/// @param iters number of iterations to run the kernel
/// @return time taken to run the kernel iters times
float VertexCentricPushDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters);

/// @brief Driver function to setup and call the vertex-centric pull kernel to run BFS on the GPU.
/// @param adj_matrix graph to perform BFS on
/// @param[out] result_h result_h of the BFS
/// @param iters number of iterations to run the kernel
/// @return time taken to run the kernel iters times
float VertexCentricPullDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters);

/// @brief Driver function to setup and call the vertex-centric push-pull kernel to run BFS on the
/// GPU.
/// @param adj_matrix graph to perform BFS on
/// @param[out] result_h result_h of the BFS
/// @param iters number of iterations to run the kernel
/// @return time taken to run the kernel iters times
float VertexCentricPushPullDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters);

/// @brief Driver function to setup and call the vertex-centric push with frontiers kernel to run
/// BFS on the GPU.
/// @param adj_matrix graph to perform BFS on
/// @param[out] result_h result_h of the BFS
/// @param iters number of iterations to run the kernel
/// @return time taken to run the kernel iters times
float VertexCentricPushWithFrontiersDriver(AdjacencyMatrix& adj_matrix, int* result_h);

#endif  // CHAPTER_15_BFS_DRIVER_H
