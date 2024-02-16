/// @file bfs.h
/// @brief Declaration of available kernels for computing the sparse matrix-vector multiplication.

#ifndef CHAPTER_15_BFS_H
#define CHAPTER_15_BFS_H

#include <cuda_runtime.h>
#include "types/graph_coo.h"
#include "types/graph_csc.h"
#include "types/graph_csr.h"

/// @brief Edge-centric BFS kernel.
/// @param graph The graph to traverse stored in COO format.
/// @param[in,out] level The level of each vertex.
/// @param[out] new_vertex_visited 1 if a new vertex was visited, 0 otherwise.
/// @param curr_level The current level.
__global__ void EdgeCentricBFS(const GraphCoo* graph, int* level, int* new_vertex_visited,
                               const int curr_level);

/// @brief Vertex-centric top-down BFS kernel.
/// @param graph The graph to traverse stored in CSR format.
/// @param[in,out] level The level of each vertex.
/// @param[out] new_vertex_visited 1 if a new vertex was visited, 0 otherwise.
/// @param curr_level The current level.
__global__ void VertexCentricPushBFS(const GraphCsr* graph, int* level, int* new_vertex_visited,
                                     const int curr_level);

/// @brief Vertex-centric bottom-up BFS kernel.
/// @param graph The graph to traverse stored in CSC format.
/// @param[in,out] level The level of each vertex.
/// @param[out] new_vertex_visited 1 if a new vertex was visited, 0 otherwise.
/// @param curr_level The current level.
__global__ void VertexCentricPullBFS(const GraphCsc* graph, int* level, int* new_vertex_visited,
                                     const int curr_level);

/// @brief Vertex-centric push BFS kernel with frontiers.
/// @param graph The graph to traverse stored in CSR format.
/// @param[in,out] level The level of each vertex.
/// @param prev_frontier The previous frontier.
/// @param[out] curr_frontier The current frontier.
/// @param n_prev_frontier The number of vertices in the previous frontier.
/// @param[out] n_curr_frontier The number of vertices in the current frontier.
/// @param curr_level The current level.
__global__ void VertexCentricPushBFSWithFrontiers(const GraphCsr* graph, int* level,
                                                  const int* prev_frontier, int* curr_frontier,
                                                  const int n_prev_frontier, int* n_curr_frontier,
                                                  const int curr_level);

/// @brief Vertex-centric push BFS kernel with frontiers.
/// @param graph The graph to traverse stored in CSR format.
/// @param[in,out] level The level of each vertex.
/// @param prev_frontier The previous frontier.
/// @param[out] curr_frontier The current frontier.
/// @param n_prev_frontier The number of vertices in the previous frontier.
/// @param[out] n_curr_frontier The number of vertices in the current frontier.
/// @param curr_level The current level.
__global__ void VertexCentricPushBFSWithFrontiersPrivatized(
    const GraphCsr* graph, int* level, const int* prev_frontier, int* curr_frontier,
    const int n_prev_frontier, int* n_curr_frontier, const int curr_level);

/// @brief Single-block vertex-centric push BFS kernel with frontiers.
/// @param graph The graph to traverse stored in CSR format.
/// @param[in,out] level The level of each vertex.
/// @param prev_frontier The previous frontier.
/// @param[out] curr_frontier The current frontier.
/// @param n_prev_frontier The number of vertices in the previous frontier.
/// @param[out] n_curr_frontier The number of vertices in the current frontier.
/// @param curr_level The current level.
__global__ void SingleBlockVertexCentricPushBFSWithFrontiersPrivatized(
    const GraphCsr* graph, int* level, const int* prev_frontier, int* curr_frontier,
    const int n_prev_frontier, int* n_curr_frontier, int* curr_level);

#endif  // CHAPTER_15_BFS_H