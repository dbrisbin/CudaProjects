/// @file bfs.h
/// @brief Declaration of available kernels for computing the sparse matrix-vector multiplication.

#ifndef CHAPTER_15_BFS_H
#define CHAPTER_15_BFS_H

#include <cuda_runtime.h>
#include "types/graph_coo.h"

/// @brief Edge-centric BFS kernel.
/// @param graph The graph to traverse stored in COO format.
/// @param[in,out] level The level of each vertex.
/// @param[out] new_vertex_visited The number of new vertices visited.
/// @param curr_level The current level.
__global__ void EdgeCentricBFS(const GraphCoo* graph, int* level, int* new_vertex_visited,
                               const int curr_level);

#endif  // CHAPTER_15_BFS_H