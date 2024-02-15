/// @file bfs.cu
/// @brief Implementation of kernels declared in bfs.h.

#include <cuda_runtime.h>
#include "bfs.h"
#include "types/graph_coo.h"

__global__ void EdgeCentricBFS(const GraphCoo* graph, int* level, int* new_vertex_visited,
                               const int curr_level)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < graph->num_edges)
    {
        int vertex = graph->src[edge];
        if (level[vertex] == curr_level - 1)
        {
            int neighbor = graph->dst[edge];
            if (level[neighbor] == -1)
            {
                level[neighbor] = curr_level;
                *new_vertex_visited = 1;
            }
        }
    }
}