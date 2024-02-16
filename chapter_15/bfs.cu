/// @file bfs.cu
/// @brief Implementation of kernels declared in bfs.h.

#include <cuda_runtime.h>
#include "bfs.h"
#include "types/graph_coo.h"
#include "types/graph_csc.h"
#include "types/graph_csr.h"

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

__global__ void VertexCentricPushBFS(const GraphCsr* graph, int* level, int* new_vertex_visited,
                                     const int curr_level)
{
    __shared__ int new_vertex_visited_s;
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    int num_new_vertices_visited{0};
    if (threadIdx.x == 0)
    {
        new_vertex_visited_s = 0;
    }
    __syncthreads();
    if (vertex < graph->n)
    {
        if (level[vertex] == curr_level - 1)
        {
            for (int edge = graph->row_ptrs[vertex]; edge < graph->row_ptrs[vertex + 1]; ++edge)
            {
                int neighbor = graph->col_idx[edge];
                if (level[neighbor] == -1)
                {
                    level[neighbor] = curr_level;
                    ++num_new_vertices_visited;
                }
            }
        }
    }
    atomicAdd(&new_vertex_visited_s, num_new_vertices_visited);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(new_vertex_visited, new_vertex_visited_s);
    }
}

__global__ void VertexCentricPullBFS(const GraphCsc* graph, int* level, int* new_vertex_visited,
                                     const int curr_level)
{
    __shared__ int new_vertex_visited_s;
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0)
    {
        new_vertex_visited_s = 0;
    }
    __syncthreads();
    if (vertex < graph->n)
    {
        if (level[vertex] == -1)
        {
            for (int edge = graph->col_ptrs[vertex]; edge < graph->col_ptrs[vertex + 1]; ++edge)
            {
                int neighbor = graph->row_idx[edge];
                if (level[neighbor] == curr_level - 1)
                {
                    level[vertex] = curr_level;
                    atomicAdd(&new_vertex_visited_s, 1);
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(new_vertex_visited, new_vertex_visited_s);
    }
}

__global__ void VertexCentricPushBFSWithFrontiers(const GraphCsr* graph, int* level,
                                                  const int* prev_frontier, int* curr_frontier,
                                                  const int n_prev_frontier, int* n_curr_frontier,
                                                  const int curr_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_prev_frontier)
    {
        int vertex = prev_frontier[i];
        for (int edge = graph->row_ptrs[vertex]; edge < graph->row_ptrs[vertex + 1]; ++edge)
        {
            int neighbor = graph->col_idx[edge];
            if (atomicCAS(&level[neighbor], -1, curr_level) == -1)
            {
                int j = atomicAdd(n_curr_frontier, 1);
                curr_frontier[j] = neighbor;
            }
        }
    }
}