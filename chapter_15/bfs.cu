/// @file bfs.cu
/// @brief Implementation of kernels declared in bfs.h.

#include <cuda_runtime.h>
#include "bfs.h"
#include "types/constants.h"
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

__global__ void VertexCentricPushBFSWithFrontiersPrivatized(
    const GraphCsr* graph, int* level, const int* prev_frontier, int* curr_frontier,
    const int n_prev_frontier, int* n_curr_frontier, const int curr_level)
{
    __shared__ int num_curr_frontier_s;
    __shared__ int curr_frontier_s[LOCAL_FRONTIER_CAPACITY];

    if (threadIdx.x == 0)
    {
        num_curr_frontier_s = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_prev_frontier)
    {
        int vertex = prev_frontier[i];
        for (int edge = graph->row_ptrs[vertex]; edge < graph->row_ptrs[vertex + 1]; ++edge)
        {
            int neighbor = graph->col_idx[edge];
            if (atomicCAS(&level[neighbor], -1, curr_level) == -1)
            {
                int j = atomicAdd(&num_curr_frontier_s, 1);
                if (j < LOCAL_FRONTIER_CAPACITY)
                {
                    curr_frontier_s[j] = neighbor;
                }
                else
                {
                    num_curr_frontier_s = LOCAL_FRONTIER_CAPACITY;
                    int k = atomicAdd(n_curr_frontier, 1);
                    curr_frontier[k] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    __shared__ int idx_curr_frontier_start_s;
    if (threadIdx.x == 0)
    {
        idx_curr_frontier_start_s = atomicAdd(n_curr_frontier, num_curr_frontier_s);
    }
    __syncthreads();

    for (int curr_frontier_idx = threadIdx.x; curr_frontier_idx < num_curr_frontier_s;
         curr_frontier_idx += blockDim.x)
    {
        curr_frontier[idx_curr_frontier_start_s + curr_frontier_idx] =
            curr_frontier_s[curr_frontier_idx];
    }
}

__global__ void SingleBlockVertexCentricPushBFSWithFrontiersPrivatized(
    const GraphCsr* graph, int* level, const int* prev_frontier, int* curr_frontier,
    const int n_prev_frontier, int* n_curr_frontier, int* curr_level)
{
    __shared__ int num_curr_frontier_s;
    __shared__ int curr_frontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ int num_prev_frontier_s;
    __shared__ int prev_frontier_s[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0)
    {
        num_prev_frontier_s = n_prev_frontier;
    }
    if (i < n_prev_frontier)
    {
        prev_frontier_s[i] = prev_frontier[i];
    }
    __syncthreads();

    do
    {
        if (threadIdx.x == 0)
        {
            num_curr_frontier_s = 0;
        }
        __syncthreads();

        if (i < num_prev_frontier_s)
        {
            int vertex = prev_frontier_s[i];
            for (int edge = graph->row_ptrs[vertex]; edge < graph->row_ptrs[vertex + 1]; ++edge)
            {
                int neighbor = graph->col_idx[edge];
                if (atomicCAS(&level[neighbor], -1, *curr_level) == -1)
                {
                    int j = atomicAdd(&num_curr_frontier_s, 1);
                    if (j < LOCAL_FRONTIER_CAPACITY)
                    {
                        curr_frontier_s[j] = neighbor;
                    }
                    else
                    {
                        num_curr_frontier_s = LOCAL_FRONTIER_CAPACITY;
                        int k = atomicAdd(n_curr_frontier, 1);
                        curr_frontier[k] = neighbor;
                    }
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_prev_frontier_s = num_curr_frontier_s;
        }
        __syncthreads();

        if (i < num_prev_frontier_s && num_prev_frontier_s < SECTION_SIZE)
        {
            if (threadIdx.x == 0)
            {
                atomicAdd(curr_level, 1);
            }
            prev_frontier_s[i] = curr_frontier_s[i];
        }
        __syncthreads();
    } while (num_prev_frontier_s < SECTION_SIZE && num_curr_frontier_s > 0);
    __syncthreads();

    __shared__ int idx_curr_frontier_start_s;
    if (threadIdx.x == 0)
    {
        idx_curr_frontier_start_s = atomicAdd(n_curr_frontier, num_curr_frontier_s);
    }
    __syncthreads();

    for (int curr_frontier_idx = threadIdx.x; curr_frontier_idx < num_curr_frontier_s;
         curr_frontier_idx += blockDim.x)
    {
        curr_frontier[idx_curr_frontier_start_s + curr_frontier_idx] =
            curr_frontier_s[curr_frontier_idx];
    }
}