/// @file bfs_driver.cu
/// @brief Definition of driver function declared in bfs_driver.h.

#include <stdio.h>
#include <algorithm>
#include "bfs.h"
#include "bfs_driver.h"
#include "types/adjacency_matrix.h"
#include "types/constants.h"
#include "types/graph_coo.h"

float EdgeCentricDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters)
{
    GraphCoo graph_coo_h{adj_matrix.ToCoo()};
    GraphCoo graph_coo_h_to_copy_to_d{};
    graph_coo_h_to_copy_to_d.num_edges = graph_coo_h.num_edges;
    cudaMalloc((void**)&graph_coo_h_to_copy_to_d.src, graph_coo_h.num_edges * sizeof(int));
    cudaMalloc((void**)&graph_coo_h_to_copy_to_d.dst, graph_coo_h.num_edges * sizeof(int));
    cudaMalloc((void**)&graph_coo_h_to_copy_to_d.val, graph_coo_h.num_edges * sizeof(int));
    cudaMemcpy(graph_coo_h_to_copy_to_d.src, graph_coo_h.src, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_coo_h_to_copy_to_d.dst, graph_coo_h.dst, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_coo_h_to_copy_to_d.val, graph_coo_h.val, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCoo* graph_coo_d{};
    cudaMalloc((void**)&graph_coo_d, sizeof(GraphCoo));
    cudaMemcpy(graph_coo_d, &graph_coo_h_to_copy_to_d, sizeof(GraphCoo), cudaMemcpyHostToDevice);
    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetN() * sizeof(int));

    float time{};
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i)
    {
        cudaMemset(result_d, -1, adj_matrix.GetN() * sizeof(int));
        cudaMemset(result_d, 0, sizeof(int));
        dim3 block_dim{SECTION_SIZE, 1, 1};
        dim3 grid_dim{static_cast<unsigned int>(
                          ceil(static_cast<float>(graph_coo_h.num_edges) / SECTION_SIZE)),
                      1, 1};
        int new_vertex_visited{0};
        int curr_level{1};
        do
        {
            new_vertex_visited = 0;
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyHostToDevice);
            EdgeCentricBFS<<<grid_dim, block_dim>>>(graph_coo_d, result_d, &new_vertex_visited,
                                                    curr_level);
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyDeviceToHost);
            ++curr_level;
        } while (new_vertex_visited != 0);

        cudaMemcpy(result_h, result_d, adj_matrix.GetN() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(graph_coo_d);
    cudaFree(graph_coo_h_to_copy_to_d.src);
    cudaFree(graph_coo_h_to_copy_to_d.dst);
    cudaFree(graph_coo_h_to_copy_to_d.val);
    cudaFree(result_d);

    delete[] graph_coo_h.src;
    delete[] graph_coo_h.dst;
    delete[] graph_coo_h.val;

    return time;
}

float VertexCentricPushDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters)
{
    GraphCsr graph_csr_h{adj_matrix.ToCsr()};
    GraphCsr graph_csr_h_to_copy_to_d{};
    graph_csr_h_to_copy_to_d.n = graph_csr_h.n;
    const int num_edges{graph_csr_h.row_ptrs[graph_csr_h.n]};
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.row_ptrs, (graph_csr_h.n + 1) * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.col_idx, num_edges * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.val, num_edges * sizeof(int));
    cudaMemcpy(graph_csr_h_to_copy_to_d.row_ptrs, graph_csr_h.row_ptrs,
               (graph_csr_h.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.col_idx, graph_csr_h.col_idx, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.val, graph_csr_h.val, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCsr* graph_csr_d{};
    cudaMalloc((void**)&graph_csr_d, sizeof(GraphCsr));
    cudaMemcpy(graph_csr_d, &graph_csr_h_to_copy_to_d, sizeof(GraphCsr), cudaMemcpyHostToDevice);
    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetN() * sizeof(int));

    float time{};
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i)
    {
        cudaMemset(result_d, -1, adj_matrix.GetN() * sizeof(int));
        cudaMemset(result_d, 0, sizeof(int));
        dim3 block_dim{SECTION_SIZE, 1, 1};
        dim3 grid_dim{
            static_cast<unsigned int>(ceil(static_cast<float>(graph_csr_h.n) / SECTION_SIZE)), 1,
            1};
        int new_vertex_visited{0};
        int curr_level{1};
        do
        {
            new_vertex_visited = 0;
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyHostToDevice);
            VertexCentricPushBFS<<<grid_dim, block_dim>>>(graph_csr_d, result_d,
                                                          &new_vertex_visited, curr_level);
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyDeviceToHost);
            ++curr_level;
        } while (new_vertex_visited != 0);

        cudaMemcpy(result_h, result_d, adj_matrix.GetN() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(graph_csr_d);
    cudaFree(graph_csr_h_to_copy_to_d.row_ptrs);
    cudaFree(graph_csr_h_to_copy_to_d.col_idx);
    cudaFree(graph_csr_h_to_copy_to_d.val);
    cudaFree(result_d);

    delete[] graph_csr_h.row_ptrs;
    delete[] graph_csr_h.col_idx;
    delete[] graph_csr_h.val;

    return time;
}

float VertexCentricPullDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters)
{
    GraphCsc graph_csc_h{adj_matrix.ToCsc()};
    GraphCsc graph_csc_h_to_copy_to_d{};
    graph_csc_h_to_copy_to_d.n = graph_csc_h.n;
    const int num_edges{graph_csc_h.col_ptrs[graph_csc_h.n]};
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.col_ptrs, (graph_csc_h.n + 1) * sizeof(int));
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.row_idx, num_edges * sizeof(int));
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.val, num_edges * sizeof(int));
    cudaMemcpy(graph_csc_h_to_copy_to_d.col_ptrs, graph_csc_h.col_ptrs,
               (graph_csc_h.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csc_h_to_copy_to_d.row_idx, graph_csc_h.row_idx, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csc_h_to_copy_to_d.val, graph_csc_h.val, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCsc* graph_csc_d{};
    cudaMalloc((void**)&graph_csc_d, sizeof(GraphCsc));
    cudaMemcpy(graph_csc_d, &graph_csc_h_to_copy_to_d, sizeof(GraphCsc), cudaMemcpyHostToDevice);
    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetN() * sizeof(int));

    float time{};
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i)
    {
        cudaMemset(result_d, -1, adj_matrix.GetN() * sizeof(int));
        cudaMemset(result_d, 0, sizeof(int));
        dim3 block_dim{SECTION_SIZE, 1, 1};
        dim3 grid_dim{
            static_cast<unsigned int>(ceil(static_cast<float>(graph_csc_h.n) / SECTION_SIZE)), 1,
            1};
        int new_vertex_visited{0};
        int curr_level{1};
        do
        {
            new_vertex_visited = 0;
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyHostToDevice);
            VertexCentricPullBFS<<<grid_dim, block_dim>>>(graph_csc_d, result_d,
                                                          &new_vertex_visited, curr_level);
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyDeviceToHost);
            ++curr_level;
        } while (new_vertex_visited != 0);

        cudaMemcpy(result_h, result_d, adj_matrix.GetN() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(graph_csc_d);
    cudaFree(graph_csc_h_to_copy_to_d.col_ptrs);
    cudaFree(graph_csc_h_to_copy_to_d.row_idx);
    cudaFree(graph_csc_h_to_copy_to_d.val);
    cudaFree(result_d);

    delete[] graph_csc_h.col_ptrs;
    delete[] graph_csc_h.row_idx;
    delete[] graph_csc_h.val;

    return time;
}

float VertexCentricPushPullDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters)
{
    GraphCsc graph_csc_h{adj_matrix.ToCsc()};
    GraphCsc graph_csc_h_to_copy_to_d{};
    graph_csc_h_to_copy_to_d.n = graph_csc_h.n;
    const int num_edges{graph_csc_h.col_ptrs[graph_csc_h.n]};
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.col_ptrs, (graph_csc_h.n + 1) * sizeof(int));
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.row_idx, num_edges * sizeof(int));
    cudaMalloc((void**)&graph_csc_h_to_copy_to_d.val, num_edges * sizeof(int));
    cudaMemcpy(graph_csc_h_to_copy_to_d.col_ptrs, graph_csc_h.col_ptrs,
               (graph_csc_h.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csc_h_to_copy_to_d.row_idx, graph_csc_h.row_idx, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csc_h_to_copy_to_d.val, graph_csc_h.val, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCsc* graph_csc_d{};
    cudaMalloc((void**)&graph_csc_d, sizeof(GraphCsc));
    cudaMemcpy(graph_csc_d, &graph_csc_h_to_copy_to_d, sizeof(GraphCsc), cudaMemcpyHostToDevice);

    GraphCsr graph_csr_h{adj_matrix.ToCsr()};
    GraphCsr graph_csr_h_to_copy_to_d{};
    graph_csr_h_to_copy_to_d.n = graph_csr_h.n;
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.row_ptrs, (graph_csr_h.n + 1) * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.col_idx, num_edges * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.val, num_edges * sizeof(int));
    cudaMemcpy(graph_csr_h_to_copy_to_d.row_ptrs, graph_csr_h.row_ptrs,
               (graph_csr_h.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.col_idx, graph_csr_h.col_idx, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.val, graph_csr_h.val, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCsr* graph_csr_d{};
    cudaMalloc((void**)&graph_csr_d, sizeof(GraphCsr));
    cudaMemcpy(graph_csr_d, &graph_csr_h_to_copy_to_d, sizeof(GraphCsr), cudaMemcpyHostToDevice);
    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetN() * sizeof(int));

    float time{};
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i)
    {
        cudaMemset(result_d, -1, adj_matrix.GetN() * sizeof(int));
        cudaMemset(result_d, 0, sizeof(int));

        dim3 block_dim{SECTION_SIZE, 1, 1};
        dim3 grid_dim{
            static_cast<unsigned int>(ceil(static_cast<float>(graph_csc_h.n) / SECTION_SIZE)), 1,
            1};
        int new_vertex_visited{0};
        int total_vertices_visited{0};
        int curr_level{1};
        do
        {
            new_vertex_visited = 0;
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyHostToDevice);
            if (total_vertices_visited < adj_matrix.GetN() / 2)
            {
                VertexCentricPushBFS<<<grid_dim, block_dim>>>(graph_csr_d, result_d,
                                                              &new_vertex_visited, curr_level);
            }
            else
            {
                VertexCentricPullBFS<<<grid_dim, block_dim>>>(graph_csc_d, result_d,
                                                              &new_vertex_visited, curr_level);
            }
            cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                       cudaMemcpyDeviceToHost);
            ++curr_level;
            total_vertices_visited += new_vertex_visited;
        } while (new_vertex_visited != 0);

        cudaMemcpy(result_h, result_d, adj_matrix.GetN() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(graph_csc_d);
    cudaFree(graph_csc_h_to_copy_to_d.col_ptrs);
    cudaFree(graph_csc_h_to_copy_to_d.row_idx);
    cudaFree(graph_csc_h_to_copy_to_d.val);
    cudaFree(graph_csr_d);
    cudaFree(graph_csr_h_to_copy_to_d.row_ptrs);
    cudaFree(graph_csr_h_to_copy_to_d.col_idx);
    cudaFree(graph_csr_h_to_copy_to_d.val);
    cudaFree(result_d);

    delete[] graph_csr_h.row_ptrs;
    delete[] graph_csr_h.col_idx;
    delete[] graph_csr_h.val;
    delete[] graph_csc_h.col_ptrs;
    delete[] graph_csc_h.row_idx;
    delete[] graph_csc_h.val;

    return time;
}

float VertexCentricPushWithFrontiersDriver(AdjacencyMatrix& adj_matrix, int* result_h)
{
    GraphCsr graph_csr_h{adj_matrix.ToCsr()};
    GraphCsr graph_csr_h_to_copy_to_d{};
    graph_csr_h_to_copy_to_d.n = graph_csr_h.n;
    const int num_edges{graph_csr_h.row_ptrs[graph_csr_h.n]};
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.row_ptrs, (graph_csr_h.n + 1) * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.col_idx, num_edges * sizeof(int));
    cudaMalloc((void**)&graph_csr_h_to_copy_to_d.val, num_edges * sizeof(int));
    cudaMemcpy(graph_csr_h_to_copy_to_d.row_ptrs, graph_csr_h.row_ptrs,
               (graph_csr_h.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.col_idx, graph_csr_h.col_idx, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_csr_h_to_copy_to_d.val, graph_csr_h.val, num_edges * sizeof(int),
               cudaMemcpyHostToDevice);

    GraphCsr* graph_csr_d{};
    cudaMalloc((void**)&graph_csr_d, sizeof(GraphCsr));
    cudaMemcpy(graph_csr_d, &graph_csr_h_to_copy_to_d, sizeof(GraphCsr), cudaMemcpyHostToDevice);

    int* prev_frontier_d{};
    int* curr_frontier_d{};

    cudaMalloc((void**)&prev_frontier_d, adj_matrix.GetN() * sizeof(int));
    cudaMalloc((void**)&curr_frontier_d, adj_matrix.GetN() * sizeof(int));

    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetN() * sizeof(int));

    float time{};
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int n_prev_frontier{1};
    int* n_curr_frontier{};
    cudaMalloc((void**)&n_curr_frontier, sizeof(int));
    cudaMemset(n_curr_frontier, 0, sizeof(int));
    cudaMemset(prev_frontier_d, 0, sizeof(int));
    cudaMemset(result_d, -1, adj_matrix.GetN() * sizeof(int));
    cudaMemset(result_d, 0, sizeof(int));
    dim3 block_dim{SECTION_SIZE, 1, 1};
    dim3 grid_dim{static_cast<unsigned int>(ceil(static_cast<float>(graph_csr_h.n) / SECTION_SIZE)),
                  1, 1};
    int curr_level{1};
    do
    {
        VertexCentricPushBFSWithFrontiers<<<grid_dim, block_dim>>>(
            graph_csr_d, result_d, prev_frontier_d, curr_frontier_d, n_prev_frontier,
            n_curr_frontier, curr_level);
        cudaMemcpy(&n_prev_frontier, n_curr_frontier, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_frontier_d, curr_frontier_d, n_prev_frontier * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        cudaMemset(n_curr_frontier, 0, sizeof(int));

        ++curr_level;
    } while (n_prev_frontier != 0);

    cudaMemcpy(result_h, result_d, adj_matrix.GetN() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(graph_csr_d);
    cudaFree(graph_csr_h_to_copy_to_d.row_ptrs);
    cudaFree(graph_csr_h_to_copy_to_d.col_idx);
    cudaFree(graph_csr_h_to_copy_to_d.val);
    cudaFree(result_d);
    cudaFree(prev_frontier_d);
    cudaFree(curr_frontier_d);
    cudaFree(n_curr_frontier);

    delete[] graph_csr_h.row_ptrs;
    delete[] graph_csr_h.col_idx;
    delete[] graph_csr_h.val;

    return time;
}