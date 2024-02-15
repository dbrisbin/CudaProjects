/// @file bfs_driver.cu
/// @brief Definition of driver function declared in bfs_driver.h.

#include <stdio.h>
#include <algorithm>
#include "bfs.h"
#include "bfs_driver.h"
#include "types/adjacency_matrix.h"
#include "types/constants.h"
#include "types/graph_coo.h"

float BfsDriver(AdjacencyMatrix& adj_matrix, int* result_h, const int iters,
                const BfsKernel kernel_to_use)
{
    GraphCoo graph_coo_h{adj_matrix.ToCoo()};
    GraphCoo* graph_coo_d{};
    cudaMalloc((void**)&graph_coo_d, sizeof(GraphCoo));
    cudaMemcpy(graph_coo_d, &graph_coo_h, sizeof(GraphCoo), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&graph_coo_d->src, graph_coo_h.num_edges * sizeof(int));
    cudaMalloc((void**)&graph_coo_d->dst, graph_coo_h.num_edges * sizeof(int));
    cudaMalloc((void**)&graph_coo_d->val, graph_coo_h.num_edges * sizeof(int));
    cudaMemcpy(graph_coo_d->src, graph_coo_h.src, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_coo_d->dst, graph_coo_h.dst, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(graph_coo_d->val, graph_coo_h.val, graph_coo_h.num_edges * sizeof(int),
               cudaMemcpyHostToDevice);
    int* result_d{};

    cudaMalloc((void**)&result_d, adj_matrix.GetNumNnz() * sizeof(int));
    cudaMemset(result_d, -1, adj_matrix.GetNumNnz() * sizeof(int));
    cudaMemset(result_d, 0, sizeof(int));

    switch (kernel_to_use)
    {
        case BfsKernel::kEdgeCentric:
        {
            for (int i = 0; i < iters; ++i)
            {
                dim3 block_dim{SECTION_SIZE, 1, 1};
                dim3 grid_dim{static_cast<unsigned int>(
                                  ceil(static_cast<float>(adj_matrix.GetNumNnz()) / SECTION_SIZE)),
                              1, 1};
                int new_vertex_visited{0};
                int curr_level{1};
                do
                {
                    new_vertex_visited = 0;
                    cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                               cudaMemcpyHostToDevice);
                    EdgeCentricBFS<<<grid_dim, block_dim>>>(graph_coo_d, result_d,
                                                            &new_vertex_visited, curr_level);
                    cudaMemcpy(&new_vertex_visited, &new_vertex_visited, sizeof(int),
                               cudaMemcpyDeviceToHost);
                    ++curr_level;
                } while (new_vertex_visited != 0);

                cudaMemcpy(result_h, result_d, adj_matrix.GetNumNnz() * sizeof(int),
                           cudaMemcpyDeviceToHost);
            }
            break;
        }
        case BfsKernel::kNumKernels:
        default:
        {
            printf("Invalid kernel type\n");
            return -1.0f;
        }
    }

    cudaFree(graph_coo_d->src);
    cudaFree(graph_coo_d->dst);
    cudaFree(graph_coo_d->val);
    cudaFree(graph_coo_d);
    cudaFree(result_d);

    float total_time{0.0f};
    return total_time;
}
