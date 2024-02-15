#include "adjacency_matrix.h"
#include <numeric>
#include "graph_coo.h"
#include "graph_csc.h"
#include "graph_csr.h"

AdjacencyMatrix::AdjacencyMatrix(const int* graph, const int n) : graph_(new int[n * n]), n_(n)
{
    for (int i = 0; i < n * n; ++i)
    {
        graph_[i] = graph[i];
    }
}

AdjacencyMatrix::~AdjacencyMatrix() { delete[] graph_; }

GraphCoo AdjacencyMatrix::ToCoo() const
{
    GraphCoo coo{};
    coo.num_edges = GetNumNnz();
    coo.src = new int[coo.num_edges];
    coo.dst = new int[coo.num_edges];
    coo.val = new int[coo.num_edges];

    int curr_edge{0};
    for (int i = 0; i < n_; ++i)
    {
        for (int j = 0; j < n_; ++j)
        {
            if (graph_[i * n_ + j] != 0)
            {
                coo.src[curr_edge] = i;
                coo.dst[curr_edge] = j;
                coo.val[curr_edge] = graph_[i * n_ + j];
                ++curr_edge;
            }
        }
    }
    return coo;
}

int AdjacencyMatrix::GetNumNnz() const { return std::accumulate(graph_, graph_ + n_ * n_, 0); }