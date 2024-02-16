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

GraphCsr AdjacencyMatrix::ToCsr() const
{
    GraphCsr csr{};
    const auto num_nnz{GetNumNnz()};
    csr.n = n_;
    csr.col_idx = new int[num_nnz];
    csr.val = new int[num_nnz];
    csr.row_ptrs = new int[n_ + 1];
    csr.row_ptrs[0] = 0;

    for (int i = 0; i < n_; ++i)
    {
        int nnz_per_row{0};
        for (int j = 0; j < n_; ++j)
        {
            if (graph_[i * n_ + j] != 0)
            {
                csr.col_idx[csr.row_ptrs[i] + nnz_per_row] = j;
                csr.val[csr.row_ptrs[i] + nnz_per_row] = graph_[i * n_ + j];
                ++nnz_per_row;
            }
        }
        csr.row_ptrs[i + 1] = csr.row_ptrs[i] + nnz_per_row;
    }
    return csr;
}

GraphCsc AdjacencyMatrix::ToCsc() const
{
    GraphCsc csc{};
    const auto num_nnz{GetNumNnz()};
    csc.n = n_;
    csc.row_idx = new int[num_nnz];
    csc.val = new int[num_nnz];
    csc.col_ptrs = new int[n_ + 1];
    csc.col_ptrs[0] = 0;

    for (int j = 0; j < n_; ++j)
    {
        int nnz_per_col{0};
        for (int i = 0; i < n_; ++i)
        {
            if (graph_[i * n_ + j] != 0)
            {
                csc.row_idx[csc.col_ptrs[j] + nnz_per_col] = i;
                csc.val[csc.col_ptrs[j] + nnz_per_col] = graph_[i * n_ + j];
                ++nnz_per_col;
            }
        }
        csc.col_ptrs[j + 1] = csc.col_ptrs[j] + nnz_per_col;
    }
    return csc;
}

int AdjacencyMatrix::GetNumNnz() const { return std::accumulate(graph_, graph_ + n_ * n_, 0); }