#ifndef CHAPTER_15_TYPES_GRAPH_COO_H
#define CHAPTER_15_TYPES_GRAPH_COO_H

#include <memory>

/// @brief Class to represent a graph in COO format
struct GraphCoo
{
    /// @brief Row indices of the non-zero elements of the matrix
    int* src;
    /// @brief Column indices of the non-zero elements of the matrix
    int* dst;
    /// @brief Values of the non-zero elements of the matrix
    int* val;
    /// @brief Number of edges in the graph
    int num_edges;
};

#endif  // CHAPTER_15_TYPES_GRAPH_COO_H