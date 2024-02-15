#ifndef CHAPTER_15_TYPES_GRAPH_CSC_H
#define CHAPTER_15_TYPES_GRAPH_CSC_H

/// @brief Class to represent a graph in CSC format
struct GraphCsc
{
    /// @brief Row indices of the non-zero elements of the matrix
    int* row_idx;
    /// @brief Values of the non-zero elements of the matrix
    int* val;
    /// @brief Column pointers of the matrix
    int* col_ptrs;
    /// @brief Number of columns of the matrix
    int n;
};

#endif  // CHAPTER_15_TYPES_GRAPH_CSC_H