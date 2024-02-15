#ifndef CHAPTER_15_TYPES_GRAPH_CSR_H
#define CHAPTER_15_TYPES_GRAPH_CSR_H

/// @brief Class to represent a graph in CSR format
struct GraphCsr
{
    /// @brief Column indices of the non-zero elements of the matrix
    int* col_idx;
    /// @brief Values of the non-zero elements of the matrix
    int* val;
    /// @brief Row pointers of the matrix
    int* row_ptrs;
    /// @brief Number of rows of the matrix
    int n;
};

#endif  // CHAPTER_15_TYPES_GRAPH_CSR_H