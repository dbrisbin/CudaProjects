#ifndef CHAPTER_15_TYPES_ADJACENCY_MATRIX_H
#define CHAPTER_15_TYPES_ADJACENCY_MATRIX_H

#include "graph_coo.h"
#include "graph_csc.h"
#include "graph_csr.h"

class AdjacencyMatrix
{
   public:
    AdjacencyMatrix(const int* graph, const int n);
    ~AdjacencyMatrix();

    int GetNumNnz() const;

    GraphCoo ToCoo() const;
    GraphCsr ToCsr() const;
    GraphCsc ToCsc() const;

   private:
    int* graph_;
    const int n_;
};
#endif  // CHAPTER_15_TYPES_ADJACENCY_MATRIX_H