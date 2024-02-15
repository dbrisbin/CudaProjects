/// @file main.cpp
/// @brief Main function to call the kernel driver to call the appropriate kernel, as determined by
/// input argument, on an input file.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include "bfs_driver.h"
#include "types/adjacency_matrix.h"
#include "types/constants.h"

/// @brief Compute BFS on the CPU.
/// @param graph graph to perform BFS on
/// @param result result of the BFS
/// @param n number of nodes in the graph
void BfsCPU(const int* graph, int* result, const int n)
{
    (void)graph;
    (void)result;
    (void)n;
}

/// @brief Compares two vectors for equality.
/// @param vec1 first vector
/// @param vec2 second vector
/// @param n length of the vectors
/// @return true if the vectors are equal, false otherwise
bool VectorsAreEqual(const int* vec1, const int* vec2, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (vec1[i] != vec2[i])
        {
            return false;
        }
    }
    return true;
}

/// @brief Prints a vector to stdout.
/// @param vec vector to print
/// @param n length of the vector
void PrintVector(const int* vec, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n",
               static_cast<int>(BfsKernel::kNumKernels) - 1);
        return 1;
    }

    // First line of file should contain the dimensions of the dense matrix, subsequent lines should
    // contain the data to perform BFS.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    BfsKernel kernel_to_use{atoi(argv[2])};
    if (kernel_to_use >= BfsKernel::kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int n{};

    int scanf_result = fscanf(file_ptr, "%d", &n);

    int *graph{}, *result{}, *expected{};

    graph = new int[n * n];
    result = new int[n];
    expected = new int[n];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            scanf_result = fscanf(file_ptr, "%d", &graph[i * n + j]);
        }
    }
    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }
    AdjacencyMatrix adj_matrix(graph, n);
    int iters{1};

    const auto time_to_compute = BfsDriver(adj_matrix, result, iters, kernel_to_use);
    printf("Took %.2f msec to compute %d iterations.", time_to_compute, iters);

    BfsCPU(graph, expected, n);
    if (!VectorsAreEqual(result, expected, n))
    {
        printf("\nResults are not equal!\n");
        printf("Expected:\n");
        PrintVector(expected, n);

        printf("\nActual:\n");
        PrintVector(result, n);
    }
    else
    {
        printf("\nResult is correct!\n");
        PrintVector(result, n);
    }

    delete[] graph;
    delete[] result;
    delete[] expected;

    return 0;
}