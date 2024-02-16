/// @file main.cpp
/// @brief Main function to call the kernel driver to call the appropriate kernel, as determined by
/// input argument, on an input file.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
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
    std::fill(result, result + n, -1);
    std::vector<int> queue;
    queue.push_back(0);
    result[0] = 0;
    int curr_level{1};
    while (!queue.empty())
    {
        std::vector<int> frontier(std::begin(queue), std::end(queue));
        queue.clear();
        for (auto src : frontier)
        {
            for (int i = 0; i < n; ++i)
            {
                if (graph[src * n + i] && result[i] == -1)
                {
                    queue.push_back(i);
                    result[i] = curr_level;
                }
            }
        }
        ++curr_level;
    }
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
    expected = new int[n];
    result = new int[n];

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
    int iters{100};

    float time_to_compute{};
    switch (kernel_to_use)
    {
        case BfsKernel::kEdgeCentric:
            time_to_compute = EdgeCentricDriver(adj_matrix, result, iters);
            break;
        case BfsKernel::kVertexCentricPush:
            time_to_compute = VertexCentricPushDriver(adj_matrix, result, iters);
            break;
        case BfsKernel::kVertexCentricPull:
            time_to_compute = VertexCentricPullDriver(adj_matrix, result, iters);
            break;
        case BfsKernel::kVertexCentricPushPull:
            time_to_compute = VertexCentricPushPullDriver(adj_matrix, result, iters);
            break;
        case BfsKernel::kVertexCentricPushWithFrontier:
            time_to_compute =
                VertexCentricPushWithFrontiersDriver(adj_matrix, result, iters, false);
            break;
        case BfsKernel::kVertexCentricPushWithFrontierPrivatized:
            time_to_compute = VertexCentricPushWithFrontiersDriver(adj_matrix, result, iters, true);
            break;
        case BfsKernel::kSingleBlockVertexCentricPushFrontierPrivatized:
            time_to_compute = SingleBlockVertexCentricPushDriver(adj_matrix, result, iters);
            break;
        case BfsKernel::kNumKernels:
        default:
            printf("Invalid kernel selected. Exiting.\n");
            break;
    }

    printf("Took %.2f msec to compute %d iterations on GPU.\n", time_to_compute, iters);

    auto start = std::chrono::high_resolution_clock::now();
    BfsCPU(graph, expected, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Took %.2f msec to compute 1 iteration on CPU.\n", duration.count());

    if (!VectorsAreEqual(result, expected, n))
    {
        printf("\nResults are not equal (result may be truncated)!\n");
        printf("Expected:\n");
        PrintVector(expected, std::min(n, 100));

        printf("\nActual:\n");
        PrintVector(result, std::min(n, 100));
    }
    else
    {
        printf("\nResult is correct (result may be truncated)!\n");
        PrintVector(result, std::min(n, 100));
    }

    delete[] expected;
    delete[] result;
    delete[] graph;

    return 0;
}