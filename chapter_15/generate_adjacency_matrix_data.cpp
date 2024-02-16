/// @file generate_adjacency_matrix_data.cpp
/// @brief A utility to generate data to be used by BFS kernels. The file produces an uncompressed
/// adjacency matrix with the requested number of vertices. The matrix represents a graph with at
/// least one path to every vertex from each source vertex along with other edges randomly inserted.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <vector>

#define MAX_VAL 100.f

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage: <output_filename> <number of vertices> <sparseness factor>\n");
        return 1;
    }

    // First line of file should be the number of vertices in the graph.
    // Remaining lines are the data.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int n = atoi(argv[2]);
    double sparseness = atof(argv[3]);
    fprintf(file_ptr, "%d\n\n", n);

    std::vector<std::vector<int>> adj_matrix(n, std::vector<int>(n, 0));

    // Create a path from each source vertex to every other vertex.
    std::vector<int> dsts;
    for (int i = 0; i < n; ++i)
    {
        dsts.push_back(i);
    }
    std::random_shuffle(std::begin(dsts), std::end(dsts));

    int src{0};
    for (auto dst : dsts)
    {
        adj_matrix[src][dst] = 1;
        src = dst;
    }
    adj_matrix[src][0] = 1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, n);

    // Add other edges randomly.
    for (int i = 0; i < sparseness * n * n; ++i)
    {
        int src = static_cast<int>(dis(gen));
        int dst = static_cast<int>(dis(gen));
        adj_matrix[src][dst] = 1;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            fprintf(file_ptr, "%d ", adj_matrix[i][j]);
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);
    return 0;
}
