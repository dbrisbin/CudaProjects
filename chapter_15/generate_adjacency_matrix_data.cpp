/// @file generate_adjacency_matrix_data.cpp
/// @brief A utility to generate data to be used by BFS kernels. The file produces an uncompressed
/// adjacency matrix with the requested number of vertices. The matrix represents a graph with at
/// least one path to every vertex from each source vertex along with other edges randomly inserted.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <random>

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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-MAX_VAL, MAX_VAL);

    fprintf(file_ptr, "%d\n\n", n);

    fclose(file_ptr);
    return 0;
}
