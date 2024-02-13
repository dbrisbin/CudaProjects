/// @file generate_sparse_matrix_data.cpp
/// @brief A utility to generate data to be used by SpMV kernels. The file produces an uncompressed
/// sparse matrix of the requested dimensions and density factor.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <random>

#define MAX_VAL 100.f

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        printf(
            "Usage: <output_filename> <number of rows> <number of columns> <sparseness factor>\n");
        return 1;
    }

    // First line of file should be the dimensions of the dense matrix/length of the vector.
    // Remaining lines are the data.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    double sparseness = atof(argv[4]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-MAX_VAL, MAX_VAL);

    fprintf(file_ptr, "%d %d\n\n", m, n);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if ((static_cast<int>(std::fabs(dis(gen))) % 100) < sparseness * 100)
            {
                fprintf(file_ptr, "%.2f ", dis(gen));
            }
            else
            {
                fprintf(file_ptr, "0.0 ");
            }
        }
        fprintf(file_ptr, "\n");
    }

    fprintf(file_ptr, "\n");
    for (int i = 0; i < n; ++i)
    {
        fprintf(file_ptr, "%f ", dis(gen));
    }

    fclose(file_ptr);
    return 0;
}
