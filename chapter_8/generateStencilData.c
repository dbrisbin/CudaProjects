#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    if (argc < 6)
    {
        printf(
            "Usage: <output_filename> <input matrix w> <input matrix h> <input matrix d> <kernel "
            "to use>.\n");
        return 1;
    }

    // First line of file should be the number of cols in N, number of rows in N, number of
    // layers in N, respectively.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Third line should be the integer representation of a StencilKernelToUse.
    // Remaining lines should be values for the matrices, N then c.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int depth = atoi(argv[4]);
    int kernel_num = atoi(argv[5]);

    if (kernel_num < 0 || kernel_num > 3)
    {
        printf(
            "Please input a valid kernel number to use (valid range 0-3, inclusive. Input: %d).\n",
            kernel_num);
        fclose(file_ptr);
        return 1;
    }

    fprintf(file_ptr, "%d %d %d\n", width, height, depth);
    fprintf(file_ptr, "0\n");
    fprintf(file_ptr, "%d\n", kernel_num);

    float max_matrix_val = 100.0;
    float denom = (float)(RAND_MAX / max_matrix_val);

    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                fprintf(file_ptr, "%f ", (float)rand() / denom);
            }
            fprintf(file_ptr, "\n");
        }
        fprintf(file_ptr, "\n");
    }

    for (int x = 0; x < 7; ++x)
    {
        fprintf(file_ptr, "%f ", (float)rand() / denom);
    }

    fclose(file_ptr);
    return 0;
}
