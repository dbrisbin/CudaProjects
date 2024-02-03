#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    if (argc < 7)
    {
        printf(
            "Usage: <output_filename> <input matrix w> <input matrix h> <input matrix d> <filter "
            "radius> <kernel to use>.\n");
        return 1;
    }

    // First line of file should be the number of cols in N, number of rows in N, number of
    // layers in N, and the radius of the convolution kernel F, respectively.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Third line should be the integer representation of a KernelToUse.
    // Remaining lines should be values for the matrices, N then F.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int depth = atoi(argv[4]);
    int radius = atoi(argv[5]);
    int kernel_num = atoi(argv[6]);

    if (kernel_num < 0 || kernel_num > 3)
    {
        printf(
            "Please input a valid kernel number to use (valid range 0-3, inclusive. Input: %d).\n",
            kernel_num);
        fclose(file_ptr);
        return 1;
    }

    fprintf(file_ptr, "%d %d %d %d\n", width, height, depth, radius);
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

    for (int z = 0; z < 2 * radius + 1; ++z)
    {
        for (int y = 0; y < 2 * radius + 1; ++y)
        {
            for (int x = 0; x < 2 * radius + 1; ++x)
            {
                fprintf(file_ptr, "%f ", (float)rand() / denom);
            }
            fprintf(file_ptr, "\n");
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);
    return 0;
}
