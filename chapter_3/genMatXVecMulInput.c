#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Please provide an output file name and size.\n");
        return 1;
    }

    // First line of file should be the width/height of the square matrices.
    // Second line should be 1 or 0 indicating whether or not to print the matrices.
    // Remaining lines should be values for the matrices.
    FILE *file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int width = atoi(argv[2]);

    fprintf(file_ptr, "%d\n", width);
    fprintf(file_ptr, "0\n");

    float max_matrix_val = 100.0;
    float denom = (float)(RAND_MAX / max_matrix_val);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width - 1; ++j)
        {
            fprintf(file_ptr, "%f ", (float)rand() / denom);
        }
        fprintf(file_ptr, "%f\n", (float)rand() / denom);
    }
    for (int j = 0; j < width; ++j)
    {
        fprintf(file_ptr, "%f\n", (float)rand() / denom);
    }
    fclose(file_ptr);
    return 0;
}
