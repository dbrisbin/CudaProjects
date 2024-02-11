/// @file generate_sorting_data.cpp
/// @brief A utility to generate data to be used by sorting kernels. The file produces an unsorted
/// array of the provided length.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_VAL 100000

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: <output_filename> <length of array>\n");
        return 1;
    }

    // First line of file should be the lengths
    // Remaining lines are the data.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int length = atoi(argv[2]);

    fprintf(file_ptr, "%d\n\n", length);

    for (int i = 0; i < length; ++i)
    {
        fprintf(file_ptr, "%d ", rand() % MAX_VAL);
    }

    fclose(file_ptr);
    return 0;
}
