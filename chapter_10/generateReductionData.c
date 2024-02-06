#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "types/constants.h"

#define MAX_VAL 5

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <output_filename> <length>\n");
        return 1;
    }

    // First line of file should be the length
    // Remaining lines are the data.
    FILE* file_ptr = fopen(argv[1], "w");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    int length = atoi(argv[2]);

    fprintf(file_ptr, "%d", length);

    for (int i = 0; i < length; ++i)
    {
        if (i % 1000 == 0)
        {
            fprintf(file_ptr, "\n");
        }
        fprintf(file_ptr, "%" FORMAT_TYPE " ", rand() % MAX_VAL);
    }
    fclose(file_ptr);
    return 0;
}
