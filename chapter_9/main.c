#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "parallelHistogramDriver.h"
#include "types/constants.h"

bool histogramsAreEqual(const int* hist, const int* expected_hist, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        if (hist[i] != expected_hist[i])
        {
            return false;
        }
    }
    return true;
}

void printHistogram(const int* hist, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        printf("%d ", hist[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-5)>.\n");
        return 1;
    }

    // First line of file should contain the length of the data, Subsequent lines should contain
    // the data to generate histogram for.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }
    enum parallelHistogramKernelToUse kernel_to_use = atoi(argv[2]);

    if (kernel_to_use >= kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int length;
    int scanf_result = fscanf(file_ptr, "%d", &length);
    int hist[NUM_BINS];
    int* data;
    data = (int*)malloc(length * sizeof(int));
    for (int i = 0; i < length; ++i)
    {
        scanf_result = fscanf(file_ptr, "%d", &data[i]);
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    printf("Computing parallel histogram:\n");
    int iters = 100;
    // compute the histogram.
    float time_to_compute_histogram =
        parallelHistogramDriver(data, length, hist, kernel_to_use, iters);

    printf("Took %.1f msec for %d iterations.\n", time_to_compute_histogram, iters);

    int expected_hist[NUM_BINS];
    (void)parallelHistogramDriver(data, length, expected_hist, kBasic, 1);
    if (!histogramsAreEqual(hist, expected_hist, NUM_BINS))
    {
        printf("\nHistograms are not equal!\n");
        printf("Expected histogram:\n");
        printHistogram(expected_hist, NUM_BINS);

        printf("\nActual histogram:\n");
        printHistogram(hist, NUM_BINS);
    }
    else
    {
        printf("\nHistogram is correct!\n");
        printHistogram(hist, NUM_BINS);
    }
    return 0;
}