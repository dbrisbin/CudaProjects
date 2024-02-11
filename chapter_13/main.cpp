/// @file main.cpp
/// @brief Main function to call the kernel driver to call the appropriate kernel, as determined by
/// input argument, on an input file.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "sort_driver.h"
#include "sort_utils.h"
#include "types/constants.h"

/// @brief Compare two arrays for equality. Does not check that lengths are equal.
/// @param actual first array to check for equality
/// @param expected second array to check for equality
/// @param length length up to which to check for equality
/// @return true if the arrays are equal up to length length, false otherwise
/// @tparam Cont1 type of first container raw pointer or stl container.
/// @tparam Cont2 type of second container raw pointer or stl container.
template <typename Cont1, typename Cont2>
bool ArraysAreEqual(const Cont1& actual, const Cont2& expected, const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        if (actual[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

/// @brief Print a histogram to standard output.
/// @param arr array to print
/// @param length length of array to print
/// @tparam Cont container type be it raw pointer or stl container
template <typename Cont>
void PrintArr(const Cont& arr, const unsigned int length)
{
    for (unsigned int i = 0; i < length; ++i)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <input file> <kernel to use (0-%d)>.\n",
               static_cast<int>(SortKernel::kNumKernels) - 1);
        return 1;
    }

    // First line of file should contain the length of the data, subsequent lines should contain
    // the data to sort.
    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    SortKernel kernel_to_use{atoi(argv[2])};
    if (kernel_to_use >= SortKernel::kNumKernels)
    {
        printf("Please select a valid kernel to use!\n");
        return 1;
    }

    int length{};

    int scanf_result = fscanf(file_ptr, "%d", &length);

    unsigned int* input{};
    unsigned int* result{};

    input = (unsigned int*)malloc(length * sizeof(unsigned int));
    result = (unsigned int*)malloc(length * sizeof(unsigned int));

    for (int i = 0; i < length; ++i)
    {
        scanf_result = fscanf(file_ptr, "%u", &input[i]);
    }

    fclose(file_ptr);

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        return 1;
    }

    int iters{1};
    const auto time_to_compute = SortDriver(input, result, length, kernel_to_use, iters);
    printf("Took %.2f msec to compute %d iterations.", time_to_compute, iters);
    std::vector<unsigned int> expected(input, input + length);
    std::sort(std::begin(expected), std::end(expected), std::less<unsigned int>());
    if (!ArraysAreEqual(result, expected, length))
    {
        printf("\nResults are not equal!\n");
        printf("Expected:\n");
        if (length < 100)
        {
            PrintArr(expected, length);
        }
        else
        {
            PrintArr(expected, 100);
            printf("...\n");
        }

        printf("\nActual:\n");
        if (length < 100)
        {
            PrintArr(result, length);
        }
        else
        {
            PrintArr(result, 100);
            printf("...\n");
        }
    }
    else
    {
        printf("\nResult is correct!\n");
        if (length < 100)
        {
            PrintArr(result, length);
        }
        else
        {
            PrintArr(result, 100);
            printf("...\n");
        }
    }

    free(input);
    free(result);

    return 0;
}