#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "types/constants.h"
#include "types/types.h"

#define MAX_VAL 10000

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: <output_filename> <length of first array> <length of second array>\n");
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

    int m = atoi(argv[2]);
    int n = atoi(argv[3]);

    fprintf(file_ptr, "%d %d\n\n", m, n);

    std::vector<std::pair<int, int>> A{};
    std::vector<std::pair<int, int>> B{};

    for (int i = 0; i < m; ++i)
    {
        A.push_back(std::pair<int, int>{rand() % MAX_VAL, rand() % MAX_VAL});
    }
    for (int i = 0; i < n; ++i)
    {
        B.push_back(std::pair<int, int>{rand() % MAX_VAL, rand() % MAX_VAL});
    }
    auto firstIsLt = [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
        return a.first < b.first;
    };
    std::sort(std::begin(A), std::end(A), firstIsLt);
    std::sort(std::begin(B), std::end(B), firstIsLt);

    for (auto& elt : A)
    {
        fprintf(file_ptr, "%d %d ", elt.first, elt.second);
    }
    fprintf(file_ptr, "\n\n");
    for (auto& elt : B)
    {
        fprintf(file_ptr, "%d %d ", elt.first, elt.second);
    }

    fclose(file_ptr);
    return 0;
}
