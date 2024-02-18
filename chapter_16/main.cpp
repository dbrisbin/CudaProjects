/// @file main.cpp
/// @brief Main function

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "le_net_5.h"
#include "types/constants.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: <input file>>.\n");
        return 1;
    }

    FILE* file_ptr = fopen(argv[1], "r");
    if (file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    LeNet5 le_net_5{};
    auto output_size = le_net_5.DetermineOutputSize();
    float* Y = new float[output_size];
    float* T = new float[output_size];
    float* X = new float[32 * 32];

    int n{};

    int scanf_result = fscanf(file_ptr, "%d", &n);
    fclose(file_ptr);

    for (int i = 0; i < 32 * 32; ++i)
    {
        scanf_result = fscanf(file_ptr, "%f", &X[i]);
    }

    for (int i = 0; i < output_size; ++i)
    {
        scanf_result = fscanf(file_ptr, "%f", &T[i]);
    }

    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        delete[] Y;
        delete[] T;
        delete[] X;

        return 1;
    }
    float delta_loss{1.0f};
    float prev_loss{0.0f};
    int iter{0};
    while (delta_loss > 0.0001f && iter < 1000)
    {
        le_net_5.Forward(X, Y);
        auto loss = le_net_5.ComputeLoss(Y, T);
        printf("Loss: %f\n", loss);
        delta_loss = abs(prev_loss - loss);
        prev_loss = loss;
        le_net_5.Backward(X, Y, T);
        iter++;
    }

    le_net_5.Forward(X, Y);
    auto loss = le_net_5.ComputeLoss(Y, T);
    printf("Loss: %f\n", loss);

    le_net_5.Backward(X, Y, T);

    delete[] Y;
    delete[] T;
    delete[] X;

    return 0;
}