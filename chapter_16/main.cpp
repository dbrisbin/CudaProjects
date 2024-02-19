/// @file main.cpp
/// @brief Main function

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include "le_net_5_modified.h"
#include "types/constants.h"
#include "utils.h"

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage: <data input file> <data labels file> <number of samples to load>\n");
        return 1;
    }

    FILE* data_file_ptr = fopen(argv[1], "r");
    if (data_file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[1]);
        return 1;
    }

    FILE* labels_file_ptr = fopen(argv[2], "r");
    if (labels_file_ptr == NULL)
    {
        printf("No such file %s.\n", argv[2]);
        return 1;
    }

    const int num_samples{atoi(argv[3])};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-.5, .5);
    const int batch_size{std::min(32, num_samples)};
    LeNet5Modified le_net_5(batch_size, gen, dis);

    auto input_size = le_net_5.DetermineInputSize();
    auto output_size = le_net_5.DetermineOutputSize();
    unsigned char* T = new unsigned char[num_samples];
    float* X = new float[num_samples * input_size];
    float* Y = new float[output_size];

    int scanf_result{};

    fseek(labels_file_ptr, 16, SEEK_SET);
    char temp;
    for (int i = 0; i < num_samples * input_size; ++i)
    {
        scanf_result = fscanf(data_file_ptr, "%c", &temp);
        X[i] = static_cast<float>(reinterpret_cast<unsigned char&>(temp)) / 255.0f;
    }
    fclose(data_file_ptr);
    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        delete[] Y;
        delete[] T;
        delete[] X;

        return 1;
    }

    fseek(labels_file_ptr, 8, SEEK_SET);
    for (int i = 0; i < num_samples; ++i)
    {
        scanf_result = fscanf(labels_file_ptr, "%c", &T[i]);
    }

    fclose(labels_file_ptr);
    if (scanf_result == EOF)
    {
        printf("Error reading file. Exiting.\n");
        delete[] Y;
        delete[] T;
        delete[] X;

        return 1;
    }

    int iter{0};
    int batch{0};

    float* expected_output = new float[output_size];
    while (iter < 100)
    {
        le_net_5.Forward(&X[batch * batch_size * input_size], Y);

        ConvertBatchToOneHot(expected_output, &T[batch * batch_size], 10, batch_size);
        auto loss = le_net_5.ComputeLoss(Y, expected_output);
        printf("Iteration: %d\n", iter);
        printf("Loss: %f\n", loss);

        le_net_5.Backward(Y, expected_output);
        iter++;
    }

    // Print the predictions for up to the first 10 samples
    le_net_5.Forward(X, Y);
    for (int i = 0; i < std::min(10, batch_size); ++i)
    {
        printf("Prediction: ");
        printf("%ld ", std::max_element(Y + i * 10, Y + (i + 1) * 10) - (Y + i * 10));
        printf("Actual: ");
        printf("%d\n", T[i]);
        printf("\n");
    }

    delete[] Y;
    delete[] expected_output;
    delete[] T;
    delete[] X;

    return 0;
}