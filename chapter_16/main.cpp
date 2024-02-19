/// @file main.cpp
/// @brief Main function

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "le_net_5_modified.h"
#include "types/constants.h"

void ConvertToOneHot(float* one_hot, int label, int num_classes)
{
    for (int i = 0; i < num_classes; ++i)
    {
        one_hot[i] = 0.0;
    }
    one_hot[label] = 1.0;
}

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
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    const int N{1};
    LeNet5Modified le_net_5(N, gen, dis);

    auto input_size = le_net_5.DetermineInputSize();
    auto output_size = le_net_5.DetermineOutputSize();
    float* Y = new float[output_size];
    unsigned char* T = new unsigned char[num_samples];
    float* X = new float[num_samples * input_size];

    int scanf_result;

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

    // float delta_loss{1.0f};
    // float prev_loss{0.0f};
    int iter{0};
    int sample_to_use{0};

    float* expected_output = new float[output_size];
    while (iter < 100)
    {
        le_net_5.Forward(&X[sample_to_use * input_size], Y);
        // print prediction
        printf("Prediction: ");
        for (int i = 0; i < output_size; ++i)
        {
            printf("%f ", Y[i]);
        }
        printf("\n");

        ConvertToOneHot(expected_output, T[sample_to_use], output_size);
        auto loss = le_net_5.ComputeLoss(Y, expected_output);
        printf("Loss: %f\n", loss);
        // delta_loss = abs(prev_loss - loss);
        // prev_loss = loss;

        le_net_5.Backward(Y, expected_output);
        iter++;
    }

    delete[] expected_output;
    delete[] Y;
    delete[] T;
    delete[] X;

    return 0;
}