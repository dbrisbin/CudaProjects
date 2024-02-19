#include "cnn_cpu.h"
#include <memory>

void CNN::Forward(const float* X, float* Y) const
{
    const auto y_size = DetermineRequiredSizeOfY();
    float* Y_curr = new float[y_size];
    float* Y_prev = new float[y_size];

    for (int i{0}; i < DetermineInputSize(); ++i)
    {
        Y_prev[i] = X[i];
    }

    for (const auto& layer : layers)
    {
        layer->Forward(Y_prev, Y_curr);
        std::swap(Y_prev, Y_curr);
    }
    for (int i{0}; i < DetermineOutputSize(); ++i)
    {
        Y[i] = Y_prev[i];
    }
    delete[] Y_curr;
    delete[] Y_prev;
}

float CNN::ComputeLoss(const float* Y, const float* T) const
{
    float loss{0.0f};
    for (int i{0}; i < DetermineOutputSize(); ++i)
    {
        loss += (Y[i] - T[i]) * (Y[i] - T[i]) / 2;
    }
    return loss;
}

void CNN::Backward(const float* Y, const float* T)
{
    const auto y_size = DetermineRequiredSizeOfY();

    float* dE_dY_curr = new float[y_size];
    float* dE_dY_prev = new float[y_size];
    for (int i{0}; i < DetermineOutputSize(); ++i)
    {
        dE_dY_prev[i] = Y[i] - T[i];
    }

    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
    {
        (*layer)->Backward(dE_dY_prev, dE_dY_curr);
        std::swap(dE_dY_prev, dE_dY_curr);
    }
    delete[] dE_dY_curr;
    delete[] dE_dY_prev;
}

int CNN::DetermineRequiredSizeOfY() const
{
    int max_output_size{0};
    for (const auto& layer : layers)
    {
        max_output_size = std::max(layer->DetermineOutputSize(), max_output_size);
    }
    return max_output_size;
}
