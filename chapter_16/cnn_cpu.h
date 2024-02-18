/// @file cnn_cpu.h
/// @brief CPU implementation of a driver for a CNN.

#ifndef CHAPTER_16_CNN_CPU_H
#define CHAPTER_16_CNN_CPU_H

#include <vector>
#include "layers_cpu.h"

class CNN
{
   public:
    void AddLayer(CNNLayer* layer) { layers.push_back(layer); }

    void Forward(const float* X, float* Y)
    {
        const auto y_size = DetermineRequiredSizeOfY();
        float* Y_curr = new float[y_size];
        const float* Y_prev;
        Y_prev = X;

        for (const auto layer : layers)
        {
            layer->Forward(Y_prev, Y_curr);
            Y_prev = Y_curr;
        }
        for (int i{0}; i < DetermineOutputSize(); ++i)
        {
            Y[i] = Y_curr[i];
        }
        delete[] Y_curr;
    }

    float ComputeLoss(const float* Y, const float* T) const
    {
        float loss{0.0f};
        for (int i{0}; i < DetermineOutputSize(); ++i)
        {
            loss += (Y[i] - T[i]) * (Y[i] - T[i]) / 2;
        }
        return loss;
    }

    void Backward(const float* X, const float* Y, const float* T)
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
            (*layer)->Backward(dE_dY_prev, X, dE_dY_curr);
            dE_dY_prev = dE_dY_curr;
        }
        delete[] dE_dY_curr;
        delete[] dE_dY_prev;
    }

    int DetermineOutputSize() const { return layers.back()->DetermineOutputSize(); }

   protected:
    std::vector<CNNLayer*> layers;

   private:
    int DetermineRequiredSizeOfY() const
    {
        int max_output_size{0};
        for (const auto layer : layers)
        {
            max_output_size = std::max(layer->DetermineOutputSize(), max_output_size);
        }
        return max_output_size;
    }
};

#endif  // CHAPTER_16_CNN_CPU_H