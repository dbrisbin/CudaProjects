/// @file cnn_cpu.h
/// @brief CPU implementation of a driver for a CNN.

#ifndef CHAPTER_16_CNN_CPU_H
#define CHAPTER_16_CNN_CPU_H

#include <memory>
#include <vector>
#include "layers_cpu.h"

class CNN
{
   public:
    CNN(const int N) : N{N} {}

    void AddLayer(std::unique_ptr<CNNLayer>&& layer) { layers.push_back(std::move(layer)); }

    void Forward(const float* X, float* Y) const;

    float ComputeLoss(const float* Y, const float* T) const;

    void Backward(const float* Y, const float* T);

    int DetermineOutputSize() const { return layers.back()->DetermineOutputSize(); }

    virtual int DetermineInputSize() const = 0;

   protected:
    std::vector<std::unique_ptr<CNNLayer>> layers;

   private:
    int DetermineRequiredSizeOfY() const;
    const int N;
};

#endif  // CHAPTER_16_CNN_CPU_H