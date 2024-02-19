/// @file utils.h
/// @brief Utility functions for the CNN.

#ifndef CHAPTER_16_UTILS_H
#define CHAPTER_16_UTILS_H

#include <cuda_runtime.h>
#include <cmath>

/// @brief Linearize a 2D index into a 1D index.
/// @param h horizontal index
/// @param w vertical index
/// @param W width
/// @return 1D index corresponding to the 2D index
inline int LinearizeIndex(const int h, const int w, const int W) { return h * W + w; }

/// @brief Linearize a 3D index into a 1D index.
/// @param m number of channels
/// @param h horizontal index
/// @param w vertical index
/// @param H height
/// @param W width
/// @return 1D index corresponding to the 3D index
inline int LinearizeIndex(const int m, const int h, const int w, const int H, const int W)
{
    return m * H * W + h * W + w;
}

/// @brief Linearize a 4D index into a 1D index.
/// @param n number of samples
/// @param c number of channels
/// @param h horizontal index
/// @param w vertical index
/// @param C number of channels
/// @param H height
/// @param W width
/// @return 1D index corresponding to the 4D index
inline int LinearizeIndex(const int n, const int c, const int h, const int w, const int C,
                          const int H, const int W)
{
    return n * C * H * W + c * H * W + h * W + w;
}

/// @brief Compute tanh activation function.
/// @param x input
/// @return tanh(x)
inline float Tanh(const float x) { return std::tanh(x); }

/// @brief Compute the derivative of the tanh activation function.
/// @param x input
/// @return 1 - tanh(x)^2
inline float dTanh(const float x) { return 1.0f - std::pow(Tanh(x), 2); }

#endif  // CHAPTER_16_UTILS_H