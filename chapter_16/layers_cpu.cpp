/// @file layers_cpu.cpp
/// @brief Implementation of CNN layers on the CPU declared in layers.h.

#include "layers_cpu.h"
#include <cmath>
#include "types/constants.h"
#include "utils.h"

void ConvLayer::Forward(const float* X, float* Y) const
{
    const auto H_out = H_in - K + 1;
    const auto W_out = W_in - K + 1;

    for (int i{0}; i < M * H_out * W_out; ++i)
    {
        Z[i] = 0.0f;
    }
    for (int i{0}; i < C * H_in * W_in; ++i)
    {
        float temp = X[i];
        X_in[i] = temp;
    }

    for (int m{0}; m < M; ++m)
    {
        for (int h{0}; h < H_out; ++h)
        {
            for (int w{0}; w < W_out; ++w)
            {
                Y[LinearizeIndex(m, h, w, H_out, W_out)] = 0.0f;
                for (int c{0}; c < C; ++c)
                {
                    for (int i{0}; i < K; ++i)
                    {
                        for (int j{0}; j < K; ++j)
                        {
                            Z[LinearizeIndex(m, h, w, H_out, W_out)] +=
                                X[LinearizeIndex(c, h + i, w + j, H_in, W_in)] *
                                W[LinearizeIndex(m, c, i, j, C, K, K)];
                        }
                    }
                }
                Y[LinearizeIndex(m, h, w, H_out, W_out)] =
                    Tanh(Z[LinearizeIndex(m, h, w, H_out, W_out)]);
            }
        }
    }
}

void ConvLayer::Backward(const float* dE_dY, float* dE_dX)
{
    float* dE_dW = new float[M * C * K * K];
    BackwardW(dE_dY, dE_dW);
    for (int i{0}; i < M * C * K * K; ++i)
    {
        W[i] -= kLearningRate * dE_dW[i];
    }
    delete[] dE_dW;
    BackwardX(dE_dY, dE_dX);
}

void ConvLayer::BackwardX(const float* dE_dY, float* dE_dX)
{
    const auto H_out = H_in - K + 1;
    const auto W_out = W_in - K + 1;

    for (int i{0}; i < C * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int m{0}; m < M; ++m)
    {
        for (int h{0}; h < H_out; ++h)
        {
            for (int w{0}; w < W_out; ++w)
            {
                for (int c{0}; c < C; ++c)
                {
                    for (int i{0}; i < K; ++i)
                    {
                        for (int j{0}; j < K; ++j)
                        {
                            if (h - i > 0 && h - i < H_out && w - j > 0 && w - j < W_out)
                            {
                                dE_dX[LinearizeIndex(c, h - i, w - j, H_in, W_in)] +=
                                    dE_dY[LinearizeIndex(m, h, w, H_out, W_out)] *
                                    W[LinearizeIndex(m, c, i, j, C, K, K)] *
                                    dTanh(Z[LinearizeIndex(m, h - i, w - j, H_out, W_out)]);
                            }
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::BackwardW(const float* dE_dY, float* dE_dW)
{
    const auto H_out = H_in - K + 1;
    const auto W_out = W_in - K + 1;

    for (int i{0}; i < M * C * K * K; ++i)
    {
        dE_dW[i] = 0.0f;
    }

    for (int m{0}; m < M; ++m)
    {
        for (int h{0}; h < H_out; ++h)
        {
            for (int w{0}; w < W_out; ++w)
            {
                for (int c{0}; c < C; ++c)
                {
                    for (int i{0}; i < K; ++i)
                    {
                        for (int j{0}; j < K; ++j)
                        {
                            dE_dW[LinearizeIndex(m, c, i, j, C, K, K)] +=
                                dE_dY[LinearizeIndex(m, h, w, H_out, W_out)] *
                                X_in[LinearizeIndex(c, h + i, w + j, H_in, W_in)];
                        }
                    }
                }
            }
        }
    }
}

void SubsamplingLayer::Forward(const float* X, float* S) const
{
    const auto H_out = H_in / K;
    const auto W_out = W_in / K;

    for (int m{0}; m < M; ++m)
    {
        for (int h{0}; h < H_out; ++h)
        {
            for (int w{0}; w < W_out; ++w)
            {
                S[LinearizeIndex(m, h, w, H_out, W_out)] = 0.0f;
                for (int i{0}; i < K; ++i)
                {
                    for (int j{0}; j < K; ++j)
                    {
                        S[LinearizeIndex(m, h, w, H_out, W_out)] +=
                            X[LinearizeIndex(m, h * K + i, w * K + j, H_in, W_in)];
                    }
                }
                S[LinearizeIndex(m, h, w, H_out, W_out)] /= K * K;
                S[LinearizeIndex(m, h, w, H_out, W_out)] =
                    Tanh(S[LinearizeIndex(m, h, w, H_out, W_out)]);
            }
        }
    }
}

void SubsamplingLayer::Backward(const float* dE_dS, float* dE_dX)
{
    const auto H_out = H_in / K;
    const auto W_out = W_in / K;

    for (int i{0}; i < M * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int m{0}; m < M; ++m)
    {
        for (int h{0}; h < H_out; ++h)
        {
            for (int w{0}; w < W_out; ++w)
            {
                for (int i{0}; i < K; ++i)
                {
                    for (int j{0}; j < K; ++j)
                    {
                        dE_dX[LinearizeIndex(m, h * K + i, w * K + j, H_in, W_in)] =
                            dE_dS[LinearizeIndex(m, h, w, H_out, W_out)];
                    }
                }
            }
        }
    }
}

void FullyConnectedLayer::Forward(const float* X, float* Y) const
{
    for (int i{0}; i < M; ++i)
    {
        Z[i] = 0.0f;
    }
    for (int i{0}; i < C * H_in * W_in; ++i)
    {
        X_in[i] = X[i];
    }

    for (int m{0}; m < M; ++m)
    {
        Y[m] = 0.0f;
        for (int c{0}; c < C; ++c)
        {
            for (int h{0}; h < H_in; ++h)
            {
                for (int w{0}; w < W_in; ++w)
                {
                    Z[m] += X[LinearizeIndex(c, h, w, H_in, W_in)] *
                            W[LinearizeIndex(m, c, h, w, C, H_in, W_in)];
                    Y[m] += X[LinearizeIndex(c, h, w, H_in, W_in)] *
                            W[LinearizeIndex(m, c, h, w, C, H_in, W_in)];
                }
            }
        }
        Y[m] = Tanh(Y[m]);
    }
}

void FullyConnectedLayer::Backward(const float* dE_dY, float* dE_dX)
{
    float* dE_dW = new float[M * C * H_in * W_in];
    BackwardW(dE_dY, dE_dW);
    for (int i{0}; i < M * C * H_in * W_in; ++i)
    {
        W[i] -= kLearningRate * dE_dW[i];
    }
    delete[] dE_dW;
    BackwardX(dE_dY, dE_dX);
}

void FullyConnectedLayer::BackwardX(const float* dE_dY, float* dE_dX)
{
    for (int i{0}; i < C * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int c{0}; c < C; ++c)
    {
        for (int h{0}; h < H_in; ++h)
        {
            for (int w{0}; w < W_in; ++w)
            {
                for (int m{0}; m < M; ++m)
                {
                    dE_dX[LinearizeIndex(c, h, w, H_in, W_in)] +=
                        dE_dY[m] * W[LinearizeIndex(m, c, h, w, C, H_in, W_in)];
                }
            }
        }
    }
}

void FullyConnectedLayer::BackwardW(const float* dE_dY, float* dE_dW)
{
    for (int i{0}; i < M * C * H_in * W_in; ++i)
    {
        dE_dW[i] = 0.0f;
    }

    for (int m{0}; m < M; ++m)
    {
        for (int c{0}; c < C; ++c)
        {
            for (int h{0}; h < H_in; ++h)
            {
                for (int w{0}; w < W_in; ++w)
                {
                    dE_dW[LinearizeIndex(m, c, h, w, C, H_in, W_in)] +=
                        dE_dY[m] * X_in[LinearizeIndex(c, h, w, H_in, W_in)];
                }
            }
        }
    }
}

void SoftmaxLayer::Forward(const float* X, float* Y) const
{
    float sum{0.0f};
    for (int m{0}; m < M; ++m)
    {
        Y[m] = std::exp(X[m]);
        sum += Y[m];
    }
    for (int m{0}; m < M; ++m)
    {
        Y[m] /= sum;
    }

    for (int m{0}; m < M; ++m)
    {
        Y_out[m] = Y[m];
    }
}

void SoftmaxLayer::Backward(const float* dE_dY, float* dE_dX)
{
    for (int i{0}; i < M * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int m{0}; m < M; ++m)
    {
        dE_dX[m] = dE_dY[m] * Y_out[m] * (1 - Y_out[m]);
    }
}