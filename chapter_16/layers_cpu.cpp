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

    for (int i{0}; i < N * M * H_out * W_out; ++i)
    {
        Z[i] = 0.0f;
    }
    for (int i{0}; i < N * C * H_in * W_in; ++i)
    {
        X_in[i] = X[i];
    }

    for (int n = 0; n < N; ++n)
    {
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
                                auto x_val = X[LinearizeIndex(n, c, h + i, w + j, C, H_in, W_in)];
                                auto w_val = W[LinearizeIndex(m, c, i, j, C, K, K)];
                                Z[LinearizeIndex(n, m, h, w, M, H_out, W_out)] += x_val * w_val;
                            }
                        }
                    }
                    Y[LinearizeIndex(n, m, h, w, M, H_out, W_out)] =
                        Tanh(Z[LinearizeIndex(n, m, h, w, M, H_out, W_out)]);
                }
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

    for (int i{0}; i < N * C * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int n{0}; n < N; ++n)
    {
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
                                    dE_dX[LinearizeIndex(n, c, h - i, w - j, C, H_in, W_in)] +=
                                        dE_dY[LinearizeIndex(n, m, h, w, M, H_out, W_out)] *
                                        W[LinearizeIndex(m, c, i, j, C, K, K)] *
                                        dTanh(
                                            Z[LinearizeIndex(n, m, h - i, w - j, M, H_out, W_out)]);
                                }
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
        for (int c{0}; c < C; ++c)
        {
            for (int i{0}; i < K; ++i)
            {
                for (int j{0}; j < K; ++j)
                {
                    for (int n{0}; n < N; ++n)
                    {
                        for (int h{0}; h < H_out; ++h)
                        {
                            for (int w{0}; w < W_out; ++w)
                            {
                                dE_dW[LinearizeIndex(m, c, i, j, C, K, K)] +=
                                    dE_dY[LinearizeIndex(n, m, h, w, M, H_out, W_out)] *
                                    X_in[LinearizeIndex(n, c, h + i, w + j, C, H_in, W_in)] *
                                    dTanh(Z[LinearizeIndex(n, m, h, w, M, H_out, W_out)]);
                            }
                        }
                    }
                    dE_dW[LinearizeIndex(m, c, i, j, C, K, K)] /= N;
                }
            }
        }
    }
}

void SubsamplingLayer::Forward(const float* X, float* S) const
{
    const auto H_out = H_in / K;
    const auto W_out = W_in / K;

    for (int n{0}; n < N; ++n)
    {
        for (int m{0}; m < M; ++m)
        {
            for (int h{0}; h < H_out; ++h)
            {
                for (int w{0}; w < W_out; ++w)
                {
                    S[LinearizeIndex(n, m, h, w, M, H_out, W_out)] = 0.0f;
                    for (int i{0}; i < K; ++i)
                    {
                        for (int j{0}; j < K; ++j)
                        {
                            auto x_val =
                                X[LinearizeIndex(n, m, h * K + i, w * K + j, M, H_in, W_in)];
                            S[LinearizeIndex(n, m, h, w, M, H_out, W_out)] += x_val;
                        }
                    }
                    S[LinearizeIndex(n, m, h, w, M, H_out, W_out)] /= K * K;
                    S[LinearizeIndex(n, m, h, w, M, H_out, W_out)] =
                        Tanh(S[LinearizeIndex(n, m, h, w, M, H_out, W_out)]);
                }
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

    for (int n{0}; n < N; ++n)
    {
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
                            dE_dX[LinearizeIndex(n, m, h * K + i, w * K + j, M, H_in, W_in)] =
                                dE_dS[LinearizeIndex(n, m, h, w, M, H_out, W_out)];
                        }
                    }
                }
            }
        }
    }
}

void FullyConnectedLayer::Forward(const float* X, float* Y) const
{
    for (int i{0}; i < N * M; ++i)
    {
        Z[i] = 0.0f;
    }
    for (int i{0}; i < N * C * H_in * W_in; ++i)
    {
        X_in[i] = X[i];
    }

    for (int n{0}; n < N; ++n)
    {
        for (int m{0}; m < M; ++m)
        {
            for (int c{0}; c < C; ++c)
            {
                for (int h{0}; h < H_in; ++h)
                {
                    for (int w{0}; w < W_in; ++w)
                    {
                        auto x_val = X[LinearizeIndex(n, c, h, w, C, H_in, W_in)];
                        auto w_val = W[LinearizeIndex(m, c, h, w, C, H_in, W_in)];
                        Z[LinearizeIndex(n, m, M)] += x_val * w_val;
                    }
                }
            }
            Y[LinearizeIndex(n, m, M)] = Tanh(Z[LinearizeIndex(n, m, M)]);
        }
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
    for (int i{0}; i < N * C * H_in * W_in; ++i)
    {
        dE_dX[i] = 0.0f;
    }

    for (int n{0}; n < N; ++n)
    {
        for (int c{0}; c < C; ++c)
        {
            for (int h{0}; h < H_in; ++h)
            {
                for (int w{0}; w < W_in; ++w)
                {
                    for (int m{0}; m < M; ++m)
                    {
                        dE_dX[LinearizeIndex(n, c, h, w, C, H_in, W_in)] +=
                            dE_dY[LinearizeIndex(n, m, M)] *
                            W[LinearizeIndex(m, c, h, w, C, H_in, W_in)] *
                            dTanh(Z[LinearizeIndex(n, m, M)]);
                    }
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
                    for (int n{0}; n < N; ++n)
                    {
                        dE_dW[LinearizeIndex(m, c, h, w, C, H_in, W_in)] +=
                            dE_dY[LinearizeIndex(n, m, M)] *
                            X_in[LinearizeIndex(n, c, h, w, M, H_in, W_in)] *
                            dTanh(Z[LinearizeIndex(n, m, M)]);
                    }
                    dE_dW[LinearizeIndex(m, c, h, w, C, H_in, W_in)] /= N;
                }
            }
        }
    }
}

void SoftmaxLayer::Forward(const float* X, float* Y) const
{
    for (int n{0}; n < N; ++n)
    {
        float sum{0.0f};
        for (int m{0}; m < M; ++m)
        {
            Y[LinearizeIndex(n, m, M)] = std::exp(X[LinearizeIndex(n, m, M)]);
            sum += Y[m];
        }
        for (int m{0}; m < M; ++m)
        {
            Y[LinearizeIndex(n, m, M)] /= sum;
        }

        for (int m{0}; m < M; ++m)
        {
            Y_out[LinearizeIndex(n, m, M)] = Y[LinearizeIndex(n, m, M)];
        }
    }
}

void SoftmaxLayer::Backward(const float* dE_dY, float* dE_dX)
{
    for (int n{0}; n < N; ++n)
    {
        for (int m{0}; m < M; ++m)
        {
            dE_dX[LinearizeIndex(n, m, M)] = dE_dY[LinearizeIndex(n, m, M)] *
                                             Y_out[LinearizeIndex(n, m, M)] *
                                             (1 - Y_out[LinearizeIndex(n, m, M)]);
        }
    }
}