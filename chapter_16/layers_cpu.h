/// @file layers_cpu.h
/// @brief CPU implementation of CNN layers.

#ifndef CHAPTER_16_LAYERS_CPU_H
#define CHAPTER_16_LAYERS_CPU_H
#include <memory>

class CNN;

class CNNLayer
{
   public:
    CNNLayer(const int N, const int M, const int C, const int H_in, const int W_in)
        : N{N}, M{M}, C{C}, H_in{H_in}, W_in{W_in} {};

    CNNLayer() = delete;
    CNNLayer(const CNNLayer&) = delete;
    CNNLayer& operator=(const CNNLayer&) = delete;

    virtual ~CNNLayer() = default;

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    virtual void Forward(const float* X, float* Y) const = 0;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    virtual void Backward(const float* dE_dY, float* dE_dX) = 0;

    /// @brief Determine the output size of the layer.
    /// @return Output size
    virtual int DetermineOutputSize() const = 0;

   protected:
    /// @brief Batch size
    const int N;
    /// @brief Number of output feature maps
    const int M;
    /// @brief Number of input feature maps
    const int C;
    /// @brief Height of input feature maps
    const int H_in;
    /// @brief Width of input feature maps
    const int W_in;
};

/// @brief Convolutional layer.
/// @details Convolution operation with a KxK kernel using Tanh activation function.
class ConvLayer : public CNNLayer
{
   public:
    ConvLayer() = delete;
    ConvLayer(const ConvLayer&) = delete;
    ConvLayer& operator=(const ConvLayer&) = delete;

    /// @brief Constructor.
    /// @param M Number of output feature maps
    /// @param C Number of input feature maps
    /// @param H_in Height of input feature maps
    /// @param W_in Width of input feature maps
    /// @param K Kernel size
    /// @tparam TGen Random number generator type
    /// @tparam TDist Random number distribution type
    template <typename TGen, typename TDist>
    ConvLayer(const int N, const int M, const int C, const int H_in, const int W_in, const int K,
              TGen& gen, TDist& dist)
        : CNNLayer{N, M, C, H_in, W_in}, K{K}
    {
        W = std::make_unique<float[]>(M * C * K * K);
        for (int i{0}; i < M * C * K * K; ++i)
        {
            W[i] = dist(gen);
        }
        Z = std::make_unique<float[]>(M * (H_in - K + 1) * (W_in - K + 1));
        X_in = std::make_unique<float[]>(C * H_in * W_in);
    };

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * (H_in - K + 1) * (W_in - K + 1); }

   private:
    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void BackwardX(const float* dE_dY, float* dE_dX);

    /// @brief Backward pass with respect to the weights.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dW Gradient of the loss with respect to the weights
    void BackwardW(const float* dE_dY, float* dE_dW);

    /// @brief Weights
    std::unique_ptr<float[]> W;
    /// @brief Pre-actvation values
    std::unique_ptr<float[]> Z;
    /// @brief input values from forward pass
    std::unique_ptr<float[]> X_in;
    /// @brief Kernel size
    const int K;

    friend CNN;
};

/// @brief Subsampling layer.
/// @details Subsampling operation with a subsampling factor of K using Tanh activation function.
class SubsamplingLayer : public CNNLayer
{
   public:
    SubsamplingLayer() = delete;

    /// @brief Constructor.
    /// @param M Number of output feature maps
    /// @param H_in Height of input feature maps
    /// @param W_in Width of input feature maps
    /// @param K Factor by which to subsample
    SubsamplingLayer(const int N, const int M, const int H_in, const int W_in, const int K)
        : CNNLayer{N, M, M, H_in, W_in}, K{K} {};

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] S Output feature maps
    void Forward(const float* X, float* S) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dS Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * H_in / K * W_in / K; }

   private:
    /// @brief Factor by which to subsample
    const int K;

    friend CNN;
};

/// @brief Fully connected layer.
/// @details Fully connected layer with Tanh activation function.
class FullyConnectedLayer : public CNNLayer
{
   public:
    FullyConnectedLayer() = delete;

    /// @brief Constructor.
    /// @param M Number of output feature maps
    /// @param C Number of input feature maps
    /// @param H_in Height of input feature maps
    /// @param W_in Width of input feature maps
    /// @tparam TGen Random number generator type
    /// @tparam TDist Random number distribution type
    template <typename TGen, typename TDist>
    FullyConnectedLayer(const int N, const int M, const int C, const int H_in, const int W_in,
                        TGen& gen, TDist& dist)
        : CNNLayer{N, M, C, H_in, W_in}
    {
        W = new float[M * C * H_in * W_in];
        for (int i{0}; i < M * C * H_in * W_in; ++i)
        {
            W[i] = dist(gen);
        }
        Z = new float[M];
        X_in = new float[C * H_in * W_in];
    };

    ~FullyConnectedLayer()
    {
        delete[] W;
        delete[] Z;
        delete[] X_in;
    }

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, float* dE_dX) override;

    int DetermineOutputSize() const override { return M; }

   private:
    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void BackwardX(const float* dE_dY, float* dE_dX);

    /// @brief Backward pass with respect to the weights.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dW Gradient of the loss with respect to the weights
    void BackwardW(const float* dE_dY, float* dE_dW);

    /// @brief Weights
    float* W;

    /// @brief Pre-actvation values
    float* Z;

    /// @brief input values from forward pass
    float* X_in;

    friend CNN;
};

class SoftmaxLayer : public CNNLayer
{
   public:
    SoftmaxLayer() = delete;

    /// @brief Constructor.
    /// @param N Batch size
    /// @param M Number of feature maps
    SoftmaxLayer(const int N, const int M) : CNNLayer{N, M, M, 1, 1} { Y_out = new float[M]; };

    ~SoftmaxLayer() { delete[] Y_out; }
    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * H_in * W_in; }

   private:
    /// @brief outputs from forward pass
    float* Y_out;

    friend CNN;
};

#endif  // CHAPTER_16_LAYERS_CPU_H