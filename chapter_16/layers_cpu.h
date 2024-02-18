/// @file layers_cpu.h
/// @brief CPU implementation of CNN layers.

#ifndef CHAPTER_16_LAYERS_CPU_H
#define CHAPTER_16_LAYERS_CPU_H

class CNN;

class CNNLayer
{
   public:
    CNNLayer(const int M, const int C, const int H_in, const int W_in)
        : M{M}, C{C}, H_in{H_in}, W_in{W_in} {};

    virtual ~CNNLayer() = default;

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    virtual void Forward(const float* X, float* Y) const = 0;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    virtual void Backward(const float* dE_dY, const float* X, float* dE_dX) = 0;

    /// @brief Determine the output size of the layer.
    /// @return Output size
    virtual int DetermineOutputSize() const = 0;

   protected:
    /// @brief Number of output feature maps
    const int M;
    /// @brief Number of input feature maps
    const int C;
    /// @brief Height of input feature maps
    const int H_in;
    /// @brief Width of input feature maps
    const int W_in;
};

class ConvLayer : public CNNLayer
{
   public:
    ConvLayer() = delete;

    /// @brief Constructor.
    /// @param M Number of output feature maps
    /// @param C Number of input feature maps
    /// @param H_in Height of input feature maps
    /// @param W_in Width of input feature maps
    /// @param K Kernel size
    /// @tparam TGen Random number generator type
    /// @tparam TDist Random number distribution type
    template <typename TGen, typename TDist>
    ConvLayer(const int M, const int C, const int H_in, const int W_in, const int K, TGen& gen,
              TDist& dist)
        : CNNLayer(M, C, H_in, W_in), K{K}
    {
        W = new float[M * C * K * K];
        for (int i{0}; i < M * C * K * K; ++i)
        {
            W[i] = dist(gen);
        }
    };

    ~ConvLayer() { delete[] W; }
    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, const float* X, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * (H_in - K + 1) * (W_in - K + 1); }

   private:
    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void BackwardX(const float* dE_dY, float* dE_dX);

    /// @brief Backward pass with respect to the weights.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dW Gradient of the loss with respect to the weights
    void BackwardW(const float* dE_dY, const float* X, float* dE_dW);

    /// @brief Weights
    float* W;

    /// @brief Kernel size
    const int K;

    friend CNN;
};

class SubsamplingLayer : public CNNLayer
{
   public:
    SubsamplingLayer() = delete;

    /// @brief Constructor.
    /// @param M Number of output feature maps
    /// @param H Height of input feature maps
    /// @param W Width of input feature maps
    /// @param K Factor by which to subsample
    SubsamplingLayer(const int M, const int H_in, const int W_in, const int K)
        : CNNLayer{M, M, H_in, W_in}, K{K} {};

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] S Output feature maps
    void Forward(const float* X, float* S) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dS Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, const float* X, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * H_in / K * W_in / K; }

   private:
    /// @brief Factor by which to subsample
    const int K;

    friend CNN;
};

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
    FullyConnectedLayer(const int M, const int C, const int H_in, const int W_in, TGen& gen,
                        TDist& dist)
        : CNNLayer{M, C, H_in, W_in}
    {
        W = new float[M * C * H_in * W_in];
        for (int i{0}; i < M * C * H_in * W_in; ++i)
        {
            W[i] = dist(gen);
        }
    };

    ~FullyConnectedLayer() { delete[] W; }

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, const float* X, float* dE_dX) override;

    int DetermineOutputSize() const override { return C * H_in * W_in; }

   private:
    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void BackwardX(const float* dE_dY, float* dE_dX);

    /// @brief Backward pass with respect to the weights.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dW Gradient of the loss with respect to the weights
    void BackwardW(const float* dE_dY, const float* X, float* dE_dW);

    /// @brief Weights
    float* W;

    friend CNN;
};

class SigmoidLayer : public CNNLayer
{
   public:
    SigmoidLayer() = delete;

    /// @brief Constructor.
    /// @param M Number of feature maps
    /// @param H Height of input feature maps
    /// @param W Width of input feature maps
    SigmoidLayer(const int M, const int H_in, const int W_in) : CNNLayer{M, M, H_in, W_in} {};

    /// @brief Forward pass.
    /// @param X Input feature maps
    /// @param[out] Y Output feature maps
    void Forward(const float* X, float* Y) const override;

    /// @brief Backward pass with respect to the inputs.
    /// @param dE_dY Gradient of the loss with respect to the output feature maps
    /// @param X Input feature maps
    /// @param[out] dE_dX Gradient of the loss with respect to the input
    void Backward(const float* dE_dY, const float* X, float* dE_dX) override;

    int DetermineOutputSize() const override { return M * H_in * W_in; }

   private:
    friend CNN;
};

#endif  // CHAPTER_16_LAYERS_CPU_H