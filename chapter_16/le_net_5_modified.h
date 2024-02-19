#ifndef CHAPTER_16_LE_NET_5_MODIFIED_H
#define CHAPTER_16_LE_NET_5_MODIFIED_H

#include <memory>
#include <random>
#include "cnn_cpu.h"
#include "layers_cpu.h"

class LeNet5Modified : public CNN
{
   public:
    template <typename TGen, typename TDist>
    LeNet5Modified(const int N, TGen& gen, TDist& dist)
    {
        std::unique_ptr<CNNLayer> C1 = std::make_unique<ConvLayer>(N, 6, 1, 28, 28, 5, gen, dist);
        AddLayer(std::move(C1));
        std::unique_ptr<CNNLayer> S2 = std::make_unique<SubsamplingLayer>(N, 6, 24, 24, 2);
        AddLayer(std::move(S2));
        std::unique_ptr<CNNLayer> C3 = std::make_unique<ConvLayer>(N, 16, 6, 12, 12, 5, gen, dist);
        AddLayer(std::move(C3));
        std::unique_ptr<CNNLayer> S4 = std::make_unique<SubsamplingLayer>(N, 16, 8, 8, 2);
        AddLayer(std::move(S4));
        std::unique_ptr<CNNLayer> C5 = std::make_unique<ConvLayer>(N, 120, 16, 4, 4, 4, gen, dist);
        AddLayer(std::move(C5));
        std::unique_ptr<CNNLayer> F6 =
            std::make_unique<FullyConnectedLayer>(N, 84, 120, 1, 1, gen, dist);
        AddLayer(std::move(F6));
        std::unique_ptr<CNNLayer> F7 =
            std::make_unique<FullyConnectedLayer>(N, 10, 84, 1, 1, gen, dist);
        AddLayer(std::move(F7));
        std::unique_ptr<CNNLayer> S8 = std::make_unique<SoftmaxLayer>(N, 10);
        AddLayer(std::move(S8));
    }

    ~LeNet5Modified();

    int DetermineInputSize() const override { return 28 * 28; }
};

#endif  // CHAPTER_16_LE_NET_5_MODIFIED_H