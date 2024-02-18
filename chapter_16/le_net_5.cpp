#include "le_net_5.h"
#include <vector>

LeNet5::LeNet5()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    ConvLayer* C1 = new ConvLayer(6, 1, 32, 32, 5, gen, dist);
    AddLayer(C1);
    SubsamplingLayer* S2 = new SubsamplingLayer(6, 28, 28, 2);
    AddLayer(S2);
    ConvLayer* C3 = new ConvLayer(16, 6, 14, 14, 5, gen, dist);
    AddLayer(C3);
    SubsamplingLayer* S4 = new SubsamplingLayer(16, 10, 10, 2);
    AddLayer(S4);
    ConvLayer* C5 = new ConvLayer(120, 16, 5, 5, 5, gen, dist);
    AddLayer(C5);
    FullyConnectedLayer* F6 = new FullyConnectedLayer(120, 84, 1, 1, gen, dist);
    AddLayer(F6);
    FullyConnectedLayer* F7 = new FullyConnectedLayer(84, 10, 1, 1, gen, dist);
    AddLayer(F7);
}

LeNet5::~LeNet5()
{
    for (const auto layer : layers)
    {
        delete layer;
    }
}