#include "le_net_5_modified.h"
#include <vector>

LeNet5Modified::~LeNet5Modified()
{
    for (auto& layer : layers)
    {
        layer.reset();
    }
}