#ifndef CHAPTER_16_LE_NET_5_H
#define CHAPTER_16_LE_NET_5_H

#include <random>
#include "cnn_cpu.h"
#include "layers_cpu.h"

class LeNet5 : public CNN
{
   public:
    // template <typename TGen, typename TDist>
    // LeNet5(const TGen& gen, const TDist& dist);

    LeNet5();

    ~LeNet5();
};

#endif  // CHAPTER_16_LE_NET_5_H