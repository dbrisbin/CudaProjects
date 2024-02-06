#ifndef CHAPTER_10_TYPES_CONSTANTS_H
#define CHAPTER_10_TYPES_CONSTANTS_H

#include "types.h"

#define TILE_WIDTH 1024

#define CFACTOR 64

enum reductionKernelToUse
{
    kBasic,
    kCoalescing,
    kSharedMemory,
    kSegmented,
    kCoarsening,
    kNumKernels
};

#endif  // CHAPTER_10_TYPES_CONSTANTS_H