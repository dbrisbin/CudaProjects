#ifndef CHAPTER_11_TYPES_CONSTANTS_H
#define CHAPTER_11_TYPES_CONSTANTS_H

#include "types.h"

#define SECTION_SIZE 1024

#define CFACTOR 64

enum parallelScanKernelToUse
{
    kKoggeStone,
    kKoggeStoneDoubleBuffering,
    kBrentKung,
    kNumKernels
};

#endif  // CHAPTER_11_TYPES_CONSTANTS_H