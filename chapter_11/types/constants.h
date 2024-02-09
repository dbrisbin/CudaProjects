#ifndef CHAPTER_11_TYPES_CONSTANTS_H
#define CHAPTER_11_TYPES_CONSTANTS_H

#include "types.h"

#define SECTION_SIZE 1024

#define CFACTOR 4

/// @brief List of available kernels which are supported.
enum parallelScanKernelToUse
{
    kKoggeStoneInclusive,
    kKoggeStoneExclusive,
    kKoggeStoneDoubleBufferingInclusive,
    kKoggeStoneDoubleBufferingExclusive,
    kBrentKungInclusive,
    kBrentKungExclusive,
    kCoarseningInclusive,
    kCoarseningExclusive,
    kCoarseningSegmented,
    kStreaming,
    kNumKernels
};

#endif  // CHAPTER_11_TYPES_CONSTANTS_H