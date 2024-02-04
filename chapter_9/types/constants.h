#ifndef CHAPTER_9_TYPES_CONSTANTS_H
#define CHAPTER_9_TYPES_CONSTANTS_H

#include <cuda_runtime.h>

#define CEILING(x, y) (((x) + (y)-1) / (y))

#define TILE_WIDTH 256
#define MAX_VAL 1000
#define VALS_PER_BIN 10
#define NUM_BINS CEILING(MAX_VAL, VALS_PER_BIN)

#define CFACTOR 64

enum parallelHistogramKernelToUse
{
    kBasic,
    kPrivatized,
    kPrivatizedWithSharedMemory,
    kCoarsening,
    kCoarseningWithCoalescedAccess,
    kAggregated,
    kNumKernels
};

#endif  // CHAPTER_9_TYPES_CONSTANTS_H