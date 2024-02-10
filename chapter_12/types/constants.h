#ifndef CHAPTER_12_TYPES_CONSTANTS_H
#define CHAPTER_12_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define CFACTOR 4
#define NUM_ELTS_PER_THREAD 128
#define NUM_ELTS_PER_TILE 2048
#define BLOCKSIZE_FOR_TILED 128
#define OUTPUT_ELTS_PER_BLOCK NUM_ELTS_PER_TILE * 4

/// @brief List of available kernels which are supported.
enum class MergeKernel : int
{
    kBasic,
    kTiled,
    kNumKernels
};

#endif  // CHAPTER_12_TYPES_CONSTANTS_H