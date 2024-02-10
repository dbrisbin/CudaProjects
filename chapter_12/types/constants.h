/// @file constants.h
/// @brief Constants including available kernels and kernel launch parameters.

#ifndef CHAPTER_12_TYPES_CONSTANTS_H
#define CHAPTER_12_TYPES_CONSTANTS_H

/// @brief Default block size to use for basic kernel.
#define SECTION_SIZE 1024
/// @brief Number of elements to process per thread for the basic kernel.
#define NUM_ELTS_PER_THREAD 128
/// @brief Number of elements to store in shared memory PER circular buffer.
#define NUM_ELTS_PER_TILE 2048
/// @brief Blocksize to use for tiled kernels.
#define BLOCKSIZE_FOR_TILED 128
/// @brief Number of elements each block should process in tiled kernels.
#define OUTPUT_ELTS_PER_BLOCK NUM_ELTS_PER_TILE * 4

/// @brief List of available kernels which are supported.
enum class MergeKernel : int
{
    kBasic = 0,
    kTiled = 1,
    kModifiedTiled = 2,
    kCircularBuffer = 3,
    kNumKernels = 4
};

#endif  // CHAPTER_12_TYPES_CONSTANTS_H