/// @file constants.h
/// @brief Constants including available kernels and kernel launch parameters.

#ifndef CHAPTER_13_TYPES_CONSTANTS_H
#define CHAPTER_13_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define CFACTOR 4

/// @brief List of available kernels which are supported.
enum class SortKernel : int
{
    kRadix = 0,
    kRadixSplit = 1,
    kNumKernels = 2
};

#endif  // CHAPTER_13_TYPES_CONSTANTS_H