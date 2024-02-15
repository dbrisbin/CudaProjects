/// @file constants.h
/// @brief Constants including available kernels and kernel launch parameters.

#ifndef CHAPTER_15_TYPES_CONSTANTS_H
#define CHAPTER_15_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define CFACTOR 4

constexpr float kEpsilon = 1.0e-3f;

/// @brief List of available kernels which are supported.
enum class BfsKernel : int
{
    kEdgeCentric = 0,
    kNumKernels = 1
};

#endif  // CHAPTER_15_TYPES_CONSTANTS_H