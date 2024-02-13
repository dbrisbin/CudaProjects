/// @file constants.h
/// @brief Constants including available kernels and kernel launch parameters.

#ifndef CHAPTER_14_TYPES_CONSTANTS_H
#define CHAPTER_14_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define CFACTOR 4
#define FLOAT_EPS 1.0e-6f

/// @brief List of available kernels which are supported.
enum class SpmvKernel : int
{
    kCooSpmv = 0,
    kCsrSpmv = 1,
    kNumKernels = 2
};

#endif  // CHAPTER_14_TYPES_CONSTANTS_H