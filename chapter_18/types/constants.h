#ifndef CHAPTER_18_TYPES_CONSTANTS_H
#define CHAPTER_18_TYPES_CONSTANTS_H

#include "types.h"

#define SECTION_SIZE 1024
#define PI 3.14159265358979323846F

/// @brief size of constant memory arrays
// Shared memory is 64KB, but we only want to use half of it to be safe.
static constexpr std::size_t kChunkSize{32U * 1024U / sizeof(Atom)};

/// @brief List of available kernels which are supported.
enum DcsKernels
{
    kBasic = 0,
    kNumKernels = 1,
};

#endif  // CHAPTER_18_TYPES_CONSTANTS_H