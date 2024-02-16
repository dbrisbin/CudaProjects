/// @file constants.h
/// @brief Constants including available kernels and kernel launch parameters.

#ifndef CHAPTER_15_TYPES_CONSTANTS_H
#define CHAPTER_15_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define CFACTOR 4
#define LOCAL_FRONTIER_CAPACITY 4096

constexpr float kEpsilon = 1.0e-3f;

/// @brief List of available kernels which are supported.
enum class BfsKernel : int
{
    kEdgeCentric = 0,
    kVertexCentricPush = 1,
    kVertexCentricPull = 2,
    kVertexCentricPushPull = 3,
    kVertexCentricPushWithFrontier = 4,
    kVertexCentricPushWithFrontierPrivatized = 5,
    kSingleBlockVertexCentricPushFrontierPrivatized = 6,
    kNumKernels = 7
};

#endif  // CHAPTER_15_TYPES_CONSTANTS_H