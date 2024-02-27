/// @file types/constants.h
/// @brief Constants used in Bezier curve tessellation and Quadtree construction.

#ifndef CHAPTER_21_TYPES_CONSTANTS_H
#define CHAPTER_21_TYPES_CONSTANTS_H

/// @brief maximum number of tessellation points.
#define MAX_TESS_POINTS 32

/// @brief List of available kernels which are supported.
enum BezierKernels
{
    kBasic = 0,
    kDynamic = 1,
    kNumKernels = 2,
};

#endif  // CHAPTER_21_TYPES_CONSTANTS_H