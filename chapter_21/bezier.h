#ifndef CHAPTER_21_BEZIER_H
#define CHAPTER_21_BEZIER_H

#include <cuda_runtime.h>
#include "types/constants.h"
#include "types/types.h"

/// @brief Compute quadratic Bezier curves using basic kernel.
/// @param lines array of quadratic Bezier curves
/// @param num_lines number of quadratic Bezier curves
__global__ void ComputeBezierLinesBasic(BezierLineFixedSize* lines, const int num_lines);

/// @brief Compute quadratic Bezier curves using dynamic kernel.
/// @param lines array of quadratic Bezier curves
/// @param num_lines number of quadratic Bezier curves
__global__ void ComputeBezierLinesDynamic(BezierLineDynamic* lines, const int num_lines);

#endif  // CHAPTER_21_BEZIER_H
