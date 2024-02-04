#ifndef TYPES_CONSTANTS_H
#define TYPES_CONSTANTS_H

#include <cuda_runtime.h>

#define TILE_WIDTH 8
#define IN_TILE_DIM_3D 8
#define OUT_TILE_DIM_3D (IN_TILE_DIM_3D - 2)
#define IN_TILE_DIM_2D 32
#define OUT_TILE_DIM_2D (IN_TILE_DIM_2D - 2)

/// @brief Define the epsilon within which floating point matrix elements must be within to be
/// considered equal.
#define EPS_FOR_MATRIX_ELEMENT_EQUALITY 0.0001
#define NUM_STENCIL_POINTS 7

enum StencilKernelToUse
{
    kBasic,
    kTiling,
    kThreadCoarsening,
    kRegisterTiling,
    kNumFilters
};

#endif