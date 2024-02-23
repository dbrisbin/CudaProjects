#ifndef CHAPTER_18_FHD_H
#define CHAPTER_18_FHD_H

#include <cuda_runtime.h>
#include "types/constants.h"
#include "types/types.h"

extern __constant__ Atom atoms_c[kChunkSize];

__global__ void DcsScatter(float* energy_grid, const dim3 grid_size, const float spacing,
                           const float z, const unsigned int num_atoms);

__global__ void DcsGatherBasic(float* energy_grid, const dim3 grid_size, const float spacing,
                               const float z, const unsigned int num_atoms);

#endif  // CHAPTER_18_FHD_H
