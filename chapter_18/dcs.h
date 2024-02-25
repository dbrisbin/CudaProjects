#ifndef CHAPTER_18_FHD_H
#define CHAPTER_18_FHD_H

#include <cuda_runtime.h>
#include "types/constants.h"
#include "types/types.h"

extern __constant__ Atom atoms_c[kChunkSize];

/// @brief Compute the energy grid for the DCS calculation using the scatter approach.
/// @param[out] energy_grid energy grid
/// @param grid_size size of the energy grid
/// @param spacing spacing of the energy grid
/// @param z z coordinate
/// @param num_atoms number of atoms
__global__ void DcsScatter(float* energy_grid, const dim3 grid_size, const float spacing,
                           const float z, const unsigned int num_atoms);

/// @brief Compute the energy grid for the DCS calculation using the gather approach.
/// @param[out] energy_grid energy grid
/// @param grid_size size of the energy grid
/// @param spacing spacing of the energy grid
/// @param z z coordinate
/// @param num_atoms number of atoms
__global__ void DcsGatherBasic(float* energy_grid, const dim3 grid_size, const float spacing,
                               const float z, const unsigned int num_atoms);

/// @brief Compute the energy grid for the DCS calculation using the gather approach with thread
/// granularity coarsening.
/// @param[out] energy_grid energy grid
/// @param grid_size size of the energy grid
/// @param spacing spacing of the energy grid
/// @param z z coordinate
/// @param num_atoms number of atoms
__global__ void DcsGatherCoarsened(float* energy_grid, const dim3 grid_size, const float spacing,
                                   const float z, const unsigned int num_atoms);

/// @brief Compute the energy grid for the DCS calculation using the gather approach with thread
/// granularity coarsening and coalesced memory access.
/// @param[out] energy_grid energy grid
/// @param grid_size size of the energy grid
/// @param spacing spacing of the energy grid
/// @param z z coordinate
/// @param num_atoms number of atoms
__global__ void DcsGatherCoarsenedCoalesced(float* energy_grid, const dim3 grid_size,
                                            const float spacing, const float z,
                                            const unsigned int num_atoms);

#endif  // CHAPTER_18_FHD_H
