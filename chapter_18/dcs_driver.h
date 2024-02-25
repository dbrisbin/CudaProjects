#ifndef CHAPTER_18_FHD_DRIVER_H
#define CHAPTER_18_FHD_DRIVER_H

#include "types/constants.h"
#include "types/types.h"

/// @brief Calls the appropriate DCS kernel based on the input parameter.
/// @param energy_grid_h The energy grid to calculate the DCS on.
/// @param grid_size The size of the energy grid.
/// @param spacing The spacing between energy grid points.
/// @param atoms_h The atoms to calculate the DCS for.
/// @param num_atoms The number of atoms in the atoms array.
/// @param kernel_to_use The kernel to use for the DCS calculation.
/// @param iters The number of iterations to run the kernel for.
/// @return The time taken to run the kernel iters times.
float DcsDriver(float* energy_grid_h, const dim3 grid_size, const float spacing,
                const Atom* atoms_h, const unsigned int num_atoms, const DcsKernels kernel_to_use,
                const int iters);

#endif  // CHAPTER_18_FHD_DRIVER_H