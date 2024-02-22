#ifndef CHAPTER_17_FHD_DRIVER_H
#define CHAPTER_17_FHD_DRIVER_H

#include "types/constants.h"
#include "types/types.h"

/// @brief This function is the driver for the FHD algorithm.
/// @param r_phi The real part of the phi array.
/// @param r_d The real part of the d array.
/// @param i_phi The imaginary part of the phi array.
/// @param i_d The imaginary part of the d array.
/// @param x The x array.
/// @param k_x The k_x array.
/// @param y The y array.
/// @param k_y The k_y array.
/// @param z The z array.
/// @param k_z The k_z array.
/// @param k_struct The array of structs containing the k_x, k_y, and k_z arrays.
/// @param M The number of elements in the phi and d arrays.
/// @param N The number of elements in the x, k_x, y, k_y, z, and k_z arrays.
/// @param r_fhd The real part of the FHD result.
/// @param i_fhd The imaginary part of the FHD result.
/// @param kernel_to_use The kernel to use for the FHD algorithm.
/// @param iters The number of iterations to run the FHD algorithm.
/// @param section_size The block size to use for the FHD algorithm.
/// @return The time it took to run the FHD algorithm iters times.
float FhdDriver(const float* r_phi, const float* r_d, const float* i_phi, const float* i_d,
                const float* x, const float* k_x, const float* y, const float* k_y, const float* z,
                const float* k_z, const KData* k_struct, const int M, const int N, float* r_fhd,
                float* i_fhd, const FhdKernels kernel_to_use, const int iters,
                const int section_size);

#endif  // CHAPTER_17_FHD_DRIVER_H