#ifndef CHAPTER_17_FHD_H
#define CHAPTER_17_FHD_H

#include <cuda_runtime.h>

/// @brief Basic kernel for the FHD algorithm with no optimizations.
/// @param r_phi The real part of the phase.
/// @param r_d The real part of the data.
/// @param i_phi The imaginary part of the phase.
/// @param i_d The imaginary part of the data.
/// @param x The x coordinates.
/// @param k_x The x wave numbers.
/// @param y The y coordinates.
/// @param k_y The y wave numbers.
/// @param z The z coordinates.
/// @param k_z The z wave numbers.
/// @param r_mu The real part of the mu.
/// @param i_mu The imaginary part of the mu.
/// @param M The number of elements in the x, y, and z arrays.
/// @param N The number of elements in the r_phi, r_d, i_phi, and i_d arrays.
/// @param r_fhd The real part of the FHD.
/// @param i_fhd The imaginary part of the FHD.
__global__ void BasicKernel(const float* r_phi, const float* r_d, const float* i_phi,
                            const float* i_d, const float* x, const float* k_x, const float* y,
                            const float* k_y, const float* z, const float* k_z, float* r_mu,
                            float* i_mu, const int M, const int N, float* r_fhd, float* i_fhd);

#endif  // CHAPTER_17_FHD_H