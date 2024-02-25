#include <cuda_runtime.h>
#include "dcs.h"
#include "types/constants.h"
#include "types/types.h"
#include "utils.h"

__constant__ Atom atoms_c[kChunkSize];

__global__ void DcsScatter(float* energy_grid, const dim3 grid_size, const float spacing,
                           const float z, const unsigned int num_atoms)
{
    const unsigned int n{blockIdx.x * blockDim.x + threadIdx.x};
    if (n >= num_atoms)
    {
        return;
    }
    const Atom& atom = atoms_c[n];
    const float dz = z - atom.z;
    const float dz2 = dz * dz;

    for (unsigned int j{0U}; j < grid_size.y; ++j)
    {
        const float dy = j * spacing - atom.y;
        const float dy2 = dy * dy;
        for (unsigned int i{0U}; i < grid_size.x; ++i)
        {
            const float dx = i * spacing - atom.x;
            const float dx2 = dx * dx;
            const float r2 = dx2 + dy2 + dz2;
            atomicAdd(&energy_grid[LinearizeIndex(i, j, static_cast<int>(grid_size.x))],
                      atom.charge / sqrtf(r2));
        }
    }
}

__global__ void DcsGatherBasic(float* energy_grid, const dim3 grid_size, const float spacing,
                               const float z, const unsigned int num_atoms)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= grid_size.x || j >= grid_size.y)
    {
        return;
    }
    float energy{0.0f};
    const float x = i * spacing;
    const float y = j * spacing;

    for (unsigned int k{0U}; k < num_atoms; ++k)
    {
        const Atom& atom = atoms_c[k];
        const float dx = x - atom.x;
        const float dy = y - atom.y;
        const float dz = z - atom.z;
        const float r2 = dx * dx + dy * dy + dz * dz;
        energy += atom.charge / sqrtf(r2);
    }
    energy_grid[LinearizeIndex(i, j, static_cast<int>(grid_size.x))] += energy;
}

__global__ void DcsGatherCoarsened(float* energy_grid, const dim3 grid_size, const float spacing,
                                   const float z, const unsigned int num_atoms)
{
    const unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * kCoarseningFactor;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= grid_size.x || j >= grid_size.y)
    {
        return;
    }
    float energy[kCoarseningFactor];
    for (unsigned int k{0U}; k < kCoarseningFactor; ++k)
    {
        energy[k] = 0.0f;
    }

    const float x = i * spacing;
    const float y = j * spacing;

    for (unsigned int k{0U}; k < num_atoms; ++k)
    {
        const Atom& atom = atoms_c[k];

        float dx = x - atom.x;
        const float dy = y - atom.y;
        const float dz = z - atom.z;
        const float dy2pdz2 = dy * dy + dz * dz;

        for (unsigned int l{0U}; l < kCoarseningFactor; ++l)
        {
            energy[l] += atom.charge / sqrtf(dx * dx + dy2pdz2);
            dx += spacing;
        }
    }
    const int idx = LinearizeIndex(i, j, static_cast<int>(grid_size.x));
    for (unsigned int k{0U}; k < kCoarseningFactor; ++k)
    {
        if (i + k < grid_size.x)
        {
            energy_grid[idx + k] += energy[k];
        }
    }
}

__global__ void DcsGatherCoarsenedCoalesced(float* energy_grid, const dim3 grid_size,
                                            const float spacing, const float z,
                                            const unsigned int num_atoms)
{
    const unsigned int i = blockIdx.x * blockDim.x * kCoarseningFactor + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= grid_size.x || j >= grid_size.y)
    {
        return;
    }
    float energy[kCoarseningFactor];
    for (unsigned int k{0U}; k < kCoarseningFactor; ++k)
    {
        energy[k] = 0.0f;
    }

    const float x = i * spacing;
    const float y = j * spacing;

    for (unsigned int k{0U}; k < num_atoms; ++k)
    {
        const Atom& atom = atoms_c[k];

        float dx = x - atom.x;
        const float dy = y - atom.y;
        const float dz = z - atom.z;
        const float dy2pdz2 = dy * dy + dz * dz;

        for (unsigned int l{0U}; l < kCoarseningFactor; ++l)
        {
            energy[l] += atom.charge / sqrtf(dx * dx + dy2pdz2);
            dx += blockDim.x * spacing;
        }
    }
    const int idx = LinearizeIndex(i, j, static_cast<int>(grid_size.x));
    for (unsigned int k{0U}; k < kCoarseningFactor; ++k)
    {
        if (i + blockDim.x * k < grid_size.x)
        {
            energy_grid[idx + blockDim.x * k] += energy[k];
        }
    }
}
