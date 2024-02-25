#ifndef CHAPTER_18_TYPES_CONSTANTS_H
#define CHAPTER_18_TYPES_CONSTANTS_H

#include "types.h"

/// @brief number of atoms per block for the scatter kernel.
static constexpr unsigned int k1DBlockSize{256U};
/// @brief number of grid points per block x dim for the gather kernels.
static constexpr unsigned int kBlockSizeX{32U};
/// @brief number of grid points per block y dim for the gather kernels.
static constexpr unsigned int kBlockSizeY{32U};
/// @brief number of grid points per block z dim for the gather kernels.
static constexpr unsigned int kBlockSizeZ{1U};
/// @brief coarsening factor for coarsening kernels.
static constexpr unsigned int kCoarseningFactor{24U};

/// @brief size of constant memory arrays
// Shared memory is 64KB, but we only want to use half of it to be safe.
static constexpr std::size_t kChunkSize{32U * 1024U / sizeof(Atom)};

/// @brief List of available kernels which are supported.
enum DcsKernels
{
    kScatter = 0,
    kGather = 1,
    kGatherCoarsened = 2,
    kGatherCoarsenedCoalesced = 3,
    kNumKernels = 4,
};

#endif  // CHAPTER_18_TYPES_CONSTANTS_H