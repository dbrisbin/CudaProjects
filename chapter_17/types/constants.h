#ifndef CHAPTER_17_TYPES_CONSTANTS_H
#define CHAPTER_17_TYPES_CONSTANTS_H

#define SECTION_SIZE 1024
#define PI 3.14159265358979323846F

/// @brief size of constant memory arrays
// Shared memory is 64KB, but we only want to use half of it to be safe.
static constexpr std::size_t kChunkSize{32U * 1024U / (sizeof(float) * 3U)};

/// @brief List of available kernels which are supported.
enum FhdKernels
{
    kBasic = 0,
    kLoopInterchangeBasic = 1,
    kLoopInterchangeWithRegisters = 2,
    kLoopInterchangeWithRegistersAndRestrict = 3,
    kLoopInterchangeWithRegistersAndConstantMem = 4,
    kLoopInterchangeWithRegistersAndConstantMemStruct = 5,
    kLoopInterchangeWithRegistersAndConstantMemStructAndDeviceTrig = 6,
    kNumKernels = 7,
};

#endif  // CHAPTER_17_TYPES_CONSTANTS_H