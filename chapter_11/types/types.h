#ifndef CHAPTER_11_TYPES_TYPES_H
#define CHAPTER_11_TYPES_TYPES_H

/// @brief Parameters to control the type operated on by kernels. The three definitions should be
/// consistent. CHECKING FOR CONSISTENCY IS NOT PERFORMED.
/// @brief Control the format in scanf and printf functions.
#define FORMAT_TYPE "d"
/// @brief Control the level of precision when printing floating point types. Set to empty string
/// when using integral types.
#define FP_PRECISION ""
/// @brief Type used for parallel scan calculations.
typedef int ParallelScanDataType;

#endif  // CHAPTER_11_TYPES_TYPES_H