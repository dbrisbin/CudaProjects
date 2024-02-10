/// @file merge_driver.h
/// @brief Declaration of driver function to setup and call the appropriate kernel to compute the
/// merge of sorted arrays on GPU.

#ifndef CHAPTER_12_MERGE_DRIVER_H
#define CHAPTER_12_MERGE_DRIVER_H

#include <utility>
#include "types/constants.h"

/// @brief Setup and call the kernel specified.
/// @param A_h first array on host to merge
/// @param m length of A_h
/// @param B_h second array on host to merge
/// @param n length of B_h
/// @param C_h to store result on host
/// @param kernel_to_use kernel to use to compute the merge
/// @param iters number of iters for timing kernel
/// @return total time spent in the kernel (excludes setup)
float MergeDriver(const std::pair<int, int>* A_h, const int m, const std::pair<int, int>* B_h,
                  const int n, std::pair<int, int>* C_h, const MergeKernel kernel_to_use,
                  const int iters);

#endif  // CHAPTER_12_MERGE_DRIVER_H