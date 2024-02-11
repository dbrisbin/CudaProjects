/// @file sort_driver.h
/// @brief Declaration of driver function to setup and call the appropriate kernel to sort an array
/// on GPU.

#ifndef CHAPTER_13_SORT_DRIVER_H
#define CHAPTER_13_SORT_DRIVER_H

#include <utility>
#include "types/constants.h"

/// @brief Setup and call the kernel specified.
/// @param data_h array on host to sort
/// @param result_h destination for result of sort
/// @param length length of array
/// @param kernel_to_use kernel to use to sort the data
/// @param iters number of iters for timing kernel
/// @return total time spent in the kernel (excludes setup)
float SortDriver(const unsigned int* data_h, unsigned int* result_h, const int length,
                 const SortKernel kernel_to_use, const int iters);

#endif  // CHAPTER_13_SORT_DRIVER_H