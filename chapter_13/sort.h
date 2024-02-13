/// @file sort.h
/// @brief Declaration of available kernels for sorting an array.

#ifndef CHAPTER_13_SORT_H
#define CHAPTER_13_SORT_H

#include <cuda_runtime.h>

/// @brief Compute an iteration of radix sort on input and store the result in output.
/// @param[out] output input and output for sorting
/// @param bits bit value of current radix
/// @param length length of input, output, and bits
/// @param iter current iteration of radix sort
/// @param flags flags used by streaming inclusive scan
/// @param scan_values scan values used by streaming inclusive scan
__global__ void RadixSortIter(unsigned int* data, unsigned int* output, unsigned int* bits_in,
                              unsigned int* bits_out, const unsigned int length,
                              const unsigned int iter, unsigned int* flags,
                              unsigned int* scan_values);

/// @brief Perform the first phase of an iteration of radix sort on input and store the result in
/// @param data the input data
/// @param output the output data
/// @param bits the bit values of the current radix
/// @param length the length of the input, output, and bits
/// @param iter the current iteration of radix sort
__global__ void RadixSortIterPhase1(const unsigned int* data, unsigned int* bits,
                                    const unsigned int length, const unsigned int iter);

/// @brief Perform the second phase of an iteration of radix sort on input and store the result in
/// @param data the input data
/// @param output the output data
/// @param bits the bit values of the current radix
/// @param length the length of the input, output, and bits
/// @param iter the current iteration of radix sort
__global__ void RadixSortIterPhase2(const unsigned int* data, unsigned int* output,
                                    const unsigned int* bits, const unsigned int length,
                                    const unsigned int iter);

#endif  // CHAPTER_13_SORT_H