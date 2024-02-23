#ifndef CHAPTER_18_UTILS_H
#define CHAPTER_18_UTILS_H

/// @brief Linearize a 2D index into a 1D index.
/// @param x horizontal index
/// @param y vertical index
/// @param X width
/// @return 1D index corresponding to the 2D index
int LinearizeIndex(const int x, const int y, const int X) { return y * X + x; }

/// @brief Linearize a 3D index into a 1D index.
/// @param x horizontal index
/// @param y vertical index
/// @param z depth index
/// @param X width
/// @param Y height
/// @return 1D index corresponding to the 3D index
int LinearizeIndex(const int x, const int y, const int z, const int X, const int Y)
{
    return z * X * Y + y * X + x;
}

#endif  // CHAPTER_18_UTILS_H