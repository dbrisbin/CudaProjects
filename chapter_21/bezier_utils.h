/// @file utils.h
/// @brief Utility functions and operators for Bezier curve tessellation.

#ifndef CHAPTER_21_BEZIER_UTILS_H
#define CHAPTER_21_BEZIER_UTILS_H

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include "types/types.h"

/// @brief Compute multiplication of a 2D vector by a scalar.
/// @param a 2D vector
/// @param b scalar
/// @return a * b
__device__ __host__ inline float2 operator*(const float2& a, const float b)
{
    return {a.x * b, a.y * b};
}

/// @brief Compute multiplication of a scalar by a 2D vector.
/// @param a scalar
/// @param b 2D vector
/// @return b * a
__device__ __host__ inline float2 operator*(const float a, const float2& b) { return b * a; }

/// @brief Compute addition of two 2D vectors.
/// @param a 2D vector
/// @param b 2D vector
/// @return a + b
__device__ __host__ inline float2 operator+(const float2& a, const float2& b)
{
    return {a.x + b.x, a.y + b.y};
}

/// @brief Compute subtraction of two 2D vectors.
/// @param a 2D vector
/// @param b 2D vector
/// @return a - b
__device__ __host__ inline float2 operator-(const float2& a, const float2& b)
{
    return {a.x - b.x, a.y - b.y};
}

/// @brief Compute norm of a 2D vector.
/// @param a 2D vector
/// @return norm of a
__device__ __host__ inline float Norm(const float2& a) { return std::sqrt(a.x * a.x + a.y * a.y); }

/// @brief Compute dot product of two 2D vectors.
/// @param a 2D vector
/// @param b 2D vector
/// @return dot product of a and b
__device__ __host__ inline float Dot(const float2& a, const float2& b)
{
    return a.x * b.x + a.y * b.y;
}

/// @brief Compute maximum curvature of a quadratic Bezier curve.
/// @tparam BezierLine type of quadratic Bezier curve
/// @param line quadratic Bezier curve
/// @return maximum curvature
template <typename BezierLine>
__device__ __host__ float ComputeCurvatureQuadratic(const BezierLine& line)
{
    const auto B = [&line](float t) -> float2 {
        return (1.F - t) * ((1.F - t) * line.CP[0] + t * line.CP[1]) +
               t * ((1.F - t) * line.CP[1] + t * line.CP[2]);
    };
    const auto dB = [&line](float t) -> float2 {
        return 2.F * ((1.F - t) * (line.CP[1] - line.CP[0]) + t * (line.CP[2] - line.CP[1]));
    };
    const auto ddB = [&line](float t) -> float2 {
        (void)t;
        return 2.F * (line.CP[2] - 2.F * line.CP[1] + line.CP[0]);
    };
    const auto numerator = [&dB, &ddB](float t) -> float {
        const auto dB_t = dB(t);
        const auto ddB_t = ddB(t);
        // Determinant of the matrix [dB_t, ddB_t]
        return dB_t.x * ddB_t.y - dB_t.y * ddB_t.x;
    };
    const auto denominator = [&B](float t) -> float {
        const auto norm_B_t = Norm(B(t));
        return norm_B_t * norm_B_t * norm_B_t;
    };

    // t such that derivative of curvature k'(t) = 0
    const float t_max{Dot(line.CP[1] - line.CP[0], line.CP[0] - 2.F * line.CP[1] + line.CP[2]) /
                      Norm(line.CP[0] - 2.F * line.CP[1] + line.CP[2])};

    // Evaluate curvature of the endpoints and at t_max
    const float curvature_t_max = abs(numerator(t_max) / denominator(t_max));
    const float curvature_0 = abs(numerator(0.F) / denominator(0.F));
    const float curvature_1 = abs(numerator(1.F) / denominator(1.F));

    const float max_curvature =
        curvature_0 > curvature_1 ? (curvature_0 > curvature_t_max ? curvature_0 : curvature_t_max)
                                  : (curvature_1 > curvature_t_max ? curvature_1 : curvature_t_max);
    return max_curvature;
}

// clamp x to range [a, b]
__device__ __host__ inline float Clamp(float x, float a, float b)
{
    if (x < a)
        return a;
    else if (x > b)
        return b;
    else
        return x;
}

/// @tparam BezierLine type of quadratic Bezier curve
template <typename BezierLine>
__device__ __host__ inline int ComputeNumberOfTessPoints(const BezierLine& line)
{
    const float max_curvature = ComputeCurvatureQuadratic(line);
    return Clamp(static_cast<int>(ceil(16.F * max_curvature)), 4, MAX_TESS_POINTS);
}
#endif  // CHAPTER_21_BEZIER_UTILS_H