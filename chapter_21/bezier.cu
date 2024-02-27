/// @file bezier.cu
/// @brief CUDA implementation of Bezier curve tessellation.

#include <cuda_runtime.h>
#include "bezier.h"
#include "bezier_utils.h"
#include "types/constants.h"
#include "types/types.h"

__global__ void ComputeBezierLinesBasic(BezierLineFixedSize* lines, const int num_lines)
{
    const unsigned int b_idx{blockIdx.x};
    if (b_idx < num_lines)
    {
        auto& line = lines[b_idx];
        const auto n_tess_points = ComputeNumberOfTessPoints(lines[b_idx]);
        line.num_vertices = n_tess_points;

        for (int inc{0}; inc < n_tess_points; inc += blockDim.x)
        {
            int idx = inc + threadIdx.x;
            if (idx < n_tess_points)
            {
                const float t = static_cast<float>(idx) / (n_tess_points - 1);
                const float t2 = t * t;
                const float one_minus_t = 1.F - t;
                const float one_minus_t2 = one_minus_t * one_minus_t;
                line.vertex_pos[idx] = one_minus_t2 * line.CP[0] +
                                       2.F * one_minus_t * t * line.CP[1] + t2 * line.CP[2];
            }
        }
    }
}

__global__ void ComputeBezierLinesDynamic(BezierLineDynamic* lines, const int num_lines)
{
    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (l_idx < num_lines)
    {
        const auto n_tess_points = ComputeNumberOfTessPoints(lines[l_idx]);
        lines[l_idx].num_vertices = n_tess_points;
        if (!cudaMalloc((void**)&lines[l_idx].vertex_pos, n_tess_points * sizeof(float2)))
        {
            ComputeBezierLineDynamic<<<ceil(static_cast<float>(n_tess_points) / 32.F), 32>>>(
                l_idx, lines, n_tess_points);
        }
    }
}

__global__ void ComputeBezierLineDynamic(const int line_idx, BezierLineDynamic* lines,
                                         const int n_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points)
    {
        const float t = static_cast<float>(idx) / (n_points - 1);
        const float t2 = t * t;
        const float one_minus_t = 1.F - t;
        const float one_minus_t2 = one_minus_t * one_minus_t;
        lines[line_idx].vertex_pos[idx] = one_minus_t2 * lines[line_idx].CP[0] +
                                          2.F * one_minus_t * t * lines[line_idx].CP[1] +
                                          t2 * lines[line_idx].CP[2];
    }
}

__global__ void FreeVertexMem(BezierLineDynamic* lines, const int num_lines)
{
    int l_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (l_idx < num_lines)
    {
        cudaFree(lines[l_idx].vertex_pos);
    }
}