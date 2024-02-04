#include "matrixUtils.h"
#include "stencil3d.h"
#include "types/constants.h"

__constant__ float c[NUM_STENCIL_POINTS];

void copyArrayToConstantMemory(float* arr, int size)
{
    cudaMemcpyToSymbol(c, arr, size);
}

__global__ void basicStencil(float* N, float* P, const int width, const int height, const int depth)
{
    const unsigned int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    const bool out_index_is_valid = (out_z >= 1 && out_z < depth - 1) &&
                                    (out_y >= 1 && out_y < height - 1) &&
                                    (out_x >= 1 && out_x < width - 1);
    if (out_index_is_valid)
    {
        P[linearized3DIndex(out_z, out_y, out_x, width, height)] =
            c[0] * N[linearized3DIndex(out_z, out_y, out_x, width, height)] +
            c[1] * N[linearized3DIndex(out_z, out_y, out_x - 1, width, height)] +
            c[2] * N[linearized3DIndex(out_z, out_y, out_x + 1, width, height)] +
            c[3] * N[linearized3DIndex(out_z, out_y - 1, out_x, width, height)] +
            c[4] * N[linearized3DIndex(out_z, out_y + 1, out_x, width, height)] +
            c[5] * N[linearized3DIndex(out_z - 1, out_y, out_x, width, height)] +
            c[6] * N[linearized3DIndex(out_z + 1, out_y, out_x, width, height)];
    }
}

__global__ void tilingStencil(float* N, float* P, const int width, const int height,
                              const int depth)
{
    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int z = blockIdx.z * OUT_TILE_DIM_3D + tz - 1;
    int y = blockIdx.y * OUT_TILE_DIM_3D + ty - 1;
    int x = blockIdx.x * OUT_TILE_DIM_3D + tx - 1;

    __shared__ float N_s[IN_TILE_DIM_3D][IN_TILE_DIM_3D][IN_TILE_DIM_3D];
    bool coords_are_in_N = (z >= 0 && z < depth && y >= 0 && y < height && x >= 0 && x < width);
    if (coords_are_in_N)
    {
        N_s[tz][ty][tx] = N[linearized3DIndex(z, y, x, width, height)];
    }
    __syncthreads();

    bool coords_are_in_output =
        (z >= 1 && z < depth - 1 && y >= 1 && y < height - 1 && x >= 1 && x < width - 1);
    bool thread_is_in_inner_N = (tz >= 1 && tz < IN_TILE_DIM_3D - 1 && ty >= 1 &&
                                 ty < IN_TILE_DIM_3D - 1 && tx >= 1 && tx < IN_TILE_DIM_3D - 1);
    if (coords_are_in_output && thread_is_in_inner_N)
    {
        // clang-format off
        P[linearized3DIndex(z, y, x, width, height)] =
            c[0] * N_s[tz][ty][tx] +
            c[1] * N_s[tz][ty][tx - 1] +
            c[2] * N_s[tz][ty][tx + 1] +
            c[3] * N_s[tz][ty - 1][tx] +
            c[4] * N_s[tz][ty + 1][tx] +
            c[5] * N_s[tz - 1][ty][tx] +
            c[6] * N_s[tz + 1][ty][tx];
        // clang-format on
    }
}

__global__ void threadCoarseningStencil(float* N, float* P, const int width, const int height,
                                        const int depth)
{
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int z_start = blockIdx.z * OUT_TILE_DIM_2D;
    int y = blockIdx.y * OUT_TILE_DIM_2D + ty - 1;
    int x = blockIdx.x * OUT_TILE_DIM_2D + tx - 1;
    __shared__ float N_prev_s[IN_TILE_DIM_2D][IN_TILE_DIM_2D];
    __shared__ float N_curr_s[IN_TILE_DIM_2D][IN_TILE_DIM_2D];
    __shared__ float N_next_s[IN_TILE_DIM_2D][IN_TILE_DIM_2D];

    bool x_and_y_in_N = y >= 0 && y < height && x >= 0 && x < width;
    bool prev_layer_is_in_N = (z_start - 1 >= 0 && z_start - 1 < depth && x_and_y_in_N);
    bool curr_layer_is_in_N = (z_start >= 0 && z_start < depth && x_and_y_in_N);
    bool next_layer_is_in_N;

    if (prev_layer_is_in_N)
    {
        N_prev_s[ty][tx] = N[linearized3DIndex(z_start - 1, y, x, width, height)];
    }
    if (curr_layer_is_in_N)
    {
        N_curr_s[ty][tx] = N[linearized3DIndex(z_start, y, x, width, height)];
    }

    bool thread_is_in_inner_N =
        (ty >= 1 && ty < IN_TILE_DIM_2D - 1 && tx >= 1 && tx < IN_TILE_DIM_2D - 1);
    bool x_and_y_in_inner_N = y >= 1 && y < height - 1 && x >= 1 && x < width - 1;
    for (int z_curr = z_start; z_curr < z_start + OUT_TILE_DIM_2D; ++z_curr)
    {
        next_layer_is_in_N = (z_curr + 1 >= 0 && z_curr + 1 < depth && x_and_y_in_N);
        if (next_layer_is_in_N)
        {
            N_next_s[ty][tx] = N[linearized3DIndex(z_curr + 1, y, x, width, height)];
        }
        __syncthreads();
        bool coords_are_in_output = (z_curr >= 1 && z_curr < depth - 1 && x_and_y_in_inner_N);
        if (coords_are_in_output && thread_is_in_inner_N)
        {
            // clang-format off
                P[linearized3DIndex(z_curr, y, x, width, height)] =
                    c[0] * N_curr_s[ty][tx] +
                    c[1] * N_curr_s[ty][tx - 1] +
                    c[2] * N_curr_s[ty][tx + 1] +
                    c[3] * N_curr_s[ty - 1][tx] +
                    c[4] * N_curr_s[ty + 1][tx] +
                    c[5] * N_prev_s[ty][tx] +
                    c[6] * N_next_s[ty][tx];
            // clang-format on
        }
        __syncthreads();
        N_prev_s[ty][tx] = N_curr_s[ty][tx];
        N_curr_s[ty][tx] = N_next_s[ty][tx];
    }
}

__global__ void registerTilingStencil(float* N, float* P, const int width, const int height,
                                      const int depth)
{
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int z_start = blockIdx.z * OUT_TILE_DIM_2D;
    int y = blockIdx.y * OUT_TILE_DIM_2D + ty - 1;
    int x = blockIdx.x * OUT_TILE_DIM_2D + tx - 1;
    float n_prev = 0.0f;
    __shared__ float N_curr_s[IN_TILE_DIM_2D][IN_TILE_DIM_2D];
    float n_next = 0.0f;

    bool x_and_y_in_N = y >= 0 && y < height && x >= 0 && x < width;
    bool prev_layer_is_in_N = (z_start - 1 >= 0 && z_start - 1 < depth && x_and_y_in_N);
    bool curr_layer_is_in_N = (z_start >= 0 && z_start < depth && x_and_y_in_N);
    bool next_layer_is_in_N;

    if (prev_layer_is_in_N)
    {
        n_prev = N[linearized3DIndex(z_start - 1, y, x, width, height)];
    }
    if (curr_layer_is_in_N)
    {
        N_curr_s[ty][tx] = N[linearized3DIndex(z_start, y, x, width, height)];
    }

    bool thread_is_in_inner_N =
        (ty >= 1 && ty < IN_TILE_DIM_2D - 1 && tx >= 1 && tx < IN_TILE_DIM_2D - 1);
    bool x_and_y_in_inner_N = y >= 1 && y < height - 1 && x >= 1 && x < width - 1;
    for (int z_curr = z_start; z_curr < z_start + OUT_TILE_DIM_2D; ++z_curr)
    {
        next_layer_is_in_N = (z_curr + 1 >= 0 && z_curr + 1 < depth && x_and_y_in_N);
        if (next_layer_is_in_N)
        {
            n_next = N[linearized3DIndex(z_curr + 1, y, x, width, height)];
        }
        __syncthreads();
        bool coords_are_in_output = (z_curr >= 1 && z_curr < depth - 1 && x_and_y_in_inner_N);
        if (coords_are_in_output && thread_is_in_inner_N)
        {
            // clang-format off
                P[linearized3DIndex(z_curr, y, x, width, height)] =
                    c[0] * N_curr_s[ty][tx] +
                    c[1] * N_curr_s[ty][tx - 1] +
                    c[2] * N_curr_s[ty][tx + 1] +
                    c[3] * N_curr_s[ty - 1][tx] +
                    c[4] * N_curr_s[ty + 1][tx] +
                    c[5] * n_prev +
                    c[6] * n_next;
            // clang-format on
        }

        n_prev = N_curr_s[ty][tx];
        __syncthreads();
        N_curr_s[ty][tx] = n_next;
    }
}
