/// @file bezier_main.cpp
/// @brief main function for bezier curve calculations

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "bezier.h"
#include "bezier_utils.h"
#include "types/constants.h"
#include "types/types.h"

void MemcpyBezierLinesToHost(BezierLineFixedSize* dst, const BezierLineFixedSize* src,
                             const unsigned int N)
{
    cudaMemcpy(dst, src, N * sizeof(BezierLineFixedSize), cudaMemcpyDeviceToHost);
    for (unsigned int i{0U}; i < N; ++i)
    {
        cudaMemcpy(&dst[i].vertex_pos, &src[i].vertex_pos, dst[i].num_vertices * sizeof(float2),
                   cudaMemcpyDeviceToHost);
    }
}

/// @brief Compute the quadratic Bezier curve.
/// @param[in,out] lines The quadratic Bezier curves.
/// @param num_lines The number of quadratic Bezier curves.
void ComputeBezierLinesCPU(BezierLineFixedSize* lines, const unsigned int num_lines)
{
    for (unsigned int i{0U}; i < num_lines; ++i)
    {
        const auto num_tess_points = ComputeNumberOfTessPoints(lines[i]);
        lines[i].num_vertices = num_tess_points;
        for (int j{0}; j < num_tess_points; ++j)
        {
            const float t = static_cast<float>(j) / (num_tess_points - 1);
            const float t2 = t * t;
            const float one_minus_t = 1.F - t;
            const float one_minus_t2 = one_minus_t * one_minus_t;
            lines[i].vertex_pos[j] = one_minus_t2 * lines[i].CP[0] +
                                     2.F * one_minus_t * t * lines[i].CP[1] + t2 * lines[i].CP[2];
        }
    }
}

/// @brief Compare two quadratic Bezier curves.
/// @param a The first quadratic Bezier curve.
/// @param b The second quadratic Bezier curve.
/// @return The maximum difference between the two curves.
template <typename TBezierLine1, typename TBezierLine2>
std::tuple<float, int, int> CompareBezierLines(const TBezierLine1* a, const TBezierLine2* b,
                                               const unsigned int num_lines)
{
    float max_diff{0.F};
    int max_diff_curve_idx{0};
    int max_diff_pt_idx{0};
    for (unsigned int i{0U}; i < num_lines; ++i)
    {
        for (int j{0}; j < b[i].num_vertices; ++j)
        {
            const float diff = Norm(a[i].vertex_pos[j] - b[i].vertex_pos[j]);
            if (diff > max_diff)
            {
                max_diff = diff;
                max_diff_curve_idx = i;
                max_diff_pt_idx = j;
            }
        }
    }
    return {max_diff, max_diff_curve_idx, max_diff_pt_idx};
}

int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3 || args.size() > 5)
    {
        std::cout << "Usage:\t" << argv[0] << "\tinput_file\tkernel_to_use (0-"
                  << BezierKernels::kNumKernels - 1 << ")\t[check_result (0/1) default=0]"
                  << std::endl;
        return 1;
    }

    std::ifstream file_ptr(args[1]);
    if (!file_ptr.is_open())
    {
        std::cout << "No such file " << args[1] << "." << std::endl;
        return 1;
    }

    const auto kernel_to_use = static_cast<BezierKernels>(std::stoi(args[2]));
    if (kernel_to_use >= BezierKernels::kNumKernels)
    {
        std::cout << "Invalid kernel number " << args[2] << "." << std::endl;
        return 1;
    }

    const bool check_result{(args.size() == 4) ? (std::stoi(args[3]) != 0) : false};

    unsigned int N{};
    file_ptr >> N;

    auto curves = std::make_unique<BezierLineFixedSize[]>(N);
    auto curves_dynamic = std::make_unique<BezierLineDynamic[]>(N);
    for (unsigned int i{0U}; i < N; ++i)
    {
        file_ptr >> curves[i].CP[0].x >> curves[i].CP[0].y >> curves[i].CP[1].x >>
            curves[i].CP[1].y >> curves[i].CP[2].x >> curves[i].CP[2].y;
        curves_dynamic[i].CP[0] = curves[i].CP[0];
        curves_dynamic[i].CP[1] = curves[i].CP[1];
        curves_dynamic[i].CP[2] = curves[i].CP[2];
    }
    file_ptr.close();

    const int iters{10};

    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    for (int i{0}; i < iters; ++i)
    {
        if (kernel_to_use == BezierKernels::kBasic)
        {
            BezierLineFixedSize* d_lines;
            cudaMalloc(&d_lines, N * sizeof(BezierLineFixedSize));
            cudaMemcpy(d_lines, curves.get(), N * sizeof(BezierLineFixedSize),
                       cudaMemcpyHostToDevice);
            ComputeBezierLinesBasic<<<N, 32>>>(d_lines, N);
            MemcpyBezierLinesToHost(curves.get(), d_lines, N);
            cudaFree(d_lines);
        }
        else
        {
            BezierLineDynamic* d_lines;
            cudaMalloc(&d_lines, N * sizeof(BezierLineDynamic));
            cudaMemcpy(d_lines, curves_dynamic.get(), N * sizeof(BezierLineDynamic),
                       cudaMemcpyHostToDevice);
            ComputeBezierLinesDynamic<<<N, 32>>>(d_lines, N);
            cudaMemcpy(curves_dynamic.get(), d_lines, N * sizeof(BezierLineDynamic),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_lines);
        }
    }
    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = end_time - start_time;

    if (check_result)
    {
        auto curves_expected = std::make_unique<BezierLineFixedSize[]>(N);
        for (unsigned int i{0U}; i < N; ++i)
        {
            curves_expected[i].CP[0] = curves[i].CP[0];
            curves_expected[i].CP[1] = curves[i].CP[1];
            curves_expected[i].CP[2] = curves[i].CP[2];
        }

        // time the CPU scan.
        start_time = std::chrono::high_resolution_clock::now();
        ComputeBezierLinesCPU(curves_expected.get(), N);
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = end_time - start_time;

        // compare the results.
        std::tuple<float, int, int> max_diff{};
        if (kernel_to_use == BezierKernels::kBasic)
        {
            max_diff = CompareBezierLines(curves.get(), curves_expected.get(), N);
        }
        else
        {
            max_diff = CompareBezierLines(curves_dynamic.get(), curves_expected.get(), N);
        }
        std::cout << "Max difference: " << std::get<0>(max_diff) << " at curve index "
                  << std::get<1>(max_diff) << " and point index " << std::get<2>(max_diff)
                  << std::endl;
        std::cout << "Time on CPU: " << cpu_duration.count() << " milliseconds for 1 iteration."
                  << std::endl;
    }
    std::cout << "Time on GPU: " << gpu_duration.count() << " milliseconds for " << iters
              << " iterations." << std::endl;

    return 0;
}