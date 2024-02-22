#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "fhd_driver.h"
#include "types/constants.h"
#include "types/types.h"

void FhdCpu(const float* r_phi, const float* r_d, const float* i_phi, const float* i_d,
            const float* x, const float* k_x, const float* y, const float* k_y, const float* z,
            const float* k_z, const int M, const int N, float* r_fhd, float* i_fhd)
{
    auto r_mu = std::make_unique<float[]>(M);
    auto i_mu = std::make_unique<float[]>(M);
    for (int m{0}; m < M; ++m)
    {
        r_mu[m] = r_phi[m] * r_d[m] + i_phi[m] * i_d[m];
        i_mu[m] = r_phi[m] * i_d[m] - i_phi[m] * r_d[m];
        for (int n{0}; n < N; ++n)
        {
            const float exp_fhd = 2 * PI * (x[n] * k_x[m] + y[n] * k_y[m] + z[n] * k_z[m]);
            const float cos_fhd = std::cos(exp_fhd);
            const float sin_fhd = std::sin(exp_fhd);

            r_fhd[n] += r_mu[m] * cos_fhd - i_mu[m] * sin_fhd;
            i_fhd[n] += r_mu[m] * sin_fhd + i_mu[m] * cos_fhd;
        }
    }
}

int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3 || args.size() > 6)
    {
        std::cout << "Usage:\t" << argv[0] << "\tinput_file\tkernel_to_use (0-"
                  << FhdKernels::kNumKernels - 1
                  << ")\t[check_result (0/1) default=0]\t[optimize_block_size (0/1) default=0]"
                  << std::endl;
        return 1;
    }

    std::ifstream file_ptr(args[1]);
    if (!file_ptr.is_open())
    {
        std::cout << "No such file " << args[1] << "." << std::endl;
        return 1;
    }

    const auto kernel_to_use = static_cast<FhdKernels>(std::stoi(args[2]));
    if (kernel_to_use >= FhdKernels::kNumKernels)
    {
        std::cout << "Invalid kernel number " << args[2] << "." << std::endl;
        return 1;
    }

    const bool check_result{(args.size() >= 4) ? (std::stoi(args[3]) != 0) : false};
    const bool optimize_block_size{(args.size() == 5) ? (std::stoi(args[4]) != 0) : false};

    int M;
    int N;
    file_ptr >> M >> N;

    auto r_phi = std::make_unique<float[]>(M);
    auto i_phi = std::make_unique<float[]>(M);
    auto r_d = std::make_unique<float[]>(M);
    auto i_d = std::make_unique<float[]>(M);

    auto k_x = std::make_unique<float[]>(M);
    auto k_y = std::make_unique<float[]>(M);
    auto k_z = std::make_unique<float[]>(M);
    auto x = std::make_unique<float[]>(N);
    auto y = std::make_unique<float[]>(N);
    auto z = std::make_unique<float[]>(N);

    auto r_fhd_actual = std::make_unique<float[]>(N);
    auto i_fhd_actual = std::make_unique<float[]>(N);

    std::copy_n(std::istream_iterator<float>(file_ptr), M, r_phi.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, i_phi.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, r_d.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, i_d.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, k_x.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, k_y.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), M, k_z.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), N, x.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), N, y.get());
    std::copy_n(std::istream_iterator<float>(file_ptr), N, z.get());

    file_ptr.close();

    auto k_struct = std::make_unique<KData[]>(M);
    for (int m{0}; m < M; ++m)
    {
        k_struct[m].x = k_x[m];
        k_struct[m].y = k_y[m];
        k_struct[m].z = k_z[m];
    }
    const int iters{10};
    float time{0.F};
    if (optimize_block_size)
    {
        std::array<int, 15> block_sizes{1,   2,   4,   8,   16,  32,  64,  128,
                                        256, 384, 512, 640, 768, 896, 1024};
        std::array<float, 15> times;
        for (int i{0}; i < 15; ++i)
        {
            times[i] = FhdDriver(r_phi.get(), r_d.get(), i_phi.get(), i_d.get(), x.get(), k_x.get(),
                                 y.get(), k_y.get(), z.get(), k_z.get(), k_struct.get(), M, N,
                                 r_fhd_actual.get(), i_fhd_actual.get(), kernel_to_use, iters,
                                 block_sizes[i]);
        }
        std::cout << "Block size:\tTime (ms)" << std::endl;
        for (int i{0}; i < 15; ++i)
        {
            std::cout << block_sizes[i] << ":\t" << times[i] << std::endl;
        }
        time = *std::min_element(times.begin(), times.end());
    }
    else
    {
        // compute the actual scan.
        time =
            FhdDriver(r_phi.get(), r_d.get(), i_phi.get(), i_d.get(), x.get(), k_x.get(), y.get(),
                      k_y.get(), z.get(), k_z.get(), k_struct.get(), M, N, r_fhd_actual.get(),
                      i_fhd_actual.get(), kernel_to_use, iters, SECTION_SIZE);
    }

    if (check_result)
    {  // compute the CPU scan.
        auto r_fhd_expected = std::make_unique<float[]>(N);
        auto i_fhd_expected = std::make_unique<float[]>(N);

        std::fill(r_fhd_expected.get(), r_fhd_expected.get() + N, 0.0f);
        std::fill(i_fhd_expected.get(), i_fhd_expected.get() + N, 0.0f);

        // time the CPU scan.

        std::chrono::high_resolution_clock::time_point start_time =
            std::chrono::high_resolution_clock::now();
        FhdCpu(r_phi.get(), r_d.get(), i_phi.get(), i_d.get(), x.get(), k_x.get(), y.get(),
               k_y.get(), z.get(), k_z.get(), M, N, r_fhd_expected.get(), i_fhd_expected.get());
        std::chrono::high_resolution_clock::time_point end_time =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end_time - start_time;

        // compare the results.
        float max_diff_r{0.F};
        float max_diff_i{0.F};
        int max_diff_index_r{0};
        int max_diff_index_i{0};

        for (int n{0}; n < N; ++n)
        {
            if (std::abs(r_fhd_expected[n] - r_fhd_actual[n]) > max_diff_r)
            {
                max_diff_r = std::abs(r_fhd_expected[n] - r_fhd_actual[n]);
                max_diff_index_r = n;
            }
            if (std::abs(i_fhd_expected[n] - i_fhd_actual[n]) > max_diff_i)
            {
                max_diff_i = std::abs(i_fhd_expected[n] - i_fhd_actual[n]);
                max_diff_index_i = n;
            }
        }

        std::cout << "Max difference in real part: " << max_diff_r << " at index "
                  << max_diff_index_r << std::endl;
        std::cout << "Max difference in imaginary part: " << max_diff_i << " at index "
                  << max_diff_index_i << std::endl;
        std::cout << "Time on CPU: " << duration.count() << " milliseconds for 1 iteration."
                  << std::endl;
    }
    std::cout << "Time on GPU: " << time << " milliseconds for " << iters << " iterations."
              << std::endl;

    return 0;
}