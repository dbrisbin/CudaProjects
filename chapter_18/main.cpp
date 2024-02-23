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
#include "dcs_driver.h"
#include "types/constants.h"
#include "types/types.h"
#include "utils.h"

void DcsCpu(float* energy_grid, const dim3 grid_size, const float spacing, const float z,
            const Atom* atoms, const unsigned int num_atoms)
{
    for (unsigned int k{0U}; k < num_atoms; ++k)
    {
        const auto& atom = atoms[k];
        const float dz = z - atom.z;
        const float dz2 = dz * dz;

        for (unsigned int j{0U}; j < grid_size.y; ++j)
        {
            const float y = static_cast<float>(j) * spacing;
            const float dy = y - atom.y;
            const float dy2 = dy * dy;

            for (unsigned int i{0U}; i < grid_size.x; ++i)
            {
                const float x = static_cast<float>(i) * spacing;
                const float dx = x - atom.x;
                const float r2 = dx * dx + dy2 + dz2;
                energy_grid[LinearizeIndex(i, j, grid_size.x)] += atom.charge / std::sqrt(r2);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3 || args.size() > 5)
    {
        std::cout << "Usage:\t" << argv[0] << "\tinput_file\tkernel_to_use (0-"
                  << DcsKernels::kNumKernels - 1 << ")\t[check_result (0/1) default=0]"
                  << std::endl;
        return 1;
    }

    std::ifstream file_ptr(args[1]);
    if (!file_ptr.is_open())
    {
        std::cout << "No such file " << args[1] << "." << std::endl;
        return 1;
    }

    const auto kernel_to_use = static_cast<DcsKernels>(std::stoi(args[2]));
    if (kernel_to_use >= DcsKernels::kNumKernels)
    {
        std::cout << "Invalid kernel number " << args[2] << "." << std::endl;
        return 1;
    }

    const bool check_result{(args.size() == 4) ? (std::stoi(args[3]) != 0) : false};

    unsigned int X;
    unsigned int Y;
    unsigned int Z;
    unsigned int N;
    file_ptr >> X >> Y >> Z >> N;

    auto atoms = std::make_unique<Atom[]>(N);

    auto energy_grid = std::make_unique<float[]>(X * Y * Z);

    for (unsigned int i{0U}; i < N; ++i)
    {
        file_ptr >> atoms[i].x >> atoms[i].y >> atoms[i].z >> atoms[i].charge;
    }
    file_ptr.close();

    const int iters{10};
    float time{0.F};
    const float spacing{0.1F};
    // compute the actual scan.
    // time =
    //     FhdDriver(r_phi.get(), r_d.get(), i_phi.get(), i_d.get(), x.get(), k_x.get(), y.get(),
    //               k_y.get(), z.get(), k_z.get(), k_struct.get(), M, N, r_fhd_actual.get(),
    //               i_fhd_actual.get(), kernel_to_use, iters, SECTION_SIZE);

    if (check_result)
    {  // compute the CPU scan.
        auto energy_grid_expected = std::make_unique<float[]>(X * Y * Z);
        std::fill(energy_grid_expected.get(), energy_grid_expected.get() + X * Y * Z, 0.F);

        // time the CPU scan.

        std::chrono::high_resolution_clock::time_point start_time =
            std::chrono::high_resolution_clock::now();
        for (unsigned int z{0}; z < Z; ++z)
        {
            DcsCpu(&energy_grid_expected[LinearizeIndex(0, 0, z, X, Y)], dim3{X, Y, Z}, spacing,
                   z * spacing, atoms.get(), N);
        }
        std::chrono::high_resolution_clock::time_point end_time =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end_time - start_time;

        // compare the results.
        float max_diff{0.F};
        int max_diff_index{0};

        for (unsigned int n{0U}; n < X * Y * Z; ++n)
        {
            if (std::abs(energy_grid[n] - energy_grid_expected[n]) > max_diff)
            {
                max_diff = std::abs(energy_grid[n] - energy_grid_expected[n]);
                max_diff_index = n;
            }
        }

        // print result
        for (unsigned int z{0}; z < Z; ++z)
        {
            std::cout << "z = " << z * spacing << std::endl;
            for (unsigned int j{0U}; j < Y; ++j)
            {
                for (unsigned int i{0U}; i < X; ++i)
                {
                    std::cout << energy_grid_expected[LinearizeIndex(i, j, z, X, Y)] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "Max difference: " << max_diff << " at index " << max_diff_index << std::endl;
        std::cout << "Time on CPU: " << duration.count() << " milliseconds for 1 iteration."
                  << std::endl;
    }
    std::cout << "Time on GPU: " << time << " milliseconds for " << iters << " iterations."
              << std::endl;

    return 0;
}