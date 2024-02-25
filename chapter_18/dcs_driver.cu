#include <iostream>
#include "dcs.h"
#include "dcs_driver.h"

float DcsDriver(float* energy_grid_h, const dim3 grid_size, const float spacing,
                const Atom* atoms_h, const unsigned int num_atoms, const DcsKernels kernel_to_use,
                const int iters)
{
    float* energy_grid_d{};

    cudaMalloc(&energy_grid_d, grid_size.x * grid_size.y * grid_size.z * sizeof(float));

    auto* kernel = &DcsScatter;
    dim3 block_dim{};
    dim3 grid_dim{};

    switch (kernel_to_use)
    {
        case DcsKernels::kScatter:
            kernel = &DcsScatter;
            block_dim = dim3(k1DBlockSize, 1U, 1U);
            grid_dim = dim3(
                static_cast<unsigned int>(std::ceil(
                    static_cast<float>(std::min(static_cast<std::size_t>(num_atoms), kChunkSize)) /
                    block_dim.x)),
                1U, 1U);
            break;
        case DcsKernels::kGather:
            kernel = &DcsGatherBasic;
            block_dim = dim3(kBlockSizeX, kBlockSizeY, 1U);
            grid_dim = dim3(
                static_cast<unsigned int>(std::ceil(static_cast<float>(grid_size.x) / block_dim.x)),
                static_cast<unsigned int>(std::ceil(static_cast<float>(grid_size.y) / block_dim.y)),
                1U);
            break;
        case DcsKernels::kGatherCoarsened:
        case DcsKernels::kGatherCoarsenedCoalesced:
            kernel = &DcsGatherCoarsened;
            block_dim = dim3(std::min(kBlockSizeX, grid_size.x / kCoarseningFactor),
                             std::min(kBlockSizeY, grid_size.y), 1U);
            grid_dim = dim3(
                static_cast<unsigned int>(
                    std::ceil(static_cast<float>(grid_size.x) / block_dim.x / kCoarseningFactor)),
                static_cast<unsigned int>(std::ceil(static_cast<float>(grid_size.y) / block_dim.y)),
                1U);
            break;
        case DcsKernels::kNumKernels:
        default:
            std::cerr << "Invalid kernel type" << std::endl;
            return -1.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i{0}; i < iters; ++i)
    {
        cudaMemset(energy_grid_d, 0, grid_size.x * grid_size.y * grid_size.z * sizeof(float));

        switch (kernel_to_use)
        {
            case DcsKernels::kScatter:
            case DcsKernels::kGather:
            case DcsKernels::kGatherCoarsened:
            case DcsKernels::kGatherCoarsenedCoalesced:
                for (std::size_t k = 0; k < grid_size.z; ++k)
                {
                    const float z = k * spacing;
                    const unsigned int energy_grid_offset = k * grid_size.x * grid_size.y;
                    for (std::size_t i = 0; i < num_atoms / kChunkSize; ++i)
                    {
                        const int start = i * kChunkSize;
                        cudaMemcpyToSymbol(atoms_c, &atoms_h[start], kChunkSize * sizeof(Atom));
                        (*kernel)<<<grid_dim, block_dim>>>(energy_grid_d + energy_grid_offset,
                                                           grid_size, spacing, z, kChunkSize);
                    }
                    if (num_atoms % kChunkSize != 0)
                    {
                        const std::size_t remaining = num_atoms % kChunkSize;
                        const int start = (num_atoms / kChunkSize) * kChunkSize;
                        cudaMemcpyToSymbol(atoms_c, &atoms_h[start], remaining * sizeof(Atom));
                        (*kernel)<<<grid_dim, block_dim>>>(energy_grid_d + energy_grid_offset,
                                                           grid_size, spacing, z, remaining);
                    }
                }
                break;
            case DcsKernels::kNumKernels:
            default:
                std::cerr << "Invalid kernel type" << std::endl;
                return -1.0f;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(energy_grid_h, energy_grid_d,
               grid_size.x * grid_size.y * grid_size.z * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(energy_grid_d);

    return milliseconds;
}