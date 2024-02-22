#include <iostream>
#include "fhd.h"
#include "fhd_driver.h"

float FhdDriver(const float* r_phi, const float* r_d, const float* i_phi, const float* i_d,
                const float* x, const float* k_x, const float* y, const float* k_y, const float* z,
                const float* k_z, const int M, const int N, float* r_fhd, float* i_fhd,
                const FhdKernels kernel_to_use, const int iters)
{
    // copy data to device,
    float *d_r_phi, *d_r_d, *d_i_phi, *d_i_d, *d_x, *d_k_x, *d_y, *d_k_y, *d_z, *d_k_z, *d_r_mu,
        *d_i_mu, *d_r_fhd, *d_i_fhd;
    cudaMalloc((void**)&d_r_phi, M * sizeof(float));
    cudaMalloc((void**)&d_r_d, M * sizeof(float));
    cudaMalloc((void**)&d_i_phi, M * sizeof(float));
    cudaMalloc((void**)&d_i_d, M * sizeof(float));
    cudaMalloc((void**)&d_k_x, M * sizeof(float));
    cudaMalloc((void**)&d_k_y, M * sizeof(float));
    cudaMalloc((void**)&d_k_z, M * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_z, N * sizeof(float));

    cudaMalloc((void**)&d_r_mu, M * sizeof(float));
    cudaMalloc((void**)&d_i_mu, M * sizeof(float));
    cudaMalloc((void**)&d_r_fhd, N * sizeof(float));
    cudaMalloc((void**)&d_i_fhd, N * sizeof(float));

    cudaMemcpy(d_r_phi, r_phi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_d, r_d, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_phi, i_phi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_d, i_d, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_x, k_x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_y, k_y, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_z, k_z, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
    {
        cudaMemset(d_r_fhd, 0, N * sizeof(float));
        cudaMemset(d_i_fhd, 0, N * sizeof(float));

        switch (kernel_to_use)
        {
            case FhdKernels::kBasic:
            {
                dim3 block_dim(SECTION_SIZE, 1, 1);
                dim3 grid_dim(std::ceil(static_cast<float>(M) / SECTION_SIZE), 1, 1);

                BasicKernel<<<grid_dim, block_dim>>>(d_r_phi, d_r_d, d_i_phi, d_i_d, d_x, d_k_x,
                                                     d_y, d_k_y, d_z, d_k_z, d_r_mu, d_i_mu, M, N,
                                                     d_r_fhd, d_i_fhd);
                break;
            }
            case FhdKernels::kLoopInterchangeBasic:
            {
                dim3 block_dim(SECTION_SIZE, 1, 1);
                dim3 grid_dim(std::ceil(static_cast<float>(M) / SECTION_SIZE), 1, 1);
                ComputeMu<<<grid_dim, block_dim>>>(d_r_phi, d_r_d, d_i_phi, d_i_d, d_r_mu, d_i_mu,
                                                   M);

                grid_dim.x = std::ceil(static_cast<float>(N) / SECTION_SIZE);
                ComputeFHDWithNThreads<<<grid_dim, block_dim>>>(
                    d_x, d_k_x, d_y, d_k_y, d_z, d_k_z, d_r_mu, d_i_mu, M, N, d_r_fhd, d_i_fhd);
                break;
            }
            case FhdKernels::kLoopInterchangeWithRegisters:
            {
                dim3 block_dim(SECTION_SIZE, 1, 1);
                dim3 grid_dim(std::ceil(static_cast<float>(M) / SECTION_SIZE), 1, 1);
                ComputeMu<<<grid_dim, block_dim>>>(d_r_phi, d_r_d, d_i_phi, d_i_d, d_r_mu, d_i_mu,
                                                   M);

                grid_dim.x = std::ceil(static_cast<float>(N) / SECTION_SIZE);
                ComputeFHDWithNThreadsAndRegisters<<<grid_dim, block_dim>>>(
                    d_x, d_k_x, d_y, d_k_y, d_z, d_k_z, d_r_mu, d_i_mu, M, N, d_r_fhd, d_i_fhd);
                break;
            }
            case FhdKernels::kLoopInterchangeWithRegistersAndRestrict:
            {
                dim3 block_dim(SECTION_SIZE, 1, 1);
                dim3 grid_dim(std::ceil(static_cast<float>(M) / SECTION_SIZE), 1, 1);
                ComputeMu<<<grid_dim, block_dim>>>(d_r_phi, d_r_d, d_i_phi, d_i_d, d_r_mu, d_i_mu,
                                                   M);

                grid_dim.x = std::ceil(static_cast<float>(N) / SECTION_SIZE);
                ComputeFHDWithNThreadsRegistersAndRestrict<<<grid_dim, block_dim>>>(
                    d_x, d_k_x, d_y, d_k_y, d_z, d_k_z, d_r_mu, d_i_mu, M, N, d_r_fhd, d_i_fhd);
                break;
            }
            case FhdKernels::kLoopInterchangeWithRegistersAndConstantMem:
            {
                dim3 block_dim(SECTION_SIZE, 1, 1);
                dim3 grid_dim(std::ceil(static_cast<float>(M) / SECTION_SIZE), 1, 1);
                ComputeMu<<<grid_dim, block_dim>>>(d_r_phi, d_r_d, d_i_phi, d_i_d, d_r_mu, d_i_mu,
                                                   M);

                grid_dim.x = std::ceil(static_cast<float>(N) / SECTION_SIZE);
                for (size_t i{0}; i < M / kChunkSize; i++)
                {
                    const auto M_offset = i * kChunkSize;
                    if (cudaMemcpyToSymbol(k_x_c, &k_x[M_offset], kChunkSize * sizeof(float)) ||
                        cudaMemcpyToSymbol(k_y_c, &k_y[M_offset], kChunkSize * sizeof(float)) ||
                        cudaMemcpyToSymbol(k_z_c, &k_z[M_offset], kChunkSize * sizeof(float)))
                    {
                        std::cerr << "Error copying to constant memory" << std::endl;
                        return -1.F;
                    }

                    ComputeFHDWithNThreadsRegistersAndConstantMem<<<grid_dim, block_dim>>>(
                        d_x, d_y, d_z, d_r_mu, d_i_mu, static_cast<int>(kChunkSize), N,
                        static_cast<int>(M_offset), d_r_fhd, d_i_fhd);
                }
                if (M % kChunkSize != 0)
                {
                    const auto M_offset = kChunkSize * (M / kChunkSize);
                    const int remaining = M % kChunkSize;
                    if (cudaMemcpyToSymbol(k_x_c, &k_x[M_offset], remaining * sizeof(float)) ||
                        cudaMemcpyToSymbol(k_y_c, &k_y[M_offset], remaining * sizeof(float)) ||
                        cudaMemcpyToSymbol(k_z_c, &k_z[M_offset], remaining * sizeof(float)))
                    {
                        std::cerr << "Error copying to constant memory" << std::endl;
                        return -1.F;
                    }

                    ComputeFHDWithNThreadsRegistersAndConstantMem<<<grid_dim, block_dim>>>(
                        d_x, d_y, d_z, d_r_mu, d_i_mu, remaining, N, static_cast<int>(M_offset),
                        d_r_fhd, d_i_fhd);
                }
                break;
            }
            case FhdKernels::kNumKernels:
            default:
                std::cerr << "Invalid kernel to use" << std::endl;
                break;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(r_fhd, d_r_fhd, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(i_fhd, d_i_fhd, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_r_phi);
    cudaFree(d_r_d);
    cudaFree(d_i_phi);
    cudaFree(d_i_d);
    cudaFree(d_x);
    cudaFree(d_k_x);
    cudaFree(d_y);
    cudaFree(d_k_y);
    cudaFree(d_z);
    cudaFree(d_k_z);
    cudaFree(d_r_fhd);
    cudaFree(d_i_fhd);

    return milliseconds;
}