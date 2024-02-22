#include <cuda_runtime.h>
#include "fhd.h"
#include "types/constants.h"
#include "types/types.h"

#ifdef USE_STRUCT
__constant__ KData k_c[kChunkSize];
__constant__ float k_x_c[1], k_y_c[1], k_z_c[1];
#else
__constant__ KData k_c[1];
__constant__ float k_x_c[kChunkSize], k_y_c[kChunkSize], k_z_c[kChunkSize];
#endif

__global__ void BasicKernel(const float* r_phi, const float* r_d, const float* i_phi,
                            const float* i_d, const float* x, const float* k_x, const float* y,
                            const float* k_y, const float* z, const float* k_z, float* r_mu,
                            float* i_mu, const int M, const int N, float* r_fhd, float* i_fhd)
{
    const auto m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M)
    {
        r_mu[m] = r_phi[m] * r_d[m] + i_phi[m] * i_d[m];
        i_mu[m] = r_phi[m] * i_d[m] - i_phi[m] * r_d[m];
        for (int n{0}; n < N; n++)
        {
            const auto exp_fhd = 2 * PI * (k_x[m] * x[n] + k_y[m] * y[n] + k_z[m] * z[n]);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            atomicAdd(&r_fhd[n], r_mu[m] * cos_fhd - i_mu[m] * sin_fhd);
            atomicAdd(&i_fhd[n], i_mu[m] * cos_fhd + r_mu[m] * sin_fhd);
        }
    }
}

__global__ void ComputeMu(const float* r_phi, const float* r_d, const float* i_phi,
                          const float* i_d, float* r_mu, float* i_mu, const int M)
{
    const auto m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M)
    {
        r_mu[m] = r_phi[m] * r_d[m] + i_phi[m] * i_d[m];
        i_mu[m] = r_phi[m] * i_d[m] - i_phi[m] * r_d[m];
    }
}

__global__ void ComputeFHDWithNThreads(const float* x, const float* k_x, const float* y,
                                       const float* k_y, const float* z, const float* k_z,
                                       const float* r_mu, const float* i_mu, const int M,
                                       const int N, float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_x[m] * x[n] + k_y[m] * y[n] + k_z[m] * z[n]);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            r_fhd[n] += r_mu[m] * cos_fhd - i_mu[m] * sin_fhd;
            i_fhd[n] += i_mu[m] * cos_fhd + r_mu[m] * sin_fhd;
        }
    }
}

__global__ void ComputeFHDWithNThreadsAndRegisters(const float* x, const float* k_x, const float* y,
                                                   const float* k_y, const float* z,
                                                   const float* k_z, const float* r_mu,
                                                   const float* i_mu, const int M, const int N,
                                                   float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N)
    {
        const auto x_n_r = x[n];
        const auto y_n_r = y[n];
        const auto z_n_r = z[n];
        float r_fhd_n_r{0.F};
        float i_fhd_n_r{0.F};
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_x[m] * x_n_r + k_y[m] * y_n_r + k_z[m] * z_n_r);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            r_fhd_n_r += r_mu[m] * cos_fhd - i_mu[m] * sin_fhd;
            i_fhd_n_r += i_mu[m] * cos_fhd + r_mu[m] * sin_fhd;
        }
        r_fhd[n] = r_fhd_n_r;
        i_fhd[n] = i_fhd_n_r;
    }
}

__global__ void ComputeFHDWithNThreadsRegistersAndRestrict(
    const float* x, const float* __restrict__ k_x, const float* y, const float* __restrict__ k_y,
    const float* z, const float* __restrict__ k_z, const float* __restrict__ r_mu,
    const float* __restrict__ i_mu, const int M, const int N, float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N)
    {
        const auto x_n_r = x[n];
        const auto y_n_r = y[n];
        const auto z_n_r = z[n];
        float r_fhd_n_r{0.F};
        float i_fhd_n_r{0.F};
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_x[m] * x_n_r + k_y[m] * y_n_r + k_z[m] * z_n_r);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            r_fhd_n_r += r_mu[m] * cos_fhd - i_mu[m] * sin_fhd;
            i_fhd_n_r += i_mu[m] * cos_fhd + r_mu[m] * sin_fhd;
        }
        r_fhd[n] = r_fhd_n_r;
        i_fhd[n] = i_fhd_n_r;
    }
}

__global__ void ComputeFHDWithNThreadsRegistersAndConstantMem(const float* x, const float* y,
                                                              const float* z, const float* r_mu,
                                                              const float* i_mu, const int M,
                                                              const int N, const int M_offset,
                                                              float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N)
    {
        const auto x_n_r = x[n];
        const auto y_n_r = y[n];
        const auto z_n_r = z[n];
        float r_fhd_n_r{0.F};
        float i_fhd_n_r{0.F};
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_x_c[m] * x_n_r + k_y_c[m] * y_n_r + k_z_c[m] * z_n_r);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            r_fhd_n_r += r_mu[m + M_offset] * cos_fhd - i_mu[m + M_offset] * sin_fhd;
            i_fhd_n_r += i_mu[m + M_offset] * cos_fhd + r_mu[m + M_offset] * sin_fhd;
        }
        r_fhd[n] += r_fhd_n_r;
        i_fhd[n] += i_fhd_n_r;
    }
}

__global__ void ComputeFHDWithNThreadsRegistersAndConstantMemStruct(
    const float* x, const float* y, const float* z, const float* r_mu, const float* i_mu,
    const int M, const int N, const int M_offset, float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N)
    {
        const auto x_n_r = x[n];
        const auto y_n_r = y[n];
        const auto z_n_r = z[n];
        float r_fhd_n_r{0.F};
        float i_fhd_n_r{0.F};
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_c[m].x * x_n_r + k_c[m].y * y_n_r + k_c[m].z * z_n_r);
            const auto cos_fhd = cos(exp_fhd);
            const auto sin_fhd = sin(exp_fhd);

            r_fhd_n_r += r_mu[m + M_offset] * cos_fhd - i_mu[m + M_offset] * sin_fhd;
            i_fhd_n_r += i_mu[m + M_offset] * cos_fhd + r_mu[m + M_offset] * sin_fhd;
        }
        r_fhd[n] += r_fhd_n_r;
        i_fhd[n] += i_fhd_n_r;
    }
}

__global__ void ComputeFHDWithNThreadsRegistersAndConstantMemStructDeviceTrig(
    const float* x, const float* y, const float* z, const float* r_mu, const float* i_mu,
    const int M, const int N, const int M_offset, float* r_fhd, float* i_fhd)
{
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N)
    {
        const auto x_n_r = x[n];
        const auto y_n_r = y[n];
        const auto z_n_r = z[n];
        float r_fhd_n_r{0.F};
        float i_fhd_n_r{0.F};
        for (int m{0}; m < M; m++)
        {
            const auto exp_fhd = 2 * PI * (k_c[m].x * x_n_r + k_c[m].y * y_n_r + k_c[m].z * z_n_r);
            const auto cos_fhd = __cosf(exp_fhd);
            const auto sin_fhd = __sinf(exp_fhd);

            r_fhd_n_r += r_mu[m + M_offset] * cos_fhd - i_mu[m + M_offset] * sin_fhd;
            i_fhd_n_r += i_mu[m + M_offset] * cos_fhd + r_mu[m + M_offset] * sin_fhd;
        }
        r_fhd[n] += r_fhd_n_r;
        i_fhd[n] += i_fhd_n_r;
    }
}
