#include <cuda_runtime.h>
#include "fhd.h"
#include "types/constants.h"

__global__ void BasicKernel(const float* r_phi, const float* r_d, const float* i_phi,
                            const float* i_d, const float* x, const float* k_x, const float* y,
                            const float* k_y, const float* z, const float* k_z, float* r_mu,
                            float* i_mu, const int M, const int N, float* r_fhd, float* i_fhd)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M)
    {
        r_mu[m] = r_phi[m] * r_d[m] + i_phi[m] * i_d[m];
        i_mu[m] = r_phi[m] * i_d[m] - i_phi[m] * r_d[m];
        for (int n = 0; n < N; n++)
        {
            const float exp_fhd = 2 * PI * (k_x[m] * x[n] + k_y[m] * y[n] + k_z[m] * z[n]);
            const float cos_fhd = cos(exp_fhd);
            const float sin_fhd = sin(exp_fhd);

            atomicAdd(&r_fhd[n], r_mu[m] * cos_fhd - i_mu[m] * sin_fhd);
            atomicAdd(&i_fhd[n], i_mu[m] * cos_fhd + r_mu[m] * sin_fhd);
        }
    }
}