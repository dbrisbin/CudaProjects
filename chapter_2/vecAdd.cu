#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void vecAddOnHost(float *A_h, float *B_h, float *C_h, int n)
{
    for (int i = 0; i < n; ++i)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAddOnDevice(float *A_h, float *B_h, float *C_h, int n)
{
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n / 256.0), 256.0>>>(A_d, B_d, C_d, n);

    cudaError_t err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d.", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);
}

int main()
{
    float *A, *B, *C, *D;
    int n;

    printf("Enter number of elements:");
    scanf("%d", &n);

    A = (float *)malloc(n * sizeof(float));
    B = (float *)malloc(n * sizeof(float));
    C = (float *)malloc(n * sizeof(float));
    D = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i)
    {
        A[i] = i;
        B[i] = (n - i) * 2;
    }

    vecAddOnHost(A, B, C, n);
    vecAddOnDevice(A, B, D, n);

    printf("A: [");
    for (int i = 0; i < n; ++i)
    {
        printf("%0.1f, ", A[i]);
    }
    printf("]\n");

    printf("B: [");
    for (int i = 0; i < n; ++i)
    {
        printf("%0.1f, ", B[i]);
    }
    printf("]\n");

    printf("C: [");
    for (int i = 0; i < n; ++i)
    {
        printf("%0.1f, ", C[i]);
    }
    printf("]\n");

    printf("D: [");
    for (int i = 0; i < n; ++i)
    {
        printf("%0.1f, ", D[i]);
    }
    printf("]\n");
}
