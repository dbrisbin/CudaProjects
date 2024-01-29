#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void HelloWorld() {
    // data[0] = 5;
    printf("Hello world\n");
}

int main() {
    // int d_data[5];
    // cudaMalloc((void**)&d_data, 5 * sizeof(int));
    HelloWorld<<<1, 1>>>();
    // printf("%d\n", d_data[0]);
}