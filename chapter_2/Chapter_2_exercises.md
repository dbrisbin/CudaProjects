1) If we want to use each thread in a grid to calculate one output element of a vector addition,
what would be the expression for mapping the thread/block indices to the data index (i)?
(A) i=threadIdx.x + threadIdx.y;
(B) i=blockIdx.x + threadIdx.x;
(C) i=blockIdx.x * blockDim.x + threadIdx.x;
(D) i=blockIdx.x * threadIdx.x
Answer: C

2) Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?
(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;
(B) i=blockIdx.x* threadIdx.x*2;
(C) i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
(D) i=blockIdx.x*blockDim.x*2 + threadIdx.x;
Answer: C

3) We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?
(A) i=blockIdx.x*blockDim.x + threadIdx.x +2;
(B) i=blockIdx.x*threadIdx.x*2;
(C) i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
(D) i=blockIdx.x*blockDim.x*2 + threadIdx.x;
Answer: C?

4) For a vector addition, assume that the vector length is 8000, each thread
calculates one output element, and the thread block size is 1024 threads. The
programmer configures the kernel call to have a minimum number of thread
(A) 8000
(B) 8196
(C) 8192
(D) 8200
Answer: C

5) If we want to allocate an array of v integer elements in the CUDA device
global memory, what would be an appropriate expression for the second
argument of the cudaMalloc call?
(A) n
(B) v
(C) n * sizeof(int)
(D) v * sizeof(int)
Answer: D

6) If we want to allocate an array of n floating-point elements and have a
floating-point pointer variable A_d to point to the allocated memory, what
would be an appropriate expression for the first argument of the cudaMalloc
() call?
(A) n
(B) (void *) A_d
(C) *A_d
(D) (void **) &A_d
Answer: D

7) If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer
to element 0 of the source array) to device array A_d (A_d is a pointer to
element 0 of the destination array), what would be an appropriate API call
for this data copy in CUDA?
(A) cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
(B) cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost);
(C) cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
(D) cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);
Answer: C

8) How would one declare a variable err that can appropriately receive the
returned value of a CUDA API call?
(A) int err;
(B) cudaError err;
(C) cudaError_t err;
(D) cudaSuccess_t err;
Answer: C

9) Consider the following CUDA kernel and the corresponding host function
that calls it:
01  __global__ void foo_kernel(float* a, float* b, unsigned int N){
02      unsigned int i=blockIdx.x*blockDim.x + threadIdx.x;
03      if(i < N) {
04          b[i]=2.7f*a[i] - 4.3f;
05      }
06  }
07  void foo(float* a_d, float* b_d) {
08      unsigned int N=200000;
09      foo_kernel<<<(N + 128-1)/128, 128>>>(a_d, b_d, N);
10  }
a. What is the number of threads per block? 128
b. What is the number of threads in the grid? 200,064
c. What is the number of blocks in the grid? 1,563
d. What is the number of threads that execute the code on line 02? 200,064
e. What is the number of threads that execute the code on line 04? 200,000

10) A new summer intern was frustrated with CUDA. He has been complaining
that CUDA is very tedious. He had to declare many functions that he plans
to execute on both the host and the device twice, once as a host function and
once as a device function. What is your response?
Bruh, you can use both __device__ and __host__ before a function definition/declaration and the compiler will generate two versions, one for host, one for device.