#include "mpi.h"

void compute_node_stencil(int dimx, int dimy, int dimz, int nreps)
{
    int np, pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int server_process = np - 1;
    unsigned int num_points = dimx * dimy * (dimz + 8);
    unsigned int num_bytes = num_points * sizeof(float);
    unsigned int num_halo_points = 4 * dimx * dimy;
    unsigned int num_halo_bytes = num_halo_points * sizeof(float);
    /* Allocate host memory */
    float* h_input = (float*)malloc(num_bytes);
    /* Allocate device memory for input and output data */
    float* d_input = NULL;
    cudaMalloc((void**)&d_input, num_bytes);
    float* rcv_address = h_input + ((0 == pid) ? num_halo_points : 0);
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);
    float *h_output = NULL, *d_output = NULL;
    float* h_output = (float*)malloc(num_bytes);
    cudaMalloc((void**)&d_output, num_bytes);
    float *h_left_boundary = NULL, *h_right_boundary = NULL;
    float *h_left_halo = NULL, *h_right_halo = NULL;
    /* Allocate host memory for halo data */
    cudaHostAlloc((void**)&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_right_halo, num_halo_bytes, cudaHostAllocDefault);
    /* Create streams used for stencil computation */
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    MPI_Status status;
    int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 2) ? (pid + 1) : MPI_PROC_NULL;
    /* Upload stencil cofficients */
    upload_coefficients(coeff, 5);
    int left_halo_offset = 0;
    int right_halo_offset = dimx * dimy * (4 + dimz);
    int left_stage1_offset = 0;
    int right_stage1_offset = dimx * dimy * (dimz - 4);
    int stage2_offset = num_halo_points;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; I < nreps; i++)
    {
        /* Compute boundary values needed by other nodes first */
        call_stencil_kernel(d_output + left_stage1_offset, d_input + left_stage1_offset, dimx, dimy,
                            12, stream0);
        call_stencil_kernel(d_output + right_stage1_offset, d_input + right_stage1_offset, dimx,
                            dimy, 12, stream0);
        /* Compute the remaining points */
        call_stencil_kernel(d_output + stage2_offset, d_input + stage2_offset, dimx, dimy, dimz,
                            stream1);
        cudaStreamSynchronize(stream0);
        /* Send data to left, get data from right */
        MPI_Sendrecv(d_output + num_halo_points, num_halo_points, MPI_FLOAT, left_neighbor, i,
                     d_output + right_halo_offset, num_halo_points, MPI_FLOAT, right_neighbor, i,
                     MPI_COMM_WORLD, &status);
        /* Send data to right, get data from left */
        MPI_Sendrecv(d_output + right_stage1_offset + num_halo_points, num_halo_points, MPI_FLOAT,
                     right_neighbor, i, d_output + left_halo_offset, num_halo_points, MPI_FLOAT,
                     left_neighbor, i, MPI_COMM_WORLD, &status);
        cudaDeviceSynchronize();
        float* temp = d_output;
        d_output = d_input;
        d_input = temp;
    }
    /* Wait for previous communications */
    MPI_Barrier(MPI_COMM_WORLD);
    float* temp = d_output;
    d_output = d_input;
    d_input = temp;
    /* Send the output, skipping halo points */
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    float* send_address = h_output + num_ghost_points;
    MPI_Send(send_address, dimx * dimy * dimz, MPI_REAL, server_process, DATA_COLLECT,
             MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Release resources */
    free(h_input);
    free(h_output);
    cudaFreeHost(h_left_ghost_own);
    cudaFreeHost(h_right_ghost_own);
    cudaFreeHost(h_left_ghost);
    cudaFreeHost(h_right_ghost);
    cudaFree(d_input);
    cudaFree(d_output);
}
