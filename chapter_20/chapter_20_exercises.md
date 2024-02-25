1) Assume that the 25-point stencil computation in this chapter is applied to a grid whose size is 64 grid points in the x dimension, 64 in the y dimension, and 2048 in the z dimension. The computation is divided across 17 MPI ranks, of which 16 ranks are compute processes and 1 rank is the data server process.  
    **a)** How many output grid point values are computed by each compute process?  
    $64 \times 64 \times \frac{2048}{16} = 524,288$  
    **b)** How many halo grid points are needed:  
        **i)** By each internal compute process?  
        $64 \times 64 \times 4 \times 2 = 32,768$  
        **ii)** By each edge compute process?  
        $64 \times 64 \times 4 = 16,384$  
    **c)** How many boundary grid points are computed in stage 1 of Fig. 20.12:  
        **i)** By each internal compute process?  
        $64 \times 64 \times 4 \times 2 = 32,768$  
        **ii)** By each edge compute process?  
        $64 \times 64 \times 4 = 16,384$  
    **d)** How many internal grid points are computed in stage 2 of Fig. 20.12:  
        **i)** By each internal compute process?  
        $64 \times 64 \times (\frac{2048}{16} - 4\times 2) = 491,520$  
        **ii)** By each edge compute process?  
        $64 \times 64 \times (\frac{2048}{16} - 4) = 507,904$  
    **e)** How many bytes are sent in stage 2 of Fig. 20.12:  
        **i)** By each internal compute process?  
        $64 \times 64 \times 4 \times 2 \times $ `sizeof(float)` $= 131,072$  
        **ii)** By each edge compute process?  
        $64 \times 64 \times 4 \times $ `sizeof(float)` $= 65,536$  
2) If the MPI call `MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)` results in a data transfer of 4000 bytes, what is the size of each data element being sent?  
    4 bytes.
3) Which of the following statements is true?  
    **a)** `MPI_Send()` is blocking by default. `true`  
    **b)** `MPI_Recv()` is blocking by default. `true`  
    **c)** MPI messages must be at least 128 bytes. `false`  
    **d)** MPI processes can access the same variable through shared memory. `false`  
4) Modify the example code to remove the calls to `cudaMemcpyAsync()` from the compute processes’ code by using GPU memory addresses on MPI_Send and MPI_Recv.  
See `compute_process_modified.c`  