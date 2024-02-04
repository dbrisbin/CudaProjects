1) Consider a 3D stencil computation on a grid of size 120x120x120, including boundary cells.  
**a)** What is the number of output grid points that is computed during each stencil sweep?  
$118 \times 118 \times 118 = 1,643,032$  
**b)** For the basic kernel in Fig. 8.6, what is the number of thread blocks that are needed, assuming a block size of 8x8x8?  
$\big\lceil \frac{120}{8}\big\rceil = 15, 15 \times 15 \times 15 = 3,375$  
**c)** For the kernel with shared memory tiling in Fig. 8.8, what is the number of thread blocks that are needed, assuming a block size of 8x8x8?  
Since the input block size should be the same as IN_TILE_DIMxIN_TILE_DIMxIN_TILE_DIM, the number of thread blocks required is the same as before, $3,375$.  
**d)** For the kernel with shared memory tiling and thread coarsening in Fig. 8.10, what is the number of thread blocks that are needed, assuming a block size of 32x32?  
Each block will account for 32 layers along the *z*-dimension, hence the number of thread blocks required is $\big\lceil \frac{120}{32}\big\rceil = 4, 4 \times 4 \times 4 = 64$ thread blocks.  

2. Consider an implementation of a seven-point (3D) stencil with shared memory tiling and thread coarsening applied. The implementation is similar to those in Figs. 8.10 and 8.12, except that the tiles are not perfect cubes. Instead, a thread block size of 32x32 is used as well as a coarsening factor of 16 (i.e., each thread block processes 16 consecutive output planes in the z dimension).  
**a)** What is the size of the input tile (in number of elements) that the thread block loads throughout its lifetime?  
Even though not all elements are needed (i.e., N[z][0][0], for any valid z), all elements in each considered plane are loaded. Assuming an interior thread block (i.e., the plane in front and behind the 16 output planes must be considered), the total number of elements loaded is $32\times 32 \times 18 = 18,432$.  
**b)** What is the size of the output tile (in number of elements) that the thread block processes throughout its lifetime?  
Assuming  an interior block, $30 \times 30 \times 16 = 14,400$  
**c)** What is the floating point to global memory access ratio (in OP/B) of the kernel?  
It is $\frac{13}{4} \times {\big\lparen 1-\frac{2}{32}\big\rparen}^3 = 2.68$ OP/B.  
**d)** How much shared memory (in bytes) is needed by each thread block if register tiling is not used, as in Fig. 8.10?  
$3T^2 \times 4\text{B} = 3\times 32^2 \times 4\text{B} = 12,288\text B$.  
**e)** How much shared memory (in bytes) is needed by each thread block if register tiling is used, as in Fig. 8.12?  
$T^2 \times 4\text{B} = 4,096\text B$.

Bonus:  
1) Implement the kernels presented in the chapter.  
See `stencil3d.cu`.

The following is average runtime over 1000 iterations for an optimized build with tuned parameters.
|Kernel | Average Runtime | Pros | Cons
|---|---|---|---|
| naive | 2.38ms | Easiest to implement. | Theoretically, should have the worst performance, but in practice, it actually had the best. 
| tiled | 4.18ms | Tiling reduces global memory reads | Did not perform very well 
| tiled with thread coarsening | 2.90ms | Thread coarsening allows us to significantly increase the block size in x and y dims | 
| register tiling with thread coarsening  | 2.39ms | In theory should have the best performance. Moves 2/7 reads into even faster registers rather than shared memory | 

Yet again, the naive approach actually performed the best. I'm guessing the GPU is just really smart and knows how to optimize better than I do. Still these approaches will come in handy when dealing with "dumber" hardware.