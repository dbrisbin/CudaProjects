1) Calculate the P[0] value in Fig. 7.3.  
    $P[0] = 0 + 0 + 8\times 5 + 2 \times 3 + 5 \times 1 = 51$  

2) Consider performing a 1D convolution on array N = {4,1,3,2,3} with filter F = {2,1,4}. What is the resulting output array?  
  $[8, 21, 13, 20, 7]$  
  
3) What do you think the following 1D convolution filters are doing?  
    **a)** $[0, 1, 0]$  
    Identity.  
    **b)** $[0, 0, 1]$  
    Shift every element left.  
    **c)** $[1, 0, 0]$  
    Shift every element right.  
    **d)** $[-\frac12, 0, \frac12]$  
    The gradient.  
    **e)** $[\frac13, \frac13, \frac13]$  
    Averaging values with a window size of 3  

4) Consider performing a 1D convolution on an array of size N with a filter of size M (assumption: such that $M = 2R+1$):  
    **a)** How many ghost cells are there in total?  
    R on each end of the array, so 2R or M-1.  
    **b)** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?  
    $N\times M$  
    **c)** How many multiplications are performed if ghost cells are not treated as multiplications?  
    Computing P[0] involves R ghost cells, P[1] involves R-1 ghost cells, P[2] involves R-2 ghost cells, ..., P[R] involves R-R ghost cells. I.e., on the left 1 + 2 + ... + R = $\frac{R(R-1)}2$ times is a ghost cell considered. Likewise for the right side. Hence a total of $\frac{R(R-1)}2\times 2 = R(R-1)$ times are ghost cells considered.  
    Therefore the total number of multiplications performed is: $N\times M - R(R-1)$.  

5) Consider performing a 2D convolution on a square matrix of size $N\times N$ with a square filter of size $M\times M$ (Assumption: $M = 2R+1$):  
    **a)** How many ghost cells are there in total?  
    At each corner $R\times R$. At the edges, $R\times N$. Total $4R\times R + 4 R \times N = 4R(R + N)$  
    **b)** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?  
    $M^2N^2$  
    **c)** How many multiplications are performed if ghost cells are not treated as multiplications?  
    This is a straightforward, but too time consuming exercise that I'm choosing not to do. Although I could totally be missing a quicker solution...  

6) Consider performing a 2D convolution on a rectangular matrix of size $N_1\times N_2$ with a rectangular mask of size $M_1\times M_2$ (Assuming $M_1 = 2 R_1 + 1, M_2 = 2 R_2 + 1$):  
    **a)** How many ghost cells are there in total?  
    $4 R_1 R_2 + 2 R_2 N_1 + 2 R_1 N_2$  
    **b)** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?  $M_1 M_2 N_1 N_2$  
    **c)** How many multiplications are performed if ghost cells are not treated as multiplications?  
    Like 5c, ain't nobody got time for dat.  

7) Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size $N\times N$ with a filter of size $M\times M$ using an output tile of size $T\times T$ (Assume $M = 2R + 1$).  
    **a)** How many thread blocks are needed?  
    $\big\lceil\frac{N}{T}\big\rceil^2$  
    **b)** How many threads are needed per block?  
    $(T + 2R)^2$  
    **c)** How much shared memory is needed per block?  
    $(T + 2R)^2 \times 4B$  
    **d)** Repeat the same questions if you were using the kernel in Fig. 7.15. 
    * **a)** $\big\lceil\frac{N}{T}\big\rceil^2$  
    * **b)** $T^2$  
    * **c)** $T^2 \times 4B$


8) Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.  
See convolution3D.cu.  

9) Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.  
See convolution3D.cu.  

10) Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.  
See convolution3D.cu.  
