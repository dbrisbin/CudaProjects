1) Assume that we need to merge two lists A=(1, 7, 8, 9, 10) and B=(7, 10, 10, 12). What are the co-rank values for C[8]?  
We will compute CoRank(9, A, 5, B, 4)
i = 5  
j = 4  
i_low = 5  
j_low = 4  
Since j == n, the first branch is not executed.
Since i == m, the second branch is not executed.
The third branch is executed and we return i == 5
A Co-rank value: 5   
B Co-rank value: 4  
2) Complete the calculation of co-rank functions for thread 2 in Fig. 12.6.  
We will compute CoRank(6, A, 5, B, 4)
i = 5  
j = 1  
i_low = 2  
j_low = 1  
Since A[4] = 10 == B[1] = 10, the first branch is not executed.  
Since i == m, the second branch is not executed.  
The third branch is executed and we return i == 5
A Co-rank value: 5   
B Co-rank value: 1  
The co-rank for k = 9 has already been computed in problem 1.  
3) For the for-loops that load A and B tiles in Fig. 12.12, add a call to the co- rank function so that we can load only the A and B elements that will be consumed in the current generation of the while-loop.  
See `ModifiedTiledKernel` in `merge.cu`  
4) Consider a parallel merge of two arrays of size 1,030,400 and 608,000. Assume that each thread merges eight elements and that a thread block size of 1024 is used.  
**a)** In the basic merge kernel in Fig. 12.9, how many threads perform a binary search on the data in the global memory?  
Each thread will perform two binary searches in global memory.
The total number of blocks is $\big\lceil\frac{m + n}{elementsPerThread \times blockDim.x}\big\rceil =\big\lceil \frac{1,030,400 + 608,000}{8 \times 1024}\big\rceil = 200$. Hence, there are a total of $200 \times 1024 = 204,800$ threads and $204,800 \times 2 = 409,600$ binary searches performed in the global memory.  
**b)** In the tiled merge kernel in Figs. 12.11-12.13, how many threads perform a binary search on the data in the global memory?  
The first thread in each block will perform two binary searches in global memory in part 1 and 0 in parts 2 and 3. Since there are 200 blocks, 200 threads will perform 400 binary searches in global memory.  
**c)** In the tiled merge kernel in Figs. 12.11-12.13, how many threads perform a binary search on the data in the shared memory?  
Each iteration of the loop starting on line 25, every thread will perform 3 binary searches in shared memory. Assuming a high end device with 96KB of shared memory, the tile_size could be up to $96KB/4B/2 = 12,284$. C_length will always be $\frac{1,030,400 + 608,000}{200} = 8192$, hence there is enough shared memory such that only one iteration has to be performed. $204,800$ threads will then perform a total of $614,400$ binary searches in shared memory.