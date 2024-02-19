1) Implement the forward pass for the pooling layer described in Section 16.2.  
See `SubsamplingLayer::Forward()` in `layers_cpu.cpp`.  
2) We used an [N x C x H x W] layout for input and output features. Can we reduce the memory bandwidth by changing it to an [N x H x W x C] layout? What are potential benefits of using a [C x H x W x N] layout?  
Depending on how the data is stored, yes it could reduce the memory bandwidth, but most likely it would not.  
Some operations should be performed at the N level, needing knowledge of the current c, h, w (for example, computing average gradient). Using the mentioned format would make it just a bit easier to compute that (though the alternative would be just averaging at time of usage).
3) Implement the backward pass for the convolutional layer described in Section 16.2.  
See `ConvLayer::Backward()` in `layers_cpu.cpp`.  
4) Analyze the read access pattern to X in the unroll_Kernel in Fig. 16.18 and show whether the memory reads that are done by adjacent threads can be coalesced.  
As is, the memory accesses to X_unroll are coalesced because the lowest index (w_unroll) is sequential across adjacent threads when W_unroll is sufficiently large.  
The memory accesses to X could be coalesced by assigning threads to compute adjacent values of q.