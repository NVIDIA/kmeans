kmeans
======

A simple kmeans clustering implementation for double precision data,
written for CUDA GPUs.

There are two ideas here:
1. The relabel step of kmeans relies on computing distances between
all n points and all k centroids. This code refactors the distance
computation using the identity ||x-y||^2 = x.x + y.y - 2x.y; this
refactorization moves the x.x computation outside the kmeans loop, and
uses GEMM to compute the x.y, getting us peak performance. (y.y is
negligably small)
2. The computation of new centroids is tricky because the labels
change every iteration.  This code uses sort to group all points with
the same label, transforming the centroid accumulation into primarily
simple additions, with only a few atomic memory operations.  For many
practical problem sizes, sorting reduces the centroid computation to less
than 20% of the overall runtime of the algorithm.

The CUDA code here is purposefully non-optimized - this code is not
meant to be the fastest possible kmeans implementation, but rather to
show how using libraries like thrust and BLAS can provide reasonable
performance with high programmer productivity.