#include "labels.h"

namespace kmeans {
namespace detail {

//n: number of points
//d: dimensionality of points
//data: points, laid out in row-major order (n rows, d cols)
//dots: result vector (n rows)
// NOTE:
//Memory accesses in this function are uncoalesced!!
//This is because data is in row major order
//However, in k-means, it's called outside the optimization loop
//on the large data array, and inside the optimization loop it's
//called only on a small array, so it doesn't really matter.
//If this becomes a performance limiter, transpose the data somewhere
__global__ void self_dots(int n, int d, double* data, double* dots) {
	double accumulator = 0;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_id < n) {
        for (int i = 0; i < d; i++) {
            double value = data[i + global_id * d];
            accumulator += value * value;
        }
        dots[global_id] = accumulator;
    }    
}

}
