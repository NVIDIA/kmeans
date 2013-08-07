#pragma once
#include <thrust/device_vector.h>

namespace kmeans {

void kmeans(int iterations,
            int n, int d, int k,
            thrust::device_vector<double>& data,
            thrust::device_vector<int>& labels,
            thrust::device_vector<double>& centroids);

}
