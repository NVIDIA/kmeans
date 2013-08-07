#pragma once
#include <thrust/device_vector.h>

namespace kmeans {
namespace detail {

void labels_init();

void make_self_dots(int n, int d,
                    thrust::device_vector<double>& data,
                    thrust::device_vector<double>& dots);

void make_all_dots(int n, int k, thrust::device_vector<double>& data_dots,
                   thrust::device_vector<double>& centroid_dots,
                   thrust::device_vector<double>& dots);

void calculate_distances(int n, int d, int k,
                         thrust::device_vector<double>& data,
                         thrust::device_vector<double>& centroids,
                         thrust::device_vector<double>& data_dots,
                         thrust::device_vector<double>& centroid_dots,
                         thrust::device_vector<double>& pairwise_distances);

void relabel(int n, int k,
             thrust::device_vector<double>& pairwise_distances,
             thrust::device_vector<int>& labels); 

}
}
