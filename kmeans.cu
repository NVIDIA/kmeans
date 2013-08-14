#include "kmeans.h"
#include "util.h"
#include <thrust/reduce.h>

namespace kmeans {

int kmeans(int iterations,
           int n, int d, int k,
           thrust::device_vector<double>& data,
           thrust::device_vector<int>& labels,
           thrust::device_vector<double>& centroids,
           thrust::device_vector<double>& distances,
           bool init_from_labels,
           double threshold) {
    thrust::device_vector<double> data_dots(n);
    thrust::device_vector<double> centroid_dots(n);
    thrust::device_vector<double> pairwise_distances(n * k);
    
    detail::labels_init();
    detail::make_self_dots(n, d, data, data_dots);

    if (init_from_labels) {
        detail::find_centroids(n, d, k, data, labels, centroids);
    }   
    double prior_distance_sum = 0;
    int i = 0;
    for(; i < iterations; i++) {
        detail::calculate_distances(n, d, k,
                                    data, centroids, data_dots,
                                    centroid_dots, pairwise_distances);

        int changes = detail::relabel(n, k, pairwise_distances, labels, distances);
       
        
        detail::find_centroids(n, d, k, data, labels, centroids);
        double distance_sum = thrust::reduce(distances.begin(), distances.end());
                std::cout << "Iteration " << i << " produced " << changes
                          << " changes, and total distance is " << distance_sum << std::endl;

        if (i > 0) {
            double delta = distance_sum / prior_distance_sum;
            if (delta > 1 - threshold) {
                std::cout << "Threshold triggered, terminating iterations early" << std::endl;
                break;
            }
        }
        prior_distance_sum = distance_sum;
    }
    return i;
}

}
