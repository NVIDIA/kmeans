#include <thrust/device_vector.h>
#include "centroids.h"
#include <iostream>

template<typename T>
void print_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            typename T::value_type value = array[i * n + j];
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int n = 1e6;
    int d = 50;
    int k = 100;

    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);
    
    
    kmeans::detail::find_centroids(n, d, k, data, labels, centroids);


}
