#include <thrust/device_vector.h>
#include "kmeans.h"
#include <iostream>

#include <cstdlib>

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

template<typename T>
void fill_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2)*3 + j;
        }
    }
}

void tiny_test() {
    int iterations = 1;
    int n = 5;
    int d = 3;
    int k = 2;

    
    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);

    fill_array(data, n, d);
    std::cout << "Data: " << std::endl;
    print_array(data, n, d);

    labels[0] = 0;
    labels[1] = 0;
    labels[2] = 0;
    labels[3] = 1;
    labels[4] = 1;

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);
    
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids);

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, k, d);

}

void random_data(thrust::device_vector<double>& array, int m, int n) {
    thrust::host_vector<double> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (double)rand()/(double)RAND_MAX;
    }
    array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}


int main() {
    int iterations = 50;
    int n = 1e6;
    int d = 50;
    int k = 100;

    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);

    random_data(data, n, d);
    random_labels(labels, n, k);

    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kmeans::kmeans(iterations, n, d, k, data, labels, centroids);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time/1000.0 << " s" << std::endl;

    
}
