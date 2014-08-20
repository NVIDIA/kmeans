#include <thrust/device_vector.h>
#include "kmeans.h"
#include "timer.h"
#include "util.h"
#include <iostream>
#include "cuda.h"

#include <cstdlib>

template<typename T>
void fill_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2)*3 + j;
        }
    }
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


void tiny_test() {
    int iterations = 1;
    int n = 5;
    int d = 3;
    int k = 2;

    
    thrust::device_vector<double> *data[1];
    thrust::device_vector<int> *labels[1];
    thrust::device_vector<double> *centroids[1];
    thrust::device_vector<double> *distances[1];
    data[0] = new thrust::device_vector<double>(n * d);
    labels[0] = new thrust::device_vector<int>(n);
    centroids[0] = new thrust::device_vector<double>(k * d);
    distances[0] = new thrust::device_vector<double>(n);

    fill_array(*data[0], n, d);
    std::cout << "Data: " << std::endl;
    print_array(*data[0], n, d);

    (*labels[0])[0] = 0;
    (*labels[0])[1] = 0;
    (*labels[0])[2] = 0;
    (*labels[0])[3] = 1;
    (*labels[0])[4] = 1;

    std::cout << "Labels: " << std::endl;
    print_array(*labels[0], n, 1);
    
    int i = kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances, 1);

    std::cout << "Labels: " << std::endl;
    print_array(*labels[0], n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(*centroids[0], k, d);

    std::cout << "Distances:" << std::endl;
    print_array(*distances[0], n, 1);
    delete(data[0]);
    delete(labels[0]);
    delete(centroids[0]);
    delete(distances[0]);
}


void more_tiny_test() {
	double dataset[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5,
		1.1, 1.2,
		0.5, 15.5,
		1.5, 15.5,
		1.5, 16.5,
		0.5, 16.5,
		1.2, 16.1,
		15.5, 15.5,
		16.5, 15.5,
		16.5, 16.5,
		15.5, 16.5,
		15.6, 16.2,
		15.5, 0.5,
		16.5, 0.5,
		16.5, 1.5,
		15.5, 1.5,
		15.7, 1.6};
	double centers[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5};
	 
    int iterations = 3;
    int n = 20;
    int d = 2;
    int k = 4;
	
    thrust::device_vector<double> *data[1];
    thrust::device_vector<int> *labels[1];
    thrust::device_vector<double> *centroids[1];
    thrust::device_vector<double> *distances[1];
    data[0] = new thrust::device_vector<double>(dataset, dataset+n*d);
    labels[0] = new thrust::device_vector<int>(n);
    centroids[0] = new thrust::device_vector<double>(centers, centers+k*d);
    distances[0] = new thrust::device_vector<double>(n);

    
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances, 1, false);

    std::cout << "Labels: " << std::endl;
    print_array(*labels[0], n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(*centroids[0], k, d);

}



int main() {
    std::cout << "Input a character to choose a test:" << std::endl;
    std::cout << "Tiny test: t" << std::endl;
    std::cout << "More tiny test: m" << std::endl;
    std::cout << "Huge test: h: " << std::endl;
    char c;
    //std::cin >> c;
    c = 'h';
    switch (c) {
    case 't':
        tiny_test();
        exit(0);
    case 'm':
        more_tiny_test();
        exit(0);
    case 'h':
        break;
    default:
        std::cout << "Choice not understood, running huge test" << std::endl;
    }
    int iterations = 100;
    int n = 5e6;
    int d = 50;
    int k = 100;

    int n_gpu;
    
    cudaGetDeviceCount(&n_gpu);

    //n_gpu = 1;
    std::cout << n_gpu << " gpus." << std::endl;

    thrust::device_vector<double> *data[16];
    thrust::device_vector<int> *labels[16];
    thrust::device_vector<double> *centroids[16];
    thrust::device_vector<double> *distances[16];
    for (int q = 0; q < n_gpu; q++) {
       cudaSetDevice(q);
       data[q] = new thrust::device_vector<double>(n/n_gpu*d);
       labels[q] = new thrust::device_vector<int>(n/n_gpu*d);
       centroids[q] = new thrust::device_vector<double>(k * d);
       distances[q] = new thrust::device_vector<double>(n);
    }

    std::cout << "Generating random data" << std::endl;
    std::cout << "Number of points: " << n << std::endl;
    std::cout << "Number of dimensions: " << d << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    
    for (int q = 0; q < n_gpu; q++) {
       random_data(*data[q], n/n_gpu, d);
       random_labels(*labels[q], n/n_gpu, k);
    }
    kmeans::timer t;
    t.start();
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances, n_gpu);
    float time = t.stop();
    std::cout << "  Time: " << time/1000.0 << " s" << std::endl;

    for (int q = 0; q < n_gpu; q++) {
       delete(data[q]);
       delete(labels[q]);
       delete(centroids[q]);
    }
}
