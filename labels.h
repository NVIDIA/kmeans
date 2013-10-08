#pragma once
#include <thrust/device_vector.h>
#include <cub/cub.cuh>

extern cudaStream_t cuda_stream[16];

template<unsigned int i>
extern __global__ void debugMark(){};

namespace kmeans {
namespace detail {

void labels_init();
void labels_close();

template<typename T>
void memcpy(thrust::host_vector<T, std::allocator<T> > &H, 
            thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaMemcpyAsync(thrust::raw_pointer_cast(H.data()), 
               thrust::raw_pointer_cast(D.data()), 
               sizeof(T) * D.size(), cudaMemcpyDeviceToHost, cuda_stream[dev_num]);
}

template<typename T>
void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D, 
            thrust::host_vector<T, std::allocator<T> > &H) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaMemcpyAsync(thrust::raw_pointer_cast(D.data()), 
               thrust::raw_pointer_cast(H.data()), 
               sizeof(T) * H.size(), cudaMemcpyHostToDevice, cuda_stream[dev_num]);
}
template<typename T>
void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Do, 
            thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Di) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaMemcpyAsync(thrust::raw_pointer_cast(Do.data()), 
               thrust::raw_pointer_cast(Di.data()), 
               sizeof(T) * Di.size(), cudaMemcpyDeviceToDevice, cuda_stream[dev_num]);
}
template<typename T>
void memzero(thrust::device_vector<T, thrust::device_malloc_allocator<T> >& D) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaMemsetAsync(thrust::raw_pointer_cast(D.data()), 0, sizeof(T)*D.size(), cuda_stream[dev_num]);
}
void streamsync(int dev_num);

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
             thrust::device_vector<int>& labels,
             thrust::device_vector<double>& distances,
             int *d_changes); 

}
}
namespace mycub {

extern void *d_key_alt_buf[16];
extern unsigned int key_alt_buf_bytes[16];
extern void *d_value_alt_buf[16];
extern unsigned int value_alt_buf_bytes[16];
extern void *d_temp_storage[16];
extern size_t temp_storage_bytes[16];
extern void *d_temp_storage2[16];
extern size_t temp_storage_bytes2[16];
extern bool cub_initted;

void sort_by_key_int(thrust::device_vector<int>& keys, thrust::device_vector<int>& values);

template <typename T, typename U>
void sort_by_key(thrust::device_vector<T>& keys, thrust::device_vector<U>& values) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaStream_t this_stream = cuda_stream[dev_num]; 
    int SIZE = keys.size();
    if (key_alt_buf_bytes[dev_num] < sizeof(T)*SIZE) {
       if (d_key_alt_buf[dev_num]) cudaFree(d_key_alt_buf[dev_num]);
       cudaMalloc(&d_key_alt_buf[dev_num], sizeof(T)*SIZE);
       key_alt_buf_bytes[dev_num] = sizeof(T)*SIZE;
       std::cout << "Malloc key_alt_buf" << std::endl;
    }
    if (value_alt_buf_bytes[dev_num] < sizeof(U)*SIZE) {
       if (d_value_alt_buf[dev_num]) cudaFree(d_value_alt_buf[dev_num]);
       cudaMalloc(&d_value_alt_buf[dev_num], sizeof(U)*SIZE);
       value_alt_buf_bytes[dev_num] = sizeof(U)*SIZE;
       std::cout << "Malloc value_alt_buf" << std::endl;
    }
    cub::DoubleBuffer<T> d_keys(thrust::raw_pointer_cast(keys.data()), (T*)d_key_alt_buf[dev_num]);
    cub::DoubleBuffer<U> d_values(thrust::raw_pointer_cast(values.data()), (U*)d_value_alt_buf[dev_num]);
    cudaError_t err;

    // Determine temporary device storage requirements for sorting operation
    //if (temp_storage_bytes[dev_num] == 0) {
        void *d_temp;
        size_t temp_bytes;
        err = cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_bytes, d_keys, 
                                        d_values, SIZE, 0, sizeof(T)*8, this_stream);
        // Allocate temporary storage for sorting operation
        cudaMalloc(&d_temp, temp_bytes);
        d_temp_storage[dev_num] = d_temp;
        temp_storage_bytes[dev_num] = temp_bytes;
        std::cout << "Malloc temp_storage. " << temp_storage_bytes[dev_num] << " bytes" << std::endl;
        std::cout << "d_temp_storage[" << dev_num << "] = " << d_temp_storage[dev_num] << std::endl;
        if (err) {
           std::cout <<"Error " << err << " in SortPairs 1" << std::endl;
           std::cout << cudaGetErrorString(err) << std::endl;
        }
    //}
    // Run sorting operation
    err = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, 
                                    d_values, SIZE, 0, sizeof(T)*8, this_stream);
    if (err) std::cout <<"Error in SortPairs 2" << std::endl;
    //cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys, 
    //                                d_values, SIZE, 0, sizeof(T)*8, this_stream);

}
template <typename T>
void sum_reduce(thrust::device_vector<T>& values, T* sum) {
    int dev_num;
    cudaGetDevice(&dev_num);
    if (!d_temp_storage2[dev_num]) {
        cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
                              sum, values.size(), cuda_stream[dev_num]); 
        // Allocate temporary storage for sorting operation
        cudaMalloc(&d_temp_storage2[dev_num], temp_storage_bytes2[dev_num]);
    }
    cub::DeviceReduce::Sum(d_temp_storage2[dev_num], temp_storage_bytes2[dev_num], thrust::raw_pointer_cast(values.data()),
                              sum, values.size(), cuda_stream[dev_num]); 
}
void cub_init();
void cub_close();
}
