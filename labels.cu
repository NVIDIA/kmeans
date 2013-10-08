#include "labels.h"
#include <cublas_v2.h>
#include <cfloat>

cudaStream_t cuda_stream[16];
namespace kmeans {
namespace detail {

cublasHandle_t cublas_handle[16];

void labels_init() {
    cublasStatus_t stat;
    cudaError_t err;
    int dev_num;
    cudaGetDevice(&dev_num);
    stat = cublasCreate(&detail::cublas_handle[dev_num]);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        exit(1);
    }
    err = cudaStreamCreate(&cuda_stream[dev_num]);
    if (err != cudaSuccess) {
        std::cout << "Stream creation failed" << std::endl;
        exit(1);
    }
    cublasSetStream(cublas_handle[dev_num], cuda_stream[dev_num]);
    mycub::cub_init();
}
void labels_close() {
    int dev_num;
    cudaGetDevice(&dev_num);
    cublasDestroy(cublas_handle[dev_num]);
    cudaStreamDestroy(cuda_stream[dev_num]);
    mycub::cub_close();
}


void streamsync(int dev_num) {

    cudaStreamSynchronize(cuda_stream[dev_num]);
}

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


void make_self_dots(int n, int d, thrust::device_vector<double>& data, thrust::device_vector<double>& dots) {
    int dev_num;
    cudaGetDevice(&dev_num);
    self_dots<<<(n-1)/256+1, 256, 0, cuda_stream[dev_num]>>>(n, d, thrust::raw_pointer_cast(data.data()),
                                    thrust::raw_pointer_cast(dots.data()));
}

__global__ void all_dots(int n, int k, double* data_dots, double* centroid_dots, double* dots) {
	__shared__ double local_data_dots[32];
	__shared__ double local_centroid_dots[32];

    int data_index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((data_index < n) && (threadIdx.y == 0)) {
        local_data_dots[threadIdx.x] = data_dots[data_index];
    }

    

    
    int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
    if ((centroid_index < k) && (threadIdx.y == 1)) {
        local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
    }
       
   	__syncthreads();

	centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
    if ((data_index < n) && (centroid_index < k)) {
        dots[data_index + centroid_index * n] = local_data_dots[threadIdx.x] +
            local_centroid_dots[threadIdx.y];
    }
}

void make_all_dots(int n, int k, thrust::device_vector<double>& data_dots,
                   thrust::device_vector<double>& centroid_dots,
                   thrust::device_vector<double>& dots) {
    int dev_num;
    cudaGetDevice(&dev_num);
    all_dots<<<
        dim3((n-1)/32+1,
             (k-1)/32+1),
        dim3(32, 32), 0,
        cuda_stream[dev_num]>>>(n, k, thrust::raw_pointer_cast(data_dots.data()),
                                 thrust::raw_pointer_cast(centroid_dots.data()),
                                 thrust::raw_pointer_cast(dots.data()));
};

void calculate_distances(int n, int d, int k,
                         thrust::device_vector<double>& data,
                         thrust::device_vector<double>& centroids,
                         thrust::device_vector<double>& data_dots,
                         thrust::device_vector<double>& centroid_dots,
                         thrust::device_vector<double>& pairwise_distances) {
    detail::make_self_dots(k, d, centroids, centroid_dots);
    detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
    //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
    //The dgemm calculates x.y for all x and y, so alpha = -2.0
    double alpha = -2.0;
    double beta = 1.0;
    //If the data were in standard column major order, we'd do a
    //centroids * data ^ T
    //But the data is in row major order, so we have to permute
    //the arguments a little
    int dev_num;
    cudaGetDevice(&dev_num);
    cublasStatus_t stat =
        cublasDgemm(detail::cublas_handle[dev_num],
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, k, d, &alpha,
                    thrust::raw_pointer_cast(data.data()),
                    d,//Has to be n or d
                    thrust::raw_pointer_cast(centroids.data()),
                    d,//Has to be k or d
                    &beta,
                    thrust::raw_pointer_cast(pairwise_distances.data()),
                    n); //Has to be n or k
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Dgemm" << std::endl;
        exit(1);
    }

}
                         
__global__ void make_new_labels(int n, int k, double* pairwise_distances,
                                int* labels, int* changes,
                                double* distances) {
    double min_distance = DBL_MAX;
    double min_idx = -1;
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < n) {
        int old_label = labels[global_id];
        for(int c = 0; c < k; c++) {
            double distance = pairwise_distances[c * n + global_id];
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = c;
            }
        }
        labels[global_id] = min_idx;
        distances[global_id] = min_distance;
        if (old_label != min_idx) {
            atomicAdd(changes, 1);
        }
    }
}


void relabel(int n, int k,
             thrust::device_vector<double>& pairwise_distances,
             thrust::device_vector<int>& labels, 
             thrust::device_vector<double>& distances, 
             int *d_changes) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaMemsetAsync(d_changes, 0, sizeof(int), cuda_stream[dev_num]);
    make_new_labels<<<(n-1)/256+1,256,0,cuda_stream[dev_num]>>>(
        n, k,
        thrust::raw_pointer_cast(pairwise_distances.data()),
        thrust::raw_pointer_cast(labels.data()),
        d_changes,
        thrust::raw_pointer_cast(distances.data()));
}

}
}
namespace mycub {
void *d_key_alt_buf[16];
unsigned int key_alt_buf_bytes[16];
void *d_value_alt_buf[16];
unsigned int value_alt_buf_bytes[16];
void *d_temp_storage[16];
size_t temp_storage_bytes[16];
void *d_temp_storage2[16];
size_t temp_storage_bytes2[16];
bool cub_initted;
void cub_init() {
    std::cout <<"CUB init" << std::endl;
    for (int q=0; q<16; q++) {
        d_key_alt_buf[q] = NULL;
        key_alt_buf_bytes[q] = 0;
        d_value_alt_buf[q] = NULL;
        value_alt_buf_bytes[q] = 0;
        d_temp_storage[q] = NULL;
        temp_storage_bytes[q] = 0;
        d_temp_storage2[q] = NULL;
        temp_storage_bytes2[q] = 0;
    }
    cub_initted = true;
}
void cub_close() {
    for (int q=0; q<16; q++) {
        if(d_key_alt_buf[q]) cudaFree(d_key_alt_buf[q]);
        if(d_value_alt_buf[q]) cudaFree(d_value_alt_buf[q]);
        if(d_temp_storage[q]) cudaFree(d_temp_storage[q]);
        if(d_temp_storage2[q]) cudaFree(d_temp_storage2[q]);
        d_temp_storage[q] = NULL;
        d_temp_storage2[q] = NULL;
    }
    cub_initted = false;
}
void sort_by_key_int(thrust::device_vector<int>& keys, thrust::device_vector<int>& values) {
    int dev_num;
    cudaGetDevice(&dev_num);
    cudaStream_t this_stream = cuda_stream[dev_num]; 
    int SIZE = keys.size();
    //int *d_key_alt_buf, *d_value_alt_buf;
    if (key_alt_buf_bytes[dev_num] < sizeof(int)*SIZE) {
       if (d_key_alt_buf[dev_num]) cudaFree(d_key_alt_buf[dev_num]);
       cudaMalloc(&d_key_alt_buf[dev_num], sizeof(int)*SIZE);
       key_alt_buf_bytes[dev_num] = sizeof(int)*SIZE;
    }
    if (value_alt_buf_bytes[dev_num] < sizeof(int)*SIZE) {
       if (d_value_alt_buf[dev_num]) cudaFree(d_value_alt_buf[dev_num]);
       cudaMalloc(&d_value_alt_buf[dev_num], sizeof(int)*SIZE);
       value_alt_buf_bytes[dev_num] = sizeof(int)*SIZE;
    }
    cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys.data()), (int*)d_key_alt_buf[dev_num]);
    cub::DoubleBuffer<int> d_values(thrust::raw_pointer_cast(values.data()), (int*)d_value_alt_buf[dev_num]);

    // Determine temporary device storage requirements for sorting operation
    if (!d_temp_storage[dev_num]) {
        cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys, 
                                        d_values, SIZE, 0, sizeof(int)*8, this_stream);
        // Allocate temporary storage for sorting operation
        cudaMalloc(&d_temp_storage[dev_num], temp_storage_bytes[dev_num]);
    }
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys, 
                                    d_values, SIZE, 0, sizeof(int)*8, this_stream);
    // Sorted keys and values are referenced by d_keys.Current() and d_values.Current()


}
}
