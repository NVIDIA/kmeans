#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

namespace kmeans {
namespace detail {

__device__ __forceinline__ void update_centroid(int label, int dimension,
                                                int d,
                                                double accumulator, double* centroids,
                                                int count, int* counts) {
    int index = label * d + dimension;
    double* target = centroids + index;
    atomicAdd(target, accumulator);
    if (dimension == 0) {
        atomicAdd(counts + label, count);
    }             
}

__global__ void calculate_centroids(int n, int d, int k,
                                    double* data,
                                    int* ordered_labels,
                                    int* ordered_indices,
                                    double* centroids,
                                    int* counts) {
    int in_flight = blockDim.y * gridDim.y;
    int labels_per_row = (n - 1) / in_flight + 1; 
    for(int dimension = threadIdx.x; dimension < d; dimension += blockDim.x) {
        double accumulator = 0;
        int count = 0;
        int global_id = threadIdx.y + blockIdx.y * blockDim.y;
        int start = global_id * labels_per_row;
        int end = (global_id + 1) * labels_per_row;
        end = (end > n) ? n : end;
        int prior_label;
        if (start < n) {
            prior_label = ordered_labels[start];
        
            for(int label_number = start; label_number < end; label_number++) {
                int label = ordered_labels[label_number];
                if (label != prior_label) {
                    update_centroid(prior_label, dimension,
                                    d,
                                    accumulator, centroids,
                                    count, counts);
                    accumulator = 0;
                    count = 0;
                }
  
                double value = data[dimension + ordered_indices[label_number] * d];
                accumulator += value;
                prior_label = label;
                count++;
            }
            update_centroid(prior_label, dimension,
                            d,
                            accumulator, centroids,
                            count, counts);
        }
    }
}

__global__ void scale_centroids(int d, int k, int* counts, double* centroids) {
    int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((global_id_x < d) && (global_id_y < k)) {
        int count = counts[global_id_y];
        //To avoid introducing divide by zero errors
        //If a centroid has no weight, we'll do no normalization
        //This will keep its coordinates defined.
        if (count < 1) {
            count = 1;
        }
        double scale = 1.0/double(count);
        centroids[global_id_x + d * global_id_y] *= scale;
    }
}

void find_centroids(int n, int d, int k,
                    thrust::device_vector<double>& data,
                    //Labels are taken by value because
                    //they get destroyed in sort_by_key
                    //So we need to make a copy of them
                    thrust::device_vector<int> labels,
                    thrust::device_vector<double>& centroids) {
    thrust::device_vector<int> indices(n);
    thrust::device_vector<int> counts(k);
    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(n),
                 indices.begin());
    //Bring all labels with the same value together
    thrust::sort_by_key(labels.begin(),
                        labels.end(),
                        indices.begin());

    //Initialize centroids to all zeros
    thrust::fill(centroids.begin(),
                 centroids.end(),
                 0);
    
    //Calculate centroids 
    int n_threads_x = 64;
    int n_threads_y = 16;
    //XXX Number of blocks here is hard coded at 30
    //This should be taken care of more thoughtfully.
    detail::calculate_centroids<<<dim3(1, 30), dim3(n_threads_x, n_threads_y)>>>
        (n, d, k,
         thrust::raw_pointer_cast(data.data()),
         thrust::raw_pointer_cast(labels.data()),
         thrust::raw_pointer_cast(indices.data()),
         thrust::raw_pointer_cast(centroids.data()),
         thrust::raw_pointer_cast(counts.data()));
    
    //Scale centroids
    detail::scale_centroids<<<dim3((d-1)/32+1, (k-1)/32+1), dim3(32, 32)>>>
        (d, k,
         thrust::raw_pointer_cast(counts.data()),
         thrust::raw_pointer_cast(centroids.data()));
}

}
}
