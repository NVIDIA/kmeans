#pragma once
#include <thrust/device_vector.h>
#include "centroids.h"
#include "labels.h"

namespace kmeans {


//! kmeans clusters data into k groups
/*! 
  
  \param iterations How many iterations to run
  \param n Number of data points
  \param d Number of dimensions
  \param k Number of clusters
  \param data Data points, in row-major order. This vector must have
  size n * d, and since it's in row-major order, data point x occupies
  positions [x * d, (x + 1) * d) in the vector. The vector is passed
  by reference since it is shared with the caller and not copied.
  \param labels Cluster labels. This vector has size n.
  The vector is passed by reference since it is shared with the caller
  and not copied.
  \param centroids Centroid locations, in row-major order. This
  vector must have size k * d, and since it's in row-major order,
  centroid x occupies positions [x * d, (x + 1) * d) in the
  vector. The vector is passed by reference since it is shared
  with the caller and not copied.
  \param distances Distances from points to centroids. This vector has
  size n. It is passed by reference since it is shared with the caller
  and not copied.
  \param init_from_labels If true, the labels need to be initialized
  before calling kmeans. If false, the centroids need to be
  initialized before calling kmeans. Defaults to true, which means
  the labels must be initialized.
  \param threshold This controls early termination of the kmeans
  iterations. If the ratio of the sum of distances from points to
  centroids from this iteration to the previous iteration changes by
  less than the threshold, than the iterations are
  terminated. Defaults to 0.000001
  \return The number of iterations actually performed.
*/

int kmeans(int iterations,
            int n, int d, int k,
            thrust::device_vector<double>** data,
            thrust::device_vector<int>** labels,
            thrust::device_vector<double>** centroids,
            thrust::device_vector<double>** distances,
            int n_gpu=1,
            bool init_from_labels=true,
            double threshold=0.000001
    );

}
