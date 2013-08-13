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
  The vector is passed
  by reference since it is shared with the caller and not copied.
  \param centroids Centroid locations, in row-major order. This
  vector must have size k * d, and since it's in row-major order,
  centroid x occupies positions [x * d, (x + 1) * d) in the
  vector. The vector is passed by reference since it is shared
  with the caller and not copied.
  \param init_from_labels If true, the labels need to be initialized
  before calling kmeans. If false, the centroids need to be
  initialized before calling kmeans.
*/

void kmeans(int iterations,
            int n, int d, int k,
            thrust::device_vector<double>& data,
            thrust::device_vector<int>& labels,
            thrust::device_vector<double>& centroids,
            bool init_from_labels=true);

}
