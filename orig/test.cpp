#include <iostream>
#include <vector>
#include "stdlib.h"
#include "string.h" //memset
//#include "mkl.h"
#include "cblas.h"

//extern "C" void dgemm_( char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int * );

void random_data(std::vector<double>& array, int n) {
   for (int i=0; i<n; i++) array[i] = (double)rand()/(double)RAND_MAX;
}

void random_labels(std::vector<int>& array, int n, int max) {
   for (int i=0; i<n; i++) array[i] = rand()%max;
}

void self_dot(std::vector<double> array_in, int n, int dim, 
              std::vector<double>& dots) {
   for (int pt = 0; pt<n; pt++) {
      double sum = 0.0;
      for (int i=0; i<dim; i++) {
         sum += array_in[pt*dim+i]*array_in[pt*dim+i];
      }
      dots[pt] = sum;
   }
}

void find_centroids(std::vector<double> array_in, int n, int dim, 
                    std::vector<int> labels_in,
                    std::vector<double>& centroids, int n_cluster) {
   std::vector<int> members(n_cluster); //Number of points in each cluster
   memset(&members[0], 0, n_cluster*sizeof(int));
   memset(&centroids[0], 0, n_cluster*dim*sizeof(double));
   //Add all vectors in the cluster
   for(int pt=0; pt<n; pt++) {
      int this_cluster = labels_in[pt];
      members[this_cluster]++;
      for (int i=0; i<dim; i++) centroids[this_cluster*dim+i] += 
                                                           array_in[pt*dim+i];
   }
   //Divide by the number of points in the cluster 
   for(int cluster=0; cluster < n_cluster; cluster++) {
      if (dim < 6) std::cout << cluster << "(" << members[cluster] << " members):  ";
      for (int i=0; i<dim; i++) { 
         centroids[cluster*dim+i] /= members[cluster];
         if (dim < 6) std::cout << centroids[cluster*dim+i] << "  ";
      }
      if (dim < 6) std::cout << std::endl;
   }
}

void compute_distances(std::vector<double> data_in, 
                       std::vector<double> data_dots_in, 
                       int n, int dim, std::vector<double> centroids_in, 
                       std::vector<double> centroid_dots, int n_cluster, 
                       std::vector<double>& pairwise_distances) {
   self_dot(centroids_in, n_cluster, dim, centroid_dots);
   for (int nn=0; nn<n; nn++) 
      for (int c=0; c<n_cluster; c++) {
         pairwise_distances[nn*n_cluster+c] = data_dots_in[nn] + 
                                                        centroid_dots[c];
      }
   double alpha = -2.0;
   double beta = 1.0;
   char transa = 'N';
   char transb = 'N';
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n_cluster, 
               dim, alpha, &data_in[0], dim, &centroids_in[0], dim, 
               beta, &pairwise_distances[0], n_cluster);
}
int relabel(std::vector<double> data_in, int n, 
             std::vector<double> pairwise_distances_in,
             int n_cluster, std::vector<int>& labels) {
   int changes = 0; 
   for (int nn=0; nn<n; nn++) {
      double min = pairwise_distances_in[nn*n_cluster];
      int idx = 0;
      for (int cc=1; cc<n_cluster; cc++) {
         double this_dist = pairwise_distances_in[nn*n_cluster+cc];
         if (this_dist < min) {
            idx=cc;
            min=this_dist;
         }
      }
      if (labels[nn] != idx) {
         changes ++; 
         labels[nn] = idx;
      }
   }
   return changes;
}
int main(int argc, char** argv) {

   int n = 5e6;
   int d = 50;
   int n_cluster = 100;
   int iterations = 100;

   if (argc>1) {
     if (0==strcmp(argv[1], "--help")) {
        std::cout << "Usage: test <number of points> <dimension of space>"
                     " <number of clusters>" << std::endl;
        return 0;
     }
     else n = atoi(argv[1]);
   }
   if (argc>2) d = atoi(argv[2]);
   if (argc>3) n_cluster = atoi(argv[3]);
   if (argc>4) iterations = atoi(argv[4]);

   std::cout << "Generating random data" << std::endl;
   std::cout << n << " points of dimension " << d << std::endl;
   std::cout << n_cluster << " clusters" << std::endl;
   
   std::vector<double> data(n*d); //input data
   std::vector<double> centroids(n_cluster*d); //centroids for each cluster
   std::vector<int> labels(n); //cluster labels for each point
   std::vector<double> distances(n); //distances from point from a centroid

   random_data(data, n*d);
 

   std::vector<double> data_dots(n);
   std::vector<double> centroid_dots(n_cluster);
   std::vector<double> pairwise_distances(n_cluster * n);
   std::vector<int> labels_copy(n);

   self_dot(data, n, d, data_dots);

   //Let the first n_cluster points be the centroids of the clusters
   memcpy(&centroids[0], &data[0], sizeof(double)*n_cluster*d);
   
   for(int i=0; i<iterations; i++) {
      compute_distances(data, data_dots, n, d, centroids, centroid_dots, 
                        n_cluster, pairwise_distances);
      int movers = relabel(data, n, pairwise_distances, n_cluster, labels);
      std::cout <<std::endl << "*** Iteration " << i << " ***" << std::endl;
      std::cout << movers << " points moved between clusters." << std::endl;
      if (0 == movers) break;
      find_centroids(data, n, d, labels, centroids, n_cluster);
   }
}
