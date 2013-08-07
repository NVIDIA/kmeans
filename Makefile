test: test.cu centroids.o labels.o kmeans.o
	nvcc -arch=sm_20 -Xptxas -v -o test test.cu centroids.o labels.o kmeans.o -lcublas

centroids.o: centroids.cu centroids.h
	nvcc -arch=sm_20 -Xptxas -v -c -o centroids.o centroids.cu

labels.o: labels.cu labels.h
	nvcc -arch=sm_20 -Xptxas -v -c -o labels.o labels.cu

kmeans.o: kmeans.cu kmeans.h centroids.h labels.h
	nvcc -arch=sm_20 -Xptxas -v -c -o kmeans.o kmeans.cu