test: test.cu centroids.o
	nvcc -arch=sm_20 -Xptxas -v -o test test.cu centroids.o

centroids.o: centroids.cu centroids.h
	nvcc -arch=sm_20 -Xptxas -v -c -o centroids.o centroids.cu