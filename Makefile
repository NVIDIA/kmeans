CUDA_ARCH ?= sm_35
CUB_HOME ?= /home/lbarnes/kmeans/cub/

test: test.cu centroids.o labels.o kmeans.o timer.o
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -o test test.cu centroids.o labels.o kmeans.o timer.o -lcublas

clean:
	rm *.o test

centroids.o: centroids.cu centroids.h labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -c -o centroids.o centroids.cu

labels.o: labels.cu labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -c -o labels.o labels.cu

kmeans.o: kmeans.cu kmeans.h centroids.h labels.h timer.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -c -o kmeans.o kmeans.cu

timer.o:
	nvcc -arch=$(CUDA_ARCH) -c -o timer.o timer.cu
