#ifeq ($(BLAS), INTEL_MKL) 
#CC = icpc
#LIBS = -L/shared/apps/rhel-6.2/intel/parallel-studio-xe-2013/composer_xe_2013_sp1/mkl/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm
#endif
#ifeq ($(BLAS), OPENBLAS) 
CC = g++
INCL = -I/usr/include/openblas/
LIBS = -lopenblas -L/shared/apps/rhel-6.2/libs/openblas-0.2.8/sandybridge/lib
#endif


test: test.cpp
	$(CC) -g -o test test.cpp $(INCL) $(LIBS)
gemm_test: gemm_test.cpp
	$(CC) -g -o gemm_test gemm_test.cpp $(LIBS)
example-1: example-1.cpp
	$(CC) -g -o example-1 example-1.cpp $(LIBS)
