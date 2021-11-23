NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS_DEBUG=-Xcompiler "-march=native -mtune=native -fopenmp -mavx" -O0 -g -G --std c++17 -gencode=arch=compute_61,code=sm_61 -lcublas -lcusolver
NVCC_FLAGS=-Xcompiler "-march=native -mtune=native -fopenmp -mavx" -O2 --std c++17 -gencode=arch=compute_61,code=sm_61 -lcublas -lcusolver


all: benchmark-create benchmark-create-debug benchmark-query benchmark-query-debug

benchmark-create: benchmark_create.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS) -o benchmark-create benchmark_create.cu

benchmark-create-debug: benchmark_create.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS_DEBUG) -o benchmark-create-debug benchmark_create.cu

benchmark-query: benchmark_query.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS) -o benchmark-query benchmark_query.cu

benchmark-query-debug: benchmark_query.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS_DEBUG) -o benchmark-query-debug benchmark_query.cu

clean:
	rm -f benchmark-create benchmark-create-debug benchmark-query benchmark-query-debug

.PHONY: all clean
