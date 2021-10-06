NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS_DEBUG=-Xcompiler "-Wall -Wextra -march=native -mtune=native" -O0 -g -G --std c++17 -gencode=arch=compute_61,code=sm_61 -lcublas -lcusolver
NVCC_FLAGS=-Xcompiler "-Wall -Wextra -march=native" -O3 --std c++17 -arch=sm_61 -lcublas -lcusolver


all: benchmark.out benchmark-debug.out

benchmark.out: benchmark.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS) -o benchmark.out benchmark.cu

benchmark-debug.out: benchmark.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS_DEBUG) -o benchmark-debug.out benchmark.cu

clean:
	rm -f benchmark*.out

.PHONY: all clean
