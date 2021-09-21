NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS=-Xcompiler "-Wall -march=native -Wextra -fopenmp -mtune=native" -O2 --std c++17 -arch=sm_61 -lcublas -lcusolver

.PHONY: all clean

all: benchmark

benchmark: benchmark.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS) -c benchmark.cu -o benchmark

clean:
	rm -f benchmark
