NVCC=/usr/local/cuda/bin/nvcc
#NVCC_FLAGS=-Xcompiler "-Wall -Wextra -march=native -mtune=native" -O0 -g -G --std c++17 -gencode=arch=compute_61,code=sm_61 -lcublas -lcusolver
NVCC_FLAGS=-Xcompiler "-Wall -Wextra -march=native" -O2 --std c++17 -arch=sm_61 -lcublas -lcusolver


.PHONY: all clean

all: benchmark.out

benchmark.out: benchmark.cu kernels.cuh globals.h helpers.h sindex.h
	$(NVCC) $(NVCC_FLAGS) -o benchmark.out benchmark.cu

clean:
	rm -f benchmark.out

