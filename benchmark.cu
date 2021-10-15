
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <cuda.h>
#include "kernels.cuh"
#include "helpers.h"
#include "globals.h"
#include "sindex.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cusolverDn.h"
#include <assert.h>
#include <limits>


#include "helpers.h"
#include <chrono>








int main() {  
    
    // cuda debug
    cudaError_t cudaStat1 = cudaSuccess;
    
    ky_t* keys = (ky_t*) malloc(NUMKEYS * KEYLEN);
    read_keys(keys, FILENAME);
    
    // parameters
    const fp_t et = 32;
    const ix_size_t pt = 16;
    const ix_size_t fstep = 60'000'000;
    const ix_size_t bstep = 10'000;
    const ix_size_t min_size = CUDACORES;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // group meta data   
    std::vector<group_t> groups;
    grouping(keys, NUMKEYS, et, pt, fstep, bstep, min_size, groups);
    ix_size_t group_n = groups.size();
    ky_t* pivots = (ky_t*) malloc(group_n * sizeof(ky_t));
    
    for (ix_size_t group_i = 0; group_i < group_n; ++group_i) {
        group_t* group = groups.data() + group_i;
        memcpy(*(pivots + group_i), group->pivot, sizeof(ky_t));
        void* tmp_ptr;
        assert(cudaMalloc(&tmp_ptr, group->n * sizeof(ky_size_t))  == cudaSuccess);
        group->dev_feat_indices = (ky_size_t*) tmp_ptr;
        assert(cudaMalloc(&tmp_ptr, (group->n + 1) * sizeof(fp_t)) == cudaSuccess);
        group->dev_weights = (fp_t*) tmp_ptr;
        assert(cudaMemcpy(group->dev_feat_indices, group->feat_indices, group->n * sizeof(ky_size_t),  cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(group->dev_weights,      group->weights,      (group->n + 1) * sizeof(fp_t), cudaMemcpyHostToDevice) == cudaSuccess);

    }

    ky_t* dev_group_pivots;
    assert(cudaMalloc(&dev_group_pivots, group_n * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMemcpy(dev_group_pivots, pivots, group_n * sizeof(ky_t), cudaMemcpyHostToDevice) == cudaSuccess);
    free(pivots);

    std::vector<group_t> roots;
    ix_size_t root_n;
    if (group_n > min_size) {
        grouping(pivots, group_n, et, pt, fstep, bstep, min_size, roots);
    }
    root_n = roots.size();


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    index_t index = { root_n, roots.data(), group_n, groups.data(), dev_group_pivots };
    ch_t key[sizeof(ky_t)] = {'A', 'A', 'A', 'B', 'B', 'A', 'B', 'D', 'B', 'D', 'C', 'D', 'D', 'D', 'B', 'B'};
    while (1)
        get_position(index, key, keys);



    

    return 0;
}
