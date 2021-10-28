
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
    const fp_t et = 1e-12;
    const ix_size_t pt = 16;
    ix_size_t fstep = 1'000'000;
    ix_size_t bstep = 100'000;
    const ix_size_t min_size = CUDACORES;

    assert(pt <= KEYLEN);
    assert(bstep <= MINSIZE);
    assert(fstep <= NUMKEYS);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // group meta data   

    std::vector<group_t> groups;
    //grouping(keys, NUMKEYS, et, pt, fstep, bstep, MINSIZE, groups);
    ix_size_t group_n = groups.size();
    
    ky_t group_pivots[group_n];
    for (ix_size_t group_i = 0; group_i < group_n; ++group_i) {
        memcpy(group_pivots + group_i, keys + groups.at(group_i).start, sizeof(ky_t));
    }
    
    std::vector<group_t> roots;
    if (group_n > 1) {
        grouping(group_pivots, group_n, et, pt, 10, 5, 5, roots);
    }
    ix_size_t root_n = roots.size();
    
    ky_t root_pivots[root_n];
    if (root_n > 0) {
        for (ix_size_t root_i = 0; root_i < root_n; ++root_i) {
            memcpy(root_pivots + root_i, group_pivots + roots.at(root_i).start, sizeof(ky_t));
        }
    }


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
 
    index_t index = { root_n, roots.data(), group_n, groups.data(), root_pivots, group_pivots};

    char filename[] = "test.bin";
    //serialize(index, filename);
    index_t* index2 = deserialize(filename);


    /*
    ***********************
    *** query benchmark ***
    ***********************
    */

    // initalization
    ky_t* dev_key;
    int_t* dev_pos;

    assert(cudaMalloc(&dev_key, sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_pos, sizeof(int_t)) == cudaSuccess);

    cudaStream_t stream_left;
    cudaStream_t stream_mid;
    cudaStream_t stream_right;

    assert(cudaStreamCreate(&stream_left) == cudaSuccess);
    assert(cudaStreamCreate(&stream_mid) == cudaSuccess);
    assert(cudaStreamCreate(&stream_right) == cudaSuccess);

    ky_t* buffer_left;
    ky_t* buffer_mid;
    ky_t* buffer_right;

    assert(cudaMalloc(&buffer_left, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&buffer_mid, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&buffer_right, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);


    for (ix_size_t i = 5065118; i < NUMKEYS; ++i) {
        ky_t* key = keys + i;
        auto tmp = get_position_from_index(index2, *key, keys, dev_key, dev_pos,
            stream_left, stream_mid, stream_right,
            buffer_left, buffer_mid, buffer_right
        );
        if (tmp >= NUMKEYS || i != tmp && (memcmp(key, keys + tmp, sizeof(ky_t)) != 0)) {
            //get_position_from_index(index2, *key, keys, dev_key, dev_pos,
            //    stream_left, stream_mid, stream_right,
            //    buffer_left, buffer_mid, buffer_right    
            //);
            exit(1);
        }
        printf("%u\n", i);
    }

    assert(cudaFree(dev_key) == cudaSuccess);
    assert(cudaFree(dev_pos) == cudaSuccess);

    assert(cudaStreamDestroy(stream_left) == cudaSuccess);
    assert(cudaStreamDestroy(stream_mid) == cudaSuccess);
    assert(cudaStreamDestroy(stream_right) == cudaSuccess);

    assert(cudaFree(buffer_left) == cudaSuccess);
    assert(cudaFree(buffer_mid) == cudaSuccess);
    assert(cudaFree(buffer_right) == cudaSuccess);

    

    return 0;
}
