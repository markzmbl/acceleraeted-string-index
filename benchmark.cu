
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
    uint32_t numkeys = read_keys(keys, FILENAME);
    
    // parameters
    const fp_t et = 1e-12;
    const uint32_t pt = 16;
    uint32_t fstep = 1'000'000;
    uint32_t bstep = 100'000;
    const uint32_t minsize = 1'000'000;


    assert(pt <= KEYLEN);
    assert(bstep <= MINSIZE);
    assert(fstep <= NUMKEYS);



    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //index_t* index = create_index(
    //    keys, numkeys, et, pt,
    //    fstep, bstep, minsize
    //);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
 
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

    ky_t* buffer_mid;

    assert(cudaMalloc(&buffer_mid, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);

    for (uint32_t i = 0; i < NUMKEYS; ++i) {
        ky_t* key = keys + i;
        auto tmp1 = get_position_from_index2(index2, *key, keys);
        auto tmp = get_position_from_index(index2, *key, keys, dev_key, dev_pos, buffer_mid);
        if (tmp >= NUMKEYS || i != tmp && (memcmp(key, keys + tmp, sizeof(ky_t)) != 0)) {
            exit(1);
        }
        printf("%u\n", i);
    }

    assert(cudaFree(dev_key) == cudaSuccess);
    assert(cudaFree(dev_pos) == cudaSuccess);

    assert(cudaFree(buffer_mid) == cudaSuccess);

    

    return 0;
}
