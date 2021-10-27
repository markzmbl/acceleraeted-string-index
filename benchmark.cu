
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
    
    ky_t pivots[group_n];
    for (ix_size_t group_i = 0; group_i < group_n; ++group_i) {
        memcpy(pivots + group_i, keys + groups.at(group_i).start, sizeof(ky_t));
    }

    std::vector<group_t> roots;
    if (group_n > 1) {
        grouping(pivots, group_n, et, pt, 10, 5, 5, roots);
    }
    ix_size_t root_n = roots.size();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
 
    index_t index = { root_n, roots.data(), group_n, groups.data(), pivots};

    char filename[] = "test.bin";
    //serialize(index, filename);
    index_t* index2 = deserialize(filename);

    for (ix_size_t i = 0; i < NUMKEYS; ++i) {
        ky_t* key = keys + i;
        auto tmp = get_position(index2, *key, keys);
        if (tmp >= NUMKEYS || i != tmp && (memcmp(key, keys + tmp, sizeof(ky_t)) != 0)) {
            get_position(index2, *key, keys);
        }
        printf("%u\n", i);
    }


    

    return 0;
}
