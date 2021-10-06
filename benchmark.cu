
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
    const fp_t et = 3.5E-8;
    const ix_size_t pt = 16;
    const ix_size_t fstep = 10'000;
    const ix_size_t bstep = 1'000;
    const ix_size_t min_size = CUDACORES;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // group meta data   
    std::vector<group_t> groups;
    grouping(keys, NUMKEYS, et, pt, fstep, bstep, min_size, groups);
    ix_size_t group_n = groups.size();
    ky_t* pivots = (ky_t*) malloc(group_n * sizeof(ky_t));
    for (ix_size_t group_i = 0; group_i < group_n; ++group_i) {
        strncpy(*(pivots + group_i), groups.at(group_i).pivot, sizeof(ky_t));
    } 
    print_keys(pivots, 0, 2);
    std::vector<group_t> root;
    root.reserve(group_n);
    grouping(pivots, group_n, et, pt, fstep, bstep, min_size, root);
    free(pivots);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;


    

    return 0;
}
