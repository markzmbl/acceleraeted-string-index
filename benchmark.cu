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











int main() {  
    
    // cuda debug
    cudaError_t cudaStat1 = cudaSuccess;
    
    ky_t* keys = (ky_t*) malloc(NUMKEYS * KEYLEN);
    read_keys(keys, FILENAME);
    
    // parameters
    const ix_size_t et = 200;
    const ix_size_t pt = 16;
    const ix_size_t fstep = 5000;
    const ix_size_t bstep = 50;

    // first batch size
    // todo benchmarks. first attempt 50% keys

    // group meta data   
    std::vector<group_t> groups;


    grouping(keys, et, pt, fstep, bstep, groups);

    return 0;
}
                        