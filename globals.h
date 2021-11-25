// globals.h
#ifndef _GLOBALS_
#define _GLOBALS_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cusolverDn.h"
#include <thread>
#include <omp.h>
#include <thread>



inline uint64_t safe_division(uint64_t divident, uint64_t divisor) {
    return (uint64_t) (divident + divisor - 1) / divisor;
}

inline cudaDeviceProp get_device_prop() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop;
}

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int get_block_size(cudaDeviceProp prop, int cudacores) {
    int blocksize = prop.maxThreadsPerBlock;
    // todo assert blocksizes power of two
    while (cudacores % blocksize != 0) {
        blocksize /= 2;
    }
    return blocksize;
}


// return enum
enum GroupStatus {threshold_success, threshold_exceed, out_of_memory, batch_exceed, batch_load, finished, too_small};


// debug flag
bool verbose = false;
bool debug = false;
bool sanity_check = false;
bool csv = false;

// gpu specific types
typedef double fp_t;
typedef uint32_t int_t;
cudaDataType cuda_float = CUDA_R_64F;
const fp_t float_max = DBL_MAX;
const fp_t float_min = -float_max;
const fp_t float_eps = DBL_EPSILON;
const fp_t eps = 0.2;
#define SINGLE false
// bias value A
const fp_t bias = 1;

// index size
const uint32_t NUMKEYS = 800'000'000;
const uint32_t int_max = UINT32_MAX;

// character
typedef char ch_t;

// key size
typedef uint8_t ky_size_t;
const ky_size_t KEYLEN = 32; 
const ky_size_t ky_size_max = UINT8_MAX;

// key
typedef ch_t ky_t[KEYLEN];
const ky_size_t KEYSIZE = sizeof(ky_t);
// group type
struct group_t {
    uint32_t start;
    uint32_t m;
    ky_size_t n;
    ky_size_t* feat_indices;
    fp_t* weights;
    fp_t avg_err;
    fp_t left_err;
    fp_t right_err;
    unsigned int fsteps;
    unsigned int bsteps;
};

// index type
struct index_t {
    uint32_t n;
    uint32_t root_n;
    group_t* roots;
    uint32_t group_n;
    group_t* groups;
    ky_t* root_pivots;
    ky_t* group_pivots;
};



// gpu

cudaDeviceProp prop = get_device_prop();
const uint32_t CUDACORES = getSPcores(prop);
const uint32_t CPUCORES = std::thread::hardware_concurrency();
const uint32_t BLOCKSIZE = get_block_size(prop, CUDACORES);
const uint32_t BLOCKNUM = (uint32_t) (CUDACORES / BLOCKSIZE);
//const uint32_t VRAM = 4.2331E+9; // whole capacity
const uint32_t VRAM = prop.totalGlobalMem;
const float LOADFACTOR = 0.12;
//const uint32_t BATCHLEN = safe_division(VRAM * LOADFACTOR, KEYSIZE);
const uint32_t BATCHLEN = 200'000'000;
uint32_t MINSIZE = 1'000'000;
const uint32_t QUERYSIZE = CUDACORES;
const uint32_t MAXSAMPLES = 40'000'000;

const char FILENAME[] = "./gene/gene200normal.txt";

// create handles and params
// create handles





#endif // _GLOBALS_