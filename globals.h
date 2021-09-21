// globals.h
#ifndef _GLOBALS_
#define _GLOBALS_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cusolverDn.h"

inline uint64_t safe_division(uint64_t divident, uint64_t divisor) {
    return (uint64_t) (divident + divisor - 1) / divisor;
}


// return enum
enum Status {success, threshold_exceed, out_of_memory, batch_end_reached};

// debug flag
bool verbose = true;
const bool debug = true;
const bool sanity_check = false;

// gpu specific types
typedef double fp_t;
typedef uint32_t int_t;
cudaDataType cuda_float = CUDA_R_64F;
const fp_t float_max = DBL_MAX;
const fp_t float_min = -float_max;
const fp_t float_eps = DBL_EPSILON;
const fp_t eps = 0.05;
int_t int_max = UINT32_MAX;
#define SINGLE false

// index size
typedef uint64_t ix_size_t;
const ix_size_t NUMKEYS = 200'000'000;

// character
typedef char ch_t;

// key size
typedef uint8_t ky_size_t;
const ky_size_t KEYLEN = 1<<4; 
const ky_size_t ky_size_max = UINT8_MAX;
const ky_size_t KEYSIZE = KEYLEN * sizeof(ch_t);

// key
typedef ch_t ky_t[KEYLEN];
// value
typedef uint8_t vl_t;

// group type
struct group_t {
    ix_size_t start;
    ix_size_t m;
    ky_size_t n;
    ky_size_t* feat_indices;
    fp_t* model;
    fp_t avg_err;
    fp_t min_err;
    fp_t max_err;
};

// gpu
const ix_size_t CUDACORES = 768;//1664;
const ix_size_t BLOCKSIZE = 1<<7;
const ix_size_t BLOCKNUM = (ix_size_t) (CUDACORES / BLOCKSIZE);
//const ix_size_t VRAM = 4.2331E+9; // whole capacity
const ix_size_t VRAM = 2.0919E+9 - 1.074E+9;// 4.2331E+9 - 1.074E+9; // 1 GiB reserved for graphics output
const ix_size_t BATCHLEN = safe_division(VRAM * 0.5, KEYSIZE);

const char FILENAME[] = "./gene/gene200.txt";

// create handles and params
// create handles



class GPUVar {
    public:
        void* address;
        size_t size_element;
        size_t count;
        size_t size_manual;
        bool allocated;
        GPUVar() {
            address = nullptr;
            size_element = 1;
            count = 1;
            allocated = 0;
            size_manual = 0;
        }
         size_t size () {
            if (size_manual == 0) {
                return size_element * count;
            } else {
                return size_manual;
            }
        }
        bool allocate () {
            if (cudaMalloc(&address, size()) == cudaSuccess) {
                allocated = true;
                return true;
            } else {
                return false;
            }
        }
        bool free () {
            if (cudaFree(address) == cudaSuccess) {
                allocated = false;
                return true;
            } else {
                return false;
            }            
        }
};

class GPUInt : public GPUVar {
    public:
        GPUInt () {
            size_element = sizeof(int_t);
        }
        int_t* ptr () {
            return (int_t*) address;
        }
};

class GPUFloat : public GPUVar {
    public:
        GPUFloat () {
            size_element = sizeof(fp_t);
        }
        fp_t* ptr () {
            return (fp_t*) address;
        }
};

class GPUChar : public GPUVar {
    public:
        GPUChar () {
            size_element = sizeof(ch_t);
        }
        ch_t* ptr () {
            return (ch_t*) address;
        }
};

class GPUKeySize : public GPUVar {
    public:
        GPUKeySize () {
            size_element = sizeof(ky_size_t);
        }
        ky_size_t* ptr () {
            return (ky_size_t*) address;
        }
};

class GPUInfoInt : public GPUVar {
    public:
        GPUInfoInt () {
            size_element = sizeof(int);
        }
        int* ptr () {
            return (int*) address;
        }
};


#endif // _GLOBALS_