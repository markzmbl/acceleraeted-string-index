
#include "globals.h"

#include <vector>


#ifndef _HELPERS_
#define _HELPERS_

void read_keys(ky_t* keys, const std::string filename) {

    std::string line;
    std::ifstream data(filename);

    for (ix_size_t key_i = 0; key_i < NUMKEYS; ++key_i) {
        std::getline(data, line);
        line = line.substr(0, KEYLEN);
        memcpy(keys + key_i, line.c_str(), KEYLEN);
    }

    data.close();
}

inline void print_key(const ky_t* key) {
    for(ky_size_t char_i = 0; char_i < KEYSIZE; ++char_i) {
        ch_t char0 = *(((ch_t*)key) + char_i);
        printf("%c", (char) char0);
    }
    printf("\n");
}

inline bool nearly_equal(fp_t a, fp_t b) {
    const fp_t abs_a = abs(a);
    const fp_t abs_b = abs(b);
    const fp_t diff  = abs(a - b);
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || diff < float_eps) {
        return diff < (eps * float_eps);
    } else {
        return diff / (abs_a + abs_b) < eps;
    }
}

inline uint64_t get_block_num(uint64_t work_load) {
    uint64_t optimal_block_num = safe_division(work_load, BLOCKSIZE);
    uint64_t block_num = (optimal_block_num > BLOCKNUM) ? BLOCKNUM : optimal_block_num;
    return block_num;
}

inline bool allocate_gpu_memory(std::vector<GPUVar*> &requests) {

    size_t required_memory = 0;
    size_t free_space = 0;
    size_t total_space = 0;

    for (size_t i = 0; i < requests.size(); ++i) {
        required_memory += requests[i]->size();
    }

    cudaMemGetInfo(&free_space, &total_space);
    if (required_memory > free_space) {
        return out_of_memory;
    } else {
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i]->allocate() == false) {
                for (size_t i = 0; i < requests.size(); ++i) {
                    requests[i]->free();
                }
                return false;
            }
        }
        return true;
    }
} 

void printMatrix(int m, int n, const fp_t*A, int lda, const char* name) {
    for(int row = 0 ; row < m ; row++) {
        for(int col = 0 ; col < n ; col++) {
            fp_t Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

#endif  // _HELPERS_