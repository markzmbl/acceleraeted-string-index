
#include "globals.h"

#include <vector>
#include <algorithm>
#include <string>


#ifndef _HELPERS_
#define _HELPERS_

void read_keys(ky_t* keys, const std::string filename) {

    std::string line;
    std::ifstream data(filename);

    for (ix_size_t key_i = 0; key_i < NUMKEYS; ++key_i) {
        std::getline(data, line);
        line = line.substr(0, KEYLEN);
        //std::replace(line.begin(), line.end(), 'A', '!');
        //std::replace(line.begin(), line.end(), 'B', '@');
        //std::replace(line.begin(), line.end(), 'C', '_');
        //std::replace(line.begin(), line.end(), 'D', '~');
        memcpy(keys + key_i, line.c_str(), KEYLEN);
    }

    data.close();
}

inline void print_group(int num, group_t group) {
    printf(
        "[GROUP]\t%'d\n"
        "\tstart:\t%'d\n"
        "\tm:\t%'d\n"
        "\tn:\t%'d\n"
        "\tavg:\t%.10e\n"
        "\tmin:\t%.10e\n"
        "\tmax:\t%.10e\n"
        "\tfsteps:\t%'d\n"
        "\tbsteps:\t%'d\n",
        num,
        group.start,
        group.m,
        group.n,
        group.avg_err,
        group.min_err,
        group.max_err,
        group.fsteps,
        group.bsteps
    );
}

inline void print_key(const ky_t* key) {
    for(ky_size_t char_i = 0; char_i < KEYSIZE; ++char_i) {
        ch_t char0 = *(((ch_t*)key) + char_i);
        printf("%c", (char) char0);
    }
    printf("\n");
}

inline void print_keys(const ky_t* keys, size_t start, size_t len) {
    for (size_t key_i = start; key_i < start + len; ++key_i) {
        print_key(keys + key_i);
    }
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
        return false;
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
            printf("%s(%'d,%'d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

// 0 equal, -1 less than, 1 greater than
inline int8_t compare_keys(const ky_t &key0, const ky_t &key1) {
    for (ky_size_t char_i = 0; char_i < sizeof(ky_t); ++char_i) {
        ch_t char0 = *(((ch_t*) key0) + char_i);
        ch_t char1 = *(((ch_t*) key1) + char_i);
        if (char0 < char1) {
            return -1;
        } else if (char0 > char1) {
            return 1;
        }
    }
    return 0;
}

inline void swap_buffer_and_stream(
        ky_t* buffer0, cudaStream_t* stream0,
        ky_t* buffer1, cudaStream_t* stream1) {

    ky_t* tmp_buffer = buffer0;
    cudaStream_t* tmp_stream = stream0;
    buffer0 = buffer1;
    stream0 = stream1;
    buffer1 = tmp_buffer;
    stream1 = tmp_stream;

}


#endif  // _HELPERS_