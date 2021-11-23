
#include "globals.h"

#include <vector>
#include <algorithm>
#include <string>
#include <assert.h>




#ifndef _HELPERS_
#define _HELPERS_

inline void print_key(const ky_t* key) {
    for(ky_size_t char_i = 0; char_i < KEYSIZE; ++char_i) {
        ch_t char0 = *(((ch_t*)*key) + char_i);
        printf("%c", (char) char0);
    }
    printf("\n");
}

inline void print_keys(const ky_t* keys, size_t start, size_t len) {
    for (size_t key_i = start; key_i < start + len; ++key_i) {
        print_key(keys + key_i);
    }
}

void read_keys(const std::string filename, ky_t* &keys,
        int64_t &numkeys, uint32_t step) {

    std::string line;
    std::ifstream data(filename);

    numkeys = 0;
    uint32_t key_i = 0;
    while(std::getline(data, line)) {
        if (key_i % step == 0) {
            line = line.substr(0, KEYLEN);
            memcpy(keys + numkeys, line.c_str(), KEYLEN);
            ++numkeys;
        }
        ++key_i;
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
        group.left_err,
        group.right_err,
        group.fsteps,
        group.bsteps
    );
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
        ky_t** buffer0, cudaStream_t** stream0,
        ky_t** buffer1, cudaStream_t** stream1) {

    ky_t* tmp_buffer = *buffer0;
    cudaStream_t* tmp_stream = *stream0;
    *buffer0 = *buffer1;
    *stream0 = *stream1;
    *buffer1 = tmp_buffer;
    *stream1 = tmp_stream;

}

inline void serialize(index_t* index, const char filename[]) {
    FILE* file;
    file = fopen(filename, "wb");

    fwrite(&(index->n), sizeof(uint32_t), 1, file);
    fwrite(&(index->root_n), sizeof(uint32_t), 1, file);
    fwrite(&(index->group_n), sizeof(uint32_t), 1, file);
    fwrite(index->root_pivots, sizeof(ky_t), index->root_n, file);
    fwrite(index->group_pivots, sizeof(ky_t), index->group_n, file);


    for (uint32_t group_i = 0; group_i < index->root_n + index->group_n; ++group_i) {
        group_t* group;
        if (group_i < index->root_n) {
            group = index->roots + group_i;
        } else {
            group = index->groups + group_i - index->root_n;
        }
        fwrite(&(group->start), sizeof(uint32_t), 1, file);
        fwrite(&(group->m), sizeof(uint32_t), 1, file);
        fwrite(&(group->n), sizeof(ky_size_t), 1, file);
        fwrite(group->feat_indices, sizeof(ky_size_t), group->n, file);
        fwrite(group->weights, sizeof(fp_t), group->n + 1, file);
        //fwrite(&(group->avg_err), sizeof(fp_t), 1, file);
        fwrite(&(group->left_err), sizeof(fp_t), 1, file);
        fwrite(&(group->right_err), sizeof(fp_t), 1, file);
        //fwrite(&(group->fsteps), sizeof(unsigned int), 1, file);
        //fwrite(&(group->bsteps), sizeof(unsigned int), 1, file);
   }
   fclose(file);
}

inline index_t* deserialize(const char filename[]) {
    FILE* file;
    file = fopen(filename,"rb");
    index_t* index = (index_t*) malloc(sizeof(index_t));
    fread(&(index->n), sizeof(uint32_t), 1, file);
    fread(&(index->root_n), sizeof(uint32_t), 1, file);
    fread(&(index->group_n), sizeof(uint32_t), 1, file);

    index->roots = (group_t*) malloc(index->root_n * sizeof(group_t));
    index->groups = (group_t*) malloc(index->group_n * sizeof(group_t));

    index->root_pivots = (ky_t*) malloc(index->root_n * sizeof(ky_t));
    fread(index->root_pivots, sizeof(ky_t), index->root_n, file);

    index->group_pivots = (ky_t*) malloc(index->group_n * sizeof(ky_t));
    fread(index->group_pivots, sizeof(ky_t), index->group_n, file);


    for (uint32_t group_i = 0; group_i < index->root_n + index->group_n; ++group_i) {
        group_t* group = (group_t*) malloc(sizeof(group_t));
        fread(&(group->start), sizeof(uint32_t), 1, file);
        fread(&(group->m), sizeof(uint32_t), 1, file);
        fread(&(group->n), sizeof(ky_size_t), 1, file);
        group->feat_indices = (ky_size_t*) malloc(group->n * sizeof(ky_size_t));
        fread(group->feat_indices, sizeof(ky_size_t), group->n, file);
        group->weights = (fp_t*) malloc((group->n + 1) * sizeof(fp_t));
        fread(group->weights, sizeof(fp_t), group->n + 1, file);
        //fread(&(group->avg_err), sizeof(fp_t), 1, file);
        fread(&(group->left_err), sizeof(fp_t), 1, file);
        fread(&(group->right_err), sizeof(fp_t), 1, file);
        //fread(&(group->fsteps), sizeof(unsigned int), 1, file);
        //fread(&(group->bsteps), sizeof(unsigned int), 1, file);

        if (group_i < index->root_n) {
            *(index->roots + group_i) = *group;
        } else {
            *(index->groups + group_i - index->root_n) = *group;
        }
    }
    fclose(file);
    return index;
}

inline void calculate_cusolver_buffer_size(cusolverDnHandle_t* cusolverH, cusolverDnParams_t* cusolverP, uint32_t maxsamples, ky_size_t feat_threash, fp_t* A, fp_t* tau, fp_t* B, uint64_t* d_work_size, uint64_t* h_work_size) {
    // calculate workspace size
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    

    // check memory usage for qr factorization
    uint64_t d_work_size_qr = 0;
    uint64_t h_work_size_qr = 0;
    cusolver_status = cusolverDnXgeqrf_bufferSize(
        *cusolverH, *cusolverP, maxsamples, feat_threash + 1,
        cuda_float, A, maxsamples /*lda*/,
        cuda_float, tau,
        cuda_float, &d_work_size_qr, &h_work_size_qr
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // check memory usage for transposed matrix multiplication
    int d_work_size_tm = 0;
    cusolver_status = cusolverDnDormqr_bufferSize(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        maxsamples, 1 /*nrhs*/, feat_threash + 1,
        A, maxsamples /*lda*/,
        tau, B,
        maxsamples /*ldb*/, &d_work_size_tm
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // allocate workbuffers
    *d_work_size = (d_work_size_qr > ((uint64_t) d_work_size_tm)) ? d_work_size_qr : (uint64_t) d_work_size_tm;
    *h_work_size = h_work_size_qr;
}

//https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

#endif  // _HELPERS_