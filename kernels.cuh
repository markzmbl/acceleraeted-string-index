#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>
#include <float.h>
#include "globals.h"


#ifndef _KERNELS_
#define _KERNELS_



__global__ void print_kernel(const ky_t* array, ix_size_t len) {
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid < len)
        printf("%u: %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c\n",(uint16_t) thid,
            *(*(array + thid)),
            *(*(array + thid) + 1),
            *(*(array + thid) + 2),
            *(*(array + thid) + 3),
            *(*(array + thid) + 4),
            *(*(array + thid) + 5),
            *(*(array + thid) + 6),
            *(*(array + thid) + 7),
            *(*(array + thid) + 8),
            *(*(array + thid) + 9),
            *(*(array + thid) + 10),
            *(*(array + thid) + 11),
            *(*(array + thid) + 12),
            *(*(array + thid) + 13),
            *(*(array + thid) + 14),
            *(*(array + thid) + 15)

        );
}

__global__ void pair_prefix_kernel(
        const ky_t* keys, ky_size_t* dev_pair_lens, ix_size_t batch_len) {
    
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    //if (!thid) printf("batchlen: %u\n", batch_len);

    for (ix_size_t thid_i = thid; thid_i < batch_len - 1; thid_i += gridDim.x * blockDim.x) {
        ky_t* key1 = (ky_t*) keys + thid_i;
        ky_t* key2 = (ky_t*) keys + thid_i + 1;
        for (ky_size_t char_i = 0; char_i < KEYLEN; ++char_i) {
            
            ch_t char1 = *(((ch_t*) key1) + char_i);
            ch_t char2 = *(((ch_t*) key2) + char_i);
            //printf("[pair_prefix_kernel] c1: %c c2: %c\n", char1, char2);
            if (char1 != char2) {
                *(dev_pair_lens + thid_i) = char_i;
//                if (debug == true) {
//                    //printf("[pair_prefix_kernel] thid_i: %u, key1: %s, key2: %s, prefixlen: %u\n", (uint16_t) thid_i, key1, key2, char_i);
//                    printf("[pair_prefix_kernel] thid_i: %u, prefixlen: %u\n", (uint16_t) thid_i, char_i);
//                }
                break;
            }
        }
    }        
}




__global__ void rmq_kernel(
        const ky_size_t* dev_pair_lens, const ix_size_t start_i, const ix_size_t m,
        int_t* dev_min_len, int_t* dev_max_len) {
    
    dev_pair_lens += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    // fill with neutral values
    ky_size_t loc_min_len = ky_size_max;
    ky_size_t loc_max_len = 0;
  
    __syncthreads();

    // iterate batch in strides
    for (ix_size_t thid_i = thid; thid_i < m; thid_i += gridDim.x * blockDim.x) {

        loc_min_len = *(dev_pair_lens + thid_i);
        loc_max_len = *(dev_pair_lens + thid_i);

        // reduce each warp
        for (uint8_t offset = 1; offset < 32; offset *= 2) {

            ky_size_t tmp_min_len = __shfl_down_sync(0xFFFFFFFF, loc_min_len, offset);
            ky_size_t tmp_max_len = __shfl_down_sync(0xFFFFFFFF, loc_max_len, offset);

            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m && threadIdx.x % 32 + offset < 32) {
                
                // min
                if (tmp_min_len < loc_min_len) {
                    loc_min_len = tmp_min_len;
                }
                // max
                if (tmp_max_len > loc_max_len) {
                    loc_max_len = tmp_max_len;
                }
            }
        }

        // declare shared memory for block wide communication
        extern __shared__ ky_size_t shrd_mmry0[];
        ky_size_t* blk_min_lens = shrd_mmry0;
        ky_size_t* blk_max_lens = blk_min_lens + (blockDim.x / 32);


        ix_size_t shrd_mmry_i;
        ix_size_t shrd_mmry_j;

        // write into shared memory
        if (threadIdx.x % 32 == 0) {
            shrd_mmry_i = threadIdx.x / 32;
            *(blk_min_lens + shrd_mmry_i) = loc_min_len;
            *(blk_max_lens + shrd_mmry_i) = loc_max_len;
        }

        // reduce each block to a single value
        for (ix_size_t offset = 32; offset < blockDim.x; offset *= 2) {
            __syncthreads();
            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m && threadIdx.x + offset < blockDim.x) {

                shrd_mmry_j = (threadIdx.x + offset) / 32;

                // min
                if (*(blk_min_lens + shrd_mmry_j) < *(blk_min_lens + shrd_mmry_i)) {
                    *(blk_min_lens + shrd_mmry_i) = *(blk_min_lens + shrd_mmry_j);
                }
                // max
                if (*(blk_max_lens + shrd_mmry_j)  > *(blk_max_lens + shrd_mmry_i)) {
                    *(blk_max_lens + shrd_mmry_i) = *(blk_max_lens + shrd_mmry_j) ;
                }
            }
        }

        // reduce first entries of blocks
        if (threadIdx.x == 0) {

            atomicMin(dev_min_len, (int_t) *blk_min_lens);
            atomicMax(dev_max_len, (int_t) *blk_max_lens);

        }
    }
}

__global__ void column_major_kernel(
        const ky_t* keys, fp_t* A,
        const ix_size_t start_i, const ix_size_t m,
        const ky_size_t* feat_indices, const ky_size_t n) {

    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    // move pointer to group start
    keys += start_i;

    for (ix_size_t thid_i = thid; thid_i < m * n; thid_i += gridDim.x * blockDim.x) {

        ix_size_t key_i =  thid_i / n;
        ky_size_t feat_i = (thid_i % n);
        ky_size_t char_i = *(feat_indices + feat_i);
        
        //if (thid_i < n)
        //printf("%u\n", char_i);
        
        fp_t elem = (fp_t) *(((ch_t*) *(keys + key_i)) + char_i);
        *(A + feat_i * m + key_i) = elem;
        //if (*(A + feat_i * m + key_i) == 0.0f)
        //    printf("%u\n", thid_i);
        //printf("thid_i: 0, key_i: %u, feat_i: %u, char_i: , char: %c\n",(uint16_t)key_i, (uint8_t)feat_i, (uint8_t)char_i, (uint8_t)*(A + feat_i * m + key_i));
        //printf("thid_i: 0, key_i: 0, feat_i: 0, char_i: , char: %c\n",(uint8_t)*(A + feat_i * m + key_i));

    }
}

__global__ void set_postition_kernel(
        fp_t* B, ix_size_t processed, ix_size_t start_i, ix_size_t m) {

    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (ix_size_t thid_i = thid; thid_i < m; thid_i += gridDim.x * blockDim.x) {

        *(B + thid_i) = processed + start_i + thid_i;

    }
}

__global__ void equal_column_kernel_old(
        ky_t* keys, ix_size_t start_i, ky_size_t feat_start,
        ix_size_t m, ky_size_t n_star,
        int_t* uneqs, ch_t* col_vals, int_t* mutexes) {
    
    // move pointer to group start
    keys += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("thid: %u\n", (uint16_t)thid);
    
    // neutral value
    int_t loc_uneq = 0;
    ch_t loc_val = 0;
    
    __syncthreads();
    
    
    for (ix_size_t thid_i = thid; thid_i < m * n_star; thid_i += gridDim.x * blockDim.x) {
      
        ix_size_t key_i = thid_i / n_star;
        ky_size_t feat_i = thid_i % n_star;

        // todo find efficient way to exit early
        //if (*(uneqs + feat_i) > 0) {
        //    continue;
        //}

        // load character in register
        loc_val = *(((ch_t*) *(keys + key_i)) + feat_start + feat_i);
        
        // reduce each warp
        for (uint8_t offset = 32/2; offset > 0; offset /= 2) {
            
            ch_t tmp_val = __shfl_down_sync(0xFFFFFFFF, loc_val, offset);
            
            if (thid_i % 32 < offset && key_i + offset < m) {
                loc_uneq |= (loc_val != tmp_val);
                loc_val = tmp_val; 
            }
        }

        // declare shared memory for block wide communication

        extern __shared__ ch_t shrd_mmry1[];
        ch_t* blk_vals = shrd_mmry1;
        int_t* blk_uneqs = (int_t*) (blk_vals + (blockDim.x / 32));
        
        ix_size_t shrd_mmry_i;
        ix_size_t shrd_mmry_j;
        
        // write into shared memory
        if (threadIdx.x % 32 == 0) {
            shrd_mmry_i = threadIdx.x / 32;
            *(blk_vals + shrd_mmry_i) = loc_val;
            *(blk_uneqs + shrd_mmry_i) = loc_uneq;
        }
        
        // reduce each block to a single value
        // offset is defined by the shared memory offset
        for (ix_size_t offset = 1; offset * 32 < blockDim.x; offset *= 2) {
            
            __syncthreads();

            if (threadIdx.x % (offset * 32) == 0 && key_i + offset < m && threadIdx.x + (offset * 32) < blockDim.x) {
                
                shrd_mmry_j = (shrd_mmry_i + offset) / 32;
                
                // read right values
                ch_t tmp_val = *(blk_vals + shrd_mmry_j);
                
                //printf("i: %u, shrd: %u, j: %u, shrd: %u\n", (uint8_t) shrd_mmry_i, (uint8_t)*(blk_max_lens + shrd_mmry_i), (uint8_t) shrd_mmry_j, (uint8_t)*(blk_max_lens + shrd_mmry_j));
                // write to beginning
                *(blk_uneqs + shrd_mmry_i) |= (*(blk_vals + shrd_mmry_i) != *(blk_vals + shrd_mmry_j));
                *(blk_vals + shrd_mmry_i) = *(blk_vals + shrd_mmry_j);
            }
            
        }    
        
        // first thread of block reads from shared memory
        if (threadIdx.x == 0 && *blk_uneqs > 0) {
            *(uneqs + feat_i) = 1;
        }
    }
}

__global__ void equal_column_kernel(
        ky_t* keys, ix_size_t start_i, ky_size_t feat_start,
        ix_size_t m, ky_size_t n_star,
        int_t* uneqs, ch_t* col_vals, int_t* mutexes) {

    keys += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;

    if (thid < n_star) {
        for (ix_size_t key_i = 0; key_i < m - 1; ++key_i) {
            if (*(((ch_t*) *(keys + key_i)) + feat_start + thid) != *(((ch_t*) *(keys + key_i + 1)) + feat_start + thid)) {
                *(uneqs + thid) = 1;
                break;
            }
        }
    }

}

    
__global__ void model_error_kernel(
    ky_t* keys, ix_size_t processed, ix_size_t start_i, fp_t* B,
    const ky_size_t* feat_indices, ix_size_t m, ky_size_t n,
    fp_t* dev_acc_error, fp_t* dev_min_error, fp_t* dev_max_error,
    int_t* mutex) {

    // move keys pointer
    keys += start_i;
    // safe division for columns
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    for (ix_size_t thid_i = thid; thid_i < m; thid_i += gridDim.x * blockDim.x) {
        
        fp_t loc_acc_err = 0;
       
        // dot product (prediction)
        for (ky_size_t feat_i = 0; feat_i < n; ++feat_i) {
            ky_size_t char_i = *(feat_indices + feat_i);
            loc_acc_err += ((fp_t) *(((ch_t*) *(keys + thid_i)) + char_i)) * *(B + feat_i);
            //if (thid_i == 1) printf("key_err: %f, char: %f, feat: %f\n", loc_acc_err, (fp_t) *(((ch_t*) *(keys + thid_i)) + char_i), *(B + feat_i));
        }
        
        // subtract actual position (key error)
        loc_acc_err -= (processed + start_i + thid_i);

        //printf("key_i: %lu, key_err: %f\n", (uint16_t)thid_i, loc_acc_err);

        //printf("thid_i: %u, err: %f\n", (uint16_t)thid_i, loc_acc_err);
        
        // declare min an max variable
        fp_t loc_min_err = loc_acc_err;
        fp_t loc_max_err = loc_acc_err;

        // begin warp shuffle
        for (uint8_t offset = 1; offset < 32; offset *= 2) {
            fp_t tmp_acc_err = __shfl_down_sync(0xFFFFFFFF, loc_acc_err, offset);
            fp_t tmp_min_err = __shfl_down_sync(0xFFFFFFFF, loc_min_err, offset);
            fp_t tmp_max_err = __shfl_down_sync(0xFFFFFFFF, loc_max_err, offset);

            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m && threadIdx.x % 32 + offset < 32) {
                //printf("thid_i: %u, offset: %u, loc: %f, tmp: %f, loc: %f, tmp: %f, loc: %f, tmp: %f\n", (uint16_t)thid_i, offset, loc_acc_err, tmp_acc_err, loc_min_err, tmp_min_err, loc_max_err, tmp_min_err);
                
                // sum
                loc_acc_err += tmp_acc_err;
                // min
                if (tmp_min_err < loc_min_err) {
                    loc_min_err = tmp_min_err;
                }
                // max
                if (tmp_max_err > loc_max_err) {
                    loc_max_err = tmp_max_err;
                }
            }
        }
        // declare shared memory for block wide communication
        extern __shared__ fp_t shrd_mmry3[];
        fp_t* blk_acc_errs = shrd_mmry3;
        fp_t* blk_min_errs = blk_acc_errs + (blockDim.x / 32);
        fp_t* blk_max_errs = blk_min_errs + (blockDim.x / 32);
        
        ix_size_t shrd_mmry_i;
        ix_size_t shrd_mmry_j;
        
        // write into shared memory
        if (threadIdx.x % 32 == 0) {
            shrd_mmry_i = threadIdx.x / 32;
            *(blk_acc_errs + shrd_mmry_i) = loc_acc_err;
            *(blk_min_errs + shrd_mmry_i) = loc_min_err;
            *(blk_max_errs + shrd_mmry_i) = loc_max_err;
        }
        
        
        // begin block shuffled
        for (ix_size_t offset = 32; offset < blockDim.x; offset *= 2) {
            
            __syncthreads();
            if (threadIdx.x % (offset * 2) == 0  && thid_i + offset < m && threadIdx.x + offset < blockDim.x) {
                               
                shrd_mmry_j = (threadIdx.x + offset) / 32;
                
                // sum
                *(blk_acc_errs + shrd_mmry_i) += *(blk_acc_errs + shrd_mmry_j);
                // min
                if (*(blk_min_errs + shrd_mmry_j) < *(blk_min_errs + shrd_mmry_i)) {
                    *(blk_min_errs + shrd_mmry_i) = *(blk_min_errs + shrd_mmry_j);
                }
                // max
                if (*(blk_max_errs + shrd_mmry_j) > *(blk_max_errs + shrd_mmry_i)) {
                    *(blk_max_errs + shrd_mmry_i) = *(blk_max_errs + shrd_mmry_j);
                }
            }
            
        }
        
        // begin block reduction
        if (threadIdx.x == 0) {
            //if ((thid_i / blockDim.x) < 1000) printf("thid_i: %u, acc: %f\n", (uint16_t)(thid_i / blockDim.x), *blk_acc_errs);
            // sum
            atomicAdd(dev_acc_error, *blk_acc_errs);
            
            // lock !
            while (0 != (atomicCAS(mutex, 0, 1))) {}
            
            // sum debug
            //*dev_acc_error += *blk_acc_errs;

            // min
            if (*blk_min_errs < *dev_min_error) {
                *dev_min_error = *blk_min_errs;
            }
            // max
            if (*blk_max_errs > *dev_max_error) {
                *dev_max_error = *blk_max_errs;
            }
            
            // unlock
            atomicExch(mutex, 0);
            
        }   
    }
}



#endif  // _KERNELS_