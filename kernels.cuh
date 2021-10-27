#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>
#include <float.h>
#include "globals.h"
#include "helpers.h"

#ifndef _KERNELS_
#define _KERNELS_



__global__ void print_kernel(const ky_t* array, ix_size_t len) {
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    const ky_t* key = (ky_t*) array + thid;
    if (thid < len)
        printf("%u: %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c\n",(uint16_t) thid,
            *(((ch_t*) key) + 0),
            *(((ch_t*) key) + 1),
            *(((ch_t*) key) + 2),
            *(((ch_t*) key) + 3),
            *(((ch_t*) key) + 4),
            *(((ch_t*) key) + 5),
            *(((ch_t*) key) + 6),
            *(((ch_t*) key) + 7),
            *(((ch_t*) key) + 8),
            *(((ch_t*) key) + 9),
            *(((ch_t*) key) + 10),
            *(((ch_t*) key) + 11),
            *(((ch_t*) key) + 12),
            *(((ch_t*) key) + 13),
            *(((ch_t*) key) + 14),
            *(((ch_t*) key) + 15)
        );
}

__global__ void pair_prefix_kernel(
        const ky_t* keys, ky_size_t* dev_pair_lens, ix_size_t batch_len) {
    
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    //if (!thid) printf("batchlen: %'d\n", batch_len);

    for (ix_size_t thid_i = thid; thid_i < batch_len - 1; thid_i += gridDim.x * blockDim.x) {
        ky_t* key1 = (ky_t*) keys + thid_i;
        ky_t* key2 = (ky_t*) keys + thid_i + 1;
        ky_size_t prefix_len = ky_size_max;
        for (ky_size_t char_i = 0; char_i < KEYLEN; ++char_i) {
            
            ch_t char1 = *(((ch_t*) key1) + char_i);
            ch_t char2 = *(((ch_t*) key2) + char_i);


            //printf("[pair_prefix_kernel] c1: %c c2: %c\n", char1, char2);
            if (char1 != char2) {
                prefix_len = char_i;
                break;
            }
        }
        *(dev_pair_lens + thid_i) = prefix_len;
    }        
}




__global__ void rmq_kernel(
        const ky_size_t* dev_pair_lens, const ix_size_t start_i, const ix_size_t m_star,
        int_t* dev_min_len, int_t* dev_max_len, fp_t step) {
    
    dev_pair_lens += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    // fill with neutral values
    ky_size_t loc_min_len = ky_size_max;
    ky_size_t loc_max_len = 0;
  
    __syncthreads();
    // iterate batch in strides
    for (ix_size_t thid_i = thid; thid_i < m_star; thid_i += gridDim.x * blockDim.x) {

        ky_size_t len = *(dev_pair_lens + ((ix_size_t) (thid_i * step)));
        if (len != ky_size_max) {
            loc_min_len = loc_max_len = len;
        }

        // reduce each warp
        for (uint8_t offset = 1; offset < 32; offset *= 2) {

            ky_size_t tmp_min_len = __shfl_down_sync(0xFFFFFFFF, loc_min_len, offset);
            ky_size_t tmp_max_len = __shfl_down_sync(0xFFFFFFFF, loc_max_len, offset);

            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m_star && threadIdx.x % 32 + offset < 32) {
                
                // min
                if (tmp_min_len < loc_min_len ) {
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
            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m_star && threadIdx.x + offset < blockDim.x) {

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
        const ix_size_t start_i, const ix_size_t m_star,
        const ky_size_t* feat_indices, const ky_size_t n_tilde,
        const fp_t step) {

    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    // move pointer to group start
    keys += start_i;

    for (ix_size_t thid_i = thid; thid_i < m_star * n_tilde; thid_i += gridDim.x * blockDim.x) {

        ix_size_t key_i =  thid_i / n_tilde;
        ky_size_t feat_i = (thid_i % n_tilde);
        ky_size_t char_i;
        
        //if(!feat_i) printf("key_i %u, step %f, key: %u\n",(uint8_t) (key_i), (float) step,(uint32_t) (key_i * step));

        if (feat_i < n_tilde - 1) {
            char_i = *(feat_indices + feat_i);
            *(A + feat_i * m_star + key_i) = (fp_t) *(((ch_t*) *(keys + ((ix_size_t) (key_i * step)))) + char_i);
        } else {
            *(A + feat_i * m_star + key_i) = bias;
        }
        
        //if (thid_i < n)
        //printf("%'d\n", char_i);
        
        //if (*(A + feat_i * m + key_i) == 0.0f)
        //    printf("%'d\n", thid_i);
        //printf("thid_i: 0, key_i: %'d, feat_i: %'d, char_i: , char: %c\n",(uint16_t)key_i, (uint8_t)feat_i, (uint8_t)char_i, (uint8_t)*(A + feat_i * m + key_i));
        //printf("thid_i: 0, key_i: 0, feat_i: 0, char_i: , char: %c\n",(uint8_t)*(A + feat_i * m + key_i));

    }
}

__global__ void set_postition_kernel(
        fp_t* B, ix_size_t processed, ix_size_t start_i,
        ix_size_t m_star, fp_t step) {

    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (ix_size_t thid_i = thid; thid_i < m_star; thid_i += gridDim.x * blockDim.x) {

        *(B + thid_i) = processed + start_i + ((ix_size_t) (thid_i * step));

    }
}

__global__ void equal_column_kernel_old(
        ky_t* keys, ix_size_t start_i, ky_size_t feat_start,
        ix_size_t m, ky_size_t n_star,
        int_t* uneqs, ch_t* col_vals, int_t* mutexes) {
    
    // move pointer to group start
    keys += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("thid: %'d\n", (uint16_t)thid);
    
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
                
                //printf("i: %'d, shrd: %'d, j: %'d, shrd: %'d\n", (uint8_t) shrd_mmry_i, (uint8_t)*(blk_max_lens + shrd_mmry_i), (uint8_t) shrd_mmry_j, (uint8_t)*(blk_max_lens + shrd_mmry_j));
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
        ix_size_t m_star, ky_size_t n_star,
        int_t* uneqs, ch_t* col_vals, int_t* mutexes, fp_t step) {

    keys += start_i;
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;

    if (thid < n_star) {
        for (ix_size_t key_i = 0; key_i < m_star - 1; ++key_i) {
            if (*(((ch_t*) *(keys + ((ix_size_t) (key_i * step)))) + feat_start + thid) != *(((ch_t*) *(keys + ((ix_size_t) ((key_i + 1) * step)))) + feat_start + thid)) {
                *(uneqs + thid) = 1;
                break;
            }
        }
    }

}

    
__global__ void model_error_kernel(
    ky_t* keys, ix_size_t processed, ix_size_t start_i, fp_t* B,
    const ky_size_t* feat_indices, ix_size_t m, ky_size_t n_tilde,
    fp_t* dev_acc_error, fp_t* dev_min_error, fp_t* dev_max_error,
    int_t* mutex) {

    // move keys pointer
    keys += start_i;
    // safe division for columns
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    for (ix_size_t thid_i = thid; thid_i < m; thid_i += gridDim.x * blockDim.x) {
        
        fp_t loc_acc_err = 0;
       
        // dot product (prediction) + bias constant
        for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
            if (feat_i < n_tilde - 1) {
                ky_size_t char_i = *(feat_indices + feat_i);
                loc_acc_err += ((fp_t) *(((ch_t*) *(keys + thid_i)) + char_i)) * *(B + feat_i);
            } else {
                loc_acc_err += *(B + feat_i);
            }
        }
        // subtract actual position (key error)
        loc_acc_err -= (processed + start_i + thid_i);

        //printf("key_i: %'d, key_err: %f\n", (uint16_t)thid_i, loc_acc_err);

        //printf("thid_i: %'d, err: %f\n", (uint16_t)thid_i, loc_acc_err);
        
        // declare min an max variable
        fp_t loc_min_err;
        fp_t loc_max_err;
        loc_min_err = loc_max_err = loc_acc_err;

        // begin warp shuffle
        for (uint8_t offset = 1; offset < 32; offset *= 2) {
            fp_t tmp_acc_err = __shfl_down_sync(0xFFFFFFFF, loc_acc_err, offset);
            fp_t tmp_min_err = __shfl_down_sync(0xFFFFFFFF, loc_min_err, offset);
            fp_t tmp_max_err = __shfl_down_sync(0xFFFFFFFF, loc_max_err, offset);

            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < m && threadIdx.x % 32 + offset < 32) {
                //printf("thid_i: %'d, offset: %'d, loc: %f, tmp: %f, loc: %f, tmp: %f, loc: %f, tmp: %f\n", (uint16_t)thid_i, offset, loc_acc_err, tmp_acc_err, loc_min_err, tmp_min_err, loc_max_err, tmp_min_err);
                
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
        fp_t* blk_min_errs = (blk_acc_errs + (blockDim.x / 32));
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
            //if ((thid_i / blockDim.x) < 1000) printf("thid_i: %'d, acc: %f\n", (uint16_t)(thid_i / blockDim.x), *blk_acc_errs);
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


__global__ void query_kernel(
    const ky_t* query, const ky_t* keys, const ix_size_t range,
    int_t* dev_pos, bool* in_block) {
    
    const ix_size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (ix_size_t thid_i = thid; thid_i < range; thid_i += gridDim.x * blockDim.x) {

        ix_size_t loc_pos = thid_i;
        ix_size_t ref_pos = thid_i;

        const ky_t* pivot = keys + thid_i;


        for (ky_size_t char_i = 0; char_i < sizeof(ky_t); ++char_i) {
            ch_t query_char = *(((ch_t*) *query) + char_i);
            ch_t key_char = *(((ch_t*) pivot) + char_i);
            if (key_char < query_char) {
                loc_pos = int_max;
                break;
            } else if (key_char > query_char) {
                break;
            }
        }
        // begin warp shuffle
        for (uint8_t offset = 1; offset < 32; offset *= 2) {
            int_t tmp_pos = __shfl_down_sync(0xFFFFFFFF, loc_pos, offset);

            if (threadIdx.x % (offset * 2) == 0 && thid_i + offset < range && threadIdx.x % 32 + offset < 32) {  
                if (tmp_pos < int_max && (tmp_pos < loc_pos || loc_pos == int_max)) {
                    loc_pos = tmp_pos;
                }
            }
        }

        //printf("thread:\tthid_i: %u -> %u\n", (uint16_t) thid_i, (int16_t) loc_pos);

        // declare shared memory for block wide communication
        extern __shared__ int_t shrd_mmry4[];
        int_t* blk_pos = shrd_mmry4;

        ix_size_t shrd_mmry_i;
        ix_size_t shrd_mmry_j;

        // write into shared memory
        if (threadIdx.x % 32 == 0) {
            //printf("warp:\tthid_i: %u -> %u\n", (uint16_t) thid_i, (int16_t) loc_pos);
            shrd_mmry_i = threadIdx.x / 32;
            *(blk_pos + shrd_mmry_i) = loc_pos;
        }
        
        
        // begin block shuffled
        for (ix_size_t offset = 32; offset < blockDim.x; offset *= 2) {
            
            __syncthreads();
            if (threadIdx.x % (offset * 2) == 0  && thid_i + offset < range && threadIdx.x + offset < blockDim.x) {
                               
                shrd_mmry_j = (threadIdx.x + offset) / 32;
                
                loc_pos = *(blk_pos + shrd_mmry_i);
                int_t tmp_pos = *(blk_pos + shrd_mmry_j);
                if (tmp_pos < int_max && (tmp_pos < loc_pos || loc_pos == int_max)) {
                    *(blk_pos + shrd_mmry_i) = tmp_pos;
                }
            }
        }
        // begin block reduction
        if (threadIdx.x == 0) {
            //printf("block:\tthid_i: %u -> %u\n", (uint16_t) thid_i, (int16_t) *blk_pos);
            // min
            if (*blk_pos < int_max) {
                if (*blk_pos != ref_pos) {
                    *in_block = true;
                    *dev_pos = *blk_pos;
                } else {
                    atomicMin(dev_pos, (int_t) *blk_pos);
                }
                break;
            }
        }
    }
}
/*
*/


#endif  // _KERNELS_