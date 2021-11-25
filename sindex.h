#include <assert.h>
#include <math.h>

#include "helpers.h"
#include "globals.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <immintrin.h>



#ifndef _SINDEX_
#define _SINDEX_



inline GroupStatus calculate_group(
    const ky_t* hst_keys,ky_size_t* hst_pair_lens, ky_t* dev_keys, ky_size_t* pair_lens, group_t* group,
    uint32_t processed, uint32_t start_i, uint32_t m, ky_size_t feat_thresh, fp_t error_thresh, uint32_t maxsamples,
    cusolverDnHandle_t* cusolverH, cusolverDnParams_t* cusolverP, cublasHandle_t* cublasH,
    int_t* dev_min_len, int_t* dev_max_len, fp_t* A, fp_t* B, int_t* hst_uneqs, int_t* dev_uneqs,
    ky_size_t* hst_feat_indices, ky_size_t* dev_feat_indices, int_t* mutex, fp_t* tau, int* dev_info,
    void* d_work, void* h_work, uint64_t d_work_size, uint64_t h_work_size, fp_t* dev_acc_error, fp_t* dev_min_error, fp_t* dev_max_error, bool force) {
        
    
    cudaError_t cudaStat = cudaSuccess;

    //cpu
    int_t host_min_len = UINT32_MAX;
    int_t host_max_len = 0;
    ky_size_t n_star = 0;
    ky_size_t n = 0;
    ky_size_t n_tilde = 0;
    int host_info = 0;
    const fp_t one = 1;
    fp_t host_acc_error = 0.0;
    fp_t host_min_error = float_max;
    fp_t host_max_error = -float_max;
    fp_t avg_error = 0;
        
    // set step and m_star to avoid memory exhaustion
    fp_t step;
    uint32_t m_star;
    uint32_t m_1_star;
    if (m > maxsamples) {
        step = ((fp_t) m) / maxsamples;
        m_star = maxsamples;
        m_1_star = ((uint32_t) (fmod(m, step)) == 1) ? m_star - 1 : m_star;
    } else {
        step = 1;
        m_star = m;
        m_1_star = m - 1;
    }

    // sanity check variables
    fp_t* hst_A;
    fp_t* hst_B;
    // reset min and max to neutral values
    assert(cudaMemcpy(dev_min_len, &int_max, sizeof(int_t), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemset(dev_max_len, 0,       sizeof(int_t)) == cudaSuccess);

    // --- range query
    // m_star - 1 because the ith key is compared with the (i+step)th key
    rmq_kernel
        <<<get_block_num(m_1_star), BLOCKSIZE, BLOCKSIZE / 32 * 2 * sizeof(ky_size_t)>>>
        (pair_lens, start_i, m_1_star , dev_min_len, dev_max_len, step);
    cudaStat = cudaGetLastError();
    if(cudaStat != cudaSuccess) {
        printf(
            "[ASSERTION]\tAfter RMQ Kernel\n"
            "\tcudaError:\t%s\n",
            cudaGetErrorString(cudaStat)
        );
        exit(1);
    }

    // copy results to cpu
    assert(cudaMemcpy(&host_min_len, dev_min_len, sizeof(int_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_max_len, dev_max_len, sizeof(int_t), cudaMemcpyDeviceToHost) == cudaSuccess);

    // rmq sanity check
    if (sanity_check) {
        ky_size_t san_min_len = ky_size_max;
        ky_size_t san_max_len = 0;
        for (fp_t key_i = 0; key_i < m_1_star; key_i += step) {
            ky_size_t key_len = *(hst_pair_lens + start_i + ((uint32_t) key_i));

            if (key_len != ky_size_max) {
                if (key_len < san_min_len) {
                    san_min_len = key_len;
                }
                if (key_len > san_max_len) {
                    san_max_len = key_len;
                }
            }
        }
        if (san_min_len != host_min_len) {
            printf(
                "[SANITY]\tMin Len\n"
                "\tstart:\t%'d\n"
                "\tm:\t%'d\n"
                "\tsanity:\t%'d\n"
                "\tgpu:\t%'d\n",
                start_i,
                m,
                san_min_len,
                host_min_len
            );
            exit(1);
        }
        if (san_max_len != host_max_len) {
            printf(
                "[SANITY]\tMax Len\n"
                "\tstart:\t%'d\n"
                "\tm:\t%'d\n"
                "\tsanity:\t%'d\n"
                "\tgpu:\t%'d\n",
                start_i,
                m,
                san_max_len,
                host_max_len
            );
            exit(1);
        }
    }

    // feature length without pruning
    n_star = host_max_len - host_min_len + 1;

    if (host_min_len > host_max_len || m <= n_star) {
        return too_small;
    }
    
    // -- determine if columns are unequal
    assert(cudaMemset(dev_uneqs, 0, n_star * sizeof(int_t)) == cudaSuccess);
    equal_column_kernel
        <<<get_block_num(m_1_star * n_star), BLOCKSIZE, BLOCKSIZE / 32 * (sizeof(ch_t) + sizeof(int_t))>>>
        (dev_keys, start_i, host_min_len, m_1_star, n_star, dev_uneqs, step);
    assert(cudaGetLastError() == cudaSuccess);

    // copy results back
    assert(cudaMemcpy(hst_uneqs, dev_uneqs, n_star * sizeof(int_t), cudaMemcpyDeviceToHost) == cudaSuccess);

    // equal column sanity check
    if (sanity_check) {
        for (ky_size_t feat_i = 0; feat_i < n_star; ++feat_i) {
            bool is_uneq = false;
            for (uint32_t key_i = 0; key_i < m_star; ++key_i) {
                ky_size_t char_i = host_min_len + feat_i;
                const ky_t* key0 = hst_keys + processed + start_i + (uint32_t) (key_i * step);
                const ky_t* key1 = hst_keys + processed + start_i + (uint32_t) ((key_i + 1) * step);
                ch_t char0 = *(((ch_t*) *key0) + char_i);
                ch_t char1 = *(((ch_t*) *key1) + char_i);
                if (char0 != char1) {
                    is_uneq = true;
                    break;
                }
            }
            assert(is_uneq == *(hst_uneqs + feat_i));
        }
    }

    for (ky_size_t feat_i = 0; feat_i < n_star; ++feat_i) {
        if (hst_uneqs[feat_i] > 0) {
            ++n;
        }
    }

    // check feature length threshold
    if (n > feat_thresh && !force) {
        if (debug) {
            printf(
                "[DEBUG]\tFeature Length Excess\n"
                "\tstart:\t%'d\n"
                "\tm:\t%'d\n"
                "\tn:\t%'d\n",
                processed + start_i,
                m,
                n
            );
        }
        return threshold_exceed;
    } else if (force) {
        n = min(feat_thresh, n);
    }

    if (m <= n) {
        return too_small;
    }

    // take bias into account
    n_tilde = n + 1;

    // calculate feat indices
    ky_size_t useful_col_i = 0;
    for (ky_size_t col_i = 0; col_i < n_star && useful_col_i < n; ++col_i) {
        if (hst_uneqs[col_i] == true) {
            hst_feat_indices[useful_col_i] = host_min_len + col_i;
            ++useful_col_i;
        }
    }

    assert(cudaMemcpy(dev_feat_indices, hst_feat_indices, n * sizeof(ky_size_t), cudaMemcpyHostToDevice) == cudaSuccess);

    // --- write in column major format
    column_major_kernel
        <<<get_block_num(m_star * n), BLOCKSIZE>>>
        (dev_keys, A, start_i, m_star, dev_feat_indices, n_tilde, step);
    cudaStat = cudaGetLastError();
    if(cudaStat != cudaSuccess) {
        printf(
            "[ASSERTION]\tAfter Colum Major Kernel\n"
            "\tcudaError:\t%s\n",
            cudaGetErrorString(cudaStat)
        );
        exit(1);
    }
    
    if (sanity_check) {
        cudaStat = cudaMallocHost(&hst_A, m_star * n_tilde * sizeof(fp_t));
        cudaStat = cudaMemcpy(hst_A, A, m_star * n_tilde * sizeof(fp_t), cudaMemcpyDeviceToHost);
        for (uint32_t key_i = 0; key_i < m_star; ++key_i) {
            const ky_t* key0 = hst_keys + processed + start_i + ((uint32_t) (key_i * step));
            for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
                fp_t feat1 = *(hst_A + feat_i * m_star + key_i);
                fp_t feat0;
                if (feat_i < n_tilde - 1) {
                    ky_size_t char_i = *(hst_feat_indices + feat_i);
                    feat0 = (fp_t) *(((ch_t*) *key0) + char_i);
                } else {
                    feat0 = bias;
                }
                assert(feat0 == feat1);
            }
        }
        cudaFreeHost(hst_A);
    }

    // --- init B
    set_postition_kernel
        <<<get_block_num(m_star), BLOCKSIZE>>>
        (B, processed, start_i, m_star, step);
    cudaStat = cudaGetLastError();
    if(cudaStat != cudaSuccess) {
        printf(
            "[ASSERTION]\tAfter Set Position Kernel\n"
            "\tcudaError:\t%s\n",
            cudaGetErrorString(cudaStat)
        );
        exit(1);
    }

    if (sanity_check) {
        cudaMallocHost(&hst_B, m * sizeof(fp_t));
        cudaMemcpy(hst_B, B, m * sizeof(fp_t), cudaMemcpyDeviceToHost);
        for (uint32_t key_i = 0; key_i < m_star; ++key_i) {
            assert(*(hst_B + key_i) == processed + start_i + ((uint32_t) (key_i * step)));
        }
    }

    // calculate workspace size
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
   
    // start linear regression  
    
    // **** compute QR factorization ****
    // actual QR factorization
    cusolver_status = cusolverDnXgeqrf(
        *cusolverH, *cusolverP, m_star, n_tilde,
        CUDA_R_64F, A, m_star /*lda*/,
        CUDA_R_64F, tau, 
        CUDA_R_64F, d_work, d_work_size,
        h_work, h_work_size, dev_info
    );

    //assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // check result
    assert(cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (host_info != 0) {
        printf(
            "[ASSERTION]\tAfter QR Factorization\n"
            "\thost_info == %i\n",
            host_info
        );
        exit(1);
    }
    
    // **** compute Q^T*B ****
    cusolver_status= cusolverDnDormqr(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m_star, 1 /*nrhs*/, n_tilde,
        A, m_star /*lda*/, tau, B, m_star /*ldb*/,
        (fp_t*) d_work, d_work_size, dev_info
    );


    assert(cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (host_info != 0) {
        printf(
            "[ASSERTION]\tAfter Q Multiplication\n"
            "\thost_info == %i\n"
            "\tm:\t%'d\n"
            "\tn\t%'d",
            host_info,
            m_star,
            n
        );
        exit(1);
    }
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // **** solve R*x = Q^T*B ****
    cublas_status = cublasDtrsm(
        *cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n_tilde, 1 /*nrhs*/, &one, A, m_star /*lda*/, B, m_star /*ldb*/
    );
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // weights sanity check
    if (sanity_check) {
        cudaMemcpy(hst_B, B, n_tilde * sizeof(fp_t), cudaMemcpyDeviceToHost);
        for (uint32_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
            // nan check

            if ((*(hst_B + feat_i) != *(hst_B + feat_i)) != false) {
                printf(
                    "[SANITY]\tNan weights\n"
                    "\tstart:\t%'d\n"
                    "\tm:\t%'d\n"
                    "\tn:\t%'d\n"
                    "\tmodel:\t",
                    start_i,
                    m_star,
                    n
                );
                for (ky_size_t feat_j = 0; feat_j < n_tilde; ++feat_j) {
                    printf("%f", *(hst_B + feat_i));
                    if (feat_j <= n_tilde - 1) {
                        printf(", ");
                    }
                }
                printf("\n");
                exit(1);
            }
        }
    }

    // get min, max and average error
    assert(cudaMemset(dev_acc_error, 0,          sizeof(fp_t)) == cudaSuccess);
    assert(cudaMemcpy(dev_min_error, &float_max, sizeof(fp_t), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(dev_max_error, &float_min, sizeof(fp_t), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemset(mutex,         0,          sizeof(int_t))         == cudaSuccess);

    model_error_kernel
        <<<get_block_num(m), BLOCKSIZE, BLOCKSIZE / 32 * 3 * (sizeof(fp_t))>>>
        (dev_keys, processed, start_i, B, dev_feat_indices, m, n_tilde,
        dev_acc_error, dev_min_error, dev_max_error, mutex);
    assert(cudaGetLastError() == cudaSuccess);
 
    assert(cudaMemcpy(&host_acc_error, dev_acc_error, sizeof(fp_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_min_error, dev_min_error, sizeof(fp_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_max_error, dev_max_error, sizeof(fp_t), cudaMemcpyDeviceToHost) == cudaSuccess);

    // error sanity check
    if (sanity_check) {
        fp_t san_acc_err = 0;
        fp_t san_min_err = float_max;
        fp_t san_max_err = float_min;

        //debug
        fp_t blk_acc;

        for (uint32_t key_i = 0; key_i < m; ++key_i) {

            //debug
            if (key_i % BLOCKSIZE == 0) {
                blk_acc = 0;
            }

            fp_t key_err = 0;
            const ky_t* key0 = hst_keys + processed + start_i + key_i;
            for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
                fp_t fac1 = *(hst_B + feat_i);
                if (feat_i < n_tilde - 1) {
                    ky_size_t char_i = *(hst_feat_indices + feat_i);
                    fp_t fac0 = (fp_t) *(((ch_t*) *key0) + char_i);
                    key_err += fac0 * fac1;
                } else {
                    key_err += fac1;
                }
            }
            key_err -= (processed + start_i + key_i);
            san_acc_err += key_err;
            if (key_err < san_min_err) {
                san_min_err = key_err;
            }
            if (key_err > san_max_err) {
                san_max_err = key_err;
            }

            //debug
            blk_acc += key_err;
        }
        //assert(san_acc_err == host_acc_error);
        //assert(san_min_err == host_min_error);
        //assert(san_max_err == host_max_error);
        assert(nearly_equal(san_acc_err, host_acc_error));
        assert(nearly_equal(san_min_err, host_min_error));
        assert(nearly_equal(san_max_err, host_max_error));
        assert(cudaFreeHost(hst_B) == cudaSuccess);
    }

    // determine average error
    avg_error = host_acc_error / m;
    //printf("avg\t%f\n", avg_error);

    // check weights threshold
    if (abs(avg_error) > error_thresh && !force) {

        if (debug) {
            printf(
                "[DEBUG]\tError Threshold Excess\n"
                "\tstart:\t%'d\n"
                "\tm:\t%'d\n"
                "\terr:\t%.10e\n"
                "\tn:\t%'d\n",
                processed + start_i,
                m,
                avg_error,
                n
            );
        }
        return threshold_exceed;
    }

    // fill in group
    group->start        = processed + start_i;
    group->m            = m;
    group->n            = n;
    group->avg_err      = avg_error;
    group->left_err     = host_max_error;    
    group->right_err    = host_min_error;

    // exit successfully
    return threshold_success;
}




inline uint32_t grouping(
    const ky_t* keys, int64_t numkeys,
    fp_t et, ky_size_t pt,
    uint32_t fstep, uint32_t bstep, uint32_t minsize, uint32_t batchlen0, group_t* &groups,
    ky_t* dev_keys, ky_size_t* dev_pair_lens, uint32_t maxsamples,
    cusolverDnHandle_t* cusolverH, cusolverDnParams_t* cusolverP, cublasHandle_t* cublasH,
    int_t* dev_min_len, int_t* dev_max_len, fp_t* A, fp_t* B, int_t* hst_uneqs, int_t* dev_uneqs,
    ky_size_t* hst_feat_indices, ky_size_t* dev_feat_indices, int_t* mutex, fp_t* tau, int* dev_info,
    void* d_work, void* h_work, uint64_t d_work_size, uint64_t h_work_size, fp_t* dev_acc_error, fp_t* dev_min_error, fp_t* dev_max_error) {

    cudaError_t cudaStat = cudaSuccess;

    // sanity check variable
    ky_size_t* hst_pair_lens;
    if (sanity_check) {
        cudaMallocHost(&hst_pair_lens, (batchlen0 - 1) * sizeof(ky_size_t));
    }

    uint32_t processed = 0;
    uint32_t start_i, end_i;
    start_i = end_i = 0;
    std::vector<group_t> group_vector;
    GroupStatus result = threshold_success;



    while (result != finished) {

        int64_t batchlen = (numkeys - processed < batchlen0) ? numkeys - processed : batchlen0;
        assert(cudaMemcpy(dev_keys, keys + processed, batchlen * sizeof(ky_t), cudaMemcpyHostToDevice) == cudaSuccess);
        // calculate common prefix lenghts
        pair_prefix_kernel
            <<<get_block_num(batchlen - 1), BLOCKSIZE>>>(dev_keys, dev_pair_lens, batchlen);
        cudaStat = cudaGetLastError();
        if(cudaStat != cudaSuccess) {
            printf(
                "[ASSERTION]\tAfter Pair Prefix Kernel\n"
                "\tcudaError:\t%s\n",
                cudaGetErrorString(cudaStat)
            );
            exit(1);
        }


        // pair length sanity check
        if (sanity_check) {
            cudaMemcpy(hst_pair_lens, dev_pair_lens, (batchlen - 1) * sizeof(ky_size_t), cudaMemcpyDeviceToHost);
            for (uint32_t key_i = 0; key_i < (batchlen - 1); ++key_i) {
                ky_size_t prefix_len = ky_size_max;
                ky_size_t char_i;
                for (char_i = 0; char_i < KEYSIZE; ++char_i) {
                    if (*(((ch_t*) *(keys + processed + key_i)) + char_i) != *(((ch_t*) *(keys + processed + key_i + 1)) + char_i)) {
                        prefix_len = char_i;
                        break;
                    }
                }
                assert(prefix_len == *(hst_pair_lens + key_i));
            }
        }


        // while batch is sufficent
        while (result != batch_exceed) {
            

            group_t group;

            unsigned int fsteps = 0;
            // increase loop
            while (result == threshold_success) {

                bool force = false;
                // free feature indices and weights values
                // as group is still expanding
                // todo: find a more elegant solution
                if (fsteps > 0) {
                    if (debug) {
                        printf(
                            "[DEBUG]\tGroup Increase\n"
                            "\tstart:\t%'d\n"
                            "\tm:\t%'d\n"
                            "\terr:\t%.10e\n"
                            "\tn:\t%'d\n",
                            processed + start_i,
                            group.m,
                            group.avg_err,
                            group.n
                        );
                    }
                }

                if (start_i == 0 && end_i == batchlen) {
                    break;
                }

                end_i += fstep;

                if (batchlen - end_i < minsize) {
                    if (start_i == 0 || processed + end_i >= numkeys) {
                        end_i = batchlen;
                        if (processed + end_i == numkeys)
                            force = true;
                    } else {
                        result = batch_exceed;
                        break;
                    }
                } 


                result = calculate_group(
                    keys, hst_pair_lens, dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et, maxsamples,
                    cusolverH, cusolverP, cublasH,
                    dev_min_len, dev_max_len, A, B, hst_uneqs, dev_uneqs,
                    hst_feat_indices, dev_feat_indices, mutex, tau, dev_info,
                    d_work, h_work, d_work_size, h_work_size, dev_acc_error, dev_min_error, dev_max_error, force
                );



                assert (result != out_of_memory);

                ++fsteps;

                if (force == true ) {
                    break;
                }                    

            }
            unsigned int bsteps = 0;
            while (result == threshold_exceed) {

                end_i -= bstep;

                // too gready -> not working
                //assert(end_i > start_i);
                bool force = false;
                if (end_i - start_i <= minsize) {
                    end_i = start_i + minsize;
                    force = true;
                }

                result = calculate_group(
                    keys, hst_pair_lens, dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et, maxsamples,
                    cusolverH, cusolverP, cublasH,
                    dev_min_len, dev_max_len, A, B, hst_uneqs, dev_uneqs,
                    hst_feat_indices, dev_feat_indices, mutex, tau, dev_info,
                    d_work, h_work, d_work_size, h_work_size, dev_acc_error, dev_min_error, dev_max_error, force
                );
                while (result == too_small) {
                    ++end_i;
                    result = calculate_group(
                        keys, hst_pair_lens, dev_keys, dev_pair_lens, &group,
                        processed, start_i, end_i - start_i, pt, et, maxsamples,
                        cusolverH, cusolverP, cublasH,
                        dev_min_len, dev_max_len, A, B, hst_uneqs, dev_uneqs,
                        hst_feat_indices, dev_feat_indices, mutex, tau, dev_info,
                        d_work, h_work, d_work_size, h_work_size, dev_acc_error, dev_min_error, dev_max_error, force
                );
                }


                assert(result != out_of_memory);

                ++bsteps;

            }

            if (result == threshold_success) {
                group.fsteps = fsteps;
                group.bsteps = bsteps;
                assert(cudaMallocHost(&(group.feat_indices), group.n * sizeof(ky_size_t)) == cudaSuccess);
                assert(cudaMallocHost(&(group.weights), (group.n + 1) * sizeof(fp_t)) == cudaSuccess);
                memcpy(group.feat_indices, hst_feat_indices, group.n * sizeof(ky_size_t));
                cudaMemcpy(group.weights, B, (group.n + 1) * sizeof(fp_t), cudaMemcpyDeviceToHost);

                if (verbose) {
                    print_group(group_vector.size(), group);
                }
                group_vector.push_back(group);

                start_i = end_i;

                if (processed + end_i == numkeys) {
                    result = finished;
                    break;
                }
            }

        }

        // end of batch
        processed += start_i;
        end_i -= start_i;
        start_i = 0;

        if (result == batch_exceed) {
            result = threshold_success;
        }
    }
    if (sanity_check) {
        cudaFreeHost(hst_pair_lens);
    }

    uint32_t group_n = group_vector.size();
    groups = (group_t*) malloc(group_n * sizeof(group_t));
    mempcpy(groups, group_vector.data(), group_n * sizeof(group_t));
    return group_n;
}

inline index_t* create_index(
    ky_t* keys, int64_t numkeys, fp_t et, ky_size_t pt,
    uint32_t fstep, uint32_t bstep, uint32_t minsize, uint32_t batchlen) {




    // **********************************
    // *** declaration and allocation ***
    // **********************************


    // cusolver and cublas handles
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnParams_t cusolverP = nullptr;
    cublasHandle_t cublasH = nullptr;
    assert(cusolverDnCreate(&cusolverH) == cudaSuccess);
    assert(cusolverDnCreateParams(&cusolverP) == cudaSuccess);
    assert(cublasCreate(&cublasH) == cudaSuccess);
    // declare cpu variables
    int_t* hst_uneqs;
    ky_size_t* hst_feat_indices;
    void* h_work;
    fp_t* weights;
    // declare gpu variables
    ky_t* dev_keys;
    ky_size_t* dev_pair_lens;
    int_t* dev_min_len;
    int_t* dev_max_len;
    fp_t* A;
    fp_t* B;
    int_t* dev_uneqs;
    ky_size_t* dev_feat_indices;
    int_t* mutex;
    fp_t* tau;
    int* dev_info;
    void* d_work;
    fp_t* dev_acc_error;
    fp_t* dev_min_error;
    fp_t* dev_max_error;
    assert(cudaMallocHost(&hst_uneqs, KEYLEN * sizeof(int_t)) == cudaSuccess);
    assert(cudaMallocHost(&hst_feat_indices, pt * sizeof(ky_size_t)) == cudaSuccess);
    assert(cudaMallocHost(&weights, (pt + 1) * sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_keys, batchlen * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_pair_lens, (batchlen - 1) * sizeof(ky_size_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_min_len, sizeof(int_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_max_len, sizeof(int_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_uneqs, KEYLEN * sizeof(int_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_feat_indices, pt * sizeof(ky_size_t)) == cudaSuccess);
    assert(cudaMalloc(&mutex, sizeof(int_t)) == cudaSuccess);
    assert(cudaMalloc(&tau, (pt + 1) * sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_info, sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&dev_acc_error, sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_min_error, sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_max_error, sizeof(fp_t)) == cudaSuccess);    

    // ********************************
    // *** find maximum sample size ***
    // ********************************

    uint64_t d_work_size = 0;
    uint64_t h_work_size = 0;
    size_t free_space = 0;
    size_t total_space = 0;
    cudaMemGetInfo(&free_space, &total_space);
    // debug video output
    free_space -= 200'000'000;
    // A(maxsamples x (feat_threash + 1)) and B(maxsamples x 1)
    uint32_t maxsamples_max = free_space / ((pt + 2) * sizeof(fp_t));
    uint32_t maxsamples_min = maxsamples_max;

    // exponential search for boundary
    do {        
        assert(cudaMalloc(&A, maxsamples_min * (pt + 1) * sizeof(fp_t)) == cudaSuccess);
        assert(cudaMalloc(&B, maxsamples_min * sizeof(fp_t)) == cudaSuccess);
        calculate_cusolver_buffer_size(
            &cusolverH, &cusolverP,
            maxsamples_min, pt, A, tau, B,
            &d_work_size, &h_work_size
        );
        assert(cudaFree(A) == cudaSuccess);
        assert(cudaFree(B) == cudaSuccess);
        maxsamples_min /= 2;
    } while (maxsamples_min * (pt + 2) * sizeof(fp_t) + d_work_size > free_space);
 
    uint32_t maxsamples_mid = (maxsamples_min + maxsamples_max) / 2;

    while (maxsamples_min <= maxsamples_max) {
        assert(cudaMalloc(&A, maxsamples_mid * (pt + 1) * sizeof(fp_t)) == cudaSuccess);
        assert(cudaMalloc(&B, maxsamples_mid * sizeof(fp_t)) == cudaSuccess);
        calculate_cusolver_buffer_size(
            &cusolverH, &cusolverP,
            maxsamples_mid, pt, A, tau, B,
            &d_work_size, &h_work_size
        );
        assert(cudaFree(A) == cudaSuccess);
        assert(cudaFree(B) == cudaSuccess);
        if (maxsamples_mid * (pt + 2) * sizeof(fp_t) + d_work_size > free_space)
            maxsamples_max = maxsamples_mid - 1;
        else
            maxsamples_min = maxsamples_mid + 1;

        maxsamples_mid = (maxsamples_min + maxsamples_max)/2;
    }
    uint32_t maxsamples = maxsamples_mid;

    printf("[MAXSAMPLES]\t%u\n", maxsamples);

    if (maxsamples > batchlen) maxsamples = batchlen;

    assert(cudaMalloc(&A, maxsamples * (pt + 1) * sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&B, maxsamples * sizeof(fp_t)) == cudaSuccess);
    assert(cudaMalloc(&d_work, d_work_size) == cudaSuccess);
    assert(cudaMallocHost(&h_work, h_work_size) == cudaSuccess);

    // ****************
    // *** grouping ***
    // ****************
    // grouping of keys
    group_t* groups;
    uint32_t group_n = grouping(keys, numkeys, et, pt, fstep, bstep, minsize, batchlen, groups,
        dev_keys, dev_pair_lens, maxsamples,
        &cusolverH, &cusolverP, &cublasH,
        dev_min_len, dev_max_len, A, B, hst_uneqs,  dev_uneqs,
        hst_feat_indices, dev_feat_indices, mutex, tau, dev_info,
        d_work, h_work, d_work_size, h_work_size, dev_acc_error, dev_min_error, dev_max_error);
    // gather group pivots
    ky_t* group_pivots = (ky_t*) malloc(group_n * sizeof(ky_t));
    for (uint32_t group_i = 0; group_i < group_n; ++group_i) {
        group_t group = *(groups + group_i);
        memcpy(group_pivots + group_i, keys + group.start, sizeof(ky_t));
    }
    // grouping of roots
    group_t* roots;
    uint32_t root_n = 0;
    if (group_n > 1) {
        root_n = grouping(group_pivots, group_n, et, pt, fstep, bstep, minsize, batchlen, roots,
        dev_keys, dev_pair_lens, maxsamples,
        &cusolverH, &cusolverP, &cublasH,
        dev_min_len, dev_max_len, A, B, hst_uneqs,  dev_uneqs,
        hst_feat_indices, dev_feat_indices, mutex, tau, dev_info,
        d_work, h_work, d_work_size, h_work_size, dev_acc_error, dev_min_error, dev_max_error);
    }
    ky_t* root_pivots = (ky_t*) malloc(root_n * sizeof(ky_t));
    // gater root pivots
    if (root_n > 0) {
        for (uint32_t root_i = 0; root_i < root_n; ++root_i) {
            group_t root = *(roots + root_i);
            memcpy(root_pivots + root_i, group_pivots + root.start, sizeof(ky_t));
        }
    }
    // create index

    index_t* index = (index_t*) malloc(sizeof(index_t));
    index->n = numkeys;
    index->root_n = root_n;
    index->roots = roots;
    index->group_n = group_n;
    index->groups = groups;
    index->root_pivots = root_pivots;
    index->group_pivots = group_pivots;

    assert(cudaFree(dev_keys) == cudaSuccess);
    assert(cudaFree(dev_pair_lens) == cudaSuccess);
    assert(cudaFree(dev_min_len) == cudaSuccess);
    assert(cudaFree(dev_max_len) == cudaSuccess);
    assert(cudaFree(dev_uneqs) == cudaSuccess);
    assert(cudaFree(dev_feat_indices) == cudaSuccess);
    assert(cudaFree(mutex) == cudaSuccess);
    assert(cudaFree(tau) == cudaSuccess);
    assert(cudaFree(dev_info) == cudaSuccess);
    assert(cudaFree(dev_acc_error) == cudaSuccess);
    assert(cudaFree(dev_min_error) == cudaSuccess);
    assert(cudaFree(dev_max_error) == cudaSuccess);
    assert(cudaFree(A) == cudaSuccess);
    assert(cudaFree(B) == cudaSuccess);
    assert(cudaFree(d_work) == cudaSuccess); 
    assert(cusolverDnDestroy(cusolverH) == cudaSuccess);
    assert(cusolverDnDestroyParams(cusolverP) == cudaSuccess);
    assert(cublasDestroy(cublasH) == cudaSuccess);

    return index;
}

#endif  // _SINDEX_
