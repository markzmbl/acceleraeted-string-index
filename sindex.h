#include <assert.h>

#include "helpers.h"
#include "globals.h"
#include <iostream>
#include <algorithm>



#ifndef _SINDEX_
#define _SINDEX_



inline Status calculate_group(
    const ky_t* hst_keys,ky_size_t* hst_pair_lens, ky_t* dev_keys, ky_size_t* pair_lens, group_t* group,
    ix_size_t processed, ix_size_t start_i, ix_size_t m,
    ky_size_t feat_thresh, fp_t error_thresh,
    ix_size_t step,
    cusolverDnHandle_t* cusolverH,
    cusolverDnParams_t* cusolverP,
    cublasHandle_t* cublasH,
    bool force) {
        
    cudaError_t cudaStat = cudaSuccess;

    //cpu
    cudaError_t cuda_stat = cudaSuccess;
    std::vector<GPUVar*> requests;
    int_t host_min_len = int_max;
    int_t host_max_len = 0;
    int_t* host_uneqs;
    ky_size_t n_star = 0;
    ky_size_t n = 0;
    ky_size_t n_tilde = 0;
    ky_size_t* host_feat_indices;
    int host_info = 0;
    uint64_t d_work_size = 0;
    uint64_t h_work_size = 0;
    double* h_work;
    const fp_t one = 1;
    fp_t host_acc_error = 0;
    fp_t host_min_error = float_max;
    fp_t host_max_error = -float_max;
    fp_t avg_error = 0;
    fp_t* model;
        
    //gpu
    GPUInt dev_min_len;
    GPUInt dev_max_len;
    GPUFloat A;
    GPUFloat B;
    GPUInt dev_uneqs;
    GPUChar col_vals;
    GPUInt mutexes;
    GPUKeySize dev_feat_indices;
    GPUInt mutex;
    GPUFloat tau;
    GPUInfoInt dev_info;
    GPUFloat d_work;
    GPUFloat dev_acc_error;
    GPUFloat dev_min_error;
    GPUFloat dev_max_error;

    // sanity check variables
    fp_t* hst_A;
    fp_t* hst_B;
   

    // allocation
    requests.push_back(&dev_min_len);
    requests.push_back(&dev_max_len);
    if (allocate_gpu_memory(requests) == false) {
        return out_of_memory;
    }
    requests.clear();

    // reset min and max to neutral values

    assert(cudaMemset(dev_min_len.ptr(), int_max, dev_min_len.size()) == cudaSuccess);
    assert(cudaMemset(dev_max_len.ptr(), 0,       dev_max_len.size()) == cudaSuccess);

    // --- range query
    rmq_kernel
        <<<get_block_num(m), BLOCKSIZE, BLOCKSIZE / 32 * 2 * sizeof(ky_size_t)>>>
        (pair_lens, start_i, m - 1, dev_min_len.ptr(), dev_max_len.ptr());
    assert(cudaPeekAtLastError() == cudaSuccess);


    // copy results to cpu
    assert(cudaMemcpy(&host_min_len, dev_min_len.ptr(), dev_min_len.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_max_len, dev_max_len.ptr(), dev_max_len.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    // rmq sanity check
    if (sanity_check) {
        ky_size_t san_min_len = *std::min_element(hst_pair_lens + start_i, hst_pair_lens + start_i + m - 1);
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
        ky_size_t san_max_len = *std::max_element(hst_pair_lens + start_i, hst_pair_lens + start_i + m - 1);
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

    // free lens
    dev_min_len.free();
    dev_max_len.free();

    // feature length without pruning
    n_star = host_max_len - host_min_len + 1;

    // allocation
    dev_uneqs.count = n_star;
    col_vals.count  = n_star;
    mutexes.count   = n_star;
    requests.push_back(&dev_uneqs);
    requests.push_back(&col_vals);
    requests.push_back(&mutexes);
    if (allocate_gpu_memory(requests) == false) {
        return out_of_memory;
    }
    requests.clear();

    // cpu allocation
    assert(cudaMallocHost(&host_uneqs, dev_uneqs.size()) == cudaSuccess);
    
    // -- determine if columns are unequal
    assert(cudaMemset(dev_uneqs.ptr(), 0, dev_uneqs.size()) == cudaSuccess);
    assert(cudaMemset(col_vals.ptr(),  0, col_vals.size())  == cudaSuccess);
    equal_column_kernel
        <<<get_block_num(m * n_star), BLOCKSIZE, BLOCKSIZE / 32 * (sizeof(ch_t) + sizeof(int_t))>>>
        (dev_keys, start_i, host_min_len, m, n_star, dev_uneqs.ptr(), col_vals.ptr(), mutexes.ptr());
    assert(cudaGetLastError() == cudaSuccess);


    // free unnecessary
    assert(col_vals.free() == true);
    assert(mutexes.free()  == true);

    // copy results back
    assert(cudaMemcpy(host_uneqs, dev_uneqs.ptr(), dev_uneqs.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    // equal column sanity check
    if (sanity_check) {
        for (ky_size_t feat_i = 0; feat_i < n_star; ++feat_i) {
            bool is_uneq = false;
            for (ix_size_t key_i = 0; key_i < m; ++key_i) {
                ky_size_t char_i = host_min_len + feat_i;
                const ky_t* key0 = hst_keys + processed + start_i + key_i;
                const ky_t* key1 = hst_keys + processed + start_i + key_i + 1;
                ch_t char0 = *(((ch_t*) *key0) + char_i);
                ch_t char1 = *(((ch_t*) *key1) + char_i);
                if (char0 != char1) {
                    is_uneq = true;
                    break;
                }
            }
            assert(is_uneq == *(host_uneqs + feat_i));
        }
    }

    // free dev_uneqs
    assert(dev_uneqs.free() == true);

    for (ky_size_t feat_i = 0; feat_i < n_star; ++feat_i) {
        if (host_uneqs[feat_i] > 0) {
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
                start_i,
                m,
                n
            );
        }
        assert(cudaFreeHost(host_uneqs) == cudaSuccess);
        return threshold_exceed;
    }

    // take bias into account
    n_tilde = n + 1;

    // set correct count
    A.count                = m * n_tilde;
    dev_feat_indices.count = n;

    // allocation
    requests.push_back(&A);
    requests.push_back(&dev_feat_indices);
    if (allocate_gpu_memory(requests) == false) {
        return out_of_memory;
    }
    requests.clear();

    // calculate feat indices
    assert(cudaMallocHost(&host_feat_indices, dev_feat_indices.size()) == cudaSuccess);
    ky_size_t useful_col_i = 0;
    for (ky_size_t col_i = 0; col_i < n_star; ++col_i) {
        if (host_uneqs[col_i] == true) {
            host_feat_indices[useful_col_i] = host_min_len + col_i;
            ++useful_col_i;
        }
    }
    assert(cudaFreeHost(host_uneqs) == cudaSuccess);

    assert(cudaMemcpy(dev_feat_indices.ptr(), host_feat_indices, dev_feat_indices.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    // debug
    //print_kernel<<<1, n>>>(dev_feat_indices.ptr(), n);
    //cudaDeviceSynchronize();

    // --- write in column major format
    column_major_kernel
        <<<get_block_num(m * n), BLOCKSIZE>>>
        (dev_keys, A.ptr(), start_i, m, dev_feat_indices.ptr(), n_tilde);
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
        cudaMallocHost(&hst_A, A.size());
        cudaMemcpy(hst_A, A.ptr(), A.size(), cudaMemcpyDeviceToHost);
        for (ix_size_t key_i = 0; key_i < m; ++key_i) {
            const ky_t* key0 = hst_keys + processed + start_i + key_i;
            for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
                fp_t feat1 = *(hst_A + feat_i * m + key_i);
                fp_t feat0;
                if (feat_i < n_tilde - 1) {
                    ky_size_t char_i = *(host_feat_indices + feat_i);
                    feat0 = (fp_t) *(((ch_t*) *key0) + char_i);
                } else {
                    feat0 = bias;
                }
                assert(feat0 == feat1);
            }
        }
        cudaFreeHost(hst_A);
    }



    // set correct count
    B.count   = m;
    tau.count = n_tilde;

    // allocation
    requests.push_back(&B);
    requests.push_back(&tau);
    requests.push_back(&dev_info);
    if (allocate_gpu_memory(requests) == false) {
        return out_of_memory;
    }
    requests.clear();


    // --- init B
    set_postition_kernel
        <<<get_block_num(m), BLOCKSIZE>>>
        (B.ptr(), processed, start_i, m);

    if (sanity_check) {
        cudaMallocHost(&hst_B, B.size());
        cudaMemcpy(hst_B, B.ptr(), B.size(), cudaMemcpyDeviceToHost);
        for (ix_size_t key_i = 0; key_i < m; ++key_i) {
            assert(*(hst_B + key_i) == processed + start_i + key_i);
        }
    }

    // calculate workspace size
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    

    // check memory usage for qr factorization
    uint64_t d_work_size_qr = 0;
    uint64_t h_work_size_qr = 0;
    cusolver_status = cusolverDnXgeqrf_bufferSize(
        *cusolverH, *cusolverP, m, n_tilde,
        cuda_float, A.ptr(), m /*lda*/,
        cuda_float, tau.ptr(),
        cuda_float, &d_work_size_qr, &h_work_size_qr
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // check memory usage for transposed matrix multiplication
    int d_work_size_tm = 0;
    #if SINGLE
    cusolver_status = cusolverDnSormqr_bufferSize(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m, 1 /*nrhs*/, n_tilde,
        A.ptr(), m /*lda*/,
        tau.ptr(), B.ptr(),
        m /*ldb*/, &d_work_size_tm
    );
    #else
    cusolver_status = cusolverDnDormqr_bufferSize(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m, 1 /*nrhs*/, n_tilde,
        A.ptr(), m /*lda*/,
        tau.ptr(), B.ptr(),
        m /*ldb*/, &d_work_size_tm
    );
    #endif
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // allocate workbuffers
    d_work_size = (d_work_size_qr > ((uint64_t) d_work_size_tm)) ? d_work_size_qr : (uint64_t) d_work_size_tm;
    h_work_size = h_work_size_qr;
    cudaMallocHost(&h_work, h_work_size);

    d_work.size_manual = d_work_size;
    requests.push_back(&d_work);
    if (allocate_gpu_memory(requests) == false) {
        assert(cudaFreeHost(host_feat_indices) == cudaSuccess);
        return out_of_memory;
    }
    requests.clear();

       
    // start linear regression  
    
    // **** compute QR factorization ****
    // actual QR factorization
    cusolver_status = cusolverDnXgeqrf(
        *cusolverH, *cusolverP, m, n_tilde,
        cuda_float, A.ptr(), m /*lda*/,
        cuda_float, tau.ptr(), 
        cuda_float, d_work.ptr(), d_work_size,
        h_work, h_work_size, dev_info.ptr()
    );
    
    cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // check result
    assert(cudaMemcpy(&host_info, dev_info.ptr(), dev_info.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (host_info != 0) {
        printf(
            "[ASSERTION]\tAfter QR Factorization\n"
            "\thost_info == %i\n",
            host_info
        );
        exit(1);
    }
    
    // **** compute Q^T*B ****
    #if SINGLE
    cusolver_status= cusolverDnSormqr(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m, 1 /*nrhs*/, n_tilde,
        A.ptr(), m /*lda*/, tau.ptr(), B.ptr(), m /*ldb*/,
        d_work.ptr(), d_work_size, dev_info.ptr()
    );
    #else
    cusolver_status= cusolverDnDormqr(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m, 1 /*nrhs*/, n_tilde,
        A.ptr(), m /*lda*/, tau.ptr(), B.ptr(), m /*ldb*/,
        d_work.ptr(), d_work_size, dev_info.ptr()
    );
    #endif

    // free tau and d_work
    assert(tau.free() == true);
    assert(d_work.free() == true);
    if (h_work_size > 0) {
        assert(cudaFreeHost(h_work) == cudaSuccess);
    }

    assert(cudaMemcpy(&host_info, dev_info.ptr(), dev_info.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (host_info != 0) {
        printf(
            "[ASSERTION]\tAfter Q Multiplication\n"
            "\thost_info == %i\n"
            "\tm:\t%'d\n"
            "\tn\t%'d",
            host_info,
            m,
            n
        );
        exit(1);
    }
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    assert(dev_info.free() == true);

    // **** solve R*x = Q^T*B ****
    #if SINGLE
    cublas_status = cublasStrsm(
        *cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n_tilde, 1 /*nrhs*/, &one, A.ptr(), m /*lda*/, B.ptr(), m /*ldb*/
    );
    #else
    cublas_status = cublasDtrsm(
        *cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n_tilde, 1 /*nrhs*/, &one, A.ptr(), m /*lda*/, B.ptr(), m /*ldb*/
    );
    #endif
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // model sanity check
    if (sanity_check) {
        cudaMemcpy(hst_B, B.ptr(), n_tilde * sizeof(fp_t), cudaMemcpyDeviceToHost);
        for (ix_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
            // nan check

            if ((*(hst_B + feat_i) != *(hst_B + feat_i)) != false) {
                printf(
                    "[SANITY]\tNan Model\n"
                    "\tstart:\t%'d\n"
                    "\tm:\t%'d\n"
                    "\tn:\t%'d\n"
                    "\tmodel:\t",
                    start_i,
                    m,
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

    // free A
    assert(A.free() == true);

    // allocation
    requests.push_back(&dev_acc_error);
    requests.push_back(&dev_min_error);
    requests.push_back(&dev_max_error);
    requests.push_back(&mutex);
    if (allocate_gpu_memory(requests) == false) {
        return out_of_memory;
    }
    requests.clear();

    // get min, max and average error
    assert(cudaMemset(dev_acc_error.ptr(), 0,          dev_acc_error.size()) == cudaSuccess);
    assert(cudaMemcpy(dev_min_error.ptr(), &float_max, dev_min_error.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(dev_max_error.ptr(), &float_min, dev_max_error.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemset(mutex.ptr(),         0,          mutex.size())         == cudaSuccess);

    model_error_kernel
        <<<get_block_num(m), BLOCKSIZE, BLOCKSIZE / 32 * 3 * sizeof(fp_t)>>>
        (dev_keys, processed, start_i, B.ptr(), dev_feat_indices.ptr(), m, n_tilde,
        dev_acc_error.ptr(), dev_min_error.ptr(), dev_max_error.ptr(), mutex.ptr());
    assert(cudaGetLastError() == cudaSuccess);
    cudaDeviceSynchronize();

    assert(mutex.free() == true);
    
    assert(cudaMemcpy(&host_acc_error, dev_acc_error.ptr(), dev_acc_error.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_min_error, dev_min_error.ptr(), dev_min_error.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_max_error, dev_max_error.ptr(), dev_max_error.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    // error sanity check
    if (sanity_check) {
        fp_t san_acc_err = 0;
        fp_t san_min_err = float_max;
        fp_t san_max_err = float_min;

        //debug
        fp_t blk_acc;

        for (ix_size_t key_i = 0; key_i < m; ++key_i) {

            //debug
            if (key_i % BLOCKSIZE == 0) {
                blk_acc = 0;
            }

            fp_t key_err = 0;
            const ky_t* key0 = hst_keys + processed + start_i + key_i;
            for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
                fp_t fac1 = *(hst_B + feat_i);
                if (feat_i < n_tilde - 1) {
                    ky_size_t char_i = *(host_feat_indices + feat_i);
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
        cudaFreeHost(hst_B);
    }

    // free errors
    assert(dev_acc_error.free() == true);
    assert(dev_min_error.free() == true);
    assert(dev_max_error.free() == true);

    // determine average error
    avg_error = host_acc_error / m;

    // check model threshold
    if (abs(avg_error) > error_thresh && !force) {

        if (debug) {
            printf(
                "[DEBUG]\tError Threshold Excess\n"
                "\tstart:\t%'d\n"
                "\tm:\t%'d\n"
                "\terr:\t%.10e\n"
                "\tn:\t%'d\n",
                start_i,
                m,
                avg_error,
                n
            );
        }
        assert(B.free() == true);
        assert(cudaFreeHost(host_feat_indices) == cudaSuccess);
        assert(dev_feat_indices.free() == true); 
        return threshold_exceed;
    }

    // copy final model
    assert(cudaMallocHost(&model, B.size()) == cudaSuccess);
    assert(cudaMemcpy(model, B.ptr(), B.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    // free B
    assert(B.free() == true);
    assert(dev_feat_indices.free() == true); 

    // fill in group
    group->start        = processed + start_i;
    strncpy(group->pivot, *(hst_keys + group->start), sizeof(ky_t));
    group->m            = m;
    group->n            = n;
    group->feat_indices = host_feat_indices;
    group->model        = model;
    group->avg_err      = avg_error;
    group->min_err      = host_min_error;    
    group->max_err      = host_max_error;

    // exit successfully
    return success;

}




inline void grouping(
    const ky_t* keys, ix_size_t num_keys,
    fp_t et, ky_size_t pt,
    ix_size_t fstep, ix_size_t bstep, ix_size_t min_size,
    std::vector<group_t> &groups) {


    // cusolver and cublas handles
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnParams_t cusolverP = nullptr;
    cublasHandle_t cublasH = nullptr;
    cusolverDnCreate(&cusolverH);
    cusolverDnCreateParams(&cusolverP);
    cublasCreate(&cublasH);

    // declare cpu variables
    ix_size_t processed = 0;
    ix_size_t start_i = 0;
    ix_size_t end_i = 0;

    // declare gpu variables

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    ky_t* dev_keys;
    ky_size_t* dev_pair_lens;
    assert(cudaMalloc(&dev_keys, BATCHLEN * KEYSIZE) == cudaSuccess);
    assert(cudaMalloc(&dev_pair_lens, (BATCHLEN - 1) * sizeof(ky_size_t)) == cudaSuccess);

    // sanity check variable
    ky_size_t* hst_pair_lens;

    // while keys are left

    //start_i = 389'078;
    //end_i = 389'479;

    ix_size_t batchlen = (num_keys - processed < BATCHLEN) ? num_keys - processed : BATCHLEN;
    assert(cudaMemcpy(dev_keys, keys + processed, batchlen * KEYSIZE, cudaMemcpyHostToDevice) == cudaSuccess); 
    // calculate common prefix lenghts
    pair_prefix_kernel
        <<<get_block_num(batchlen - 1), BLOCKSIZE>>>(dev_keys, dev_pair_lens, batchlen);
    assert(cudaGetLastError() == cudaSuccess);

    while (processed + end_i + fstep < num_keys) {

        if (processed > 0) {
            batchlen = (num_keys - processed < BATCHLEN) ? num_keys - processed : BATCHLEN;
            assert(cudaMemcpy(dev_keys, keys + processed, batchlen * KEYSIZE, cudaMemcpyHostToDevice) == cudaSuccess);
            // calculate common prefix lenghts
            pair_prefix_kernel
                <<<get_block_num(batchlen - 1), BLOCKSIZE>>>(dev_keys, dev_pair_lens, batchlen);
            assert(cudaGetLastError() == cudaSuccess);
        }

        //print_kernel<<<1, 100>>>(dev_keys, 100);
        //cudaDeviceSynchronize();

        //printf("%'d\n", sizeof(ky_t));
        //printf("%c, %c\n", *(((ch_t*)*keys) + 13), *(((ch_t*)*(keys + 1)) + 13));
        //print_kernel<<<1, 16>>>(dev_keys, 16);
        //cudaDeviceSynchronize();


        
        // pair length sanity check
        if (sanity_check) {
            cudaMallocHost(&hst_pair_lens, (batchlen - 1) * sizeof(ky_size_t));
            cudaMemcpy(hst_pair_lens, dev_pair_lens, (batchlen - 1) * sizeof(ky_size_t), cudaMemcpyDeviceToHost);
            for (ix_size_t key_i = 0; key_i < (batchlen - 1); ++key_i) {
                ky_size_t char_i;
                for (char_i = 0; char_i < KEYSIZE; ++char_i) {
                    if (*(((ch_t*) *(keys + processed + key_i)) + char_i) != *(((ch_t*) *(keys + processed + key_i + 1)) + char_i)) {
                        break;
                    }
                }
                assert(char_i == *(hst_pair_lens + key_i));
                //if (key_i >= 3061350) {
                //    printf("key_1: %'d, pairlen: %'d\n", key_i, char_i);
                //    print_key(keys + key_i);
                //    print_key(keys + key_i + 1);
                //}
            }
            //cudaFreeHost(hst_pair_lens);
        }


        //print_keys(keys, 389'078, 401);
        //group_t group0;
        //auto result0 = calculate_group(keys, hst_pair_lens,
        //    dev_keys, dev_pair_lens, &group0,
        //    0, 2735155, 14, pt, et, 1,
        //    &cusolverH, &cusolverP, &cublasH, false);

        // while batch is sufficent
        while (end_i + fstep < batchlen) {
            
            group_t group;
            Status result = success;

            unsigned int fsteps = 0;
            // increase loop
            while (end_i + fstep < batchlen && result == success) {

                // free feature indices and model values
                // as group is still expanding
                if (fsteps > 0) {
                    assert(cudaFreeHost(group.feat_indices)  == cudaSuccess);
                    assert(cudaFreeHost(group.model)         == cudaSuccess);
                    if (debug) {
                        printf(
                            "[DEBUG]\tGroup Increase\n"
                            "\tstart:\t%'d\n"
                            "\tm:\t%'d\n"
                            "\terr:\t%.10e\n"
                            "\tn:\t%'d\n",
                            start_i,
                            group.m,
                            group.avg_err,
                            group.n
                        );
                    }
                }

                end_i += fstep;

                ix_size_t step = 1;
                
                assert(batchlen > end_i - start_i);

                result = calculate_group(keys, hst_pair_lens,
                    dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et, step,
                    &cusolverH, &cusolverP, &cublasH, false);


                assert(result != out_of_memory);

                ++fsteps;

            }
            unsigned int bsteps = 0;
            while (end_i > bstep && result == threshold_exceed) {

                ix_size_t step = 1;

                end_i -= bstep;

                // too gready -> not working
                //assert(end_i > start_i);
                bool force = false;
                if (end_i - start_i <= min_size) {
                    end_i = start_i + min_size;
                    force = true;
                }
                result = calculate_group(keys, hst_pair_lens,
                    dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et, step,
                    &cusolverH, &cusolverP, &cublasH, force);

                assert(result != out_of_memory);

                ++bsteps;

                if (force) {
                    break;
                }
            }

            if (end_i + fstep < batchlen) {
                group.fsteps = fsteps;
                group.bsteps = bsteps;

                if (verbose) {
                    print_group(groups.size(), group);
                }
                groups.push_back(group);

                start_i = end_i;
            }
        }

        // end of batch
        processed += start_i;
        end_i -= start_i;
        start_i = 0;

        // 

    }
    // last group
    group_t group;
    ix_size_t step = 1;
    Status result = calculate_group(keys, hst_pair_lens,
        dev_keys, dev_pair_lens, &group,
        processed, start_i, num_keys - processed - start_i, pt, et, step,
        &cusolverH, &cusolverP, &cublasH, true);
    group.fsteps = 1;
    group.bsteps = 0;
    if (verbose) {
        print_group(groups.size(), group);
    }
    groups.push_back(group);

    assert(cudaFree(dev_keys) == cudaSuccess);
    assert(cudaFree(dev_pair_lens) == cudaSuccess);

    cusolverDnDestroy(cusolverH);
    cusolverDnDestroyParams(cusolverP);
    cublasDestroy(cublasH);
}

#endif  // _SINDEX_