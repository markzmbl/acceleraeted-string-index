#include <assert.h>
#include <math.h>

#include "helpers.h"
#include "globals.h"
#include <iostream>
#include <algorithm>



#ifndef _SINDEX_
#define _SINDEX_



inline GroupStatus calculate_group(
    const ky_t* hst_keys,ky_size_t* hst_pair_lens, ky_t* dev_keys, ky_size_t* pair_lens, group_t* group,
    ix_size_t processed, ix_size_t start_i, ix_size_t m,
    ky_size_t feat_thresh, fp_t error_thresh,
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
    fp_t* weights;
        
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

    // set step and m_star to avoid memory exhaustion
    fp_t step;
    ix_size_t m_star;
    ix_size_t m_1_star;
    if (m > MAXSAMPLES) {
        step = ((fp_t) m) / MAXSAMPLES;
        m_star = MAXSAMPLES;
        m_1_star = ((ix_size_t) (fmod(m, step)) == 1) ? m_star - 1 : m_star;
    } else {
        step = 1;
        m_star = m;
        m_1_star = m - 1;
    }

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
    // m_star - 1 because the ith key is compared with the (i+step)th key
    rmq_kernel
        <<<get_block_num(m_star), BLOCKSIZE, BLOCKSIZE / 32 * 2 * sizeof(ky_size_t)>>>
        (pair_lens, start_i, m_star , dev_min_len.ptr(), dev_max_len.ptr(), step);
    cudaStat = cudaGetLastError();
    if(cudaStat != cudaSuccess) {
        printf(
            "[ASSERTION]\tAfter Colum Major Kernel\n"
            "\tcudaError:\t%s\n",
            cudaGetErrorString(cudaStat)
        );
        exit(1);
    }

    // copy results to cpu
    assert(cudaMemcpy(&host_min_len, dev_min_len.ptr(), dev_min_len.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&host_max_len, dev_max_len.ptr(), dev_max_len.size(), cudaMemcpyDeviceToHost) == cudaSuccess);

    // rmq sanity check
    if (sanity_check) {
        ky_size_t san_min_len = ky_size_max;
        ky_size_t san_max_len = 0;
        for (fp_t key_i = 0; key_i < m - 1; key_i += step) {
            ky_size_t key_len = *(hst_pair_lens + ((ix_size_t) key_i));
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
        <<<get_block_num(m_star * n_star), BLOCKSIZE, BLOCKSIZE / 32 * (sizeof(ch_t) + sizeof(int_t))>>>
        (dev_keys, start_i, host_min_len, m_star, n_star, dev_uneqs.ptr(), col_vals.ptr(), mutexes.ptr(), step);
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
            for (ix_size_t key_i = 0; key_i < m_star; ++key_i) {
                ky_size_t char_i = host_min_len + feat_i;
                const ky_t* key0 = hst_keys + processed + start_i + (ix_size_t) (key_i * step);
                const ky_t* key1 = hst_keys + processed + start_i + (ix_size_t) ((key_i + 1) * step);
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
    A.count                = m_star * n_tilde;
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
        <<<get_block_num(m_star * n), BLOCKSIZE>>>
        (dev_keys, A.ptr(), start_i, m_star, dev_feat_indices.ptr(), n_tilde, step);
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
        for (ix_size_t key_i = 0; key_i < m_star; ++key_i) {
            const ky_t* key0 = hst_keys + processed + start_i + ((ix_size_t) (key_i * step));
            for (ky_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
                fp_t feat1 = *(hst_A + feat_i * m_star + key_i);
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
    B.count   = m_star;
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
        <<<get_block_num(m_star), BLOCKSIZE>>>
        (B.ptr(), processed, start_i, m_star, step);

    if (sanity_check) {
        cudaMallocHost(&hst_B, B.size());
        cudaMemcpy(hst_B, B.ptr(), B.size(), cudaMemcpyDeviceToHost);
        for (ix_size_t key_i = 0; key_i < m_star; ++key_i) {
            assert(*(hst_B + key_i) == processed + start_i + ((ix_size_t) (key_i * step)));
        }
    }

    // calculate workspace size
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    

    // check memory usage for qr factorization
    uint64_t d_work_size_qr = 0;
    uint64_t h_work_size_qr = 0;
    cusolver_status = cusolverDnXgeqrf_bufferSize(
        *cusolverH, *cusolverP, m_star, n_tilde,
        cuda_float, A.ptr(), m_star /*lda*/,
        cuda_float, tau.ptr(),
        cuda_float, &d_work_size_qr, &h_work_size_qr
    );
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // check memory usage for transposed matrix multiplication
    int d_work_size_tm = 0;
    #if SINGLE
    cusolver_status = cusolverDnSormqr_bufferSize(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m_star, 1 /*nrhs*/, n_tilde,
        A.ptr(), m_star /*lda*/,
        tau.ptr(), B.ptr(),
        m_star /*ldb*/, &d_work_size_tmfalse
    );
    #endif
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // allocate workbuffers
    d_work_size = (d_work_size_qr > ((uint64_t) d_work_size_tm)) ? d_work_size_qr : (uint64_t) d_work_size_tm;
    h_work_size = h_work_size_qr;
    assert(cudaMallocHost(&h_work, h_work_size) == cudaSuccess);

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
        *cusolverH, *cusolverP, m_star, n_tilde,
        cuda_float, A.ptr(), m_star /*lda*/,
        cuda_float, tau.ptr(), 
        cuda_float, d_work.ptr(), d_work_size,
        h_work, h_work_size, dev_info.ptr()
    );
    
    assert(cudaDeviceSynchronize() == cudaSuccess);
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
        m_star, 1 /*nrhs*/, n_tilde,
        A.ptr(), m_star /*lda*/, tau.ptr(), B.ptr(), m_star /*ldb*/,
        d_work.ptr(), d_work_size, dev_info.ptr()
    );
    #else
    cusolver_status= cusolverDnDormqr(
        *cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        m_star, 1 /*nrhs*/, n_tilde,
        A.ptr(), m_star /*lda*/, tau.ptr(), B.ptr(), m_star /*ldb*/,
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
            m_star,
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
        n_tilde, 1 /*nrhs*/, &one, A.ptr(), m_star /*lda*/, B.ptr(), m_star /*ldb*/
    );
    #else
    cublas_status = cublasDtrsm(
        *cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n_tilde, 1 /*nrhs*/, &one, A.ptr(), m_star /*lda*/, B.ptr(), m_star /*ldb*/
    );
    #endif
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // weights sanity check
    if (sanity_check) {
        cudaMemcpy(hst_B, B.ptr(), n_tilde * sizeof(fp_t), cudaMemcpyDeviceToHost);
        for (ix_size_t feat_i = 0; feat_i < n_tilde; ++feat_i) {
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
        <<<get_block_num(m), BLOCKSIZE, BLOCKSIZE / 32 * 3 * (sizeof(fp_t))>>>
        (dev_keys, processed, start_i, B.ptr(), dev_feat_indices.ptr(), m, n_tilde,
        dev_acc_error.ptr(), dev_min_error.ptr(), dev_max_error.ptr(), mutex.ptr());
    assert(cudaGetLastError() == cudaSuccess);

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
        assert(cudaFreeHost(hst_B) == cudaSuccess);
    }

    // free errors
    assert(dev_acc_error.free() == true);
    assert(dev_min_error.free() == true);
    assert(dev_max_error.free() == true);

    // determine average error
    avg_error = host_acc_error / m;

    // check weights threshold
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

    // copy final weights
    assert(cudaMallocHost(&weights, n_tilde * sizeof(fp_t)) == cudaSuccess);
    assert(cudaMemcpy(weights, B.ptr(), n_tilde * sizeof(fp_t), cudaMemcpyDeviceToHost) == cudaSuccess);

    // free B
    assert(B.free() == true);
    assert(dev_feat_indices.free() == true); 

    // fill in group
    group->start        = processed + start_i;
    group->m            = m;
    group->n            = n;
    group->feat_indices = host_feat_indices;
    group->weights      = weights;
    group->avg_err      = avg_error;
    group->left_err     = abs(host_max_error);    
    group->right_err    = abs(host_min_error);

    // exit successfully
    return threshold_success;

}




inline void grouping(
    const ky_t* keys, ix_size_t numkeys,
    fp_t et, ky_size_t pt,
    ix_size_t fstep, ix_size_t bstep, ix_size_t minsize,
    std::vector<group_t> &groups) {

    cudaError_t cudaStat = cudaSuccess;

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

    ky_t* dev_keys;
    ky_size_t* dev_pair_lens;
    assert(cudaMalloc(&dev_keys, BATCHLEN * KEYSIZE) == cudaSuccess);
    assert(cudaMalloc(&dev_pair_lens, (BATCHLEN - 1) * sizeof(ky_size_t)) == cudaSuccess);

    // sanity check variable
    ky_size_t* hst_pair_lens;


    if (sanity_check) {
        cudaMallocHost(&hst_pair_lens, (BATCHLEN - 1) * sizeof(ky_size_t));
    }

    GroupStatus result = threshold_success;
    while (result != finished) {

        ix_size_t batchlen = (numkeys - processed < BATCHLEN) ? numkeys - processed : BATCHLEN;
        assert(cudaMemcpy(dev_keys, keys + processed, batchlen * KEYSIZE, cudaMemcpyHostToDevice) == cudaSuccess);
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
            for (ix_size_t key_i = 0; key_i < (batchlen - 1); ++key_i) {
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

                // free feature indices and weights values
                // as group is still expanding
                // todo: find a more elegant solution
                if (fsteps > 0) {
                    assert(cudaFreeHost(group.feat_indices) == cudaSuccess);
                    assert(cudaFreeHost(group.weights)      == cudaSuccess);
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
                end_i = (end_i > batchlen) ? batchlen : end_i;

                result = calculate_group(keys, hst_pair_lens,
                    dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et,
                    &cusolverH, &cusolverP, &cublasH, false);

                assert (result != out_of_memory);

                ++fsteps;

                if (result == threshold_success && end_i == batchlen) {
                    if (start_i == 0 || processed + end_i == numkeys) {
                        break; 
                    } else if (start_i > 0) {
                        result = batch_exceed;
                    }
                    // else result stays threshold success
                    // since this is the biggest group possible to add
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

                result = calculate_group(keys, hst_pair_lens,
                    dev_keys, dev_pair_lens, &group,
                    processed, start_i, end_i - start_i, pt, et,
                    &cusolverH, &cusolverP, &cublasH, force);

                assert(result != out_of_memory);

                ++bsteps;

                if (force) {
                    break;
                }
            }

            if (result == threshold_success) {
                group.fsteps = fsteps;
                group.bsteps = bsteps;

                if (groups.size() == 199) {
                    printf("\n");
                }

                if (verbose) {
                    print_group(groups.size(), group);
                }
                groups.push_back(group);



                start_i = end_i;
            }

            if (end_i == batchlen) {
                result = batch_exceed;
            }
        }

        // end of batch
        processed += start_i;

        if (processed == numkeys) {
            result = finished;
        } else {
            end_i -= start_i;
            start_i = 0;
        }



    }

    assert(cudaFree(dev_keys) == cudaSuccess);
    assert(cudaFree(dev_pair_lens) == cudaSuccess);
    if (sanity_check) {
        cudaFreeHost(hst_pair_lens);
    }

    assert(cusolverDnDestroy(cusolverH) == cudaSuccess);
    assert(cusolverDnDestroyParams(cusolverP) == cudaSuccess);
    assert(cublasDestroy(cublasH) == cudaSuccess);
}

inline ix_size_t query_range(
        ky_t* dev_key, ky_t &key, ky_t* keys,
        ix_size_t query_start, ix_size_t query_end,
        ix_size_t left, ix_size_t mid, ix_size_t right,
        cudaStream_t* stream_left, cudaStream_t* stream_mid, cudaStream_t* stream_right,
        ky_t* buffer_left, ky_t* buffer_mid, ky_t* buffer_right,
        ix_size_t querysize, int_t* dev_pos, int_t &hst_pos,
        bool* dev_inblk, bool hst_inblk) {

    QueryStatus result;
    do {

        // fill left and write buffer
        if (left < ix_max) {
            querysize = (mid - left < QUERYSIZE) ? mid - left : QUERYSIZE;
            assert(cudaMemcpyAsync(buffer_left,  keys + left,  querysize * sizeof(ky_t), cudaMemcpyHostToDevice, *stream_left) == cudaSuccess);
        }
        if (right < ix_max) {
            querysize = (query_end - right < QUERYSIZE) ? query_end - right : QUERYSIZE;
            assert(cudaMemcpyAsync(buffer_right, keys + right, querysize * sizeof(ky_t), cudaMemcpyHostToDevice, *stream_right) == cudaSuccess);
        }
        assert(cudaMemcpy(dev_pos, &int_max, sizeof(int_t), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemset(dev_inblk, false, sizeof(bool)) == cudaSuccess);
        // run kernel for mid buffer in the mean time
        if (right != ix_max) {
            querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
        } else {
            querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
        }
        query_kernel
            <<<get_block_num(querysize), BLOCKSIZE, BLOCKSIZE / 32 * sizeof(int_t), *stream_mid>>>
            (dev_key, buffer_mid, querysize, dev_pos, dev_inblk);
        assert(cudaGetLastError() == cudaSuccess);
        // get result from mid buffer
        assert(cudaMemcpy(&hst_pos, dev_pos, sizeof(int_t), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(cudaMemcpy(&hst_inblk, dev_inblk, sizeof(bool), cudaMemcpyDeviceToHost) == cudaSuccess);


        // evaluate result
        if (hst_pos != int_max) {
            if (memcmp(&key, keys + mid + hst_pos, sizeof(ky_t)) == 0) {
                result = found_target;
                return mid + hst_pos;
            } else if (hst_inblk == true) {
                if (memcmp(&key, keys + mid + hst_pos, sizeof(ky_t)) < 0) {
                    --hst_pos;
                }
                result = found_target;
                return mid + hst_pos;
            } else {
                result = target_left;
            }
        } else {
            result = target_right;
        }

        if (result != found_target) {
            switch (result) {
                case target_left:
                    swap_buffer_and_stream(&buffer_mid, &stream_mid, &buffer_left, &stream_left);
                    query_end = mid;
                    mid = left;
                    break;
                
                case target_right:
                    swap_buffer_and_stream(&buffer_mid, &stream_mid, &buffer_right, &stream_right);
                    query_start = mid + QUERYSIZE;
                    mid = right;
                    break;
            }

            // kernel indices
            if (query_end - query_start <= QUERYSIZE) {
                if (mid > query_start) {
                    left = query_start;
                } else {
                    left = ix_max;
                }
                right = ix_max;
            } else if (query_end - query_start <= 2 * QUERYSIZE) {
                if (query_start < mid) {
                    left = query_start;
                } else {
                    left = ix_max;
                }
                if (query_start + QUERYSIZE < mid) {
                    right = query_start + QUERYSIZE;
                } else {
                    right = ix_max;
                }
            } else {
                left = (mid + query_start - QUERYSIZE) / 2;
                right = (query_end + mid + QUERYSIZE) / 2;
            }
        }

    } while (result != found_target);
}

inline ix_size_t get_position(
        const index_t* index, ky_t &key, ky_t* keys) {
    
    ky_t* dev_key;
    int_t hst_pos = int_max;
    int_t* dev_pos;
    bool hst_inblk = false;
    bool* dev_inblk;

    assert(cudaMalloc(&dev_key, sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_pos, sizeof(int_t)) == cudaSuccess);
    assert(cudaMalloc(&dev_inblk, sizeof(bool)) == cudaSuccess);
    assert(cudaMemcpy(dev_key, &key, sizeof(ky_t), cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream_left;
    cudaStream_t stream_mid;
    cudaStream_t stream_right;

    assert(cudaStreamCreate(&stream_left) == cudaSuccess);
    assert(cudaStreamCreate(&stream_mid) == cudaSuccess);
    assert(cudaStreamCreate(&stream_right) == cudaSuccess);

    ky_t* buffer_left;
    ky_t* buffer_mid;
    ky_t* buffer_right;

    assert(cudaMalloc(&buffer_left, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&buffer_mid, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);
    assert(cudaMalloc(&buffer_right, QUERYSIZE * sizeof(ky_t)) == cudaSuccess);

    ix_size_t query_start;
    ix_size_t query_end;

    ix_size_t left;
    ix_size_t right;
    ix_size_t mid;

    ix_size_t querysize;
    if (index->root_n > 0) {
        
        query_start = 0;
        query_end = index->root_n;

        // kernel indices
        if (query_end - query_start <= QUERYSIZE) {
            left = ix_max;
            right = ix_max;
            mid = query_start;
        } else if (query_end - query_start <= 2 * QUERYSIZE) {
            left = ix_max;
            mid = query_start;
            right = query_end - QUERYSIZE;
        } else {
            mid = (ix_size_t) (query_end - query_start - QUERYSIZE / 2);
            left = (mid + query_start - QUERYSIZE) / 2;
            right = (query_end + mid + QUERYSIZE) / 2;
        }

        if (right != ix_max) {
            querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
        } else {
            querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
        }
        assert(cudaMemcpyAsync(buffer_mid, ((ky_t*) index->pivots) + mid, querysize * sizeof(ky_t), cudaMemcpyHostToDevice, stream_mid) == cudaSuccess);
        hst_pos = query_range(
            dev_key, key, index->pivots,
            query_start, query_end,
            left, mid, right,
            &stream_left, &stream_mid, &stream_right,
            buffer_left, buffer_mid, buffer_right,
            querysize, dev_pos, hst_pos,
            dev_inblk, hst_inblk
        );


    } else {
        hst_pos = 0;
    }
    group_t* group = index->groups + hst_pos;
    // determine query range
    fp_t prediction = 0;
    for (ky_size_t feat_i = 0; feat_i < group->n + 1; ++feat_i) {
        if (feat_i == group->n) {
            prediction += *(group->weights + feat_i);
        } else {
            ky_size_t char_i = *(group->feat_indices + feat_i);
            ch_t character = *(((ch_t*) key) + char_i);
            fp_t weight = *(group->weights + feat_i);
            prediction += weight * ((fp_t) character);
        }
    }

    // shift query borders
    if ((int64_t) (prediction - group->left_err) < (int64_t) group->start || prediction - group->left_err < 0) {
        query_start = group->start;
    } else if ((int64_t) (prediction - group->left_err) > (int64_t) (group->start + group->m)) {
        return group->start + group->m - 1;
    } else {
        auto tmp = (prediction - group->left_err);
        query_start = (ix_size_t) (prediction - group->left_err);
    }
    if ((int64_t) ceil(prediction + group->right_err + 0.5) < (int64_t) group->start || prediction + group->right_err + 0.5 < 0) {
        return group->start;
    } else if ((int64_t) ceil(prediction + group->right_err + 0.5) > (int64_t) (group->start + group->m)) {
        query_end = group->start + group->m;
    } else {
        auto tmp = ceil(prediction + group->right_err + 0.5);
        query_end = ceil(prediction + group->right_err + 0.5);
    }

    if (prediction < group->start) {
        prediction = group->start;
    } else if (prediction >= group->start + group->m) {
        prediction = group->start + group->m - 1;
    }


    // kernel indices
    if (query_end - query_start <= QUERYSIZE) {
        left = ix_max;
        right = ix_max;
        mid = query_start;
    } else if (query_end - query_start <= 2 * QUERYSIZE) {
        if (prediction < query_start + QUERYSIZE) {
            left = ix_max;
            mid = query_start;
            right = query_start + QUERYSIZE;
        } else {
            left = query_start;
            mid = query_end - QUERYSIZE;
            right = ix_max;
        }
    } else {
        if (prediction - query_start < 0.5 * QUERYSIZE) {
            left = ix_max;
            mid = query_start;
            right = (query_end + mid + QUERYSIZE) / 2;
        } else if (query_end - prediction < 0.5 * QUERYSIZE) {
            right = ix_max;
            mid = query_end - QUERYSIZE - 1;
            left = (mid + query_start - QUERYSIZE) / 2;
        } else {
            mid = (ix_size_t) (prediction - QUERYSIZE / 2);
            querysize = (mid - query_start < QUERYSIZE) ? mid - query_start : QUERYSIZE;
            left = (mid + query_start - querysize) / 2;
            querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
            right = (query_end + mid + querysize) / 2;
        }
    }


    QueryStatus result;
    if (right != ix_max) {
        querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
    } else {
        querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
    }
    assert(cudaMemcpyAsync(buffer_mid, keys + mid, querysize * sizeof(ky_t), cudaMemcpyHostToDevice, stream_mid) == cudaSuccess);

    hst_pos = query_range(
        dev_key, key, keys,
        query_start, query_end,
        left, mid, right,
        &stream_left, &stream_mid, &stream_right,
        buffer_left, buffer_mid, buffer_right,
        querysize, dev_pos, hst_pos,
        dev_inblk, hst_inblk
    );

    assert(cudaFree(dev_key) == cudaSuccess);
    assert(cudaFree(dev_pos) == cudaSuccess);
    assert(cudaFree(dev_inblk) == cudaSuccess);

    assert(cudaStreamDestroy(stream_left) == cudaSuccess);
    assert(cudaStreamDestroy(stream_mid) == cudaSuccess);
    assert(cudaStreamDestroy(stream_right) == cudaSuccess);

    assert(cudaFree(buffer_left) == cudaSuccess);
    assert(cudaFree(buffer_mid) == cudaSuccess);
    assert(cudaFree(buffer_right) == cudaSuccess);

    return hst_pos;

}
#endif  // _SINDEX_