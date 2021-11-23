#include <assert.h>
#include <math.h>

#include "helpers.h"
#include "globals.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

#ifndef _SINDEXQUERYOLD_
#define _SINDEXQUERYOLD_




inline uint32_t query_range(
        ky_t* dev_key, ky_t &key, ky_t* keys,
        uint32_t query_start, uint32_t query_end,
        uint32_t left, uint32_t mid, uint32_t right,
        ky_t* query_buffer, uint32_t querysize,
        int_t* dev_pos, int_t &hst_pos) {

    QueryStatus result;
    do {

        assert(cudaMemcpy(query_buffer, keys + mid, querysize * sizeof(ky_t), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(dev_pos, &int_max, sizeof(int_t), cudaMemcpyHostToDevice) == cudaSuccess);
        // run kernel for mid buffer in the mean time
        if (right != UINT32_MAX) {
            querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
        } else {
            querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
        }
        query_kernel
            <<<get_block_num(querysize), BLOCKSIZE, BLOCKSIZE / 32 * sizeof(int_t)>>>
            (dev_key, query_buffer, querysize, dev_pos);
        assert(cudaGetLastError() == cudaSuccess);
        // get result from mid buffer
        assert(cudaMemcpy(&hst_pos, dev_pos, sizeof(int_t), cudaMemcpyDeviceToHost) == cudaSuccess);


        // evaluate result
        if (hst_pos != UINT32_MAX) {
            if (memcmp(&key, keys + mid + hst_pos, sizeof(ky_t)) == 0) {
                result = found_target;
                return mid + hst_pos;
            } else if (hst_pos > 0) {
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
                    query_end = mid;
                    mid = left;
                    break;
                
                case target_right:
                    query_start = mid + QUERYSIZE;
                    mid = right;
                    break;
            }

            if (query_end - query_start <= QUERYSIZE) {
                if (mid > query_start) {
                    left = query_start;
                } else {
                    left = UINT32_MAX;
                }
                right = UINT32_MAX;
            } else if (query_end - query_start <= 2 * QUERYSIZE) {
                if (query_start < mid) {
                    left = query_start;
                } else {
                    left = UINT32_MAX;
                }
                if (query_start + QUERYSIZE < mid) {
                    right = query_start + QUERYSIZE;
                } else {
                    right = UINT32_MAX;
                }
            } else {
                left = (mid + query_start - QUERYSIZE) / 2;
                right = (query_end + mid + QUERYSIZE) / 2;
            }
        }

    } while (result != found_target);
}

inline uint32_t get_position_from_group(
        const group_t* group, ky_t &key, ky_t* keys,
        ky_t* dev_key, int_t* dev_pos, ky_t* query_buffer) {


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

    uint32_t query_start;
    uint32_t query_end;

    uint32_t left;
    uint32_t right;
    uint32_t mid;

    uint32_t querysize;

    // shift query borders
    if ((int64_t) (prediction - group->left_err) - 1 < (int64_t) group->start || prediction - group->left_err - 1 < 0) {
        query_start = group->start;
    } else if ((int64_t) (prediction - group->left_err) - 1 > (int64_t) (group->start + group->m)) {
        return group->start + group->m - 1;
    } else {
        query_start = (uint32_t) (prediction - group->left_err) - 1;
    }
    if ((int64_t) ceil(prediction - group->right_err) + 1 < (int64_t) group->start || ceil(prediction - group->right_err) + 1 < 0) {
        return group->start;
    } else if ((int64_t) ceil(prediction - group->right_err) + 1 > (int64_t) (group->start + group->m)) {
        query_end = group->start + group->m;
    } else {
        query_end = ceil(prediction - group->right_err) + 1;
    }

    int_t hst_pos;

    if (query_start == query_end - 1) {
        hst_pos = query_start;
    } else {


        if (prediction < group->start) {
            prediction = group->start;
        } else if (prediction >= group->start + group->m) {
            prediction = group->start + group->m - 1;
        }


        // kernel indices
        if (query_end - query_start <= QUERYSIZE) {
            left = UINT32_MAX;
            right = UINT32_MAX;
            mid = query_start;
        } else if (query_end - query_start <= 2 * QUERYSIZE) {
            if (prediction < query_start + QUERYSIZE) {
                left = UINT32_MAX;
                mid = query_start;
                right = query_start + QUERYSIZE;
            } else {
                left = query_start;
                mid = query_end - QUERYSIZE;
                right = UINT32_MAX;
            }
        } else {
            if (prediction - query_start < 0.5 * QUERYSIZE) {
                left = UINT32_MAX;
                mid = query_start;
                right = (query_end + mid + QUERYSIZE) / 2;
            } else if (query_end - prediction < 0.5 * QUERYSIZE) {
                right = UINT32_MAX;
                mid = query_end - QUERYSIZE - 1;
                left = (mid + query_start - QUERYSIZE) / 2;
            } else {
                mid = (uint32_t) (prediction - QUERYSIZE / 2);
                querysize = (mid - query_start < QUERYSIZE) ? mid - query_start : QUERYSIZE;
                left = (mid + query_start - querysize) / 2;
                querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
                right = (query_end + mid + querysize) / 2;
            }
        }


        if (right != UINT32_MAX) {
            querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
        } else {
            querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
        }

        hst_pos = query_range(
            dev_key, key, keys,
            query_start, query_end,
            left, mid, right, query_buffer,
            querysize, dev_pos, hst_pos
        );
    }

    return hst_pos;
}


inline uint32_t get_position_from_index(
        const index_t* index, ky_t &key, ky_t* keys,
        ky_t* dev_key, int_t* dev_pos, ky_t* query_buffer) {
    
    int_t hst_pos = UINT32_MAX;

    assert(cudaMemcpy(dev_key, &key, sizeof(ky_t), cudaMemcpyHostToDevice) == cudaSuccess);

    uint32_t query_start;
    uint32_t query_end;

    uint32_t left;
    uint32_t right;
    uint32_t mid;

    uint32_t querysize;
    if (index->root_n > 0) {
        
        query_start = 0;
        query_end = index->root_n;

        if (query_start == query_end - 1) {
            hst_pos = query_start;
        } else {

            // kernel indices
            if (query_end - query_start <= QUERYSIZE) {
                left = UINT32_MAX;
                right = UINT32_MAX;
                mid = query_start;
            } else if (query_end - query_start <= 2 * QUERYSIZE) {
                left = UINT32_MAX;
                mid = query_start;
                right = query_end - QUERYSIZE;
            } else {
                mid = (uint32_t) (query_end - query_start - QUERYSIZE / 2);
                left = (mid + query_start - QUERYSIZE) / 2;
                right = (query_end + mid + QUERYSIZE) / 2;
            }

            if (right != UINT32_MAX) {
                querysize = (right - mid < QUERYSIZE) ? right - mid : QUERYSIZE;
            } else {
                querysize = (query_end - mid < QUERYSIZE) ? query_end - mid : QUERYSIZE;
            }
            hst_pos = query_range(
                dev_key, key, index->root_pivots,
                query_start, query_end,
                left, mid, right, query_buffer,
                querysize, dev_pos, hst_pos
            );
        }
    } else {
        hst_pos = 0;
    }


    group_t* root = index->roots + hst_pos;

    hst_pos = get_position_from_group(
        root, key, index->group_pivots, dev_key, dev_pos, query_buffer  
    );

    group_t* group = index->groups + hst_pos;

    hst_pos = get_position_from_group(
        group, key, keys, dev_key, dev_pos,query_buffer
    );

    return hst_pos;

}
/////////////////////////////
/////////////////////////////
inline uint32_t search(ky_t* key, ky_t* keys,
        uint32_t query_start, uint32_t query_end) {
    uint32_t pos = query_start;

    assert(query_start <= query_end);
    assert(query_end - query_start <= CPUCORES);

    //#pragma omp parallel for num_threads(query_end - query_start) reduction(max:pos)
    for (uint32_t thread_i = query_start; thread_i < query_end; ++thread_i) {
        if (memcmp(keys + thread_i, key, sizeof(ky_t)) <= 0) {
            pos = std::max(pos, thread_i);
        }
    }
    return pos;
}

inline uint32_t exponential(
        ky_t *key, ky_t* keys,
        uint32_t query_start, uint32_t query_end) {

    uint32_t pos;
    uint32_t exponent = 0;
    uint32_t boundary_exponent = 0;
    uint32_t boundary = query_start;

    bool finished = false;

    while (!finished) {

        //#pragma omp parallel for num_threads(CPUCORES) reduction(max:boundary_exponent)
        for (uint32_t thread_i = 0; thread_i < CPUCORES; ++thread_i) {
            int64_t index;
            int64_t cmp;
            if (query_start < query_end) {
                index = query_start + pow(2, exponent + thread_i);
                if (index < query_end) {
                    cmp = memcmp(keys + index, key, sizeof(ky_t));
                } else {
                    cmp = 1;
                }
            } else {
                index = query_start - pow(2, exponent + thread_i);
                if (index > query_end) {
                    cmp = memcmp(key, keys + index, sizeof(ky_t));
                } else {
                    cmp = 1;
                }
            }

            if (cmp <= 0) {
                boundary_exponent = std::max(exponent + thread_i, boundary_exponent);
            } else {
                finished = true;
            }
        }
        exponent += CPUCORES;
    }

    uint32_t power = pow(2, boundary_exponent + 1);
    if (query_start > query_end) {
        boundary -= power;
        if (boundary < query_end) {
            boundary = query_end;
        }
    } else {
        boundary += power;
        if (boundary > query_end) {
            boundary = query_end;
        }
    }

    return boundary;
}

inline uint32_t binary(
        ky_t* key, ky_t* keys,
        uint32_t query_start, uint32_t query_end) {
    
    assert(query_start <= query_end);

    uint32_t pos = query_start;

    while(query_start + CPUCORES < query_end) {

        uint32_t interval_len = safe_division(query_end - query_start, CPUCORES);

        #pragma omp parallel for num_threads(CPUCORES) reduction(max:pos)
        for (uint32_t thread_i = 0; thread_i < CPUCORES; ++thread_i) {
            uint32_t index = query_start + thread_i * interval_len;
            if (memcmp(keys + index, key, sizeof(ky_t)) <= 0) {
                pos = std::max(pos, index);
            }
        }
        query_start = pos;
        query_end = query_start + interval_len;
    }
    pos = search(key, keys, query_start, query_end);

    return pos;
}

inline fp_t predict256(
        ky_t* key, ky_size_t* feat_indices, fp_t* weights, ky_size_t n) {

    // debug
    fp_t prediction2 = 0;
    for (ky_size_t feat_i = 0; feat_i < n + 1; ++feat_i) {
        if (feat_i == n) {
            prediction2 += *(weights + feat_i);
        } else {
            ky_size_t char_i = *(feat_indices + feat_i);
            ch_t character = *(((ch_t*) key) + char_i);
            fp_t weight = *(weights + feat_i);
            prediction2 += weight * ((fp_t) character);
        }
    }


    // copy key values into double array
    fp_t key_vals[n];
    //#pragma omp parallel for num_threads(CPUCORES)
    for (ky_size_t feat_i = 0; feat_i < n; ++feat_i) {
        *(key_vals + feat_i) =
            *(((ch_t*) key) + *(feat_indices + feat_i));
    }

    fp_t prediction = 0;

    // check if registers are necessary
    if (n < 4) {
        //#pragma omp parallel for num_threads(3) reduction(+:prediction)
        for (ky_size_t feat_i = 0; feat_i < n; ++feat_i) {
            prediction +=
                *(key_vals + feat_i) *
                *(weights + feat_i);
        }
    } else {
        __m256d K[CPUCORES];
        __m256d W[CPUCORES];
        __m256d S = _mm256_setzero_pd();

        // n divided by 4
        //#pragma omp parallel for num_threads(CPUCORES)
        for (ky_size_t vector_i = 0; vector_i < (n >> 2); ++vector_i) {
            // copy to double array
            // load registers
            *(K + vector_i) = _mm256_loadu_pd(key_vals + 4 * vector_i);
            *(W + vector_i) = _mm256_loadu_pd(weights +  4 * vector_i);
            // fused multiply add
            S = _mm256_fmadd_pd(*(K + vector_i), *(W + vector_i), S);
        }
        S = _mm256_hadd_pd(S, S);
        prediction += *((fp_t*) &S) + *(((fp_t*) &S) + 2);


        //#pragma omp parallel for num_threads(3) reduction(+:prediction)
        for (size_t feat_i = (n & (~3)); feat_i < n; ++feat_i) {
            prediction +=
                *(key_vals + feat_i) *
                *(weights  + feat_i);
        }
    }
    // add y-shift
    prediction += *(weights + n);
    return prediction;

}

inline uint32_t query_group(ky_t* key, group_t* group, ky_t* keys) {

    uint32_t pos;

    // calculate prediction
    fp_t prediction = predict256(key, group->feat_indices, group->weights, group->n);
    // set boundaries by error

    // boundaries are last possible indices !

//    int64_t left_boundary  = floor(prediction - group->left_err)  - 1;
//    int64_t right_boundary = ceil (prediction - group->right_err) + 2;
//    // shift boundaries into group
//    if (left_boundary < group->start) {
//        left_boundary = group->start;
//    } else if (left_boundary > group->start + group->m - 1) {
//        left_boundary = group->start + group->m - 1;
//    }
//    if (right_boundary < group->start) {
//        right_boundary = group->start;
//    } else if (right_boundary > group->start + group->m - 1) {
//        right_boundary = group->start + group->m - 1;
//    }
//
//    // get result if boundaries are small
//    if (right_boundary - left_boundary < CPUCORES) {
//        return search(key, keys, left_boundary, right_boundary + 1);
//    }

    uint32_t left_boundary = group->start;
    uint32_t right_boundary = group->start + group->m - 1;

    // query start is first element
    int64_t query_start = round(prediction) - CPUCORES / 2;

    // shift query start
    if (query_start < left_boundary) {
        query_start = left_boundary;
    } else if (query_start > right_boundary + 1) {
        query_start = right_boundary + 1;
    }

    // query end is last element not in query
    int64_t query_end = query_start + CPUCORES;

    // shift query end
    if (query_end < left_boundary) {
        query_end = left_boundary;
    } else if (query_end > right_boundary + 1) {
        query_end = right_boundary + 1;
    }

    // search around prediction
    pos = search(key, keys, query_start, query_end);

    // position found
    if (pos > query_start && pos < query_end - 1 ||
            memcmp(key, keys + pos, sizeof(ky_t)) == 0 ||
            pos == right_boundary ||
            pos < right_boundary &&
            memcmp(key, keys + pos, sizeof(ky_t)) > 0 &&
            memcmp(key, keys + pos + 1, sizeof(ky_t)) < 0) {
        return pos;
    }

    uint32_t boundary;
    // determine range and direction
    if (pos == query_start) {
        // search left
        --query_start;
        boundary = left_boundary;
    } else if (pos == query_start + CPUCORES - 1) {
        // search right
        query_start = query_start + CPUCORES;
        boundary = right_boundary;
    }

    if (abs(boundary - query_start) < CPUCORES) {
        if (query_start < boundary) {
            pos = search(key, keys, query_start, boundary + 1);
        } else {
            pos = search(key, keys, boundary, query_start + 1);
        }
        return pos;
    }

    boundary = exponential(key, keys, query_start, boundary);

    if (query_start < boundary) {
        pos = binary(key, keys, query_start, boundary + 1);
    } else {
        pos = binary(key, keys, boundary, query_start + 1);
    }

    return pos;
}

inline uint32_t get_position_from_index2(const index_t* index, ky_t* key, ky_t* keys) {
    uint32_t pos;
    pos = binary(key, index->root_pivots, 0, index->root_n);

    group_t* root_i = ((group_t*) index->roots) + pos;
    pos = query_group(key, root_i, index->group_pivots);

    group_t* group_i = ((group_t*) index->groups) + pos;
    pos = query_group(key, group_i, keys);

    return pos;
}


#endif  // _SINDEXQUERYOLD_
