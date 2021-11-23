#include <assert.h>
#include <math.h>

#include "helpers.h"
#include "globals.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

#ifndef _SINDEXQUERY_
#define _SINDEXQUERY_


inline fp_t predict256_single(
        group_t* group, ky_t* key) {

    // copy key values into double array
    fp_t key_vals[group->n];
    for (ky_size_t feat_i = 0; feat_i < group->n; ++feat_i) {
        *(key_vals + feat_i) =
            *(((ch_t*) key) + *(group->feat_indices + feat_i));
    }

    fp_t prediction = 0;

    // check if registers are necessary
    if (group->n < 4) {
        for (ky_size_t feat_i = 0; feat_i < group->n; ++feat_i) {
            prediction +=
                *(key_vals + feat_i) *
                *(group->weights + feat_i);
        }
    } else {
        __m256d K;
        __m256d W;
        __m256d S = _mm256_setzero_pd();

        // n divided by 4
        for (ky_size_t vector_i = 0; vector_i < (group->n >> 2); ++vector_i) {
            // copy to double array
            // load registers
            K = _mm256_loadu_pd(key_vals + 4 * vector_i);
            W = _mm256_loadu_pd(group->weights +  4 * vector_i);
            // fused multiply add
            S = _mm256_fmadd_pd(K, W, S);
        }
        S = _mm256_hadd_pd(S, S);
        prediction += *((fp_t*) &S) + *(((fp_t*) &S) + 2);


        //#pragma omp parallel for num_threads(3) reduction(+:prediction)
        for (size_t feat_i = (group->n & (~3)); feat_i < group->n; ++feat_i) {
            prediction +=
                *(key_vals + feat_i) *
                *(group->weights  + feat_i);
        }
    }
    // add y-shift
    prediction += *(group->weights + group->n);
    return prediction;
}


inline uint32_t binary_single(
        ky_t* key, ky_t* keys,
        uint32_t query_start, uint32_t query_end) {

    uint32_t pivot = query_start;

    while (query_start < query_end - 1) {

        pivot = (query_start + query_end) / 2;
        
        auto cmp = memcmp(keys + pivot, key, sizeof(ky_t));

        if (cmp == 0) {
            return pivot;
        } else if (cmp < 0) {
            query_start = pivot;
        } else {
            query_end = pivot;
        }
    }
    return query_start;
    
}

inline uint32_t query_group_single(
        group_t* group, ky_t* key, ky_t* keys, bool use_err) {
    
    // use group linear model to predict position
    fp_t prediction = predict256_single(group, key);

    int64_t first;
    int64_t last;

    // use group error to decrease margin
    if (use_err) {
        first = floor(prediction - group->left_err)  - 1;
        last  = ceil (prediction - group->right_err) + 1;
        if (first > group->start + group->m - 1) {
            return group->start + group->m - 1;
        }
        if (last < group->start) {
            return group->start;
        }
        first = (first < group->start) ? group->start : first;
        last  = (last > group->start + group->m - 1) ? group->start + group->m - 1 : last;
    } else {
        first = group->start;
        last  = group->start + group->m - 1;
    }

    if (first == last)
        return first;

    prediction = round(prediction);
    prediction = (prediction < first) ? first : prediction;
    prediction = (prediction > last)  ? last  : prediction;

    auto cmp = memcmp(keys + (uint32_t) prediction, key, sizeof(ky_t));

    uint32_t pos;
    int64_t boundary = prediction;
    uint32_t exponent = 0;
    if (cmp == 0) {
        return prediction;
    } else if (cmp < 0) {
        do {
            boundary = prediction + pow(2, exponent);
            boundary = (boundary > last) ? last : boundary;
            ++exponent;
        } while (boundary < last && memcmp(keys + boundary, key, sizeof(ky_t)) < 0);
        pos = binary_single(key, keys, first, boundary + 1);
    } else {
        do {
            boundary = prediction - pow(2, exponent);
            boundary = (boundary < first) ? first : boundary;
            ++exponent;
        } while (boundary > first && memcmp(keys + boundary, key, sizeof(ky_t)) > 0);
        pos = binary_single(key, keys, boundary, last + 1);
    }
    return pos;
}

inline uint32_t query_single(
            index_t* index, ky_t* key, ky_t* keys) {
    
    uint32_t pos;
    pos = binary_single(key, index->root_pivots, 0, index->root_n);

    group_t* root_i = ((group_t*) index->roots) + pos;
    pos = query_group_single(root_i, key, index->group_pivots, true);
    if (!(memcmp(index->group_pivots + pos, key, sizeof(ky_t)) == 0 ||
            pos < index->group_n - 1 &&
            memcmp(index->group_pivots + pos, key, sizeof(ky_t)) < 0 &&
            memcmp(index->group_pivots + pos + 1, key, sizeof(ky_t)) > 0)) {
        pos = query_group_single(root_i, key, index->group_pivots, false);
    }

    group_t* group_i = ((group_t*) index->groups) + pos;
    pos = query_group_single(group_i, key, keys, true);
    if (!(memcmp(keys + pos, key, sizeof(ky_t)) == 0 ||
            pos < index->n - 1 &&
            memcmp(keys + pos, key, sizeof(ky_t)) < 0 &&
            memcmp(keys + pos + 1, key, sizeof(ky_t)) > 0)) {
        pos = query_group_single(group_i, key, keys, false);
    }

    return pos;
}

#endif  // _SINDEXQUERY_
