
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <cuda.h>
#include "kernels.cuh"
#include "helpers.h"
#include "globals.h"
#include "sindex.h"
#include "sindex_query.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cusolverDn.h"
#include <assert.h>
#include <limits>
#include <numeric>

#include "helpers.h"
#include <chrono>








int main(int argc, char* argv[]) {  

    std::string dataset = "./data/";
    dataset.append(getCmdOption(argv, argv + argc, "-f"));
    dataset.append(".txt");
    uint32_t step = std::stoul(getCmdOption(argv, argv + argc, "-s"));
    ky_t* keys;
    cudaMallocHost(&keys, 800'000'000 / step * sizeof(ky_t));
    int64_t numkeys;
    read_keys(dataset, keys, numkeys, step);

    std::string filename = "./grouping/";
    filename.append(getCmdOption(argv, argv + argc, "-f"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-s"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-e"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-fs"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-bs"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-ms"));
    filename.append(".bin");

    
    index_t* index = deserialize(filename.c_str());
    
    //https://stackoverflow.com/questions/238603/how-can-i-get-a-files-size-in-c
    FILE* f = fopen(filename.c_str(), "r");
    fseek(f, 0L, SEEK_END);
    int64_t file_size = ftell(f);
    fclose(f);

    /*
    ***********************
    *** query benchmark ***
    ***********************
    */

    // initalization

    srand(time(nullptr));
    uint64_t run_n = 100000;
    uint64_t runs[run_n];
    uint64_t sample_n = CPUCORES * 100;
    ky_t samples[sample_n];
    
    for (uint64_t run_i = 0; run_i < run_n; ++run_i) {
        // fill samples
        for (uint32_t key_j = 0; key_j < sample_n; ++key_j) {
            uint32_t rnd = rand() % numkeys;
            memcpy(samples + key_j, keys + rnd, sizeof(ky_t));
        }
        std::chrono::steady_clock::time_point ts1 = std::chrono::steady_clock::now();
        #pragma omp parallel for num_threads(CPUCORES)
        for (uint32_t key_j = 0; key_j < sample_n; ++key_j) {
            ky_t* key = samples + key_j;
            query_single(index, key, keys);   
        }
        std::chrono::steady_clock::time_point ts2 = std::chrono::steady_clock::now();
        *(runs + run_i) = (ts2 - ts1).count();
        if (run_i % (run_n / 10) ==0) {
            printf(
                "[QUERY]\t%u\n"
                "\ttime:\t%u\n",
                run_i,
                *(runs + run_i)
            );
        }
    }

    filename = "./csv/query/";
    filename.append(getCmdOption(argv, argv + argc, "-f"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-s"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-e"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-fs"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-bs"));
    filename.append("_");
    filename.append(getCmdOption(argv, argv + argc, "-ms"));
    filename.append(".csv");
    FILE* file = fopen(filename.c_str(), "w");

    fprintf(file,"#cpus:%u\n", CPUCORES);
    fprintf(file,"#sample_n:%u\n", sample_n);
    fp_t avg = 0;
    for (uint64_t run_i = 0; run_i < run_n; ++run_i)
        avg += *(runs + run_i);
    avg /= run_n;
    fprintf(file,"#avg:%f\n", avg);
    fprintf(file,"#size:%u\n",file_size);
    fprintf(file,"run,time,time/sample_n\n");

    for(uint64_t run_i = 0; run_i < run_n; ++run_i) {
        uint64_t time_i = *(runs + run_i);
        fprintf(file,"%u,%u,%.17g\n", run_i, time_i, ((fp_t) time_i) / sample_n);
    }
    
    fclose(file);
    return 0;
}
