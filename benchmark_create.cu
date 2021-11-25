
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


#include "helpers.h"
#include <chrono>








int main(int argc, char* argv[]) {  
    

    fp_t et = std::stod(getCmdOption(argv, argv + argc, "-e"));
    uint32_t pt = std::stoul(getCmdOption(argv, argv + argc, "-p"));
    uint32_t batchlen = std::stoul(getCmdOption(argv, argv + argc, "-b"));
    uint32_t fstep = std::stod(getCmdOption(argv, argv + argc, "-fs"));
    uint32_t bstep = std::stod(getCmdOption(argv, argv + argc, "-bs"));
    uint32_t minsize = std::stoul(getCmdOption(argv, argv + argc, "-ms"));

    if (cmdOptionExists(argv, argv + argc, "-v")) {
        verbose = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-d")) {
        debug = true;
    }
    if (cmdOptionExists(argv, argv + argc, "-csv")) {
        csv = true;
    }


    // cuda debug
    cudaError_t cudaStat1 = cudaSuccess;
    
    std::string dataset = "./data/";
    dataset.append(getCmdOption(argv, argv + argc, "-f"));
    dataset.append(".txt");
    uint32_t step = std::stoul(getCmdOption(argv, argv + argc, "-s"));
    ky_t* keys;
    cudaMallocHost(&keys, 800'000'000 / step * sizeof(ky_t));
    int64_t numkeys;

    read_keys(dataset, keys, numkeys, step);
    
    // parameters
    
    
    assert(pt <= KEYLEN);
    assert(fstep <= numkeys);
    
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    index_t* index = create_index(
        keys, numkeys, et, pt,
        fstep, bstep, minsize, batchlen
    );
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    
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
    serialize(index, filename.c_str());

    if (csv) {
        filename = "./csv/grouping/";
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

        uint32_t mins = 0;
        for (uint32_t group_i = 0; group_i < index->group_n; ++group_i){
            group_t* group = index->groups + group_i;
            if (group->m == minsize) {
                ++mins;
            }
        }
        fprintf(file,"#mins:%u\n", mins);
        fprintf(file,"#group_n:%u\n", index->group_n);
        fprintf(file,"#root_n:%u\n", index->root_n);
        fprintf(file,"#time:%u\n", (end - begin).count());
        fprintf(file,"type,start,m,n,avg_err,left_err,right_err,fsteps,bsteps\n");

        for (uint32_t group_i = 0; group_i < index->root_n + index->group_n; ++group_i) {
            std::string type;
            group_t* group;
            if (group_i < index->root_n) {
                group = index->roots + group_i;
                type = "root";
            } else {
                group = index->groups + group_i - index->root_n;
                type = "group";
            }
            fprintf(file,"%s,%u,%u,%u,%.17g,%.17g,%.17g,%u,%u\n",
                type.c_str(), group->start, group->m, group->n,
                group->avg_err, group->left_err, group->right_err,
                group->fsteps, group->bsteps
            );

        }
        fclose(file);

    }

    
    return 0;
}
