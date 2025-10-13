// CUDA timer implementation for clustering benchmarks
#include "include/cuda_timer.h"
#include <iostream>

namespace stream {

// Implementations are kept in header since they're all inline
// This file exists just to ensure proper compilation and linking

void print_cuda_timer_info() {
    std::cout << "CUDA Clustering Microbenchmarking System Ready" << std::endl;
}

} // namespace stream
