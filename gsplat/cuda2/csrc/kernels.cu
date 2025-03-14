#include "kernels.cuh"
#include <cooperative_groups.h>


namespace cg = cooperative_groups;

__global__ void func_kernel(const uint32_t N) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
}

void launch_func(uint32_t N, uint32_t N_THREADS, cudaStream_t stream) {
    if (N > 0) {
        int blocks = (N + N_THREADS - 1) / N_THREADS;
        func_kernel<<<blocks, N_THREADS, 0, stream>>>(N);
    }
}
