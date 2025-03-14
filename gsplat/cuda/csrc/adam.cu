#include "adam.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void selective_adam_kernel(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (tiles_touched[g_idx]) {
        float Register_param_grad = param_grad[p_idx];
        float Register_exp_avg = exp_avg[p_idx];
        float Register_exp_avg_sq = exp_avg_sq[p_idx];
        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
        float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

        param[p_idx] += step;
        exp_avg[p_idx] = Register_exp_avg;
        exp_avg_sq[p_idx] = Register_exp_avg_sq;
    }
}


void selective_adam_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
) {
    if (n_elements <= 0) {
        return;
    }
    selective_adam_kernel<<<n_blocks_linear(n_elements), N_THREADS, shmem_size, stream>>>(
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        tiles_touched,
        lr,
        b1,
        b2,
        eps,
        N,
        M
    );   
}