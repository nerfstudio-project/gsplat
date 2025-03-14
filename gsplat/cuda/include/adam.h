#pragma once

#include "common.h"

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
);