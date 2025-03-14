#pragma once

#include "common.h"

void sh_fwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs, // [N, 3]
    const float *__restrict__ coeffs,     // [N, K, 3]
    const bool *__restrict__ masks,   // [N]
    float *__restrict__ colors            // [N, 3]
);

void sh_bwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs, // [N, 3]
    const float *__restrict__ coeffs,     // [N, K, 3]
    const bool *__restrict__ masks,   // [N]
    const float *__restrict__ v_colors,   // [N, 3
    float *__restrict__ v_coeffs,         // [N, K, 3]
    float *__restrict__ v_dirs            // [N, 3] optional
);
