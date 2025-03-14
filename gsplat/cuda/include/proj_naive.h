#pragma once

#include "common.h"

void proj_naive_fwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,  // [C, N, 3]
    const float *__restrict__ covars, // [C, N, 3, 3]
    const float *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    float *__restrict__ means2d, // [C, N, 2]
    float *__restrict__ covars2d // [C, N, 2, 2]
);

void proj_naive_bwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,  // [C, N, 3]
    const float *__restrict__ covars, // [C, N, 3, 3]
    const float *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const float *__restrict__ v_means2d,  // [C, N, 2]
    const float *__restrict__ v_covars2d, // [C, N, 2, 2]
    float *__restrict__ v_means,          // [C, N, 3]
    float *__restrict__ v_covars          // [C, N, 3, 3]
);
