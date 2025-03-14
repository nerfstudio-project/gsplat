#pragma once

#include "common.h"

void proj_fused_fwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] optional
    const float *__restrict__ quats,    // [N, 4] optional
    const float *__restrict__ scales,   // [N, 3] optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    int32_t *__restrict__ radii,  // [C, N]
    float *__restrict__ means2d,      // [C, N, 2]
    float *__restrict__ depths,       // [C, N]
    float *__restrict__ conics,       // [C, N, 3]
    float *__restrict__ compensations // [C, N] optional
);

void proj_fused_bwd_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] optional
    const float *__restrict__ quats,    // [N, 4] optional
    const float *__restrict__ scales,   // [N, 3] optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    const int32_t *__restrict__ radii,   // [C, N]
    const float *__restrict__ conics,        // [C, N, 3]
    const float *__restrict__ compensations, // [C, N] optional
    const float *__restrict__ v_means2d,       // [C, N, 2]
    const float *__restrict__ v_depths,        // [C, N]
    const float *__restrict__ v_conics,        // [C, N, 3]
    const float *__restrict__ v_compensations, // [C, N] optional
    float *__restrict__ v_means,   // [N, 3]
    float *__restrict__ v_covars,  // [N, 6] optional
    float *__restrict__ v_quats,   // [N, 4] optional
    float *__restrict__ v_scales,  // [N, 3] optional
    float *__restrict__ v_viewmats // [C, 4, 4] optional
);