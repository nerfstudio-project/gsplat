/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenCV-fisheye backward CUDA kernels for the no_external D1/D2/D3/D5 ops.
// VJPs chain through the fisheye adjoint helpers in fisheye_kernel.cuh and the
// shared pose/quaternion adapters in camera_kernel.cuh. Each kernel reduces the
// intrinsic grads (principal_point, focal_length, forward_poly — NO ab slot)
// via block_sum + one atomicAdd per slot per block. D3/D5 reorder the
// single pose rotation gradient xyzw -> wxyz at the store boundary.

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "fisheye_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kThreads = 256;

dim3 grid_for_count(int64_t count) {
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Value-cast flag unpack: scratch holds the flag VALUE, not a
// bit-reinterpreted float.
__device__ __forceinline__ void fisheye_unpack_flags(
    float packed,
    bool& behind_camera,
    bool& angle_clamped,
    bool& oob,
    bool& xy_norm_clamped) {
    uint32_t f = static_cast<uint32_t>(packed);
    behind_camera = (f & 1u) != 0u;
    angle_clamped = (f & 2u) != 0u;
    oob = (f & 4u) != 0u;
    xy_norm_clamped = (f & 8u) != 0u;
}

// Block-reduces the intrinsic grad slots via block_sum, then atomicAdds the
// result from thread 0. approx_backward_factor has no grad slot.
__device__ __forceinline__ void reduce_fisheye_intrinsic_grads(
    OpenCVFisheyeParamGrads const& local,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly) {
    static_assert(
        kFisheyeForwardPolyTerms == 4,
        "reduce_fisheye_intrinsic_grads unrolls exactly 4 forward_poly slots");
    float pp0 = block_sum<kThreads>(local.pp[0]);
    float pp1 = block_sum<kThreads>(local.pp[1]);
    float f0 = block_sum<kThreads>(local.focal[0]);
    float f1 = block_sum<kThreads>(local.focal[1]);
    float k0 = block_sum<kThreads>(local.forward_poly[0]);
    float k1 = block_sum<kThreads>(local.forward_poly[1]);
    float k2 = block_sum<kThreads>(local.forward_poly[2]);
    float k3 = block_sum<kThreads>(local.forward_poly[3]);

    if (threadIdx.x == 0) {
        if (grad_principal_point != nullptr) {
            atomicAdd(&grad_principal_point[0], pp0);
            atomicAdd(&grad_principal_point[1], pp1);
        }
        if (grad_focal_length != nullptr) {
            atomicAdd(&grad_focal_length[0], f0);
            atomicAdd(&grad_focal_length[1], f1);
        }
        if (grad_forward_poly != nullptr) {
            atomicAdd(&grad_forward_poly[0], k0);
            atomicAdd(&grad_forward_poly[1], k1);
            atomicAdd(&grad_forward_poly[2], k2);
            atomicAdd(&grad_forward_poly[3], k3);
        }
    }
}

// Fused block reduction for all BIVARIATE_NUM_DIFF_PARAMS bivariate-coeff
// gradients: one warp-shuffle pass per slot, then parallel atomicAdds from the
// first warp into the active polynomial slice (selected via bivariate_coeff_base).
__device__ __forceinline__ void reduce_fisheye_bivariate_grads(
    BivariateParamGrads const& local,
    BivariateWindshieldDistortion_KernelParameters distortion,
    bool is_undistort,
    float* __restrict__ grad_distortion_coeffs) {
    static_assert(kThreads % 32 == 0, "kThreads must be a multiple of warp size");
    constexpr int kNumWarps = kThreads / 32;
    constexpr int kSlots = BIVARIATE_NUM_DIFF_PARAMS;
    __shared__ float warp_sums[kNumWarps][kSlots];

    float values[kSlots];
#pragma unroll
    for (int i = 0; i < BIVARIATE_H_POLY_TERMS; ++i) {
        values[i] = local.h_poly[i];
    }
#pragma unroll
    for (int i = 0; i < BIVARIATE_V_POLY_TERMS; ++i) {
        values[BIVARIATE_H_POLY_TERMS + i] = local.v_poly[i];
    }

    unsigned int mask = 0xffffffffu;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < kSlots; ++i) {
        float v = values[i];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(mask, v, offset);
        }
        if (lane == 0) {
            warp_sums[warp][i] = v;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < kSlots) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total += warp_sums[w][lane];
        }
        if (grad_distortion_coeffs != nullptr) {
            uint32_t base = bivariate_coeff_base(
                distortion.reference_polynomial, is_undistort);
            atomicAdd(&grad_distortion_coeffs[base + lane], total);
        }
    }
}

// Unpacks the D1 8-slot project scratch into FisheyeProjectState. Project state
// packs theta then delta.
__device__ __forceinline__ void fisheye_load_proj_state_8(
    const float* __restrict__ scratch,
    int64_t off,
    FisheyeProjectState& state) {
    state.ray_xy_norm = scratch[off + 0];
    state.theta = scratch[off + 1];
    state.delta = scratch[off + 2];
    fisheye_unpack_flags(
        scratch[off + 3],
        state.behind_camera,
        state.angle_clamped,
        state.oob,
        state.xy_norm_clamped);
}

// Unpacks the 8-slot backproject scratch into FisheyeBackprojectState. The
// backproject state packs delta then theta.
__device__ __forceinline__ void fisheye_load_bp_state_8(
    const float* __restrict__ scratch,
    int64_t off,
    FisheyeBackprojectState& state) {
    state.normalized = make_float2(scratch[off + 0], scratch[off + 1]);
    state.delta = scratch[off + 2];
    state.theta = scratch[off + 3];
    state.ray_raw = make_float3(scratch[off + 4], scratch[off + 5], scratch[off + 6]);
    state.min2d_clamped = static_cast<uint32_t>(scratch[off + 7]) != 0u;
}

// =============================================================================
// D1 backward -- camera_rays_to_image_points_opencv_fisheye_no_external
// =============================================================================

__global__ void camera_rays_to_image_points_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ camera_rays,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};

    if (idx < count) {
        FisheyeProjectState state;
        fisheye_load_proj_state_8(scratch, idx * 8, state);
        float3 ray = read_vec3(camera_rays, idx);
        float2 d_img = make_float2(
            grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        fisheye_project_ray_bwd(ray, params, state, d_img, d_ray, d_params);
        if (grad_camera_rays != nullptr) {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// =============================================================================
// D2 backward -- image_points_to_camera_rays_opencv_fisheye_no_external
// =============================================================================

__global__ void image_points_to_camera_rays_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};

    if (idx < count) {
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, idx * 8, state);
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_ray = read_vec3(grad_camera_rays, idx);
        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_ray, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// Unpacks the D3 14-slot mean-pose scratch into FisheyeProjectState plus p_rel
// and cam_pt. Slots [10..13] are pad (the backward recomputes the slerp).
__device__ __forceinline__ void fisheye_load_meanpose_14(
    const float* __restrict__ scratch,
    int64_t off,
    float3& p_rel,
    float3& cam_pt,
    FisheyeProjectState& state) {
    p_rel = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
    cam_pt = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
    state.ray_xy_norm = scratch[off + 6];
    state.theta = scratch[off + 7];
    state.delta = scratch[off + 8];
    fisheye_unpack_flags(
        scratch[off + 9],
        state.behind_camera,
        state.angle_clamped,
        state.oob,
        state.xy_norm_clamped);
}

// =============================================================================
// D3 backward -- project_world_points_mean_pose_opencv_fisheye_no_external
//
// Feeds the UNNORMALIZED cam_pt straight into fisheye_project_ray_bwd (atan2
// path, NO normalize3_bwd), then through
// quat_inverse_rotate_bwd_xyzw_geom (mean rotation) and quat_slerp_pair_bwd to
// world-point + start/end pose grads. The single mean rotation gradient is
// stored in wxyz output order.
// =============================================================================

__global__ void project_world_points_mean_pose_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ world_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 14;
        float3 p_rel;
        float3 cam_pt;
        FisheyeProjectState state;
        fisheye_load_meanpose_14(scratch, off, p_rel, cam_pt, state);

        if (!state.behind_camera && !state.oob) {
            float2 d_img = make_float2(
                grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            // Unnormalized cam_pt feeds project_ray_bwd directly (NO
            // normalize3_bwd in the pose chain).
            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            fisheye_project_ray_bwd(cam_pt, params, state, d_img, d_cam_pt, d_params);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, &rx, &ry, &rz, &rw);
            float4 rot_mid_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_mid_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(
                rot_mid_xyzw, p_rel, d_cam_pt, d_rot_mid_xyzw, d_p_rel);

            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_mean_t = scale3(d_p_rel, -1.0f);
            d_trans0.x += 0.5f * d_mean_t.x;
            d_trans0.y += 0.5f * d_mean_t.y;
            d_trans0.z += 0.5f * d_mean_t.z;
            d_trans1.x += 0.5f * d_mean_t.x;
            d_trans1.y += 0.5f * d_mean_t.y;
            d_trans1.z += 0.5f * d_mean_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, rx, ry, rz, rw,
                d_rot_mid_xyzw.x, d_rot_mid_xyzw.y, d_rot_mid_xyzw.z, d_rot_mid_xyzw.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0.x += gq0x;
            d_rot0.y += gq0y;
            d_rot0.z += gq0z;
            d_rot0.w += gq0w;
            d_rot1.x += gq1x;
            d_rot1.y += gq1y;
            d_rot1.z += gq1z;
            d_rot1.w += gq1w;
        } else {
            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    float t0x = block_sum<kThreads>(d_trans0.x);
    float t0y = block_sum<kThreads>(d_trans0.y);
    float t0z = block_sum<kThreads>(d_trans0.z);
    float t1x = block_sum<kThreads>(d_trans1.x);
    float t1y = block_sum<kThreads>(d_trans1.y);
    float t1z = block_sum<kThreads>(d_trans1.z);
    float r0x = block_sum<kThreads>(d_rot0.x);
    float r0y = block_sum<kThreads>(d_rot0.y);
    float r0z = block_sum<kThreads>(d_rot0.z);
    float r0w = block_sum<kThreads>(d_rot0.w);
    float r1x = block_sum<kThreads>(d_rot1.x);
    float r1y = block_sum<kThreads>(d_rot1.y);
    float r1z = block_sum<kThreads>(d_rot1.z);
    float r1w = block_sum<kThreads>(d_rot1.w);

    if (threadIdx.x == 0) {
        if (grad_start_translation != nullptr) {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if (grad_end_translation != nullptr) {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        // Emit rotation grads in wxyz output order.
        if (grad_start_rotation != nullptr) {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if (grad_end_rotation != nullptr) {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// =============================================================================
// D5 backward -- image_points_to_world_rays_static_pose_opencv_fisheye_no_external
//
// d_origin passes directly to d_translation; d_direction chains through
// quat_rotate_bwd to d_rotation and d_camera_ray, then through
// fisheye_backproject_image_point_bwd to d_image_point + intrinsic grads. The
// single static rotation gradient is stored in wxyz output order.
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ translation,
    const float* __restrict__ rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_translation,
    float* __restrict__ grad_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};
    float4 d_rot_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, idx * 8, state);
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin = make_float3(
            grad_world_rays[idx * 6 + 0],
            grad_world_rays[idx * 6 + 1],
            grad_world_rays[idx * 6 + 2]);
        float3 d_direction = make_float3(
            grad_world_rays[idx * 6 + 3],
            grad_world_rays[idx * 6 + 4],
            grad_world_rays[idx * 6 + 5]);

        d_trans = d_origin;

        float4 pose_r_xyzw = read_quat_xyzw_from_wxyz(rotation, 0);
        // Reconstruct the normalized camera_ray; on the min2d short-circuit the
        // forward returned (0,0,1) and the backproject bwd contributes no grad.
        float3 camera_ray = normalize3(state.ray_raw);
        if (state.min2d_clamped) {
            camera_ray = make_float3(0.0f, 0.0f, 1.0f);
        }
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(
            pose_r_xyzw, camera_ray, d_direction, d_rot_xyzw, d_camera_ray);

        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_camera_ray, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    float tx = block_sum<kThreads>(d_trans.x);
    float ty = block_sum<kThreads>(d_trans.y);
    float tz = block_sum<kThreads>(d_trans.z);
    float rx = block_sum<kThreads>(d_rot_xyzw.x);
    float ry = block_sum<kThreads>(d_rot_xyzw.y);
    float rz = block_sum<kThreads>(d_rot_xyzw.z);
    float rw = block_sum<kThreads>(d_rot_xyzw.w);

    if (threadIdx.x == 0) {
        if (grad_translation != nullptr) {
            atomicAdd(&grad_translation[0], tx);
            atomicAdd(&grad_translation[1], ty);
            atomicAdd(&grad_translation[2], tz);
        }
        // Emit rotation grad in wxyz output order.
        if (grad_rotation != nullptr) {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// =============================================================================
// D1 bivariate backward -- camera_rays_to_image_points_opencv_fisheye_bivariate_windshield
//
// Reconstructs distorted_ray from the saved camera ray (scratch stride 8, no
// unnorm_out), runs fisheye_project_ray_bwd at the distorted ray, then chains
// d_distorted back through apply_bivariate_distortion_bwd to d_camera_ray +
// fisheye intrinsic + bivariate-coeff grads.
// =============================================================================

__global__ void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ camera_rays,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, false);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};

    if (idx < count) {
        FisheyeProjectState state;
        fisheye_load_proj_state_8(scratch, idx * 8, state);
        float3 ray = read_vec3(camera_rays, idx);
        float3 distorted_ray = apply_bivariate_distortion(ray, bivariate_params);
        float2 d_img = make_float2(
            grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
        fisheye_project_ray_bwd(distorted_ray, params, state, d_img, d_distorted, d_params);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            ray, bivariate_params, d_distorted, d_ray, d_biv);
        if (grad_camera_rays != nullptr) {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D2 bivariate backward -- image_points_to_camera_rays_opencv_fisheye_bivariate_windshield
//
// Chain (scratch stride 12, unnorm_out at [8..10]):
//   d_ray -> normalize3_bwd(unnorm_out) -> d_unnorm
//   d_unnorm -> apply_bivariate_distortion_bwd(distorted_ray=normalize3(ray_raw))
//            -> d_distorted + bivariate-coeff grads
//   d_distorted -> normalize3_bwd(ray_raw) -> d_ray_raw
//   d_ray_raw -> fisheye_backproject_image_point_bwd -> d_image_point + intrinsic
// =============================================================================

__global__ void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ image_points,
    const float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, true);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};

    if (idx < count) {
        int64_t off = idx * 12;
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, off, state);
        float3 unnorm_out = make_float3(
            scratch[off + 8], scratch[off + 9], scratch[off + 10]);
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_ray = read_vec3(grad_camera_rays, idx);

        // camera_ray = normalize3(unnorm_out).
        float3 d_unnorm = normalize3_bwd(unnorm_out, d_ray);
        // unnorm_out = apply_bivariate_distortion(distorted_ray); distorted_ray
        // is the backproject output normalize3(ray_raw).
        float3 distorted_ray = normalize3(state.ray_raw);
        float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray, bivariate_params, d_unnorm, d_distorted, d_biv);
        // distorted_ray = normalize3(ray_raw).
        float3 d_ray_raw = normalize3_bwd(state.ray_raw, d_distorted);
        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_ray_raw, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}

// =============================================================================
// D3 bivariate backward -- project_world_points_mean_pose_opencv_fisheye_bivariate_windshield
//
// Identical pose/slerp chain to D3 no_external, with the bivariate distortion
// inserted between cam_pt and the fisheye projection: recompute distorted from
// cam_pt (scratch stride 14, no unnorm_out), run fisheye_project_ray_bwd at the
// distorted ray, then apply_bivariate_distortion_bwd back to d_cam_pt + coeff
// grads. The mean rotation gradient is stored in wxyz output order.
// =============================================================================

__global__ void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ world_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, false);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 14;
        float3 p_rel;
        float3 cam_pt;
        FisheyeProjectState state;
        fisheye_load_meanpose_14(scratch, off, p_rel, cam_pt, state);

        if (!state.behind_camera && !state.oob) {
            float2 d_img = make_float2(
                grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            // Recompute distorted ray for the fisheye_project_ray_bwd primal,
            // then chain back through the bivariate distortion (NO normalize3_bwd
            // in the pose chain).
            float3 distorted = apply_bivariate_distortion(cam_pt, bivariate_params);
            float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
            fisheye_project_ray_bwd(distorted, params, state, d_img, d_distorted, d_params);
            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(
                cam_pt, bivariate_params, d_distorted, d_cam_pt, d_biv);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, &rx, &ry, &rz, &rw);
            float4 rot_mid_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_mid_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(
                rot_mid_xyzw, p_rel, d_cam_pt, d_rot_mid_xyzw, d_p_rel);

            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_mean_t = scale3(d_p_rel, -1.0f);
            d_trans0.x += 0.5f * d_mean_t.x;
            d_trans0.y += 0.5f * d_mean_t.y;
            d_trans0.z += 0.5f * d_mean_t.z;
            d_trans1.x += 0.5f * d_mean_t.x;
            d_trans1.y += 0.5f * d_mean_t.y;
            d_trans1.z += 0.5f * d_mean_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, rx, ry, rz, rw,
                d_rot_mid_xyzw.x, d_rot_mid_xyzw.y, d_rot_mid_xyzw.z, d_rot_mid_xyzw.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0.x += gq0x;
            d_rot0.y += gq0y;
            d_rot0.z += gq0z;
            d_rot0.w += gq0w;
            d_rot1.x += gq1x;
            d_rot1.y += gq1y;
            d_rot1.z += gq1z;
            d_rot1.w += gq1w;
        } else {
            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    float t0x = block_sum<kThreads>(d_trans0.x);
    float t0y = block_sum<kThreads>(d_trans0.y);
    float t0z = block_sum<kThreads>(d_trans0.z);
    float t1x = block_sum<kThreads>(d_trans1.x);
    float t1y = block_sum<kThreads>(d_trans1.y);
    float t1z = block_sum<kThreads>(d_trans1.z);
    float r0x = block_sum<kThreads>(d_rot0.x);
    float r0y = block_sum<kThreads>(d_rot0.y);
    float r0z = block_sum<kThreads>(d_rot0.z);
    float r0w = block_sum<kThreads>(d_rot0.w);
    float r1x = block_sum<kThreads>(d_rot1.x);
    float r1y = block_sum<kThreads>(d_rot1.y);
    float r1z = block_sum<kThreads>(d_rot1.z);
    float r1w = block_sum<kThreads>(d_rot1.w);

    if (threadIdx.x == 0) {
        if (grad_start_translation != nullptr) {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if (grad_end_translation != nullptr) {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        // Emit rotation grads in wxyz output order.
        if (grad_start_rotation != nullptr) {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if (grad_end_rotation != nullptr) {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D5 bivariate backward -- image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield
//
// d_origin -> d_translation; d_direction -> quat_rotate_bwd -> d_rotation +
// d_camera_ray. Then chain (scratch stride 12, unnorm_out at [8..10]):
//   d_camera_ray -> normalize3_bwd(unnorm_out) -> d_unnorm
//   d_unnorm -> apply_bivariate_distortion_bwd(normalize3(ray_raw)) -> d_distorted
//   d_distorted -> normalize3_bwd(ray_raw) -> d_ray_raw
//   d_ray_raw -> fisheye_backproject_image_point_bwd -> d_image_point + intrinsic
// The static rotation gradient is stored in wxyz output order.
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ image_points,
    const float* __restrict__ translation,
    const float* __restrict__ rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_translation,
    float* __restrict__ grad_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, true);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 12;
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, off, state);
        float3 unnorm_out = make_float3(
            scratch[off + 8], scratch[off + 9], scratch[off + 10]);
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin = make_float3(
            grad_world_rays[idx * 6 + 0],
            grad_world_rays[idx * 6 + 1],
            grad_world_rays[idx * 6 + 2]);
        float3 d_direction = make_float3(
            grad_world_rays[idx * 6 + 3],
            grad_world_rays[idx * 6 + 4],
            grad_world_rays[idx * 6 + 5]);

        d_trans = d_origin;

        float4 pose_r_xyzw = read_quat_xyzw_from_wxyz(rotation, 0);
        float3 camera_ray = normalize3(unnorm_out);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(
            pose_r_xyzw, camera_ray, d_direction, d_rot_xyzw, d_camera_ray);

        // camera_ray = normalize3(unnorm_out).
        float3 d_unnorm = normalize3_bwd(unnorm_out, d_camera_ray);
        // unnorm_out = apply_bivariate_distortion(distorted_ray).
        float3 distorted_ray = normalize3(state.ray_raw);
        float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray, bivariate_params, d_unnorm, d_distorted, d_biv);
        // distorted_ray = normalize3(ray_raw).
        float3 d_ray_raw = normalize3_bwd(state.ray_raw, d_distorted);

        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_ray_raw, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    float tx = block_sum<kThreads>(d_trans.x);
    float ty = block_sum<kThreads>(d_trans.y);
    float tz = block_sum<kThreads>(d_trans.z);
    float rx = block_sum<kThreads>(d_rot_xyzw.x);
    float ry = block_sum<kThreads>(d_rot_xyzw.y);
    float rz = block_sum<kThreads>(d_rot_xyzw.z);
    float rw = block_sum<kThreads>(d_rot_xyzw.w);

    if (threadIdx.x == 0) {
        if (grad_translation != nullptr) {
            atomicAdd(&grad_translation[0], tx);
            atomicAdd(&grad_translation[1], ty);
            atomicAdd(&grad_translation[2], tz);
        }
        // Emit rotation grad in wxyz output order.
        if (grad_rotation != nullptr) {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}

// Reduces start/end translation + rotation pose grads via block_sum and
// atomicAdds them from thread 0 in wxyz output order. Mirrors
// reduce_pose2_grads_components; both rotation grads are fed as xyzw and stored
// as wxyz here.
__device__ __forceinline__ void reduce_fisheye_pose2_grads(
    float3 d_trans0,
    float3 d_trans1,
    float4 d_rot0_xyzw,
    float4 d_rot1_xyzw,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation) {
    float t0x = block_sum<kThreads>(d_trans0.x);
    float t0y = block_sum<kThreads>(d_trans0.y);
    float t0z = block_sum<kThreads>(d_trans0.z);
    float t1x = block_sum<kThreads>(d_trans1.x);
    float t1y = block_sum<kThreads>(d_trans1.y);
    float t1z = block_sum<kThreads>(d_trans1.z);
    float r0x = block_sum<kThreads>(d_rot0_xyzw.x);
    float r0y = block_sum<kThreads>(d_rot0_xyzw.y);
    float r0z = block_sum<kThreads>(d_rot0_xyzw.z);
    float r0w = block_sum<kThreads>(d_rot0_xyzw.w);
    float r1x = block_sum<kThreads>(d_rot1_xyzw.x);
    float r1y = block_sum<kThreads>(d_rot1_xyzw.y);
    float r1z = block_sum<kThreads>(d_rot1_xyzw.z);
    float r1w = block_sum<kThreads>(d_rot1_xyzw.w);
    if (threadIdx.x == 0) {
        if (grad_start_translation != nullptr) {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if (grad_end_translation != nullptr) {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if (grad_start_rotation != nullptr) {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if (grad_end_rotation != nullptr) {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
}

// =============================================================================
// D4 backward -- project_world_points_shutter_pose_opencv_fisheye_no_external
//
// Replays ONE differentiable step at the converged alpha read from scratch
// (off+14). Gradient flow is gated on fwd_valid = !isnan(alpha) &&
// !behind_camera && !oob (the D4-only NaN-alpha gate). Feeds the
// UNNORMALIZED cam_pt straight into fisheye_project_ray_bwd (NO normalize3_bwd
// in the pose chain), then through quat_inverse_rotate_bwd to the
// world-point + start/end pose grads. Translation splits (1-alpha)/alpha; both
// rotation grads are stored in wxyz output order.
// =============================================================================

__global__ void project_world_points_shutter_pose_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 16;
        float3 p_rel;
        float3 cam_pt;
        FisheyeProjectState state;
        fisheye_load_meanpose_14(scratch, off, p_rel, cam_pt, state);
        float alpha = scratch[off + 14];
        bool const fwd_valid =
            !isnan(alpha) && !state.behind_camera && !state.oob;

        if (fwd_valid) {
            float2 d_img = make_float2(
                grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            fisheye_project_ray_bwd(cam_pt, params, state, d_img, d_cam_pt, d_params);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &rx, &ry, &rz, &rw);
            float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_alpha = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(
                rot_alpha_xyzw, p_rel, d_cam_pt, d_rot_alpha, d_p_rel);

            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_pose_t = scale3(d_p_rel, -1.0f);
            d_trans0.x += (1.0f - alpha) * d_pose_t.x;
            d_trans0.y += (1.0f - alpha) * d_pose_t.y;
            d_trans0.z += (1.0f - alpha) * d_pose_t.z;
            d_trans1.x += alpha * d_pose_t.x;
            d_trans1.y += alpha * d_pose_t.y;
            d_trans1.z += alpha * d_pose_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, rx, ry, rz, rw,
                d_rot_alpha.x, d_rot_alpha.y, d_rot_alpha.z, d_rot_alpha.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0.x += gq0x;
            d_rot0.y += gq0y;
            d_rot0.z += gq0z;
            d_rot0.w += gq0w;
            d_rot1.x += gq1x;
            d_rot1.y += gq1y;
            d_rot1.z += gq1z;
            d_rot1.w += gq1w;
        } else {
            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    reduce_fisheye_pose2_grads(
        d_trans0, d_trans1, d_rot0, d_rot1,
        grad_start_translation, grad_end_translation,
        grad_start_rotation, grad_end_rotation);
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// =============================================================================
// D6 backward -- image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external
//
// d_origin passes directly to start/end translation (split (1-alpha)/alpha);
// d_direction chains through quat_rotate_bwd at the slerp pose -> d_rotation +
// d_camera_ray, then through fisheye_backproject_image_point_bwd to
// d_image_point + intrinsic grads. NO isnan(alpha) gate; alpha read from
// scratch (off+8). Both rotation grads stored in wxyz output order.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    OpenCVFisheyeParamGrads d_params{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 12;
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, off, state);
        float alpha = scratch[off + 8];
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin = make_float3(
            grad_world_rays[idx * 6 + 0],
            grad_world_rays[idx * 6 + 1],
            grad_world_rays[idx * 6 + 2]);
        float3 d_direction = make_float3(
            grad_world_rays[idx * 6 + 3],
            grad_world_rays[idx * 6 + 4],
            grad_world_rays[idx * 6 + 5]);

        float3 d_pose_t = d_origin;
        d_trans0.x += (1.0f - alpha) * d_pose_t.x;
        d_trans0.y += (1.0f - alpha) * d_pose_t.y;
        d_trans0.z += (1.0f - alpha) * d_pose_t.z;
        d_trans1.x += alpha * d_pose_t.x;
        d_trans1.y += alpha * d_pose_t.y;
        d_trans1.z += alpha * d_pose_t.z;

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, &rx, &ry, &rz, &rw);
        float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);

        // Reconstruct the normalized camera_ray; on the min2d short-circuit the
        // forward returned (0,0,1) and the backproject bwd contributes no grad.
        float3 camera_ray = normalize3(state.ray_raw);
        if (state.min2d_clamped) {
            camera_ray = make_float3(0.0f, 0.0f, 1.0f);
        }
        float4 d_rot_alpha = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(
            rot_alpha_xyzw, camera_ray, d_direction, d_rot_alpha, d_camera_ray);

        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_camera_ray, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }

        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
        trajectory_cuda::quat_slerp_pair_bwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, rx, ry, rz, rw,
            d_rot_alpha.x, d_rot_alpha.y, d_rot_alpha.z, d_rot_alpha.w,
            &gq0x, &gq0y, &gq0z, &gq0w,
            &gq1x, &gq1y, &gq1z, &gq1w,
            &ga_unused);
        d_rot0.x += gq0x;
        d_rot0.y += gq0y;
        d_rot0.z += gq0z;
        d_rot0.w += gq0w;
        d_rot1.x += gq1x;
        d_rot1.y += gq1y;
        d_rot1.z += gq1z;
        d_rot1.w += gq1w;
    }

    reduce_fisheye_pose2_grads(
        d_trans0, d_trans1, d_rot0, d_rot1,
        grad_start_translation, grad_end_translation,
        grad_start_rotation, grad_end_rotation);
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
}

// =============================================================================
// D4 bivariate backward -- project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield
//
// Identical pose/slerp chain to D4 no_external, with the bivariate distortion
// inserted between cam_pt and the fisheye projection: recompute distorted from
// cam_pt (scratch stride 16, no unnorm_out), run fisheye_project_ray_bwd at the
// distorted ray, then apply_bivariate_distortion_bwd back to d_cam_pt + coeff
// grads. NaN-alpha gate as D4; both rotation grads stored in wxyz order.
// =============================================================================

__global__ void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, false);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 16;
        float3 p_rel;
        float3 cam_pt;
        FisheyeProjectState state;
        fisheye_load_meanpose_14(scratch, off, p_rel, cam_pt, state);
        float alpha = scratch[off + 14];
        bool const fwd_valid =
            !isnan(alpha) && !state.behind_camera && !state.oob;

        if (fwd_valid) {
            float2 d_img = make_float2(
                grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            float3 distorted = apply_bivariate_distortion(cam_pt, bivariate_params);
            float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
            fisheye_project_ray_bwd(distorted, params, state, d_img, d_distorted, d_params);
            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(
                cam_pt, bivariate_params, d_distorted, d_cam_pt, d_biv);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &rx, &ry, &rz, &rw);
            float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_alpha = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(
                rot_alpha_xyzw, p_rel, d_cam_pt, d_rot_alpha, d_p_rel);

            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_pose_t = scale3(d_p_rel, -1.0f);
            d_trans0.x += (1.0f - alpha) * d_pose_t.x;
            d_trans0.y += (1.0f - alpha) * d_pose_t.y;
            d_trans0.z += (1.0f - alpha) * d_pose_t.z;
            d_trans1.x += alpha * d_pose_t.x;
            d_trans1.y += alpha * d_pose_t.y;
            d_trans1.z += alpha * d_pose_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, rx, ry, rz, rw,
                d_rot_alpha.x, d_rot_alpha.y, d_rot_alpha.z, d_rot_alpha.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0.x += gq0x;
            d_rot0.y += gq0y;
            d_rot0.z += gq0z;
            d_rot0.w += gq0w;
            d_rot1.x += gq1x;
            d_rot1.y += gq1y;
            d_rot1.z += gq1z;
            d_rot1.w += gq1w;
        } else {
            if (grad_world_points != nullptr) {
                write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    reduce_fisheye_pose2_grads(
        d_trans0, d_trans1, d_rot0, d_rot1,
        grad_start_translation, grad_end_translation,
        grad_start_rotation, grad_end_rotation);
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D6 bivariate backward -- image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield
//
// d_origin -> start/end translation (split (1-alpha)/alpha); d_direction ->
// quat_rotate_bwd at the slerp pose -> d_rotation + d_camera_ray. Then chain
// (scratch stride 16, unnorm_out at [12..14]):
//   d_camera_ray -> normalize3_bwd(unnorm_out) -> d_unnorm
//   d_unnorm -> apply_bivariate_distortion_bwd(normalize3(ray_raw)) -> d_distorted
//   d_distorted -> normalize3_bwd(ray_raw) -> d_ray_raw
//   d_ray_raw -> fisheye_backproject_image_point_bwd -> d_image_point + intrinsic
// NO isnan(alpha) gate; both rotation grads stored in wxyz output order.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ image_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_forward_poly,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params =
        load_bivariate_windshield_params(distortion, true);
    OpenCVFisheyeParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 16;
        FisheyeBackprojectState state;
        fisheye_load_bp_state_8(scratch, off, state);
        float alpha = scratch[off + 8];
        float3 unnorm_out = make_float3(
            scratch[off + 12], scratch[off + 13], scratch[off + 14]);
        float2 img = make_float2(
            image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin = make_float3(
            grad_world_rays[idx * 6 + 0],
            grad_world_rays[idx * 6 + 1],
            grad_world_rays[idx * 6 + 2]);
        float3 d_direction = make_float3(
            grad_world_rays[idx * 6 + 3],
            grad_world_rays[idx * 6 + 4],
            grad_world_rays[idx * 6 + 5]);

        float3 d_pose_t = d_origin;
        d_trans0.x += (1.0f - alpha) * d_pose_t.x;
        d_trans0.y += (1.0f - alpha) * d_pose_t.y;
        d_trans0.z += (1.0f - alpha) * d_pose_t.z;
        d_trans1.x += alpha * d_pose_t.x;
        d_trans1.y += alpha * d_pose_t.y;
        d_trans1.z += alpha * d_pose_t.z;

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, &rx, &ry, &rz, &rw);
        float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);

        float3 camera_ray = normalize3(unnorm_out);
        float4 d_rot_alpha = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(
            rot_alpha_xyzw, camera_ray, d_direction, d_rot_alpha, d_camera_ray);

        // camera_ray = normalize3(unnorm_out).
        float3 d_unnorm = normalize3_bwd(unnorm_out, d_camera_ray);
        // unnorm_out = apply_bivariate_distortion(distorted_ray).
        float3 distorted_ray = normalize3(state.ray_raw);
        float3 d_distorted = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray, bivariate_params, d_unnorm, d_distorted, d_biv);
        // distorted_ray = normalize3(ray_raw).
        float3 d_ray_raw = normalize3_bwd(state.ray_raw, d_distorted);

        float2 d_img = make_float2(0.0f, 0.0f);
        fisheye_backproject_image_point_bwd(
            img, params, state, d_ray_raw, d_img, d_params);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }

        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
        trajectory_cuda::quat_slerp_pair_bwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, rx, ry, rz, rw,
            d_rot_alpha.x, d_rot_alpha.y, d_rot_alpha.z, d_rot_alpha.w,
            &gq0x, &gq0y, &gq0z, &gq0w,
            &gq1x, &gq1y, &gq1z, &gq1w,
            &ga_unused);
        d_rot0.x += gq0x;
        d_rot0.y += gq0y;
        d_rot0.z += gq0z;
        d_rot0.w += gq0w;
        d_rot1.x += gq1x;
        d_rot1.y += gq1y;
        d_rot1.z += gq1z;
        d_rot1.w += gq1w;
    }

    reduce_fisheye_pose2_grads(
        d_trans0, d_trans1, d_rot0, d_rot1,
        grad_start_translation, grad_end_translation,
        grad_start_rotation, grad_end_rotation);
    reduce_fisheye_intrinsic_grads(
        d_params, grad_principal_point, grad_focal_length, grad_forward_poly);
    reduce_fisheye_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}

} // namespace

// =============================================================================
// Backward launchers
// =============================================================================

void camera_rays_to_image_points_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            camera_rays,
            grad_image_points,
            grad_camera_rays,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            image_points,
            grad_camera_rays,
            grad_image_points,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_mean_pose_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            image_points,
            translation,
            rotation,
            grad_world_rays,
            grad_image_points,
            grad_translation,
            grad_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Bivariate-windshield backward launchers (D1/D2/D3/D5).
// =============================================================================

void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            camera_rays,
            grad_image_points,
            grad_camera_rays,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            grad_camera_rays,
            grad_image_points,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            translation,
            rotation,
            grad_world_rays,
            grad_image_points,
            grad_translation,
            grad_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Shutter-pose backward launchers (D4/D6, both distortions).
// =============================================================================

void project_world_points_shutter_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_shutter_pose_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            image_points,
            start_rotation,
            end_rotation,
            grad_world_rays,
            grad_image_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward_kernel
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            start_rotation,
            end_rotation,
            grad_world_rays,
            grad_image_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_focal_length,
            grad_forward_poly,
            grad_distortion_coeffs,
            scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
