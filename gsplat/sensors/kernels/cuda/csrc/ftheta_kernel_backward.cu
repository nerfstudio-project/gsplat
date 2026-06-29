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

// Backward CUDA kernels for FTheta camera models. Implements the VJPs for each
// forward kernel in ftheta_kernel.cu, chaining through the FTheta IFT adjoint
// helpers in ftheta_kernel.cuh and the shared pose/quaternion adapters in
// camera_kernel.cuh.

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "ftheta_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Block-reduces each FThetaParamGrads slot via block_sum then atomicAdds the
// result from thread 0. Called once per kernel after all per-ray gradient work.
__device__ __forceinline__ void reduce_ftheta_intrinsic_grads(
    const FThetaParamGrads &local,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv
)
{
    float pp0 = block_sum<kThreads>(local.pp[0]);
    float pp1 = block_sum<kThreads>(local.pp[1]);
    float fw0 = block_sum<kThreads>(local.fw_poly[0]);
    float fw1 = block_sum<kThreads>(local.fw_poly[1]);
    float fw2 = block_sum<kThreads>(local.fw_poly[2]);
    float fw3 = block_sum<kThreads>(local.fw_poly[3]);
    float fw4 = block_sum<kThreads>(local.fw_poly[4]);
    float fw5 = block_sum<kThreads>(local.fw_poly[5]);
    float bw0 = block_sum<kThreads>(local.bw_poly[0]);
    float bw1 = block_sum<kThreads>(local.bw_poly[1]);
    float bw2 = block_sum<kThreads>(local.bw_poly[2]);
    float bw3 = block_sum<kThreads>(local.bw_poly[3]);
    float bw4 = block_sum<kThreads>(local.bw_poly[4]);
    float bw5 = block_sum<kThreads>(local.bw_poly[5]);
    float A0  = block_sum<kThreads>(local.A[0]);
    float A1  = block_sum<kThreads>(local.A[1]);
    float A2  = block_sum<kThreads>(local.A[2]);
    float A3  = block_sum<kThreads>(local.A[3]);
    float Ai0 = block_sum<kThreads>(local.Ainv[0]);
    float Ai1 = block_sum<kThreads>(local.Ainv[1]);
    float Ai2 = block_sum<kThreads>(local.Ainv[2]);
    float Ai3 = block_sum<kThreads>(local.Ainv[3]);

    if(threadIdx.x == 0)
    {
        if(grad_principal_point != nullptr)
        {
            atomicAdd(&grad_principal_point[0], pp0);
            atomicAdd(&grad_principal_point[1], pp1);
        }
        if(grad_fw_poly != nullptr)
        {
            atomicAdd(&grad_fw_poly[0], fw0);
            atomicAdd(&grad_fw_poly[1], fw1);
            atomicAdd(&grad_fw_poly[2], fw2);
            atomicAdd(&grad_fw_poly[3], fw3);
            atomicAdd(&grad_fw_poly[4], fw4);
            atomicAdd(&grad_fw_poly[5], fw5);
        }
        if(grad_bw_poly != nullptr)
        {
            atomicAdd(&grad_bw_poly[0], bw0);
            atomicAdd(&grad_bw_poly[1], bw1);
            atomicAdd(&grad_bw_poly[2], bw2);
            atomicAdd(&grad_bw_poly[3], bw3);
            atomicAdd(&grad_bw_poly[4], bw4);
            atomicAdd(&grad_bw_poly[5], bw5);
        }
        if(grad_A != nullptr)
        {
            atomicAdd(&grad_A[0], A0);
            atomicAdd(&grad_A[1], A1);
            atomicAdd(&grad_A[2], A2);
            atomicAdd(&grad_A[3], A3);
        }
        if(grad_Ainv != nullptr)
        {
            atomicAdd(&grad_Ainv[0], Ai0);
            atomicAdd(&grad_Ainv[1], Ai1);
            atomicAdd(&grad_Ainv[2], Ai2);
            atomicAdd(&grad_Ainv[3], Ai3);
        }
    }
}

// Fused block reduction for all BIVARIATE_NUM_DIFF_PARAMS bivariate-coeff
// gradients: single __syncthreads + one warp-shuffle pass per slot, then
// parallel atomicAdds from the first warp.
__device__ __forceinline__ void reduce_ftheta_bivariate_grads(
    const BivariateParamGrads &local,
    BivariateWindshieldDistortion_KernelParameters distortion,
    bool is_undistort,
    float *__restrict__ grad_distortion_coeffs
)
{
    static_assert(kThreads % 32 == 0, "kThreads must be a multiple of warp size");
    constexpr int kNumWarps = kThreads / 32;
    constexpr int kSlots    = BIVARIATE_NUM_DIFF_PARAMS;
    __shared__ float warp_sums[kNumWarps][kSlots];

    float values[kSlots];
#pragma unroll
    for(int i = 0; i < BIVARIATE_H_POLY_TERMS; ++i)
    {
        values[i] = local.h_poly[i];
    }
#pragma unroll
    for(int i = 0; i < BIVARIATE_V_POLY_TERMS; ++i)
    {
        values[BIVARIATE_H_POLY_TERMS + i] = local.v_poly[i];
    }

    unsigned int mask = 0xFFFFFFFFu;
    int lane          = threadIdx.x & 31;
    int warp          = threadIdx.x >> 5;
#pragma unroll
    for(int i = 0; i < kSlots; ++i)
    {
        float v = values[i];
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            v += __shfl_xor_sync(mask, v, offset);
        }
        if(lane == 0)
        {
            warp_sums[warp][i] = v;
        }
    }
    __syncthreads();

    if(warp == 0 && lane < kSlots)
    {
        float total = 0.0f;
#pragma unroll
        for(int w = 0; w < kNumWarps; ++w)
        {
            total += warp_sums[w][lane];
        }
        if(grad_distortion_coeffs != nullptr)
        {
            uint32_t base = bivariate_coeff_base(distortion.reference_polynomial, is_undistort);
            atomicAdd(&grad_distortion_coeffs[base + lane], total);
        }
    }
}

// Unpacks the 8-slot forward project scratch into FThetaProjectState.
__device__ __forceinline__ void ftheta_load_proj_state_8(
    const float *__restrict__ scratch, int64_t off, FThetaProjectState &state
)
{
    state.ray_norm = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
    state.theta    = scratch[off + 3];
    state.r        = scratch[off + 4];
    state.xy_norm  = scratch[off + 5];
    ftheta_unpack_flags(scratch[off + 6], state.behind_camera, state.angle_clamped, state.min2d_clamped);
}

// Unpacks the 8-slot forward backproject scratch into FThetaBackprojectState.
__device__ __forceinline__ void ftheta_load_bp_state_8(
    const float *__restrict__ scratch, int64_t off, FThetaBackprojectState &state
)
{
    state.transformed   = make_float2(scratch[off + 0], scratch[off + 1]);
    state.rdist         = scratch[off + 2];
    state.theta         = scratch[off + 3];
    state.ray_raw       = make_float3(scratch[off + 4], scratch[off + 5], scratch[off + 6]);
    state.min2d_clamped = ftheta_bp_unpack_flags(scratch[off + 7]);
}

// =============================================================================
// D1 backward -- camera_rays_to_image_points_ftheta_no_external
//
// Backward: FTheta project camera rays -> image points (no external
// distortion). Consumes the 8-slot forward scratch and chains d_image_point
// through ftheta_project_ray_bwd to d_camera_ray + intrinsic grads.
// =============================================================================

__global__ void camera_rays_to_image_points_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};

    if(idx < count)
    {
        FThetaProjectState state;
        ftheta_load_proj_state_8(scratch, idx * 8, state);
        float3 ray   = read_vec3(camera_rays, idx);
        float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        ftheta_project_ray_bwd(ray, params, state, d_img, d_ray, d_params);
        if(grad_camera_rays != nullptr)
        {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D2 backward -- image_points_to_camera_rays_ftheta_no_external
//
// Backward: FTheta backproject image points -> camera rays (no external
// distortion). Chains d_camera_ray through ftheta_backproject_image_point_bwd
// to d_image_point + intrinsic grads.
// =============================================================================

__global__ void image_points_to_camera_rays_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};

    if(idx < count)
    {
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, idx * 8, state);
        float2 img   = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_ray = read_vec3(grad_camera_rays, idx);
        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_ray, d_img, d_params);
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D3 backward -- camera_rays_to_image_points_ftheta_bivariate_windshield
//
// Backward: FTheta project camera rays -> image points with bivariate
// windshield distortion. Chains d_image_point through ftheta_project_ray_bwd
// at the distorted ray, then through apply_bivariate_distortion_bwd to
// d_camera_ray + intrinsic + bivariate coeff grads.
// =============================================================================

__global__ void camera_rays_to_image_points_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx                          = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, false);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};

    if(idx < count)
    {
        FThetaProjectState state;
        ftheta_load_proj_state_8(scratch, idx * 8, state);
        float3 ray           = read_vec3(camera_rays, idx);
        // Recompute distorted_ray rather than reading it from scratch.
        float3 distorted_ray = apply_bivariate_distortion(ray, biv_params);
        float2 d_img         = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

        float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
        ftheta_project_ray_bwd(distorted_ray, params, state, d_img, d_distorted_ray, d_params);

        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(ray, biv_params, d_distorted_ray, d_ray, d_biv);
        if(grad_camera_rays != nullptr)
        {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D3 (inverse) backward -- image_points_to_camera_rays_ftheta_bivariate_windshield
//
// Backward: FTheta backproject image points -> camera rays with bivariate
// windshield distortion. Scratch layout (stride 12): 8-slot bp state +
// [8..10]=unnorm_out, [11]=unused.
// =============================================================================

__global__ void image_points_to_camera_rays_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx                          = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, true);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};

    if(idx < count)
    {
        int64_t off = idx * 12;
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, off, state);
        float3 unnorm_out   = make_float3(scratch[off + 8], scratch[off + 9], scratch[off + 10]);
        float2 img          = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_camera_ray = read_vec3(grad_camera_rays, idx);

        // Chain d_camera_ray through normalize3_bwd(unnorm_out), then through
        // apply_bivariate_distortion_bwd, then through ftheta_backproject_bwd
        // (which absorbs the inner normalize3_bwd of state.ray_raw).
        float3 d_unnorm_out = normalize3_bwd(unnorm_out, d_camera_ray);
        float3 ftheta_ray   = normalize3(state.ray_raw);
        float3 d_ftheta_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(ftheta_ray, biv_params, d_unnorm_out, d_ftheta_ray, d_biv);
        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_ftheta_ray, d_img, d_params);
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}

// Unpacks the 10-slot mean/shutter forward scratch
// ([p_rel(3), cam_pt(3), theta, r, xy_norm, flags]) into FThetaProjectState
// plus p_rel and cam_pt. ray_norm is left zero; callers must recompute it
// from cam_pt (or from the distorted ray on bivariate variants).
__device__ __forceinline__ void ftheta_load_meanpose_10(
    const float *__restrict__ scratch, int64_t off, float3 &p_rel, float3 &cam_pt, FThetaProjectState &state
)
{
    p_rel         = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
    cam_pt        = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
    state.theta   = scratch[off + 6];
    state.r       = scratch[off + 7];
    state.xy_norm = scratch[off + 8];
    ftheta_unpack_flags(scratch[off + 9], state.behind_camera, state.angle_clamped, state.min2d_clamped);
    state.ray_norm = make_float3(0.0f, 0.0f, 0.0f);
}

// =============================================================================
// D4 backward -- project_world_points_mean_pose_ftheta_no_external
//
// Backward: project world points -> image points with mean shutter pose
// (slerp alpha = 0.5, no external distortion). Chains d_image_point through
// ftheta_project_ray_bwd, quat_inverse_rotate_bwd (mean rotation), and
// quat_slerp_pair_bwd to world-point + start/end pose grads.
// =============================================================================

__global__ void project_world_points_mean_pose_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        int64_t off = idx * 10;
        float3 p_rel;
        float3 cam_pt;
        FThetaProjectState state;
        ftheta_load_meanpose_10(scratch, off, p_rel, cam_pt, state);

        if(!state.behind_camera)
        {
            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            float3 d_cam_pt             = make_float3(0.0f, 0.0f, 0.0f);
            // Repopulate ray_norm from cam_pt for the project bwd helper.
            FThetaProjectState replayed = state;
            replayed.ray_norm           = normalize3(cam_pt);
            ftheta_project_ray_bwd(cam_pt, params, replayed, d_img, d_cam_pt, d_params);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &rx, &ry, &rz, &rw
            );
            float4 rot_mid_xyzw   = make_float4(rx, ry, rz, rw);
            float4 d_rot_mid_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel        = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot_mid_xyzw, p_rel, d_cam_pt, d_rot_mid_xyzw, d_p_rel);

            // d_world = d_p_rel; d_mean_t = -d_p_rel, split 0.5/0.5 to trans0/trans1.
            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_mean_t  = scale3(d_p_rel, -1.0f);
            d_trans0.x      += 0.5f * d_mean_t.x;
            d_trans0.y      += 0.5f * d_mean_t.y;
            d_trans0.z      += 0.5f * d_mean_t.z;
            d_trans1.x      += 0.5f * d_mean_t.x;
            d_trans1.y      += 0.5f * d_mean_t.y;
            d_trans1.z      += 0.5f * d_mean_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                rot0.x,
                rot0.y,
                rot0.z,
                rot0.w,
                rot1.x,
                rot1.y,
                rot1.z,
                rot1.w,
                0.5f,
                rx,
                ry,
                rz,
                rw,
                d_rot_mid_xyzw.x,
                d_rot_mid_xyzw.y,
                d_rot_mid_xyzw.z,
                d_rot_mid_xyzw.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
            float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
            // Accumulate in xyzw order; conversion to wxyz happens at write time.
            d_rot0.x    += d_r0.x;
            d_rot0.y    += d_r0.y;
            d_rot0.z    += d_r0.z;
            d_rot0.w    += d_r0.w;
            d_rot1.x    += d_r1.x;
            d_rot1.y    += d_r1.y;
            d_rot1.z    += d_r1.z;
            d_rot1.w    += d_r1.w;
        }
        else
        {
            if(grad_world_points != nullptr)
            {
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        // Emit rotation grads in wxyz output order.
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D4 (bivariate) backward -- project_world_points_mean_pose_ftheta_bivariate
//
// Same as D4 above, with bivariate windshield distortion inserted between
// cam_pt and the FTheta projection. Adds bivariate coeff gradient
// accumulation.
// =============================================================================

__global__ void project_world_points_mean_pose_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx                          = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, false);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        int64_t off = idx * 10;
        float3 p_rel;
        float3 cam_pt;
        FThetaProjectState state;
        ftheta_load_meanpose_10(scratch, off, p_rel, cam_pt, state);

        if(!state.behind_camera)
        {
            float2 d_img                = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float3 distorted_ray        = apply_bivariate_distortion(cam_pt, biv_params);
            FThetaProjectState replayed = state;
            replayed.ray_norm           = normalize3(distorted_ray);

            float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
            ftheta_project_ray_bwd(distorted_ray, params, replayed, d_img, d_distorted_ray, d_params);
            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(cam_pt, biv_params, d_distorted_ray, d_cam_pt, d_biv);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &rx, &ry, &rz, &rw
            );
            float4 rot_mid_xyzw   = make_float4(rx, ry, rz, rw);
            float4 d_rot_mid_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel        = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot_mid_xyzw, p_rel, d_cam_pt, d_rot_mid_xyzw, d_p_rel);

            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_mean_t  = scale3(d_p_rel, -1.0f);
            d_trans0.x      += 0.5f * d_mean_t.x;
            d_trans0.y      += 0.5f * d_mean_t.y;
            d_trans0.z      += 0.5f * d_mean_t.z;
            d_trans1.x      += 0.5f * d_mean_t.x;
            d_trans1.y      += 0.5f * d_mean_t.y;
            d_trans1.z      += 0.5f * d_mean_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                rot0.x,
                rot0.y,
                rot0.z,
                rot0.w,
                rot1.x,
                rot1.y,
                rot1.z,
                rot1.w,
                0.5f,
                rx,
                ry,
                rz,
                rw,
                d_rot_mid_xyzw.x,
                d_rot_mid_xyzw.y,
                d_rot_mid_xyzw.z,
                d_rot_mid_xyzw.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
            float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
            d_rot0.x    += d_r0.x;
            d_rot0.y    += d_r0.y;
            d_rot0.z    += d_r0.z;
            d_rot0.w    += d_r0.w;
            d_rot1.x    += d_r1.x;
            d_rot1.y    += d_r1.y;
            d_rot1.z    += d_r1.z;
            d_rot1.w    += d_r1.w;
        }
        else
        {
            if(grad_world_points != nullptr)
            {
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D5 backward -- image_points_to_world_rays_static_pose_ftheta_no_external
//
// Backward: FTheta backproject image points -> world rays with static pose.
// d_origin passes directly to d_translation; d_direction chains through
// quat_rotate_bwd to d_rotation and d_camera_ray, then through
// ftheta_backproject_image_point_bwd to d_image_point + intrinsic grads.
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ translation,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};
    float4 d_rot_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans    = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, idx * 8, state);
        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);

        d_trans = d_origin;

        float4 pose_r_xyzw = read_quat_xyzw_from_wxyz(rotation, 0);
        float3 camera_ray  = normalize3(state.ray_raw);
        if(state.min2d_clamped)
        {
            camera_ray = make_float3(0.0f, 0.0f, 1.0f);
        }
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(pose_r_xyzw, camera_ray, d_direction, d_rot_xyzw, d_camera_ray);

        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_camera_ray, d_img, d_params);
        if(grad_image_points != nullptr)
        {
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

    if(threadIdx.x == 0)
    {
        if(grad_translation != nullptr)
        {
            atomicAdd(&grad_translation[0], tx);
            atomicAdd(&grad_translation[1], ty);
            atomicAdd(&grad_translation[2], tz);
        }
        if(grad_rotation != nullptr)
        {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D5 (bivariate) backward
//
// Same as D5 with bivariate windshield distortion between the FTheta
// backprojection output and the world-ray direction. Scratch carries the
// pre-normalized bivariate output in slots [8..10].
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ translation,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx                          = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, true);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans    = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        int64_t off = idx * 12;
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, off, state);
        float3 unnorm_out = make_float3(scratch[off + 8], scratch[off + 9], scratch[off + 10]);
        float2 img        = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);

        d_trans = d_origin;

        float4 pose_r_xyzw  = read_quat_xyzw_from_wxyz(rotation, 0);
        float3 camera_ray   = normalize3(unnorm_out);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(pose_r_xyzw, camera_ray, d_direction, d_rot_xyzw, d_camera_ray);
        float3 d_unnorm_out      = normalize3_bwd(unnorm_out, d_camera_ray);
        float3 ftheta_ray_norm   = normalize3(state.ray_raw);
        float3 d_ftheta_ray_norm = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(ftheta_ray_norm, biv_params, d_unnorm_out, d_ftheta_ray_norm, d_biv);

        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_ftheta_ray_norm, d_img, d_params);
        if(grad_image_points != nullptr)
        {
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

    if(threadIdx.x == 0)
    {
        if(grad_translation != nullptr)
        {
            atomicAdd(&grad_translation[0], tx);
            atomicAdd(&grad_translation[1], ty);
            atomicAdd(&grad_translation[2], tz);
        }
        if(grad_rotation != nullptr)
        {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}

// =============================================================================
// D6 backward -- project_world_points_shutter_pose_ftheta_no_external
//
// Backward: project world points -> image points with per-ray rolling-shutter
// alpha (no external distortion). Differentiates a single step at the saved
// alpha; the forward alpha-solve loop is not replayed. d_pose_t splits
// (1-alpha)/(alpha) between start/end translation.
// =============================================================================

__global__ void project_world_points_shutter_pose_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)max_iterations;
    (void)initial_relative_time;
    (void)shutter_type;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        bool valid = valid_flags[idx];
        if(valid)
        {
            int64_t off = idx * 11;
            float3 p_rel;
            float3 cam_pt;
            FThetaProjectState state;
            ftheta_load_meanpose_10(scratch, off, p_rel, cam_pt, state);
            float alpha = scratch[off + 10];

            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            FThetaProjectState replayed = state;
            replayed.ray_norm           = normalize3(cam_pt);
            float3 d_cam_pt             = make_float3(0.0f, 0.0f, 0.0f);
            ftheta_project_ray_bwd(cam_pt, params, replayed, d_img, d_cam_pt, d_params);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &rx, &ry, &rz, &rw
            );
            float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_alpha    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel        = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot_alpha_xyzw, p_rel, d_cam_pt, d_rot_alpha, d_p_rel);

            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_pose_t  = scale3(d_p_rel, -1.0f);
            d_trans0.x      += (1.0f - alpha) * d_pose_t.x;
            d_trans0.y      += (1.0f - alpha) * d_pose_t.y;
            d_trans0.z      += (1.0f - alpha) * d_pose_t.z;
            d_trans1.x      += alpha * d_pose_t.x;
            d_trans1.y      += alpha * d_pose_t.y;
            d_trans1.z      += alpha * d_pose_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                rot0.x,
                rot0.y,
                rot0.z,
                rot0.w,
                rot1.x,
                rot1.y,
                rot1.z,
                rot1.w,
                alpha,
                rx,
                ry,
                rz,
                rw,
                d_rot_alpha.x,
                d_rot_alpha.y,
                d_rot_alpha.z,
                d_rot_alpha.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
            float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
            d_rot0.x    += d_r0.x;
            d_rot0.y    += d_r0.y;
            d_rot0.z    += d_r0.z;
            d_rot0.w    += d_r0.w;
            d_rot1.x    += d_r1.x;
            d_rot1.y    += d_r1.y;
            d_rot1.z    += d_r1.z;
            d_rot1.w    += d_r1.w;
        }
        else
        {
            if(grad_world_points != nullptr)
            {
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D6 (bivariate) backward
//
// Same as D6 with bivariate windshield distortion inserted between
// cam_pt and the FTheta projection.
// =============================================================================

__global__ void project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)max_iterations;
    (void)initial_relative_time;
    (void)shutter_type;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, false);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        bool valid = valid_flags[idx];
        if(valid)
        {
            int64_t off = idx * 11;
            float3 p_rel;
            float3 cam_pt;
            FThetaProjectState state;
            ftheta_load_meanpose_10(scratch, off, p_rel, cam_pt, state);
            float alpha = scratch[off + 10];

            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            float3 distorted_ray        = apply_bivariate_distortion(cam_pt, biv_params);
            FThetaProjectState replayed = state;
            replayed.ray_norm           = normalize3(distorted_ray);
            float3 d_distorted_ray      = make_float3(0.0f, 0.0f, 0.0f);
            ftheta_project_ray_bwd(distorted_ray, params, replayed, d_img, d_distorted_ray, d_params);

            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(cam_pt, biv_params, d_distorted_ray, d_cam_pt, d_biv);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &rx, &ry, &rz, &rw
            );
            float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_alpha    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel        = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot_alpha_xyzw, p_rel, d_cam_pt, d_rot_alpha, d_p_rel);

            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_pose_t  = scale3(d_p_rel, -1.0f);
            d_trans0.x      += (1.0f - alpha) * d_pose_t.x;
            d_trans0.y      += (1.0f - alpha) * d_pose_t.y;
            d_trans0.z      += (1.0f - alpha) * d_pose_t.z;
            d_trans1.x      += alpha * d_pose_t.x;
            d_trans1.y      += alpha * d_pose_t.y;
            d_trans1.z      += alpha * d_pose_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                rot0.x,
                rot0.y,
                rot0.z,
                rot0.w,
                rot1.x,
                rot1.y,
                rot1.z,
                rot1.w,
                alpha,
                rx,
                ry,
                rz,
                rw,
                d_rot_alpha.x,
                d_rot_alpha.y,
                d_rot_alpha.z,
                d_rot_alpha.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
            float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
            d_rot0.x    += d_r0.x;
            d_rot0.y    += d_r0.y;
            d_rot0.z    += d_r0.z;
            d_rot0.w    += d_r0.w;
            d_rot1.x    += d_r1.x;
            d_rot1.y    += d_r1.y;
            d_rot1.z    += d_r1.z;
            d_rot1.w    += d_r1.w;
        }
        else
        {
            if(grad_world_points != nullptr)
            {
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, false, grad_distortion_coeffs);
}

// =============================================================================
// D7 backward -- image_points_to_world_rays_shutter_pose_ftheta_no_external
//
// Backward: FTheta backproject image points -> world rays with rolling-shutter
// pose (no external distortion). d_origin splits (1-alpha)/(alpha) to
// start/end translation; d_direction chains through quat_rotate_bwd and
// quat_slerp_pair_bwd into start/end rotation, then through
// ftheta_backproject_image_point_bwd to d_image_point + intrinsic grads.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    const float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)shutter_type;
    FThetaParams params = load_ftheta_params(projection);
    FThetaParamGrads d_params{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        int64_t off = idx * 9;
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, off, state);
        float alpha = scratch[off + 8];

        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);

        d_trans0.x += (1.0f - alpha) * d_origin.x;
        d_trans0.y += (1.0f - alpha) * d_origin.y;
        d_trans0.z += (1.0f - alpha) * d_origin.z;
        d_trans1.x += alpha * d_origin.x;
        d_trans1.y += alpha * d_origin.y;
        d_trans1.z += alpha * d_origin.z;

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &rx, &ry, &rz, &rw
        );
        float4 pose_r_xyzw = make_float4(rx, ry, rz, rw);
        float3 camera_ray  = normalize3(state.ray_raw);
        if(state.min2d_clamped)
        {
            camera_ray = make_float3(0.0f, 0.0f, 1.0f);
        }
        float4 d_pose_r     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(pose_r_xyzw, camera_ray, d_direction, d_pose_r, d_camera_ray);

        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
        gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
            rot0.x,
            rot0.y,
            rot0.z,
            rot0.w,
            rot1.x,
            rot1.y,
            rot1.z,
            rot1.w,
            alpha,
            rx,
            ry,
            rz,
            rw,
            d_pose_r.x,
            d_pose_r.y,
            d_pose_r.z,
            d_pose_r.w,
            &gq0x,
            &gq0y,
            &gq0z,
            &gq0w,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w
        );
        float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
        float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
        d_rot0.x    += d_r0.x;
        d_rot0.y    += d_r0.y;
        d_rot0.z    += d_r0.z;
        d_rot0.w    += d_r0.w;
        d_rot1.x    += d_r1.x;
        d_rot1.y    += d_r1.y;
        d_rot1.z    += d_r1.z;
        d_rot1.w    += d_r1.w;

        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_camera_ray, d_img, d_params);
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
}

// =============================================================================
// D7 (bivariate) backward
//
// Same as D7 with bivariate windshield distortion between the FTheta
// backprojection output and the world-ray direction.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)shutter_type;
    FThetaParams params                  = load_ftheta_params(projection);
    BivariateWindshieldParams biv_params = load_bivariate_windshield_params(distortion, true);
    FThetaParamGrads d_params{};
    BivariateParamGrads d_biv{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        int64_t off = idx * 12;
        FThetaBackprojectState state;
        ftheta_load_bp_state_8(scratch, off, state);
        float3 unnorm_out = make_float3(scratch[off + 8], scratch[off + 9], scratch[off + 10]);
        float alpha       = scratch[off + 11];

        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);

        d_trans0.x += (1.0f - alpha) * d_origin.x;
        d_trans0.y += (1.0f - alpha) * d_origin.y;
        d_trans0.z += (1.0f - alpha) * d_origin.z;
        d_trans1.x += alpha * d_origin.x;
        d_trans1.y += alpha * d_origin.y;
        d_trans1.z += alpha * d_origin.z;

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &rx, &ry, &rz, &rw
        );
        float4 pose_r_xyzw  = make_float4(rx, ry, rz, rw);
        float3 camera_ray   = normalize3(unnorm_out);
        float4 d_pose_r     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(pose_r_xyzw, camera_ray, d_direction, d_pose_r, d_camera_ray);

        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
        gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
            rot0.x,
            rot0.y,
            rot0.z,
            rot0.w,
            rot1.x,
            rot1.y,
            rot1.z,
            rot1.w,
            alpha,
            rx,
            ry,
            rz,
            rw,
            d_pose_r.x,
            d_pose_r.y,
            d_pose_r.z,
            d_pose_r.w,
            &gq0x,
            &gq0y,
            &gq0z,
            &gq0w,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w
        );
        float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
        float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
        d_rot0.x    += d_r0.x;
        d_rot0.y    += d_r0.y;
        d_rot0.z    += d_r0.z;
        d_rot0.w    += d_r0.w;
        d_rot1.x    += d_r1.x;
        d_rot1.y    += d_r1.y;
        d_rot1.z    += d_r1.z;
        d_rot1.w    += d_r1.w;

        float3 d_unnorm_out      = normalize3_bwd(unnorm_out, d_camera_ray);
        float3 ftheta_ray_norm   = normalize3(state.ray_raw);
        float3 d_ftheta_ray_norm = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(ftheta_ray_norm, biv_params, d_unnorm_out, d_ftheta_ray_norm, d_biv);

        float2 d_img = make_float2(0.0f, 0.0f);
        ftheta_backproject_image_point_bwd(img, params, state, d_ftheta_ray_norm, d_img, d_params);
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
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

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    reduce_ftheta_bivariate_grads(d_biv, distortion, true, grad_distortion_coeffs);
}
} // namespace

// =============================================================================
// Backward host launchers (12 total).
// =============================================================================

void camera_rays_to_image_points_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_no_external_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_no_external_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void camera_rays_to_image_points_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_no_external_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_no_external_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        image_points,
        translation,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *valid_flags,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_no_external_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        world_points,
        start_rotation,
        end_rotation,
        shutter_type,
        max_iterations,
        initial_relative_time,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *valid_flags,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        world_points,
        start_rotation,
        end_rotation,
        shutter_type,
        max_iterations,
        initial_relative_time,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        image_points,
        start_rotation,
        end_rotation,
        shutter_type,
        grad_world_rays,
        grad_image_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        image_points,
        start_rotation,
        end_rotation,
        shutter_type,
        grad_world_rays,
        grad_image_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_principal_point,
        grad_fw_poly,
        grad_bw_poly,
        grad_A,
        grad_Ainv,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
