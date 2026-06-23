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

// FTheta forward CUDA kernels: 6 ops x {no_external, bivariate_windshield} =
// 12 kernel entry points. Backwards live in ftheta_kernel_backward.cu and must
// match the per-op scratch strides documented at each kernel below:
//
//   D1  cr -> ip,  no_external                  : 8  floats / row
//   D2  ip -> cr,  no_external                  : 8  floats / row
//   D3  cr -> ip,  bivariate_windshield         : 8  floats / row
//   D3  ip -> cr,  bivariate_windshield         : 12 floats / row
//   D4  mean-pose, no_external                  : 10 floats / row
//   D4  mean-pose, bivariate_windshield         : 10 floats / row
//   D5  static-pose img -> world,  no_external  : 8  floats / row
//   D5  static-pose img -> world,  bivariate    : 12 floats / row
//   D6  shutter-pose, no_external               : 11 floats / row
//   D6  shutter-pose, bivariate_windshield      : 11 floats / row
//   D7  shutter-pose img -> world, no_external  : 9  floats / row
//   D7  shutter-pose img -> world, bivariate    : 12 floats / row

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "ftheta_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

// Returns a 1-D grid large enough to cover `count` threads at kThreads/block.
dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Packs FThetaProjectState into D1/D3 forward kernels' 8-slot scratch layout.
__device__ __forceinline__ void ftheta_save_proj_state_8(
    float *__restrict__ scratch, int64_t off, const FThetaProjectState &state
)
{
    scratch[off + 0] = state.ray_norm.x;
    scratch[off + 1] = state.ray_norm.y;
    scratch[off + 2] = state.ray_norm.z;
    scratch[off + 3] = state.theta;
    scratch[off + 4] = state.r;
    scratch[off + 5] = state.xy_norm;
    scratch[off + 6] = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
    scratch[off + 7] = 0.0f;
}

// Packs FThetaBackprojectState into the 8-slot backproject scratch layout.
// Slot 7 carries the packed min2d_clamped bit so the backward can bail early
// on identity rays.
__device__ __forceinline__ void ftheta_save_bp_state_8(
    float *__restrict__ scratch, int64_t off, const FThetaBackprojectState &state
)
{
    scratch[off + 0] = state.transformed.x;
    scratch[off + 1] = state.transformed.y;
    scratch[off + 2] = state.rdist;
    scratch[off + 3] = state.theta;
    scratch[off + 4] = state.ray_raw.x;
    scratch[off + 5] = state.ray_raw.y;
    scratch[off + 6] = state.ray_raw.z;
    scratch[off + 7] = ftheta_bp_pack_flags(state.min2d_clamped);
}

// Backprojects an image point through the FTheta model then applies the
// bivariate windshield distortion. Returns the normalized camera ray, its
// pre-normalization value (for the backward), and the FTheta backproject state.
__device__ __forceinline__ void ftheta_backproject_bivariate_camera_ray(
    float2 image_point,
    const FThetaParams &params,
    const BivariateWindshieldParams &bivariate_params,
    float3 &camera_ray,
    float3 &unnorm_out,
    FThetaBackprojectState &bp_state
)
{
    const float3 ftheta_ray = ftheta_backproject_image_point(image_point, params, bp_state);
    unnorm_out              = apply_bivariate_distortion(ftheta_ray, bivariate_params);
    camera_ray              = normalize3(unnorm_out);
}

// =============================================================================
// D1 -- camera_rays_to_image_points_ftheta_no_external
// =============================================================================

__global__ void camera_rays_to_image_points_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params = load_ftheta_params(projection);
    float3 ray          = read_vec3(camera_rays, idx);
    FThetaProjectState state;
    bool valid                = false;
    const float2 img          = ftheta_project_ray(ray, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid && ftheta_image_point_in_frame(img, params);
    if(scratch != nullptr)
    {
        ftheta_save_proj_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D2 -- image_points_to_camera_rays_ftheta_no_external
// =============================================================================

__global__ void image_points_to_camera_rays_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params = load_ftheta_params(projection);
    float2 image_point  = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FThetaBackprojectState state;
    const float3 ray = ftheta_backproject_image_point(image_point, params, state);
    write_vec3(camera_rays, idx, ray);
    if(scratch != nullptr)
    {
        ftheta_save_bp_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D3 -- camera_rays_to_image_points_ftheta_bivariate_windshield
// (forward: distort camera ray then project)
// =============================================================================

__global__ void camera_rays_to_image_points_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = load_ftheta_params(projection);
    // External distortion is applied BEFORE the FTheta projection.
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 ray                                 = read_vec3(camera_rays, idx);
    const float3 distorted_ray                 = apply_bivariate_distortion(ray, bivariate_params);
    FThetaProjectState state;
    bool valid                = false;
    const float2 img          = ftheta_project_ray(distorted_ray, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid && ftheta_image_point_in_frame(img, params);
    if(scratch != nullptr)
    {
        // Backward reconstructs distorted_ray from camera_rays; only the
        // FTheta primal needs to be persisted.
        ftheta_save_proj_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D3 (inverse) -- image_points_to_camera_rays_ftheta_bivariate_windshield
// (forward: backproject image point then undistort)
// =============================================================================

__global__ void image_points_to_camera_rays_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = load_ftheta_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FThetaBackprojectState bp_state;
    float3 camera_ray;
    float3 unnorm_out;
    ftheta_backproject_bivariate_camera_ray(image_point, params, bivariate_params, camera_ray, unnorm_out, bp_state);
    write_vec3(camera_rays, idx, camera_ray);
    if(scratch != nullptr)
    {
        int64_t off = idx * 12;
        ftheta_save_bp_state_8(scratch, off, bp_state);
        scratch[off + 8]  = unnorm_out.x;
        scratch[off + 9]  = unnorm_out.y;
        scratch[off + 10] = unnorm_out.z;
        scratch[off + 11] = 0.0f;
    }
}

// =============================================================================
// D4 -- project_world_points_mean_pose_ftheta_no_external
// Mean-pose: thread 0 broadcasts the LERP/SLERP midpoint pose + intrinsics via
// shared memory; each thread then projects its world point through FTheta.
// =============================================================================

__global__ void project_world_points_mean_pose_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t mean_timestamp_us,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ FThetaParams block_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if(threadIdx.x == 0)
    {
        block_params  = load_ftheta_params(projection);
        float3 trans0 = read_vec3(start_translation, 0);
        float3 trans1 = read_vec3(end_translation, 0);
        float4 rot0   = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1   = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t  = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &qx, &qy, &qz, &qw
        );
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if(idx >= count)
    {
        return;
    }
    FThetaParams params = block_params;
    float3 world_point  = read_vec3(world_points, idx);
    float3 pose_t       = block_pose_t;
    float4 pose_r_xyzw  = block_pose_r_xyzw;
    float3 p_rel        = sub3(world_point, pose_t);
    float3 cam_pt       = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    FThetaProjectState state;
    bool valid = false;
    float2 img = make_float2(0.0f, 0.0f);
    if(cam_pt.z > 0.0f)
    {
        // Pass cam_pt unnormalized; ftheta_project_ray normalizes once
        // internally, keeping fwd/bwd numerics consistent under --use_fast_math.
        img = ftheta_project_ray(cam_pt, params, state, valid);
    }
    else
    {
        state.ray_norm      = make_float3(0.0f, 0.0f, 0.0f);
        state.theta         = 0.0f;
        state.r             = 0.0f;
        state.xy_norm       = 0.0f;
        state.behind_camera = true;
        state.angle_clamped = false;
        state.min2d_clamped = false;
    }
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid && ftheta_image_point_in_frame(img, params);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = mean_timestamp_us;
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off      = idx * 10;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.theta;
        scratch[off + 7] = state.r;
        scratch[off + 8] = state.xy_norm;
        scratch[off + 9] = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
    }
}

// =============================================================================
// D4 (bivariate) -- project_world_points_mean_pose_ftheta_bivariate_windshield
// Mean-pose variant with bivariate windshield distortion applied to the
// normalized camera ray before FTheta projection.
// =============================================================================

__global__ void project_world_points_mean_pose_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t mean_timestamp_us,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ FThetaParams block_params;
    __shared__ BivariateWindshieldParams block_bivariate_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if(threadIdx.x == 0)
    {
        block_params           = load_ftheta_params(projection);
        block_bivariate_params = load_bivariate_windshield_params(distortion, false);
        float3 trans0          = read_vec3(start_translation, 0);
        float3 trans1          = read_vec3(end_translation, 0);
        float4 rot0            = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1            = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t           = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &qx, &qy, &qz, &qw
        );
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = block_params;
    BivariateWindshieldParams bivariate_params = block_bivariate_params;
    float3 world_point                         = read_vec3(world_points, idx);
    float3 pose_t                              = block_pose_t;
    float4 pose_r_xyzw                         = block_pose_r_xyzw;
    float3 p_rel                               = sub3(world_point, pose_t);
    float3 cam_pt                              = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    FThetaProjectState state;
    bool valid = false;
    float2 img = make_float2(0.0f, 0.0f);
    if(cam_pt.z > 0.0f)
    {
        const float3 camera_ray    = normalize3(cam_pt);
        const float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
        img                        = ftheta_project_ray(distorted_ray, params, state, valid);
    }
    else
    {
        state.ray_norm      = make_float3(0.0f, 0.0f, 0.0f);
        state.theta         = 0.0f;
        state.r             = 0.0f;
        state.xy_norm       = 0.0f;
        state.behind_camera = true;
        state.angle_clamped = false;
        state.min2d_clamped = false;
    }
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid && ftheta_image_point_in_frame(img, params);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = mean_timestamp_us;
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off      = idx * 10;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.theta;
        scratch[off + 7] = state.r;
        scratch[off + 8] = state.xy_norm;
        scratch[off + 9] = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
    }
}

// =============================================================================
// D5 -- image_points_to_world_rays_static_pose_ftheta_no_external
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ translations,
    const float *__restrict__ rotations,
    int64_t timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params = load_ftheta_params(projection);
    float2 image_point  = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FThetaBackprojectState state;
    const float3 camera_ray = ftheta_backproject_image_point(image_point, params, state);
    float3 pose_t           = read_vec3(translations, 0);
    float4 pose_r_xyzw      = read_quat_xyzw_from_wxyz(rotations, 0);
    float3 origin           = pose_t;
    float3 direction        = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    timestamps_us[idx]      = timestamp_us;
    write_vec3(pose_translations, idx, pose_t);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    if(scratch != nullptr)
    {
        ftheta_save_bp_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D5 (bivariate) -- image_points_to_world_rays_static_pose_ftheta_bivariate_windshield
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ translations,
    const float *__restrict__ rotations,
    int64_t timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = load_ftheta_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FThetaBackprojectState bp_state;
    float3 camera_ray;
    float3 unnorm_out;
    ftheta_backproject_bivariate_camera_ray(image_point, params, bivariate_params, camera_ray, unnorm_out, bp_state);
    float3 pose_t           = read_vec3(translations, 0);
    float4 pose_r_xyzw      = read_quat_xyzw_from_wxyz(rotations, 0);
    float3 origin           = pose_t;
    float3 direction        = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    timestamps_us[idx]      = timestamp_us;
    write_vec3(pose_translations, idx, pose_t);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    if(scratch != nullptr)
    {
        int64_t off = idx * 12;
        ftheta_save_bp_state_8(scratch, off, bp_state);
        scratch[off + 8]  = unnorm_out.x;
        scratch[off + 9]  = unnorm_out.y;
        scratch[off + 10] = unnorm_out.z;
        scratch[off + 11] = 0.0f;
    }
}

// =============================================================================
// D6 -- project_world_points_shutter_pose_ftheta_no_external
// Shutter-pose: fixed-point convergence loop reprojects each world point under
// successive LERP/SLERP poses until the relative-time/pixel error settles.
// Global-shutter mode terminates after a single iteration.
// =============================================================================

__global__ void project_world_points_shutter_pose_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params         = load_ftheta_params(projection);
    float3 world_point          = read_vec3(world_points, idx);
    float3 trans0               = read_vec3(start_translation, 0);
    float3 trans1               = read_vec3(end_translation, 0);
    float4 rot0                 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1                 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float relative_time         = initial_relative_time;
    float alpha                 = 0.0f;
    float2 previous_image_point = make_float2(0.0f, 0.0f);
    float3 pose_t               = trans0;
    float4 pose_r_xyzw          = rot0;
    float3 p_rel                = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt               = make_float3(0.0f, 0.0f, 0.0f);
    FThetaProjectState state;
    state.ray_norm         = make_float3(0.0f, 0.0f, 0.0f);
    state.theta            = 0.0f;
    state.r                = 0.0f;
    state.xy_norm          = 0.0f;
    state.behind_camera    = false;
    state.angle_clamped    = false;
    state.min2d_clamped    = false;
    float2 image_point_out = make_float2(0.0f, 0.0f);
    bool valid_out         = false;
    for(int64_t i = 0; i < max_iterations; ++i)
    {
        alpha  = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_r_xyzw = make_float4(qx, qy, qz, qw);
        p_rel       = sub3(world_point, pose_t);
        cam_pt      = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
        if(cam_pt.z <= 0.0f)
        {
            // Preserve the last successful image_point; only flip valid.
            valid_out           = false;
            state.behind_camera = true;
            break;
        }
        bool fwd_valid = false;
        FThetaProjectState iter_state;
        const float2 img = ftheta_project_ray(cam_pt, params, iter_state, fwd_valid);
        state            = iter_state;
        image_point_out  = img;
        valid_out        = fwd_valid;
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL || !fwd_valid)
        {
            break;
        }
        float next_relative_time
            = compute_relative_frame_time_opencv(img, projection.width, projection.height, shutter_type);
        float2 delta_px   = make_float2(img.x - previous_image_point.x, img.y - previous_image_point.y);
        float pixel_error = sqrtf(delta_px.x * delta_px.x + delta_px.y * delta_px.y);
        if(i > 0 && pixel_error < stop_delta_mean_error_px)
        {
            break;
        }
        float time_delta              = fabsf(next_relative_time - relative_time);
        float approximate_pixel_error = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if(approximate_pixel_error < stop_mean_error_px)
        {
            break;
        }
        previous_image_point = img;
        relative_time        = next_relative_time;
    }
    image_points[idx * 2 + 0] = image_point_out.x;
    image_points[idx * 2 + 1] = image_point_out.y;
    // Apply the bounds + finite check ONLY at the final write so the
    // rolling-shutter loop above can converge inward from a mid-iteration
    // pose that lands a few pixels outside the frame.
    valid_flags[idx]          = valid_out && ftheta_image_point_in_frame(image_point_out, params);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off       = idx * 11;
        scratch[off + 0]  = p_rel.x;
        scratch[off + 1]  = p_rel.y;
        scratch[off + 2]  = p_rel.z;
        scratch[off + 3]  = cam_pt.x;
        scratch[off + 4]  = cam_pt.y;
        scratch[off + 5]  = cam_pt.z;
        scratch[off + 6]  = state.theta;
        scratch[off + 7]  = state.r;
        scratch[off + 8]  = state.xy_norm;
        scratch[off + 9]  = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
        scratch[off + 10] = alpha;
    }
}

// =============================================================================
// D6 (bivariate) -- project_world_points_shutter_pose_ftheta_bivariate_windshield
// Shutter-pose variant with bivariate windshield distortion applied each
// iteration after normalizing cam_pt and before the FTheta projection.
// =============================================================================

__global__ void project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = load_ftheta_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 world_point                         = read_vec3(world_points, idx);
    float3 trans0                              = read_vec3(start_translation, 0);
    float3 trans1                              = read_vec3(end_translation, 0);
    float4 rot0                                = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1                                = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float relative_time                        = initial_relative_time;
    float alpha                                = 0.0f;
    float2 previous_image_point                = make_float2(0.0f, 0.0f);
    float3 pose_t                              = trans0;
    float4 pose_r_xyzw                         = rot0;
    float3 p_rel                               = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt                              = make_float3(0.0f, 0.0f, 0.0f);
    FThetaProjectState state;
    state.ray_norm         = make_float3(0.0f, 0.0f, 0.0f);
    state.theta            = 0.0f;
    state.r                = 0.0f;
    state.xy_norm          = 0.0f;
    state.behind_camera    = false;
    state.angle_clamped    = false;
    state.min2d_clamped    = false;
    float2 image_point_out = make_float2(0.0f, 0.0f);
    bool valid_out         = false;
    for(int64_t i = 0; i < max_iterations; ++i)
    {
        alpha  = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_r_xyzw = make_float4(qx, qy, qz, qw);
        p_rel       = sub3(world_point, pose_t);
        cam_pt      = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
        if(cam_pt.z <= 0.0f)
        {
            valid_out           = false;
            state.behind_camera = true;
            break;
        }
        const float3 camera_ray    = normalize3(cam_pt);
        const float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
        bool fwd_valid             = false;
        FThetaProjectState iter_state;
        const float2 img = ftheta_project_ray(distorted_ray, params, iter_state, fwd_valid);
        state            = iter_state;
        image_point_out  = img;
        valid_out        = fwd_valid;
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL || !fwd_valid)
        {
            break;
        }
        float next_relative_time
            = compute_relative_frame_time_opencv(img, projection.width, projection.height, shutter_type);
        float2 delta_px   = make_float2(img.x - previous_image_point.x, img.y - previous_image_point.y);
        float pixel_error = sqrtf(delta_px.x * delta_px.x + delta_px.y * delta_px.y);
        if(i > 0 && pixel_error < stop_delta_mean_error_px)
        {
            break;
        }
        float time_delta              = fabsf(next_relative_time - relative_time);
        float approximate_pixel_error = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if(approximate_pixel_error < stop_mean_error_px)
        {
            break;
        }
        previous_image_point = img;
        relative_time        = next_relative_time;
    }
    image_points[idx * 2 + 0] = image_point_out.x;
    image_points[idx * 2 + 1] = image_point_out.y;
    // Apply the bounds + finite check ONLY at the final write so the
    // rolling-shutter loop above can converge inward from a mid-iteration
    // pose that lands a few pixels outside the frame.
    valid_flags[idx]          = valid_out && ftheta_image_point_in_frame(image_point_out, params);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off       = idx * 11;
        scratch[off + 0]  = p_rel.x;
        scratch[off + 1]  = p_rel.y;
        scratch[off + 2]  = p_rel.z;
        scratch[off + 3]  = cam_pt.x;
        scratch[off + 4]  = cam_pt.y;
        scratch[off + 5]  = cam_pt.z;
        scratch[off + 6]  = state.theta;
        scratch[off + 7]  = state.r;
        scratch[off + 8]  = state.xy_norm;
        scratch[off + 9]  = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
        scratch[off + 10] = alpha;
    }
}

// =============================================================================
// D7 -- image_points_to_world_rays_shutter_pose_ftheta_no_external
// Image-point shutter: relative_time is derived once from the pixel's row/col,
// then a single LERP/SLERP pose is used to lift the FTheta-backprojected ray
// into world space. No fixed-point iteration.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params = load_ftheta_params(projection);
    float2 image_point  = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float relative_time
        = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);
    FThetaBackprojectState bp_state;
    const float3 camera_ray = ftheta_backproject_image_point(image_point, params, bp_state);
    float3 pose_t           = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), relative_time);
    float4 rot0             = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1             = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    trajectory_cuda::quat_slerp_pair_fwd_f(
        rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, relative_time, &qx, &qy, &qz, &qw
    );
    float4 pose_r_xyzw      = make_float4(qx, qy, qz, qw);
    float3 origin           = pose_t;
    float3 direction        = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off = idx * 9;
        ftheta_save_bp_state_8(scratch, off, bp_state);
        scratch[off + 8] = relative_time;
    }
}

// =============================================================================
// D7 (bivariate) -- image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield
// Image-point shutter variant that applies the inverse bivariate windshield
// distortion after FTheta backprojection.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params                        = load_ftheta_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float relative_time
        = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);
    FThetaBackprojectState bp_state;
    float3 camera_ray;
    float3 unnorm_out;
    ftheta_backproject_bivariate_camera_ray(image_point, params, bivariate_params, camera_ray, unnorm_out, bp_state);
    float3 pose_t = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), relative_time);
    float4 rot0   = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1   = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    trajectory_cuda::quat_slerp_pair_fwd_f(
        rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, relative_time, &qx, &qy, &qz, &qw
    );
    float4 pose_r_xyzw      = make_float4(qx, qy, qz, qw);
    float3 origin           = pose_t;
    float3 direction        = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off = idx * 12;
        ftheta_save_bp_state_8(scratch, off, bp_state);
        scratch[off + 8]  = unnorm_out.x;
        scratch[off + 9]  = unnorm_out.y;
        scratch[off + 10] = unnorm_out.z;
        scratch[off + 11] = relative_time;
    }
}
} // namespace

// =============================================================================
// Forward launchers (one per __global__): early-out on count <= 0, dispatch
// the kernel on `stream`, then check for launch errors.
// =============================================================================

void camera_rays_to_image_points_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *camera_rays,
    float *image_points,
    bool *valid_flags,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_no_external_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, camera_rays, image_points, valid_flags, scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    float *camera_rays,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_no_external_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, image_points, camera_rays, scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void camera_rays_to_image_points_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *camera_rays,
    float *image_points,
    bool *valid_flags,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    float *camera_rays,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t mean_timestamp_us,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_no_external_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        world_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        mean_timestamp_us,
        image_points,
        valid_flags,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t mean_timestamp_us,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        world_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        mean_timestamp_us,
        image_points,
        valid_flags,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *translation,
    const float *rotation,
    int64_t timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        image_points,
        translation,
        rotation,
        timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translation,
    const float *rotation,
    int64_t timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_kernel<<<
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
        timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_no_external_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        world_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        max_iterations,
        stop_mean_error_px,
        stop_delta_mean_error_px,
        initial_relative_time,
        image_points,
        valid_flags,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        world_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        max_iterations,
        stop_mean_error_px,
        stop_delta_mean_error_px,
        initial_relative_time,
        image_points,
        valid_flags,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        image_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        image_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
