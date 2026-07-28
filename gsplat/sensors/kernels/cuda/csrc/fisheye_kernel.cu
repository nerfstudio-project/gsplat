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

// OpenCV-fisheye forward CUDA kernels for the no_external D1/D2/D3/D5 ops. The
// device math primitives live in fisheye_kernel.cuh. Per-op scratch strides
// (the fwd-write == bwd-read contract):
//
//   D1  cr -> ip,    no_external               : 8  floats / row
//   D2  ip -> cr,    no_external               : 8  floats / row
//   D3  world -> ip, mean-pose, no_external    : 14 floats / row
//   D5  ip -> world rays, static-pose          : 8  floats / row

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "fisheye_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Value-cast (NOT bit-cast) flag pack for the fisheye project state.
// Bit 0 behind_camera, bit 1 angle_clamped, bit 2 oob, bit 3 xy_norm_clamped.
__device__ __forceinline__ float fisheye_pack_flags(
    bool behind_camera, bool angle_clamped, bool oob, bool xy_norm_clamped
)
{
    uint32_t f  = 0u;
    f          |= behind_camera ? 1u : 0u;
    f          |= angle_clamped ? 2u : 0u;
    f          |= oob ? 4u : 0u;
    f          |= xy_norm_clamped ? 8u : 0u;
    return static_cast<float>(f);
}

// Backproject min2d flag pack (value-cast).
__device__ __forceinline__ float fisheye_bp_pack_flags(bool min2d_clamped)
{
    return min2d_clamped ? 1.0f : 0.0f;
}

// Packs FisheyeProjectState into the D1 8-slot scratch layout. The project
// state packs theta then delta (opposite to the backproject state).
__device__ __forceinline__ void fisheye_save_proj_state_8(
    float *__restrict__ scratch, int64_t off, const FisheyeProjectState &state
)
{
    scratch[off + 0] = state.ray_xy_norm;
    scratch[off + 1] = state.theta;
    scratch[off + 2] = state.delta;
    scratch[off + 3] = fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
    scratch[off + 4] = 0.0f;
    scratch[off + 5] = 0.0f;
    scratch[off + 6] = 0.0f;
    scratch[off + 7] = 0.0f;
}

// Packs FisheyeBackprojectState into the 8-slot backproject scratch layout. The
// backproject state packs delta then theta (opposite to the project state).
__device__ __forceinline__ void fisheye_save_bp_state_8(
    float *__restrict__ scratch, int64_t off, const FisheyeBackprojectState &state
)
{
    scratch[off + 0] = state.normalized.x;
    scratch[off + 1] = state.normalized.y;
    scratch[off + 2] = state.delta;
    scratch[off + 3] = state.theta;
    scratch[off + 4] = state.ray_raw.x;
    scratch[off + 5] = state.ray_raw.y;
    scratch[off + 6] = state.ray_raw.z;
    scratch[off + 7] = fisheye_bp_pack_flags(state.min2d_clamped);
}

// =============================================================================
// D1 -- camera_rays_to_image_points_opencv_fisheye_no_external
// =============================================================================

__global__ void camera_rays_to_image_points_opencv_fisheye_no_external_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    float3 ray                 = read_vec3(camera_rays, idx);
    FisheyeProjectState state;
    bool valid                = false;
    const float2 img          = fisheye_project_ray(ray, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid;
    if(scratch != nullptr)
    {
        fisheye_save_proj_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D2 -- image_points_to_camera_rays_opencv_fisheye_no_external
// =============================================================================

__global__ void image_points_to_camera_rays_opencv_fisheye_no_external_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    float2 image_point         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FisheyeBackprojectState state;
    const float3 ray = fisheye_backproject_image_point(image_point, params, state);
    write_vec3(camera_rays, idx, ray);
    if(scratch != nullptr)
    {
        fisheye_save_bp_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D3 -- project_world_points_mean_pose_opencv_fisheye_no_external
// Mean-pose: the start/end control poses are LERP/SLERP-interpolated at
// alpha = 0.5 (broadcast through shared memory); each thread then projects its
// world point through the fisheye model on the UNNORMALIZED cam_pt (atan2 path,
// no ray normalize). Scratch stride 14: p_rel[0..2], cam_pt[3..5],
// fstate(ray_xy_norm/theta/delta)[6..8], flags[9], pad[10..13].
// =============================================================================

__global__ void project_world_points_mean_pose_opencv_fisheye_no_external_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    __shared__ OpenCVFisheyeParams block_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if(threadIdx.x == 0)
    {
        block_params  = load_opencv_fisheye_params(projection);
        float3 trans0 = read_vec3(start_translation, 0);
        float3 trans1 = read_vec3(end_translation, 0);
        float4 rot0   = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1   = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t  = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &qx, &qy, &qz, &qw
        );
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if(idx >= count)
    {
        return;
    }
    OpenCVFisheyeParams params = block_params;
    float3 world_point         = read_vec3(world_points, idx);
    float3 pose_t              = block_pose_t;
    float4 pose_r_xyzw         = block_pose_r_xyzw;
    float3 p_rel               = sub3(world_point, pose_t);
    float3 cam_pt              = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    FisheyeProjectState state;
    bool valid                = false;
    const float2 img          = fisheye_project_ray(cam_pt, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid;
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
        int64_t off      = idx * 14;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.ray_xy_norm;
        scratch[off + 7] = state.theta;
        scratch[off + 8] = state.delta;
        scratch[off + 9]
            = fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
    }
}

// =============================================================================
// D5 -- image_points_to_world_rays_static_pose_opencv_fisheye_no_external
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    float2 image_point         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FisheyeBackprojectState state;
    const float3 camera_ray = fisheye_backproject_image_point(image_point, params, state);
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
        fisheye_save_bp_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D1 bivariate -- camera_rays_to_image_points_opencv_fisheye_bivariate_windshield
// Bivariate distortion is applied to the raw camera ray BEFORE the fisheye
// projection. Scratch stride stays 8: the backward reconstructs
// distorted_ray from the saved camera_ray, so no unnorm_out is stashed.
// =============================================================================

__global__ void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params                 = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 ray                                 = read_vec3(camera_rays, idx);
    const float3 distorted_ray                 = apply_bivariate_distortion(ray, bivariate_params);
    FisheyeProjectState state;
    bool valid                = false;
    const float2 img          = fisheye_project_ray(distorted_ray, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid;
    if(scratch != nullptr)
    {
        fisheye_save_proj_state_8(scratch, idx * 8, state);
    }
}

// =============================================================================
// D2 bivariate -- image_points_to_camera_rays_opencv_fisheye_bivariate_windshield
// Backproject (undistort direction) then bivariate distort then normalize3.
// Scratch stride 12: base [0..7] + unnorm_out [8..10] + pad [11].
// =============================================================================

__global__ void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params                 = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FisheyeBackprojectState state;
    const float3 distorted_ray = fisheye_backproject_image_point(image_point, params, state);
    const float3 unnorm_out    = apply_bivariate_distortion(distorted_ray, bivariate_params);
    const float3 ray           = normalize3(unnorm_out);
    write_vec3(camera_rays, idx, ray);
    if(scratch != nullptr)
    {
        int64_t off = idx * 12;
        fisheye_save_bp_state_8(scratch, off, state);
        scratch[off + 8]  = unnorm_out.x;
        scratch[off + 9]  = unnorm_out.y;
        scratch[off + 10] = unnorm_out.z;
        scratch[off + 11] = 0.0f;
    }
}

// =============================================================================
// D3 bivariate -- project_world_points_mean_pose_opencv_fisheye_bivariate_windshield
// Mean-pose with bivariate distortion applied to the UNNORMALIZED cam_pt before
// the fisheye projection (the bivariate normalizes internally). Scratch stride
// stays 14: the backward reconstructs distorted from cam_pt.
// =============================================================================

__global__ void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    __shared__ OpenCVFisheyeParams block_params;
    __shared__ BivariateWindshieldParams block_bivariate_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if(threadIdx.x == 0)
    {
        block_params           = load_opencv_fisheye_params(projection);
        block_bivariate_params = load_bivariate_windshield_params(distortion, false);
        float3 trans0          = read_vec3(start_translation, 0);
        float3 trans1          = read_vec3(end_translation, 0);
        float4 rot0            = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1            = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t           = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, 0.5f, &qx, &qy, &qz, &qw
        );
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if(idx >= count)
    {
        return;
    }
    OpenCVFisheyeParams params                 = block_params;
    BivariateWindshieldParams bivariate_params = block_bivariate_params;
    float3 world_point                         = read_vec3(world_points, idx);
    float3 pose_t                              = block_pose_t;
    float4 pose_r_xyzw                         = block_pose_r_xyzw;
    float3 p_rel                               = sub3(world_point, pose_t);
    float3 cam_pt                              = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    const float3 distorted                     = apply_bivariate_distortion(cam_pt, bivariate_params);
    FisheyeProjectState state;
    bool valid                = false;
    const float2 img          = fisheye_project_ray(distorted, params, state, valid);
    image_points[idx * 2 + 0] = img.x;
    image_points[idx * 2 + 1] = img.y;
    valid_flags[idx]          = valid;
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
        int64_t off      = idx * 14;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.ray_xy_norm;
        scratch[off + 7] = state.theta;
        scratch[off + 8] = state.delta;
        scratch[off + 9]
            = fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
    }
}

// =============================================================================
// D5 bivariate -- image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield
// Backproject (undistort) then bivariate distort then normalize3, then rotate
// into world. Scratch stride 12: base [0..7] + unnorm_out [8..10] + pad [11].
// =============================================================================

__global__ void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params                 = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    FisheyeBackprojectState state;
    const float3 distorted_ray = fisheye_backproject_image_point(image_point, params, state);
    const float3 unnorm_out    = apply_bivariate_distortion(distorted_ray, bivariate_params);
    const float3 camera_ray    = normalize3(unnorm_out);
    float3 pose_t              = read_vec3(translations, 0);
    float4 pose_r_xyzw         = read_quat_xyzw_from_wxyz(rotations, 0);
    float3 origin              = pose_t;
    float3 direction           = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0]    = origin.x;
    world_rays[idx * 6 + 1]    = origin.y;
    world_rays[idx * 6 + 2]    = origin.z;
    world_rays[idx * 6 + 3]    = direction.x;
    world_rays[idx * 6 + 4]    = direction.y;
    world_rays[idx * 6 + 5]    = direction.z;
    timestamps_us[idx]         = timestamp_us;
    write_vec3(pose_translations, idx, pose_t);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    if(scratch != nullptr)
    {
        int64_t off = idx * 12;
        fisheye_save_bp_state_8(scratch, off, state);
        scratch[off + 8]  = unnorm_out.x;
        scratch[off + 9]  = unnorm_out.y;
        scratch[off + 10] = unnorm_out.z;
        scratch[off + 11] = 0.0f;
    }
}

// NaN sentinel written to the D4 alpha slot on convergence failure; the D4
// backward gates gradient flow on !isnan(alpha).
__device__ __forceinline__ float fisheye_alpha_nan_sentinel()
{
    return __int_as_float(0x7FC00000);
}

// =============================================================================
// D4 -- project_world_points_shutter_pose_opencv_fisheye_no_external
// Rolling shutter: non-differentiable fixed-point convergence to the
// shutter time alpha (the converged pose is interpolated by LERP/SLERP between
// the start/end control poses), then the converged primal is projected through
// the fisheye model on the UNNORMALIZED cam_pt (atan2 path, no ray normalize).
// The backward replays ONE differentiable step at the converged alpha. Scratch
// stride 16: p_rel[0..2], cam_pt[3..5], fstate(ray_xy_norm/theta/delta)[6..8],
// flags[9], pad[10..13], alpha[14] (NaN sentinel when !valid), pad[15].
// =============================================================================

// __launch_bounds__(kThreads, 4) caps this heavy shutter-iteration kernel at
// 64 regs to hold 4 blocks/SM on SM 8.0; intentionally not applied to the D4
// bivariate forward, whose higher reg footprint (~89) would spill under the cap.
__global__ void
    __launch_bounds__(kThreads, 4) project_world_points_shutter_pose_opencv_fisheye_no_external_forward_kernel(
        int64_t count,
        OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    float3 world_point         = read_vec3(world_points, idx);
    float3 trans0              = read_vec3(start_translation, 0);
    float3 trans1              = read_vec3(end_translation, 0);
    float4 rot0                = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1                = read_quat_xyzw_from_wxyz(end_rotation, 0);

    float relative_time = initial_relative_time;
    float alpha         = 0.0f;
    float2 cur_img      = make_float2(0.0f, 0.0f);
    float2 prev_img     = make_float2(0.0f, 0.0f);
    bool valid          = false;
    float3 p_rel        = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt       = make_float3(0.0f, 0.0f, 0.0f);
    float3 pose_t       = trans0;
    float4 pose_r_xyzw  = rot0;
    FisheyeProjectState state;
    state.ray_xy_norm     = 0.0f;
    state.theta           = 0.0f;
    state.delta           = 0.0f;
    state.behind_camera   = false;
    state.angle_clamped   = false;
    state.oob             = false;
    state.xy_norm_clamped = false;

    for(int64_t iter = 0; iter < max_iterations; ++iter)
    {
        alpha  = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_r_xyzw = make_float4(qx, qy, qz, qw);
        p_rel       = sub3(world_point, pose_t);
        cam_pt      = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);

        if(cam_pt.z <= 0.0f)
        {
            valid               = false;
            state.behind_camera = true;
            break;
        }

        FisheyeProjectState iter_state;
        bool fwd_valid   = false;
        const float2 img = fisheye_project_ray(cam_pt, params, iter_state, fwd_valid);
        state            = iter_state;
        cur_img          = img;

        if(!fwd_valid)
        {
            valid = false;
            break;
        }
        valid = true;

        const float new_t = compute_relative_frame_time_opencv(img, projection.width, projection.height, shutter_type);
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL)
        {
            break;
        }

        const float2 dpx      = make_float2(img.x - prev_img.x, img.y - prev_img.y);
        const float pixel_err = sqrtf(dpx.x * dpx.x + dpx.y * dpx.y);
        if(iter > 0 && pixel_err < stop_delta_mean_error_px)
        {
            break;
        }
        const float dt         = fabsf(new_t - relative_time);
        const float approx_dpx = dt * static_cast<float>(std::max(projection.width, projection.height));
        if(approx_dpx < stop_mean_error_px)
        {
            break;
        }
        prev_img      = img;
        relative_time = new_t;
    }

    image_points[idx * 2 + 0] = cur_img.x;
    image_points[idx * 2 + 1] = cur_img.y;
    valid_flags[idx]          = valid;
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
        int64_t off      = idx * 16;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.ray_xy_norm;
        scratch[off + 7] = state.theta;
        scratch[off + 8] = state.delta;
        scratch[off + 9]
            = fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
        scratch[off + 14] = valid ? alpha : fisheye_alpha_nan_sentinel();
        scratch[off + 15] = 0.0f;
    }
}

// =============================================================================
// D6 -- image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external
// Image-point shutter: relative_time (== alpha for the start/end control pair)
// is derived once from the pixel row/col, then a single LERP/SLERP pose lifts
// the fisheye-backprojected ray into world space. No fixed-point iteration and
// NO NaN-alpha gate (D6 has no behind-camera invalidation). Scratch
// stride 12: bp state[0..7], alpha[8], unused pad[9..11] (backward reads only
// alpha).
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    float2 image_point         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float alpha = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);

    float3 pose_t = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), alpha);
    float4 rot0   = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1   = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    gsplat_geometry::quat_slerp_pair_fwd<float>(
        rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
    );
    float4 pose_r_xyzw = make_float4(qx, qy, qz, qw);

    FisheyeBackprojectState state;
    const float3 camera_ray = fisheye_backproject_image_point(image_point, params, state);
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
        timestamps_us[idx] = timestamp_from_relative_time(alpha, start_timestamp_us, end_timestamp_us);
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
        fisheye_save_bp_state_8(scratch, off, state);
        // Backward reads only alpha (off+8) from the shutter slots; off+9/off+10
        // and off+11 pad are never read.
        scratch[off + 8] = alpha;
    }
}

// =============================================================================
// D4 bivariate -- project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield
// Rolling shutter as D4 no_external, with the bivariate distortion applied to
// the UNNORMALIZED cam_pt before the fisheye projection each iteration.
// Scratch stride stays 16: slot 14 is alpha, NOT unnorm_out; the
// backward reconstructs distorted from cam_pt.
// =============================================================================

__global__ void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params                 = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 world_point                         = read_vec3(world_points, idx);
    float3 trans0                              = read_vec3(start_translation, 0);
    float3 trans1                              = read_vec3(end_translation, 0);
    float4 rot0                                = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1                                = read_quat_xyzw_from_wxyz(end_rotation, 0);

    float relative_time = initial_relative_time;
    float alpha         = 0.0f;
    float2 cur_img      = make_float2(0.0f, 0.0f);
    float2 prev_img     = make_float2(0.0f, 0.0f);
    bool valid          = false;
    float3 p_rel        = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt       = make_float3(0.0f, 0.0f, 0.0f);
    float3 pose_t       = trans0;
    float4 pose_r_xyzw  = rot0;
    FisheyeProjectState state;
    state.ray_xy_norm     = 0.0f;
    state.theta           = 0.0f;
    state.delta           = 0.0f;
    state.behind_camera   = false;
    state.angle_clamped   = false;
    state.oob             = false;
    state.xy_norm_clamped = false;

    for(int64_t iter = 0; iter < max_iterations; ++iter)
    {
        alpha  = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_r_xyzw = make_float4(qx, qy, qz, qw);
        p_rel       = sub3(world_point, pose_t);
        cam_pt      = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);

        if(cam_pt.z <= 0.0f)
        {
            valid               = false;
            state.behind_camera = true;
            break;
        }

        const float3 distorted = apply_bivariate_distortion(cam_pt, bivariate_params);
        FisheyeProjectState iter_state;
        bool fwd_valid   = false;
        const float2 img = fisheye_project_ray(distorted, params, iter_state, fwd_valid);
        state            = iter_state;
        cur_img          = img;

        if(!fwd_valid)
        {
            valid = false;
            break;
        }
        valid = true;

        const float new_t = compute_relative_frame_time_opencv(img, projection.width, projection.height, shutter_type);
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL)
        {
            break;
        }

        const float2 dpx      = make_float2(img.x - prev_img.x, img.y - prev_img.y);
        const float pixel_err = sqrtf(dpx.x * dpx.x + dpx.y * dpx.y);
        if(iter > 0 && pixel_err < stop_delta_mean_error_px)
        {
            break;
        }
        const float dt         = fabsf(new_t - relative_time);
        const float approx_dpx = dt * static_cast<float>(std::max(projection.width, projection.height));
        if(approx_dpx < stop_mean_error_px)
        {
            break;
        }
        prev_img      = img;
        relative_time = new_t;
    }

    image_points[idx * 2 + 0] = cur_img.x;
    image_points[idx * 2 + 1] = cur_img.y;
    valid_flags[idx]          = valid;
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
        int64_t off      = idx * 16;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = state.ray_xy_norm;
        scratch[off + 7] = state.theta;
        scratch[off + 8] = state.delta;
        scratch[off + 9]
            = fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
        scratch[off + 14] = valid ? alpha : fisheye_alpha_nan_sentinel();
        scratch[off + 15] = 0.0f;
    }
}

// =============================================================================
// D6 bivariate -- image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield
// Image-point shutter as D6 no_external, with the bivariate distortion applied
// after fisheye backprojection and before normalize3. Scratch stride 16:
// bp state[0..7] + alpha[8] + unused pad[9..11] + unnorm_out[12..14] +
// pad[15] (backward reads only alpha and unnorm_out). NO NaN-alpha gate.
// =============================================================================

__global__ void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    OpenCVFisheyeParams params                 = load_opencv_fisheye_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point                         = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float alpha = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);

    float3 pose_t = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), alpha);
    float4 rot0   = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1   = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    gsplat_geometry::quat_slerp_pair_fwd<float>(
        rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
    );
    float4 pose_r_xyzw = make_float4(qx, qy, qz, qw);

    FisheyeBackprojectState state;
    const float3 distorted_ray = fisheye_backproject_image_point(image_point, params, state);
    const float3 unnorm_out    = apply_bivariate_distortion(distorted_ray, bivariate_params);
    const float3 camera_ray    = normalize3(unnorm_out);
    float3 origin              = pose_t;
    float3 direction           = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0]    = origin.x;
    world_rays[idx * 6 + 1]    = origin.y;
    world_rays[idx * 6 + 2]    = origin.z;
    world_rays[idx * 6 + 3]    = direction.x;
    world_rays[idx * 6 + 4]    = direction.y;
    world_rays[idx * 6 + 5]    = direction.z;
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(alpha, start_timestamp_us, end_timestamp_us);
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
        int64_t off = idx * 16;
        fisheye_save_bp_state_8(scratch, off, state);
        // Backward reads alpha (off+8) and unnorm_out (off+12..14); off+9/off+10
        // and the off+11/off+15 pads are never read.
        scratch[off + 8]  = alpha;
        scratch[off + 12] = unnorm_out.x;
        scratch[off + 13] = unnorm_out.y;
        scratch[off + 14] = unnorm_out.z;
    }
}
} // namespace

// =============================================================================
// Forward launchers (one per __global__): early-out on count <= 0, dispatch the
// kernel on `stream`, then check for launch errors.
// =============================================================================

void camera_rays_to_image_points_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    camera_rays_to_image_points_opencv_fisheye_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_camera_rays_opencv_fisheye_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_opencv_fisheye_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external_forward_kernel<<<
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

// =============================================================================
// Bivariate-windshield forward launchers (D1/D2/D3/D5).
// =============================================================================

void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_forward_kernel<<<
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

void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_forward_kernel<<<
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

// =============================================================================
// Shutter-pose forward launchers (D4/D6, both distortions).
// =============================================================================

void project_world_points_shutter_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_opencv_fisheye_no_external_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_forward_kernel<<<
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

void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_forward_kernel<<<
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

void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_forward_kernel<<<
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
