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

// OpenCV-fisheye forward kernels. Scratch layouts are an ABI with the
// hand-written backward kernels; update DistortionScratchTraits and backward
// readers together when slots move.

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "fisheye_kernel.cuh"
#include "projection_forward_impl.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

template<DistortionOpFamily Op, typename PolicyTag>
using FisheyeForwardScratch
    = DistortionScratchTraits<DistortionSensor::OpenCVFisheye, Op, DistortionDirection::Forward, PolicyTag>;

// =============================================================================
// D1 -- camera_rays_to_image_points_opencv_fisheye_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void camera_rays_to_image_points_opencv_fisheye_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    camera_rays_to_image_points_forward_impl<kThreads, OpenCVFisheyeProjectionPolicy, DistortionPolicy>(
        count, projection, distortion, camera_rays, image_points, valid_flags, scratch
    );
}

// =============================================================================
// D2 -- image_points_to_camera_rays_opencv_fisheye_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_camera_rays_opencv_fisheye_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    image_points_to_camera_rays_forward_impl<kThreads, OpenCVFisheyeProjectionPolicy, DistortionPolicy>(
        count, projection, distortion, image_points, camera_rays, scratch
    );
}

// =============================================================================
// D3 -- project_world_points_mean_pose_opencv_fisheye_{policy}
// Mean-pose uses the midpoint control pose for all points; cache it per block
// because every thread shares the same interpolation. The distortion policy
// receives unnormalized cam_pt; bivariate normalizes inside its policy before
// fisheye projection.
// Scratch follows the D3 backward ABI.
// =============================================================================

template<typename DistortionPolicy>
__global__ void project_world_points_mean_pose_opencv_fisheye_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
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
    project_world_points_mean_pose_forward_impl<kThreads, OpenCVFisheyeProjectionPolicy, DistortionPolicy>(
        count,
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
        scratch
    );
}

// =============================================================================
// D5 -- image_points_to_world_rays_static_pose_opencv_fisheye_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_static_pose_opencv_fisheye_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
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
    image_points_to_world_rays_static_pose_forward_impl<kThreads, OpenCVFisheyeProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        translations,
        rotations,
        timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
}

// D4 backward uses isnan(alpha) as its convergence/validity gate.
__device__ __forceinline__ float fisheye_alpha_nan_sentinel()
{
    return __int_as_float(0x7FC00000);
}

// =============================================================================
// D4 -- project_world_points_shutter_pose_opencv_fisheye_no_external
// Rolling-shutter alpha is solved as a non-differentiable fixed point; backward
// replays one differentiable step at the converged alpha. Store NaN alpha when
// invalid so backward can suppress gradients.
// Scratch: p_rel[0..2], cam_pt[3..5], fstate[6..8], flags[9], alpha[14].
// =============================================================================

// Only the no-external D4 kernel uses this launch bound; the bivariate D4
// kernel is left unconstrained.
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
    using Scratch
        = FisheyeForwardScratch<DistortionOpFamily::ProjectWorldPointsShutterPose, NoExternalDistortionPolicyTag>;
    float3 world_point = read_vec3(world_points, idx);
    float3 trans0      = read_vec3(start_translation, 0);
    float3 trans1      = read_vec3(end_translation, 0);
    float4 rot0        = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1        = read_quat_xyzw_from_wxyz(end_rotation, 0);

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
        int64_t off      = idx * Scratch::kScratchStride;
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
            = opencv_fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
        scratch[off + 14] = valid ? alpha : fisheye_alpha_nan_sentinel();
        scratch[off + 15] = 0.0f;
    }
}

// =============================================================================
// D6 -- image_points_to_world_rays_shutter_pose_opencv_fisheye_{policy}
// Image-point shutter alpha comes directly from the pixel, so D6 does not need
// D4's fixed-point solve or NaN-alpha gradient gate.
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_shutter_pose_opencv_fisheye_forward_kernel(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
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
    image_points_to_world_rays_shutter_pose_forward_impl<kThreads, OpenCVFisheyeProjectionPolicy, DistortionPolicy>(
        count,
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
        scratch
    );
}

// =============================================================================
// D4 bivariate -- project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield
// Same fixed-point contract as no-external D4, but distortion happens before
// fisheye projection. Slot 14 remains alpha; backward reconstructs distorted
// coordinates from the stored camera point.
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
    OpenCVFisheyeParams params = load_opencv_fisheye_params(projection);
    using Scratch
        = FisheyeForwardScratch<DistortionOpFamily::ProjectWorldPointsShutterPose, BivariateWindshieldPolicyTag>;
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, Scratch::kIsUndistort);
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
        int64_t off      = idx * Scratch::kScratchStride;
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
            = opencv_fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
        scratch[off + 14] = valid ? alpha : fisheye_alpha_nan_sentinel();
        scratch[off + 15] = 0.0f;
    }
}
} // namespace

// =============================================================================
// Public launchers keep the generated binding surface stable while the kernels
// share policy-templated implementations.
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
    camera_rays_to_image_points_opencv_fisheye_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, NoExternalDistortion_KernelParameters{}, camera_rays, image_points, valid_flags, scratch
        );
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
    image_points_to_camera_rays_opencv_fisheye_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, NoExternalDistortion_KernelParameters{}, image_points, camera_rays, scratch
        );
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
    project_world_points_mean_pose_opencv_fisheye_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
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
    image_points_to_world_rays_static_pose_opencv_fisheye_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            translation,
            rotation,
            timestamp_us,
            world_rays,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Bivariate D1/D2/D3/D5 use the same kernel bodies with a nontrivial policy.
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
    camera_rays_to_image_points_opencv_fisheye_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, distortion, camera_rays, image_points, valid_flags, scratch
        );
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
    image_points_to_camera_rays_opencv_fisheye_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, distortion, image_points, camera_rays, scratch
        );
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
    project_world_points_mean_pose_opencv_fisheye_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
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
            scratch
        );
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
    image_points_to_world_rays_static_pose_opencv_fisheye_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
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
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// D4/D6 stay split by operation because rolling and image-point shutter have
// different backward contracts.
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
    image_points_to_world_rays_shutter_pose_opencv_fisheye_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
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
            scratch
        );
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
    image_points_to_world_rays_shutter_pose_opencv_fisheye_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
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
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
