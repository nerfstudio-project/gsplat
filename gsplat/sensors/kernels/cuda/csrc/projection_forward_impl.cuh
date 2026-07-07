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

#pragma once

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"

#include <type_traits>

template<typename ProjectionPolicy, DistortionOpFamily Op, typename DistortionPolicy>
struct ProjectionScratchContract
{
    using Forward = DistortionScratchTraits<
        ProjectionPolicy::kScratchSensor,
        Op,
        DistortionDirection::Forward,
        typename DistortionPolicy::Tag
    >;
    using Backward = DistortionScratchTraits<
        ProjectionPolicy::kScratchSensor,
        Op,
        DistortionDirection::Backward,
        typename DistortionPolicy::Tag
    >;

    static_assert(Forward::kScratchStride == Backward::kScratchStride);
    static_assert(Forward::kInverseStashOffset == Backward::kInverseStashOffset);
    static_assert(Forward::kIsUndistort == Backward::kIsUndistort);

    static __device__ void validate()
    {
        ProjectionPolicy::template ScratchIO<Op>::template validate<Forward>();
        ProjectionPolicy::template ScratchIO<Op>::template validate<Backward>();
    }
};

template<int BlockThreads>
constexpr __device__ __forceinline__ void validate_projection_block_size()
{
    static_assert(BlockThreads > 0 && BlockThreads <= 1024);
    static_assert(BlockThreads % 32 == 0);
}

template<typename ProjectionPolicy, typename DistortionPolicy>
constexpr bool kSupportsGeneratedShutterImagePoints
    = ProjectionPolicy::kScratchSensor == DistortionSensor::OpenCVPinhole
   && std::is_same_v<typename DistortionPolicy::Tag, BivariateWindshieldPolicyTag>;

template<typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ float2
    load_shutter_image_point(const float *__restrict__ image_points, int64_t idx, int64_t width)
{
    if constexpr(kSupportsGeneratedShutterImagePoints<ProjectionPolicy, DistortionPolicy>)
    {
        if(image_points == nullptr)
        {
            const int64_t y = idx / width;
            const int64_t x = idx - y * width;
            return make_float2(0.5f + static_cast<float>(x), 0.5f + static_cast<float>(y));
        }
    }
    return make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void camera_rays_to_image_points_forward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    constexpr DistortionOpFamily kOp = DistortionOpFamily::CameraRaysToImagePoints;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Forward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }

    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    const float3 ray           = read_vec3(camera_rays, idx);
    const float3 projected_ray = DistortionPolicy::apply_fwd(ray, distortion_params);
    typename ProjectionPolicy::ProjectState state;
    bool projection_valid    = false;
    const float2 image_point = ProjectionPolicy::project(projected_ray, params, state, projection_valid);

    image_points[idx * 2 + 0] = image_point.x;
    image_points[idx * 2 + 1] = image_point.y;
    valid_flags[idx]          = ProjectionPolicy::final_project_valid(image_point, params, projection_valid);
    if(scratch != nullptr)
    {
        ProjectionPolicy::template ScratchIO<kOp>::template save_forward<Scratch>(
            scratch, idx * Scratch::kScratchStride, state
        );
    }
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_camera_rays_forward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToCameraRays;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Forward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }

    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    const float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    typename ProjectionPolicy::BackprojectState state;
    const float3 projected_ray = ProjectionPolicy::backproject(image_point, params, state);
    const int64_t off          = idx * Scratch::kScratchStride;
    const float3 camera_ray    = DistortionPolicy::apply_inverse(
        projected_ray, distortion_params, scratch, off + Scratch::kInverseStashOffset
    );
    write_vec3(camera_rays, idx, camera_ray);
    if(scratch != nullptr)
    {
        ProjectionPolicy::finalize_backproject_state_for_scratch(params, state);
        ProjectionPolicy::template ScratchIO<kOp>::template save_forward<Scratch>(scratch, off, state);
    }
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void project_world_points_mean_pose_forward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
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
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::Params>);
    static_assert(std::is_trivially_copyable_v<typename DistortionPolicy::Params>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ProjectWorldPointsMeanPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Forward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ typename ProjectionPolicy::Params block_params;
    __shared__ typename DistortionPolicy::Params block_distortion_params;
    __shared__ float3 block_pose_translation;
    __shared__ float4 block_pose_rotation_xyzw;
    if(threadIdx.x == 0)
    {
        block_params            = ProjectionPolicy::load(projection);
        block_distortion_params = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
        const float3 start_t    = read_vec3(start_translation, 0);
        const float3 end_t      = read_vec3(end_translation, 0);
        const float4 start_r    = read_quat_xyzw_from_wxyz(start_rotation, 0);
        const float4 end_r      = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_translation  = lerp3(start_t, end_t, 0.5f);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            start_r.x, start_r.y, start_r.z, start_r.w, end_r.x, end_r.y, end_r.z, end_r.w, 0.5f, &qx, &qy, &qz, &qw
        );
        block_pose_rotation_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if(idx >= count)
    {
        return;
    }

    const typename ProjectionPolicy::Params params            = block_params;
    const typename DistortionPolicy::Params distortion_params = block_distortion_params;
    const float3 world_point                                  = read_vec3(world_points, idx);
    const float3 pose_translation                             = block_pose_translation;
    const float4 pose_rotation_xyzw                           = block_pose_rotation_xyzw;
    const float3 p_rel                                        = sub3(world_point, pose_translation);
    const float3 camera_point = quat_inverse_rotate_xyzw_geom(pose_rotation_xyzw, p_rel);
    typename ProjectionPolicy::ProjectState state;
    float2 image_point    = make_float2(0.0f, 0.0f);
    bool projection_valid = false;
    bool rejected         = false;
    if constexpr(ProjectionPolicy::kGatePosePointBeforeDistortion)
    {
        rejected = !(camera_point.z > 0.0f);
    }
    if(rejected)
    {
        ProjectionPolicy::set_rejected_pose_project_state(state);
    }
    else
    {
        float3 camera_ray = camera_point;
        if constexpr(ProjectionPolicy::kNormalizePoseProjectInput)
        {
            camera_ray = normalize3(camera_point);
        }
        const float3 projected_ray = DistortionPolicy::apply_fwd(camera_ray, distortion_params);
        image_point                = ProjectionPolicy::project(projected_ray, params, state, projection_valid);
    }

    image_points[idx * 2 + 0] = image_point.x;
    image_points[idx * 2 + 1] = image_point.y;
    valid_flags[idx]          = ProjectionPolicy::final_project_valid(image_point, params, projection_valid);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = mean_timestamp_us;
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_translation);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_rotation_xyzw);
    }
    if(scratch != nullptr)
    {
        ProjectionPolicy::template ScratchIO<kOp>::template save_forward<Scratch>(
            scratch, idx * Scratch::kScratchStride, p_rel, camera_point, state
        );
    }
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_world_rays_static_pose_forward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
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
    validate_projection_block_size<BlockThreads>();
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToWorldRaysStaticPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Forward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }

    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    const float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    typename ProjectionPolicy::BackprojectState state;
    const float3 projected_ray = ProjectionPolicy::backproject(image_point, params, state);
    // Avoid overlapping lens-state construction with the pose-output live range.
    ProjectionPolicy::finalize_backproject_state_for_scratch(params, state);
    const int64_t off       = idx * Scratch::kScratchStride;
    const float3 camera_ray = DistortionPolicy::apply_inverse(
        projected_ray, distortion_params, scratch, off + Scratch::kInverseStashOffset
    );
    const float3 pose_translation   = read_vec3(translations, 0);
    const float4 pose_rotation_xyzw = read_quat_xyzw_from_wxyz(rotations, 0);
    const float3 direction          = quat_rotate_xyzw_geom(pose_rotation_xyzw, camera_ray);

    world_rays[idx * 6 + 0] = pose_translation.x;
    world_rays[idx * 6 + 1] = pose_translation.y;
    world_rays[idx * 6 + 2] = pose_translation.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    timestamps_us[idx]      = timestamp_us;
    write_vec3(pose_translations, idx, pose_translation);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_rotation_xyzw);
    if(scratch != nullptr)
    {
        ProjectionPolicy::template ScratchIO<kOp>::template save_forward<Scratch>(scratch, off, state);
    }
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_world_rays_shutter_pose_forward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
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
    validate_projection_block_size<BlockThreads>();
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToWorldRaysShutterPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Forward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }

    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    const float2 image_point
        = load_shutter_image_point<ProjectionPolicy, DistortionPolicy>(image_points, idx, projection.width);
    const float alpha
        = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);

    typename ProjectionPolicy::BackprojectState state;
    const float3 projected_ray = ProjectionPolicy::backproject(image_point, params, state);
    const int64_t off          = idx * Scratch::kScratchStride;
    const float3 camera_ray    = DistortionPolicy::apply_inverse(
        projected_ray, distortion_params, scratch, off + Scratch::kInverseStashOffset
    );

    const float3 start_t          = read_vec3(start_translation, 0);
    const float3 end_t            = read_vec3(end_translation, 0);
    const float3 pose_translation = lerp3(start_t, end_t, alpha);
    const float4 start_r          = read_quat_xyzw_from_wxyz(start_rotation, 0);
    const float4 end_r            = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    gsplat_geometry::quat_slerp_pair_fwd<float>(
        start_r.x, start_r.y, start_r.z, start_r.w, end_r.x, end_r.y, end_r.z, end_r.w, alpha, &qx, &qy, &qz, &qw
    );
    const float4 pose_rotation_xyzw = make_float4(qx, qy, qz, qw);
    const float3 direction          = quat_rotate_xyzw_geom(pose_rotation_xyzw, camera_ray);

    world_rays[idx * 6 + 0] = pose_translation.x;
    world_rays[idx * 6 + 1] = pose_translation.y;
    world_rays[idx * 6 + 2] = pose_translation.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(alpha, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_translation);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_rotation_xyzw);
    }
    if(scratch != nullptr)
    {
        ProjectionPolicy::finalize_backproject_state_for_scratch(params, state);
        ProjectionPolicy::template ScratchIO<kOp>::template save_forward<Scratch>(scratch, off, state, alpha);
    }
}
