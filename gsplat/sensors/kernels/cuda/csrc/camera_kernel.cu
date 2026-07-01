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

// Forward CUDA kernels for OpenCV pinhole camera models. 13 kernel entry
// points (generate_image_points + 12 differentiable forward kernels covering
// pinhole x {no_external, bivariate_windshield} x
// {camera_rays_to_image_points, image_points_to_camera_rays,
// project_world_points_mean_pose, project_world_points_shutter_pose,
// image_points_to_world_rays_static_pose,
// image_points_to_world_rays_shutter_pose}). Backwards live in
// camera_kernel_backward.cu.

#include "projection_forward_impl.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

// Returns a 1-D grid large enough to cover `count` threads at kThreads/block.
dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// ===========================================================================
// Kernel 1 — generate_image_points (non-differentiable utility).
// Fills image_points[idx] = (x + 0.5, y + 0.5) for idx = y*width + x.
// No scratch; no backward.
// ===========================================================================

__global__ void generate_image_points_kernel(int64_t width, int64_t height, float *__restrict__ image_points)
{
    int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t count = width * height;
    if(idx >= count)
    {
        return;
    }
    int64_t x                 = idx % width;
    int64_t y                 = idx / width;
    image_points[idx * 2 + 0] = static_cast<float>(x) + 0.5f;
    image_points[idx * 2 + 1] = static_cast<float>(y) + 0.5f;
}

// ===========================================================================
// Kernel 2 — camera_rays_to_image_points (no-external / pinhole only).
// Projects normalised camera-space rays through the OpenCV pinhole+distortion
// model to produce pixel image_points and a boolean valid mask.
//
// Scratch layout per-thread (stride 6, written only when scratch != nullptr):
//   [0] x     (normalised x = ray.x / ray.z after perspective divide)
//   [1] y     (normalised y = ray.y / ray.z)
//   [2] inv_z (1 / ray.z; reused by the backward divide-chain)
//   [3] r2    (r^2 = x^2 + y^2)
//   [4] icD   (combined radial factor num/den)
//   [5] den   (distortion denominator before inversion)
// ===========================================================================

__global__ void camera_rays_to_image_points_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    camera_rays_to_image_points_forward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count, projection, NoExternalDistortion_KernelParameters{}, camera_rays, image_points, valid_flags, scratch
    );
}

// ===========================================================================
// Kernel 3 — image_points_to_camera_rays (no-external / pinhole only).
// Inverts the pinhole+distortion model: pixel → normalised camera-space ray.
// Uses the early-exit fixed-point inverse strategy.
//
// Scratch layout per-thread (stride 5, written only when scratch != nullptr):
//   [0] xy.x   (undistorted normalised x)
//   [1] xy.y   (undistorted normalised y)
//   [2] d.r2   (r^2 = xy.x^2 + xy.y^2)
//   [3] d.icD  (combined radial factor num/den)
//   [4] den    (distortion_den(r2, params))
// ===========================================================================

__global__ void image_points_to_camera_rays_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    image_points_to_camera_rays_forward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count, projection, NoExternalDistortion_KernelParameters{}, image_points, camera_rays, scratch
    );
}

// ===========================================================================
// Kernel 4 — project_world_points_mean_pose (no-external / pinhole only).
// Projects 3-D world points to pixel coordinates using a single pose that is
// the LERP/SLERP midpoint (alpha=0.5) of start and end poses. Params and the
// interpolated pose are precomputed once in shared memory by thread 0.
//
// Scratch layout per-thread (stride 9, written only when scratch != nullptr):
//   [0]  p_rel.x    (world_point - pose_t, camera-frame offset x)
//   [1]  p_rel.y
//   [2]  p_rel.z
//   [3]  cam_pt.x   (p_rel rotated into camera space)
//   [4]  cam_pt.y
//   [5]  cam_pt.z
//   [6]  r2
//   [7]  icD
//   [8]  den
// ===========================================================================

__global__ void project_world_points_mean_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_forward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
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
}

// ===========================================================================
// Kernel 5 — pinhole-local project_world_points_shutter_pose body.
// Both external-distortion variants use this body. Each iteration reprojects
// the current image point to find its rolling-shutter timestamp, updates the
// interpolated pose, and re-projects until convergence or max_iterations.
// Global-shutter mode terminates after a single iteration.
//
// Scratch layout per-thread (stride 10, written only when scratch != nullptr):
//   [0]  p_rel.x    (final iteration: world_point - pose_t)
//   [1]  p_rel.y
//   [2]  p_rel.z
//   [3]  cam_pt.x   (p_rel rotated into camera space)
//   [4]  cam_pt.y
//   [5]  cam_pt.z
//   [6]  r2
//   [7]  icD
//   [8]  den
//   [9]  alpha      (final relative_time / interpolation parameter)
// ===========================================================================

template<typename DistortionPolicy>
inline __device__ void project_world_points_shutter_pose_pinhole_forward_impl(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
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
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ProjectWorldPointsShutterPose;
    using ForwardScratch             = DistortionScratchTraits<
        DistortionSensor::OpenCVPinhole,
        kOp,
        DistortionDirection::Forward,
        typename DistortionPolicy::Tag
    >;
    using BackwardScratch = DistortionScratchTraits<
        DistortionSensor::OpenCVPinhole,
        kOp,
        DistortionDirection::Backward,
        typename DistortionPolicy::Tag
    >;
    static_assert(ForwardScratch::kScratchStride == BackwardScratch::kScratchStride);
    static_assert(ForwardScratch::kInverseStashOffset == BackwardScratch::kInverseStashOffset);
    static_assert(ForwardScratch::kIsUndistort == BackwardScratch::kIsUndistort);
    OpenCVPinholeScratchIO<kOp>::validate<ForwardScratch>();
    OpenCVPinholeScratchIO<kOp>::validate<BackwardScratch>();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    const OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, ForwardScratch::kIsUndistort);
    const float3 world_point       = read_vec3(world_points, idx);
    const float3 start_t           = read_vec3(start_translation, 0);
    const float3 end_t             = read_vec3(end_translation, 0);
    const float4 start_r           = read_quat_xyzw_from_wxyz(start_rotation, 0);
    const float4 end_r             = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float relative_time            = initial_relative_time;
    float alpha                    = 0.0f;
    float2 image_point             = make_float2(0.0f, 0.0f);
    float2 previous_image_point    = make_float2(0.0f, 0.0f);
    float3 pose_translation        = start_t;
    float4 pose_rotation_xyzw      = start_r;
    float3 p_rel                   = make_float3(0.0f, 0.0f, 0.0f);
    float3 camera_point            = make_float3(0.0f, 0.0f, 0.0f);
    float compact_project_state[6] = {};
    bool valid                     = false;
    for(int64_t i = 0; i < max_iterations; ++i)
    {
        alpha            = relative_time;
        pose_translation = lerp3(start_t, end_t, alpha);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            start_r.x, start_r.y, start_r.z, start_r.w, end_r.x, end_r.y, end_r.z, end_r.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_rotation_xyzw = make_float4(qx, qy, qz, qw);
        p_rel              = sub3(world_point, pose_translation);
        camera_point       = quat_inverse_rotate_xyzw_geom(pose_rotation_xyzw, p_rel);
        if(camera_point.z <= 0.0f)
        {
            valid = false;
            break;
        }
        const float3 camera_ray    = normalize3(camera_point);
        const float3 projected_ray = DistortionPolicy::apply_fwd(camera_ray, distortion_params);
        image_point                = opencv_pinhole_project_compact(
            projected_ray, params, scratch != nullptr ? compact_project_state : nullptr, valid
        );
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL || !valid)
        {
            break;
        }
        const float next_relative_time
            = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);
        const float2 delta
            = make_float2(image_point.x - previous_image_point.x, image_point.y - previous_image_point.y);
        const float pixel_error = sqrtf(delta.x * delta.x + delta.y * delta.y);
        if(i > 0 && pixel_error < stop_delta_mean_error_px)
        {
            break;
        }
        const float time_delta = fabsf(next_relative_time - relative_time);
        const float approximate_pixel_error
            = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if(approximate_pixel_error < stop_mean_error_px)
        {
            break;
        }
        previous_image_point = image_point;
        relative_time        = next_relative_time;
    }
    image_points[idx * 2 + 0] = image_point.x;
    image_points[idx * 2 + 1] = image_point.y;
    valid_flags[idx]          = valid;
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
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
        OpenCVPinholeProjectState state{};
        state.r2  = compact_project_state[3];
        state.icD = compact_project_state[4];
        state.den = compact_project_state[5];
        OpenCVPinholeScratchIO<kOp>::save_forward<ForwardScratch>(
            scratch, idx * ForwardScratch::kScratchStride, p_rel, camera_point, state, alpha
        );
    }
}

__global__ void project_world_points_shutter_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_pinhole_forward_impl<NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
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
}

// ===========================================================================
// Kernel 6 — image_points_to_world_rays_static_pose (no-external / pinhole).
// Inverts the pinhole+distortion model and rotates the resulting camera ray
// into world space using a single static pose (no interpolation).
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 5, written only when scratch != nullptr):
//   [0] xy.x   (undistorted normalised x)
//   [1] xy.y
//   [2] d.r2
//   [3] d.icD
//   [4] distortion_den(r2, params)
// ===========================================================================

__global__ void image_points_to_world_rays_static_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_static_pose_forward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
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

// ===========================================================================
// Kernel 7 — image_points_to_world_rays_shutter_pose (no-external / pinhole).
// Computes the rolling-shutter relative time from each pixel's row/column,
// LERP/SLERPs the pose at that time, then inverts the pinhole model to emit a
// world ray. Non-iterative: single pose evaluation per pixel.
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 9, written only when scratch != nullptr):
//   [0] xy.x          (undistorted normalised x)
//   [1] xy.y
//   [2] d.r2
//   [3] d.icD
//   [4] distortion_den(r2, params)
//   [5] relative_time  (alpha used for pose interpolation)
//   [6] 0.0f           (unused; backward re-derives SLERP state)
//   [7] 1.0f           (unused)
//   [8] 0.0f           (unused)
// ===========================================================================

__global__ void image_points_to_world_rays_shutter_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_forward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
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
}

// ===========================================================================
// Kernel 8 — camera_rays_to_image_points (bivariate_windshield).
// Applies the bivariate windshield polynomial to the input camera ray, then
// projects the distorted ray through the OpenCV pinhole model.
//
// Scratch layout per-thread (stride 10, written only when scratch != nullptr):
//   [0] distorted_ray.x   (output of apply_bivariate_distortion)
//   [1] distorted_ray.y
//   [2] distorted_ray.z
//   [3] x
//   [4] y
//   [5] inv_z
//   [6] r2
//   [7] icD
//   [8] den
//   [9] front-face mask: 1.0 if distorted_ray.z > 0, else 0.0
// ===========================================================================

__global__ void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    camera_rays_to_image_points_forward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
        count, projection, distortion, camera_rays, image_points, valid_flags, scratch
    );
}

// ===========================================================================
// Kernel 9 — image_points_to_camera_rays (bivariate_windshield).
// Uses the fixed-ten intrinsic inverse followed by the inverse bivariate
// polynomial.
//
// Scratch layout per-thread (stride 9):
//   [0] xy.x
//   [1] xy.y
//   [2] d.r2
//   [3] d.icD
//   [4] den
//   [5] camera_ray_pre_norm.x
//   [6] camera_ray_pre_norm.y
//   [7] camera_ray_pre_norm.z
//   [8] 0.0f (unused, zeroed explicitly to make stride apparent)
// ===========================================================================

__global__ void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    image_points_to_camera_rays_forward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
        count, projection, distortion, image_points, camera_rays, scratch
    );
}

// ===========================================================================
// Kernel 10 — project_world_points_mean_pose (bivariate_windshield).
// Identical control flow to Kernel 4 but inserts apply_bivariate_distortion
// between the world→camera rotation and the OpenCV projection step.
// Params, bivariate params, and the midpoint pose are broadcast via shared
// memory by thread 0 exactly as in Kernel 4.
//
// Scratch layout per-thread (stride 9, matching Kernel 4 exactly):
//   [0]  p_rel.x
//   [1]  p_rel.y
//   [2]  p_rel.z
//   [3]  cam_pt.x
//   [4]  cam_pt.y
//   [5]  cam_pt.z
//   [6]  r2
//   [7]  icD
//   [8]  den
// ===========================================================================

__global__ void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_forward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
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

// ===========================================================================
// Kernel 11 — project_world_points_shutter_pose (bivariate_windshield).
// Identical iterative rolling-shutter loop to Kernel 5 but applies the
// bivariate windshield distortion to the normalised camera ray before the
// OpenCV pinhole projection on each iteration.
//
// Scratch layout per-thread (stride 10, matching Kernel 5 exactly):
//   [0]  p_rel.x
//   [1]  p_rel.y
//   [2]  p_rel.z
//   [3]  cam_pt.x
//   [4]  cam_pt.y
//   [5]  cam_pt.z
//   [6]  r2
//   [7]  icD
//   [8]  den
//   [9]  alpha  (final relative_time)
// ===========================================================================

__global__ void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_pinhole_forward_impl<BivariateWindshieldPolicy>(
        count,
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
        scratch
    );
}

// ===========================================================================
// Kernel 12 — image_points_to_world_rays_static_pose (bivariate_windshield).
// Inverts the intrinsic and bivariate models, then rotates the resulting
// camera ray into world space with a single static pose.
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 9):
//   [0] xy.x
//   [1] xy.y
//   [2] d.r2
//   [3] d.icD
//   [4] den
//   [5] camera_ray_pre_norm.x
//   [6] camera_ray_pre_norm.y
//   [7] camera_ray_pre_norm.z
//   [8] 0.0f (unused, zeroed explicitly to make stride apparent)
// ===========================================================================

__global__ void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_static_pose_forward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
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

// ===========================================================================
// Kernel 13 — image_points_to_world_rays_shutter_pose (bivariate_windshield).
// Computes the rolling-shutter pose via LERP/SLERP at the per-pixel relative
// time, inverts the intrinsic and bivariate models, and rotates into world
// space.
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 12):
//   [0] xy.x
//   [1] xy.y
//   [2] d.r2
//   [3] d.icD
//   [4] den
//   [5] camera_ray_pre_norm.x
//   [6] camera_ray_pre_norm.y
//   [7] camera_ray_pre_norm.z
//   [8] relative_time  (alpha for pose interpolation)
//   [9]  0.0f           (unused; backward re-derives SLERP state)
//   [10] 1.0f           (unused)
//   [11] 0.0f           (unused)
// ===========================================================================

__global__ void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_forward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
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
} // namespace

// ===========================================================================
// Host launch wrappers — one per kernel above. Each wrapper guards count <= 0,
// computes the 1-D grid, and forwards all tensor pointers to its kernel.
// ===========================================================================

// Launches generate_image_points_kernel; image_points: [height, width, 2] f32
// (the kernel writes a contiguous row-major buffer; equivalent flat layout is
// [height*width, 2]).
void generate_image_points_launch(int64_t width, int64_t height, float *image_points, cudaStream_t stream)
{
    int64_t count = width * height;
    if(count <= 0)
    {
        return;
    }
    generate_image_points_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(width, height, image_points);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 2; camera_rays: [N,3] f32, image_points: [N,2] f32, scratch: [N,6] f32 or nullptr.
void camera_rays_to_image_points_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    camera_rays_to_image_points_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, camera_rays, image_points, valid_flags, scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 3; image_points: [N,2] f32, camera_rays: [N,3] f32, scratch: [N,5] f32 or nullptr.
void image_points_to_camera_rays_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_camera_rays_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, image_points, camera_rays, scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 4; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,9] f32 or nullptr.
void project_world_points_mean_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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

// Launches Kernel 5; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void project_world_points_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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

// Launches Kernel 7; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_world_rays_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
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
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 6; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,5] f32 or nullptr.
void image_points_to_world_rays_static_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *image_points,
    const float *translations,
    const float *rotations,
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
    image_points_to_world_rays_static_pose_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 8; camera_rays: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 9; image_points: [N,2] f32, camera_rays: [N,3] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count, projection, distortion, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 10; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,9] f32 or nullptr.
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<
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

// Launches Kernel 11; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<
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

// Launches Kernel 12; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translations,
    const float *rotations,
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
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 13; image_points: [N,2] f32 or nullptr, world_rays: [N,6] f32, scratch: [N,12] f32 or nullptr.
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<
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
