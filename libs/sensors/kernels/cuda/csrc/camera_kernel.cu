/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Forward CUDA kernels for OpenCV pinhole camera models. 13 kernel entry
// points (generate_image_points + 12 differentiable forward kernels covering
// pinhole x {no_external, bivariate_windshield} x
// {camera_rays_to_image_points, image_points_to_camera_rays,
// project_world_points_mean_pose, project_world_points_shutter_pose,
// image_points_to_world_rays_static_pose,
// image_points_to_world_rays_shutter_pose}). Backwards live in
// camera_kernel_backward.cu.

#include "camera_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kThreads = 256;

// Returns a 1-D grid large enough to cover `count` threads at kThreads/block.
dim3 grid_for_count(int64_t count) {
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// fixed_iterative_undistort — used only by the bivariate_camera_ray helper.
// Runs exactly 10 Newton iterations unconditionally so all threads in a warp
// take the same number of steps (avoids divergence from early-exit heuristic).
__device__ __forceinline__ float2 fixed_iterative_undistort(
    float2 xy0,
    const OpenCVPinholeParams& params) {
    // Same Newton update as iterative_undistort(), but always exactly 10
    // unrolled iterations with no early exit so bivariate forward lanes do not
    // diverge on per-pixel convergence.
    float2 xy = xy0;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        DistortionResult d = compute_distortion(xy, params);
        xy.x = (xy0.x - d.delta_x) / d.icD;
        xy.y = (xy0.y - d.delta_y) / d.icD;
    }
    return xy;
}

// bivariate_camera_ray_from_image_point — shared helper for the three
// bivariate unprojection kernels (image_points_to_camera_rays,
// image_points_to_world_rays_static_pose,
// image_points_to_world_rays_shutter_pose).
// Undistorts the image point, maps to a normalized camera ray via the pinhole
// model, then applies the inverse bivariate windshield polynomial.
//
// Scratch layout per-thread (stride depends on caller; see each kernel):
//   [0] xy.x          (undistorted normalised x)
//   [1] xy.y          (undistorted normalised y)
//   [2] d.r2          (r^2 = xy.x^2 + xy.y^2)
//   [3] d.icD         (combined radial factor num/den)
//   [4] den           (denominator value before inversion)
//   [5] camera_ray_pre_norm.x  (unnormalised bivariate-distorted ray x)
//   [6] camera_ray_pre_norm.y
//   [7] camera_ray_pre_norm.z
__device__ __forceinline__ float3 bivariate_camera_ray_from_image_point(
    const OpenCVPinholeParams& params,
    const BivariateWindshieldParams& bivariate_params,
    float2 image_point,
    float* scratch,
    int64_t scratch_offset) {
    float2 xy0 = make_float2(
        (image_point.x - params.cx) / params.fx,
        (image_point.y - params.cy) / params.fy);
    float2 xy = fixed_iterative_undistort(xy0, params);
    DistortionResult d = compute_distortion(xy, params);
    float den = d.den;
    float3 distorted_ray = normalize3(make_float3(xy.x, xy.y, 1.0f));
    float3 camera_ray_pre_norm = apply_bivariate_distortion(distorted_ray, bivariate_params);
    float3 camera_ray = normalize3(camera_ray_pre_norm);
    if (scratch != nullptr) {
        scratch[scratch_offset + 0] = xy.x;
        scratch[scratch_offset + 1] = xy.y;
        scratch[scratch_offset + 2] = d.r2;
        scratch[scratch_offset + 3] = d.icD;
        scratch[scratch_offset + 4] = den;
        scratch[scratch_offset + 5] = camera_ray_pre_norm.x;
        scratch[scratch_offset + 6] = camera_ray_pre_norm.y;
        scratch[scratch_offset + 7] = camera_ray_pre_norm.z;
    }
    return camera_ray;
}

// ===========================================================================
// Kernel 1 — generate_image_points (non-differentiable utility).
// Fills image_points[idx] = (x + 0.5, y + 0.5) for idx = y*width + x.
// No scratch; no backward.
// ===========================================================================

__global__ void generate_image_points_kernel(int64_t width, int64_t height, float* __restrict__ image_points) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t count = width * height;
    if (idx >= count) {
        return;
    }
    int64_t x = idx % width;
    int64_t y = idx / width;
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
// All 6 slots are written by project_camera_ray_opencv at base idx*6.
// ===========================================================================

__global__ void camera_rays_to_image_points_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ camera_rays,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    float3 ray = read_vec3(camera_rays, idx);
    ProjectionEval eval = project_camera_ray_opencv(params, ray, scratch, idx * 6);
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
}

// ===========================================================================
// Kernel 3 — image_points_to_camera_rays (no-external / pinhole only).
// Inverts the pinhole+distortion model: pixel → normalised camera-space ray.
// Uses iterative_undistort (early-exit Newton, unlike the bivariate variant).
//
// Scratch layout per-thread (stride 5, written only when scratch != nullptr):
//   [0] xy.x   (undistorted normalised x after iterative Newton)
//   [1] xy.y   (undistorted normalised y)
//   [2] d.r2   (r^2 = xy.x^2 + xy.y^2)
//   [3] d.icD  (combined radial factor num/den)
//   [4] den    (distortion_den(r2, params))
// ===========================================================================

__global__ void image_points_to_camera_rays_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    float* __restrict__ camera_rays,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float2 xy0 = make_float2((image_point.x - params.cx) / params.fx, (image_point.y - params.cy) / params.fy);
    float2 xy = iterative_undistort(xy0, params);
    float3 ray = normalize3(make_float3(xy.x, xy.y, 1.0f));
    write_vec3(camera_rays, idx, ray);
    if (scratch != nullptr) {
        DistortionResult d = compute_distortion(xy, params);
        int64_t off = idx * 5;
        scratch[off + 0] = xy.x;
        scratch[off + 1] = xy.y;
        scratch[off + 2] = d.r2;
        scratch[off + 3] = d.icD;
        scratch[off + 4] = distortion_den(d.r2, params);
    }
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
//   [6]  r2         (from project_camera_ray_opencv, ray_scratch[3])
//   [7]  icD        (ray_scratch[4])
//   [8]  den        (ray_scratch[5])
// ===========================================================================

__global__ void project_world_points_mean_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ world_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t mean_timestamp_us,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ OpenCVPinholeParams block_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if (threadIdx.x == 0) {
        block_params = load_opencv_pinhole_params(projection);
        float3 trans0 = read_vec3(start_translation, 0);
        float3 trans1 = read_vec3(end_translation, 0);
        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            0.5f, &qx, &qy, &qz, &qw);
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = block_params;
    float3 world_point = read_vec3(world_points, idx);
    float3 pose_t = block_pose_t;
    float4 pose_r_xyzw = block_pose_r_xyzw;
    float3 p_rel = sub3(world_point, pose_t);
    float3 cam_pt = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    ProjectionEval eval = {make_float2(0.0f, 0.0f), false};
    float ray_scratch[6] = {};
    if (cam_pt.z > 0.0f) {
        float3 camera_ray = normalize3(cam_pt);
        eval = project_camera_ray_opencv(params, camera_ray, scratch != nullptr ? ray_scratch : nullptr, 0);
    }
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = mean_timestamp_us;
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        int64_t off = idx * 9;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = ray_scratch[3];
        scratch[off + 7] = ray_scratch[4];
        scratch[off + 8] = ray_scratch[5];
    }
}

// ===========================================================================
// Kernel 5 — project_world_points_shutter_pose (no-external / pinhole only).
// Projects 3-D world points iteratively: each iteration reprojects the current
// image point to find its rolling-shutter timestamp, updates the interpolated
// pose, and re-projects until pixel error converges or max_iterations reached.
// Global-shutter mode terminates after a single iteration.
//
// Scratch layout per-thread (stride 10, written only when scratch != nullptr):
//   [0]  p_rel.x    (final iteration: world_point - pose_t)
//   [1]  p_rel.y
//   [2]  p_rel.z
//   [3]  cam_pt.x   (p_rel rotated into camera space)
//   [4]  cam_pt.y
//   [5]  cam_pt.z
//   [6]  r2         (ray_scratch[3] from project_camera_ray_opencv)
//   [7]  icD        (ray_scratch[4])
//   [8]  den        (ray_scratch[5])
//   [9]  alpha      (final relative_time / interpolation parameter)
// ===========================================================================

__global__ void project_world_points_shutter_pose_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ world_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    float3 world_point = read_vec3(world_points, idx);
    float3 trans0 = read_vec3(start_translation, 0);
    float3 trans1 = read_vec3(end_translation, 0);
    float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    int64_t iterations = max_iterations;
    float relative_time = initial_relative_time;
    float alpha = 0.0f;
    float2 previous_image_point = make_float2(0.0f, 0.0f);
    float3 pose_t = trans0;
    float4 pose_r_xyzw = rot0;
    float3 p_rel = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt = make_float3(0.0f, 0.0f, 0.0f);
    ProjectionEval eval = {make_float2(0.0f, 0.0f), false};
    float ray_scratch[6] = {};
    for (int64_t i = 0; i < iterations; ++i) {
        alpha = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        {
            float qx, qy, qz, qw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &qx, &qy, &qz, &qw);
            pose_r_xyzw = make_float4(qx, qy, qz, qw);
        }
        p_rel = sub3(world_point, pose_t);
        cam_pt = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
        if (cam_pt.z <= 0.0f) {
            // Preserve the last successful image_point; only flip valid.
            eval.valid = false;
            break;
        }
        float3 camera_ray = normalize3(cam_pt);
        eval = project_camera_ray_opencv(params, camera_ray, scratch != nullptr ? ray_scratch : nullptr, 0);
        if (shutter_type == static_cast<int64_t>(gsplat_sensors::ShutterType::GLOBAL) || !eval.valid) {
            break;
        }
        float next_relative_time = compute_relative_frame_time_opencv(eval.image_point, projection.width, projection.height, shutter_type);
        float2 delta_px = make_float2(eval.image_point.x - previous_image_point.x, eval.image_point.y - previous_image_point.y);
        float pixel_error = sqrtf(delta_px.x * delta_px.x + delta_px.y * delta_px.y);
        if (i > 0 && pixel_error < stop_delta_mean_error_px) {
            break;
        }
        float time_delta = fabsf(next_relative_time - relative_time);
        float approximate_pixel_error = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if (approximate_pixel_error < stop_mean_error_px) {
            break;
        }
        previous_image_point = eval.image_point;
        relative_time = next_relative_time;
    }
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        int64_t off = idx * 10;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = ray_scratch[3];
        scratch[off + 7] = ray_scratch[4];
        scratch[off + 8] = ray_scratch[5];
        scratch[off + 9] = alpha;
    }
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
    const float* __restrict__ image_points,
    const float* __restrict__ translations,
    const float* __restrict__ rotations,
    int64_t timestamp_us,
    float* __restrict__ world_rays,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float2 xy0 = make_float2((image_point.x - params.cx) / params.fx, (image_point.y - params.cy) / params.fy);
    float2 xy = iterative_undistort(xy0, params);
    DistortionResult d = compute_distortion(xy, params);
    float3 camera_ray = normalize3(make_float3(xy.x, xy.y, 1.0f));
    float3 pose_t = read_vec3(translations, 0);
    float4 pose_r_xyzw = read_quat_xyzw_from_wxyz(rotations, 0);
    float3 origin = pose_t;
    float3 direction = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    timestamps_us[idx] = timestamp_us;
    write_vec3(pose_translations, idx, pose_t);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    if (scratch != nullptr) {
        int64_t off = idx * 5;
        scratch[off + 0] = xy.x;
        scratch[off + 1] = xy.y;
        scratch[off + 2] = d.r2;
        scratch[off + 3] = d.icD;
        scratch[off + 4] = distortion_den(d.r2, params);
    }
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
    const float* __restrict__ image_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* __restrict__ world_rays,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float relative_time = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);
    float2 xy0 = make_float2((image_point.x - params.cx) / params.fx, (image_point.y - params.cy) / params.fy);
    float2 xy = iterative_undistort(xy0, params);
    float3 camera_ray = normalize3(make_float3(xy.x, xy.y, 1.0f));
    float3 pose_t = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), relative_time);
    float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    trajectory_cuda::quat_slerp_pair_fwd_f(
        rot0.x, rot0.y, rot0.z, rot0.w,
        rot1.x, rot1.y, rot1.z, rot1.w,
        relative_time, &qx, &qy, &qz, &qw);
    float4 pose_r_xyzw = make_float4(qx, qy, qz, qw);
    float3 origin = pose_t;
    float3 direction = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        DistortionResult d = compute_distortion(xy, params);
        int64_t off = idx * 9;
        scratch[off + 0] = xy.x;
        scratch[off + 1] = xy.y;
        scratch[off + 2] = d.r2;
        scratch[off + 3] = d.icD;
        scratch[off + 4] = distortion_den(d.r2, params);
        scratch[off + 5] = relative_time;
        scratch[off + 6] = 0.0f;
        scratch[off + 7] = 1.0f;
        scratch[off + 8] = 0.0f;
    }
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
//   [3] x     (normalised x inside project_camera_ray_opencv, ray_scratch[0])
//   [4] y     (ray_scratch[1])
//   [5] inv_z (ray_scratch[2])
//   [6] r2    (ray_scratch[3])
//   [7] icD   (ray_scratch[4])
//   [8] den   (ray_scratch[5])
//   [9] front-face mask: 1.0 if distorted_ray.z > 0, else 0.0
// ===========================================================================

__global__ void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ camera_rays,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 ray = read_vec3(camera_rays, idx);
    float3 distorted_ray = apply_bivariate_distortion(ray, bivariate_params);
    float ray_scratch[6] = {};
    ProjectionEval eval = project_camera_ray_opencv(
        params,
        distorted_ray,
        scratch != nullptr ? ray_scratch : nullptr,
        0);
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
    if (scratch != nullptr) {
        int64_t off = idx * 10;
        scratch[off + 0] = distorted_ray.x;
        scratch[off + 1] = distorted_ray.y;
        scratch[off + 2] = distorted_ray.z;
        scratch[off + 3] = ray_scratch[0];
        scratch[off + 4] = ray_scratch[1];
        scratch[off + 5] = ray_scratch[2];
        scratch[off + 6] = ray_scratch[3];
        scratch[off + 7] = ray_scratch[4];
        scratch[off + 8] = ray_scratch[5];
        scratch[off + 9] = distorted_ray.z > 0.0f ? 1.0f : 0.0f;
    }
}

// ===========================================================================
// Kernel 9 — image_points_to_camera_rays (bivariate_windshield).
// Delegates to bivariate_camera_ray_from_image_point which runs fixed 10-iter
// Newton undistortion followed by inverse bivariate polynomial.
//
// Scratch layout per-thread (stride 9, via bivariate_camera_ray helper):
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
    const float* __restrict__ image_points,
    float* __restrict__ camera_rays,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float3 camera_ray = bivariate_camera_ray_from_image_point(
        params,
        bivariate_params,
        image_point,
        scratch,
        idx * 9);
    write_vec3(camera_rays, idx, camera_ray);
    if (scratch != nullptr) {
        // K9 stride is fixed at 9 floats by the wrapper-side scratch allocation.
        // Slot 8 is unused by the K9 backward; zero it to make the layout explicit.
        scratch[idx * 9 + 8] = 0.0f;
    }
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
//   [6]  r2    (ray_scratch[3] from project_camera_ray_opencv on distorted ray)
//   [7]  icD   (ray_scratch[4])
//   [8]  den   (ray_scratch[5])
// ===========================================================================

__global__ void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ world_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t mean_timestamp_us,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ OpenCVPinholeParams block_params;
    __shared__ BivariateWindshieldParams block_bivariate_params;
    __shared__ float3 block_pose_t;
    __shared__ float4 block_pose_r_xyzw;
    if (threadIdx.x == 0) {
        block_params = load_opencv_pinhole_params(projection);
        block_bivariate_params = load_bivariate_windshield_params(distortion, false);
        float3 trans0 = read_vec3(start_translation, 0);
        float3 trans1 = read_vec3(end_translation, 0);
        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        block_pose_t = lerp3(trans0, trans1, 0.5f);
        float qx, qy, qz, qw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            0.5f, &qx, &qy, &qz, &qw);
        block_pose_r_xyzw = make_float4(qx, qy, qz, qw);
    }
    __syncthreads();
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = block_params;
    BivariateWindshieldParams bivariate_params = block_bivariate_params;
    float3 world_point = read_vec3(world_points, idx);
    float3 pose_t = block_pose_t;
    float4 pose_r_xyzw = block_pose_r_xyzw;
    float3 p_rel = sub3(world_point, pose_t);
    float3 cam_pt = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
    ProjectionEval eval = {make_float2(0.0f, 0.0f), false};
    float ray_scratch[6] = {};
    if (cam_pt.z > 0.0f) {
        float3 camera_ray = normalize3(cam_pt);
        float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
        eval = project_camera_ray_opencv(params, distorted_ray, scratch != nullptr ? ray_scratch : nullptr, 0);
    }
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = mean_timestamp_us;
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        int64_t off = idx * 9;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = ray_scratch[3];
        scratch[off + 7] = ray_scratch[4];
        scratch[off + 8] = ray_scratch[5];
    }
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
//   [6]  r2    (ray_scratch[3] on distorted ray)
//   [7]  icD   (ray_scratch[4])
//   [8]  den   (ray_scratch[5])
//   [9]  alpha  (final relative_time)
// ===========================================================================

__global__ void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ world_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* __restrict__ image_points,
    bool* __restrict__ valid_flags,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    float3 world_point = read_vec3(world_points, idx);
    float3 trans0 = read_vec3(start_translation, 0);
    float3 trans1 = read_vec3(end_translation, 0);
    float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float relative_time = initial_relative_time;
    float alpha = 0.0f;
    float2 previous_image_point = make_float2(0.0f, 0.0f);
    float3 pose_t = trans0;
    float4 pose_r_xyzw = rot0;
    float3 p_rel = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt = make_float3(0.0f, 0.0f, 0.0f);
    ProjectionEval eval = {make_float2(0.0f, 0.0f), false};
    float ray_scratch[6] = {};
    for (int64_t i = 0; i < max_iterations; ++i) {
        alpha = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        {
            float qx, qy, qz, qw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &qx, &qy, &qz, &qw);
            pose_r_xyzw = make_float4(qx, qy, qz, qw);
        }
        p_rel = sub3(world_point, pose_t);
        cam_pt = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
        if (cam_pt.z <= 0.0f) {
            // Preserve the last successful image_point; only flip valid.
            eval.valid = false;
            break;
        }
        float3 camera_ray = normalize3(cam_pt);
        float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
        eval = project_camera_ray_opencv(params, distorted_ray, scratch != nullptr ? ray_scratch : nullptr, 0);
        if (shutter_type == static_cast<int64_t>(gsplat_sensors::ShutterType::GLOBAL) || !eval.valid) {
            break;
        }
        float next_relative_time = compute_relative_frame_time_opencv(eval.image_point, projection.width, projection.height, shutter_type);
        float2 delta_px = make_float2(eval.image_point.x - previous_image_point.x, eval.image_point.y - previous_image_point.y);
        float pixel_error = sqrtf(delta_px.x * delta_px.x + delta_px.y * delta_px.y);
        if (i > 0 && pixel_error < stop_delta_mean_error_px) {
            break;
        }
        float time_delta = fabsf(next_relative_time - relative_time);
        float approximate_pixel_error = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if (approximate_pixel_error < stop_mean_error_px) {
            break;
        }
        previous_image_point = eval.image_point;
        relative_time = next_relative_time;
    }
    image_points[idx * 2 + 0] = eval.image_point.x;
    image_points[idx * 2 + 1] = eval.image_point.y;
    valid_flags[idx] = eval.valid;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        int64_t off = idx * 10;
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = cam_pt.x;
        scratch[off + 4] = cam_pt.y;
        scratch[off + 5] = cam_pt.z;
        scratch[off + 6] = ray_scratch[3];
        scratch[off + 7] = ray_scratch[4];
        scratch[off + 8] = ray_scratch[5];
        scratch[off + 9] = alpha;
    }
}

// ===========================================================================
// Kernel 12 — image_points_to_world_rays_static_pose (bivariate_windshield).
// Delegates undistortion to bivariate_camera_ray_from_image_point, then
// rotates the resulting camera ray into world space with a single static pose.
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 9, via bivariate_camera_ray helper):
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
    const float* __restrict__ image_points,
    const float* __restrict__ translations,
    const float* __restrict__ rotations,
    int64_t timestamp_us,
    float* __restrict__ world_rays,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    float3 camera_ray = bivariate_camera_ray_from_image_point(
        params,
        bivariate_params,
        image_point,
        scratch,
        idx * 9);
    float3 pose_t = read_vec3(translations, 0);
    float4 pose_r_xyzw = read_quat_xyzw_from_wxyz(rotations, 0);
    float3 origin = pose_t;
    float3 direction = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    timestamps_us[idx] = timestamp_us;
    write_vec3(pose_translations, idx, pose_t);
    write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    if (scratch != nullptr) {
        // K12 stride is fixed at 9 floats by the wrapper-side scratch allocation.
        // Slot 8 is unused by the K12 backward; zero it to make the layout explicit.
        scratch[idx * 9 + 8] = 0.0f;
    }
}

// ===========================================================================
// Kernel 13 — image_points_to_world_rays_shutter_pose (bivariate_windshield).
// Computes the rolling-shutter pose via LERP/SLERP at the per-pixel relative
// time, then calls bivariate_camera_ray_from_image_point and rotates into
// world space. Accepts nullptr image_points (generates pixel centres from idx).
// Output world_rays layout: [origin.xyz, direction.xyz] per ray (stride 6).
//
// Scratch layout per-thread (stride 12, via bivariate_camera_ray helper +
// trailing slots written directly):
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
    const float* __restrict__ image_points,
    const float* __restrict__ start_translation,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_translation,
    const float* __restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* __restrict__ world_rays,
    int64_t* __restrict__ timestamps_us,
    float* __restrict__ pose_translations,
    float* __restrict__ pose_rotations,
    float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    float2 image_point;
    if (image_points != nullptr) {
        image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
    } else {
        // Generated-elements path: regenerate pixel coords from (idx % width, idx / width).
        int64_t res_x = projection.width;
        int64_t y_idx = idx / res_x;
        int64_t x_idx = idx - y_idx * res_x;
        image_point = make_float2(0.5f + static_cast<float>(x_idx), 0.5f + static_cast<float>(y_idx));
    }
    float relative_time = compute_relative_frame_time_opencv(image_point, projection.width, projection.height, shutter_type);
    float3 pose_t = lerp3(read_vec3(start_translation, 0), read_vec3(end_translation, 0), relative_time);
    float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float qx, qy, qz, qw;
    trajectory_cuda::quat_slerp_pair_fwd_f(
        rot0.x, rot0.y, rot0.z, rot0.w,
        rot1.x, rot1.y, rot1.z, rot1.w,
        relative_time, &qx, &qy, &qz, &qw);
    float4 pose_r_xyzw = make_float4(qx, qy, qz, qw);
    float3 camera_ray = bivariate_camera_ray_from_image_point(
        params,
        bivariate_params,
        image_point,
        scratch,
        idx * 12);
    float3 origin = pose_t;
    float3 direction = quat_rotate_xyzw_geom(pose_r_xyzw, camera_ray);
    world_rays[idx * 6 + 0] = origin.x;
    world_rays[idx * 6 + 1] = origin.y;
    world_rays[idx * 6 + 2] = origin.z;
    world_rays[idx * 6 + 3] = direction.x;
    world_rays[idx * 6 + 4] = direction.y;
    world_rays[idx * 6 + 5] = direction.z;
    if (timestamps_us != nullptr) {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if (pose_translations != nullptr) {
        write_vec3(pose_translations, idx, pose_t);
    }
    if (pose_rotations != nullptr) {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if (scratch != nullptr) {
        int64_t off = idx * 12;
        scratch[off + 8] = relative_time;
        scratch[off + 9] = 0.0f;
        scratch[off + 10] = 1.0f;
        scratch[off + 11] = 0.0f;
    }
}

} // namespace

// ===========================================================================
// Host launch wrappers — one per kernel above. Each wrapper guards count <= 0,
// computes the 1-D grid, and forwards all tensor pointers to its kernel.
// ===========================================================================

// Launches generate_image_points_kernel; image_points: [height, width, 2] f32
// (the kernel writes a contiguous row-major buffer; equivalent flat layout is
// [height*width, 2]).
void generate_image_points_launch(
    int64_t width,
    int64_t height,
    float* image_points,
    cudaStream_t stream) {
    int64_t count = width * height;
    if (count <= 0) {
        return;
    }
    generate_image_points_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(width, height, image_points);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 2; camera_rays: [N,3] f32, image_points: [N,2] f32, scratch: [N,6] f32 or nullptr.
void camera_rays_to_image_points_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 3; image_points: [N,2] f32, camera_rays: [N,3] f32, scratch: [N,5] f32 or nullptr.
void image_points_to_camera_rays_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 4; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,9] f32 or nullptr.
void project_world_points_mean_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 5; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void project_world_points_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 7; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_world_rays_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 6; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,5] f32 or nullptr.
void image_points_to_world_rays_static_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* translations,
    const float* rotations,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 8; camera_rays: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, distortion, camera_rays, image_points, valid_flags, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 9; image_points: [N,2] f32, camera_rays: [N,3] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count, projection, distortion, image_points, camera_rays, scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 10; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,9] f32 or nullptr.
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 11; world_points: [N,3] f32, image_points: [N,2] f32, scratch: [N,10] f32 or nullptr.
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 12; image_points: [N,2] f32, world_rays: [N,6] f32, scratch: [N,9] f32 or nullptr.
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translations,
    const float* rotations,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches Kernel 13; image_points: [N,2] f32 or nullptr, world_rays: [N,6] f32, scratch: [N,12] f32 or nullptr.
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
