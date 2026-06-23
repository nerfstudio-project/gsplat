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

// Native CUDA VJPs for OpenCV pinhole camera models, called by the PyTorch
// autograd wrappers for each forward kernel in camera_kernel.cu. Shared
// quaternion VJP adapters live in camera_kernel.cuh and delegate to
// libs/geometry.

#include "camera_kernel.cuh"

#include <c10/cuda/CUDAException.h>

namespace {

// ===========================================================================
// Launch utilities
// ===========================================================================

constexpr int kThreads = 256;

// Returns the 1-D grid required to cover `count` elements at kThreads per block.
dim3 grid_for_count(int64_t count) {
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

struct IntrinsicLocalGrads {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    DistortionParamGrads distortion{};
};

// Block-reduces each IntrinsicLocalGrads slot via block_sum then atomicAdds the
// result from thread 0. Called once per kernel after all per-ray gradient work.
__device__ __forceinline__ void reduce_intrinsic_grads(
    IntrinsicLocalGrads local,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs) {
    float fx = block_sum<kThreads>(local.fx);
    float fy = block_sum<kThreads>(local.fy);
    float cx = block_sum<kThreads>(local.cx);
    float cy = block_sum<kThreads>(local.cy);
    float k0 = block_sum<kThreads>(local.distortion.k[0]);
    float k1 = block_sum<kThreads>(local.distortion.k[1]);
    float k2 = block_sum<kThreads>(local.distortion.k[2]);
    float k3 = block_sum<kThreads>(local.distortion.k[3]);
    float k4 = block_sum<kThreads>(local.distortion.k[4]);
    float k5 = block_sum<kThreads>(local.distortion.k[5]);
    float p0 = block_sum<kThreads>(local.distortion.p[0]);
    float p1 = block_sum<kThreads>(local.distortion.p[1]);
    float s0 = block_sum<kThreads>(local.distortion.s[0]);
    float s1 = block_sum<kThreads>(local.distortion.s[1]);
    float s2 = block_sum<kThreads>(local.distortion.s[2]);
    float s3 = block_sum<kThreads>(local.distortion.s[3]);

    if (threadIdx.x == 0) {
        if (grad_focal_length != nullptr) {
            atomicAdd(&grad_focal_length[0], fx);
            atomicAdd(&grad_focal_length[1], fy);
        }
        if (grad_principal_point != nullptr) {
            atomicAdd(&grad_principal_point[0], cx);
            atomicAdd(&grad_principal_point[1], cy);
        }
        if (grad_radial_coeffs != nullptr) {
            atomicAdd(&grad_radial_coeffs[0], k0);
            atomicAdd(&grad_radial_coeffs[1], k1);
            atomicAdd(&grad_radial_coeffs[2], k2);
            atomicAdd(&grad_radial_coeffs[3], k3);
            atomicAdd(&grad_radial_coeffs[4], k4);
            atomicAdd(&grad_radial_coeffs[5], k5);
        }
        if (grad_tangential_coeffs != nullptr) {
            atomicAdd(&grad_tangential_coeffs[0], p0);
            atomicAdd(&grad_tangential_coeffs[1], p1);
        }
        if (grad_thin_prism_coeffs != nullptr) {
            atomicAdd(&grad_thin_prism_coeffs[0], s0);
            atomicAdd(&grad_thin_prism_coeffs[1], s1);
            atomicAdd(&grad_thin_prism_coeffs[2], s2);
            atomicAdd(&grad_thin_prism_coeffs[3], s3);
        }
    }
}

// Fused block reduction for all BIVARIATE_NUM_DIFF_PARAMS bivariate-coeff
// gradients: single __syncthreads + one warp-shuffle pass per slot, then
// 21 parallel atomicAdds from the first warp. Avoids 42 __syncthreads that
// sequential block_sum calls would require.
__device__ __forceinline__ void reduce_bivariate_grads(
    BivariateParamGrads local,
    BivariateWindshieldDistortion_KernelParameters distortion,
    bool is_undistort,
    float* __restrict__ grad_distortion_coeffs) {
    // Single-pass block reduction across all 21 bivariate-coeff slots:
    // - per-slot warp shuffle (no syncs)
    // - one shared write per (warp, slot)
    // - a single __syncthreads
    // - 21 parallel atomicAdds from lane 0..20 of warp 0
    // (was 21 sequential block_sum calls = 42 __syncthreads.)
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
            uint32_t base = bivariate_coeff_base(distortion.reference_polynomial, is_undistort);
            atomicAdd(&grad_distortion_coeffs[base + lane], total);
        }
    }
}

// ===========================================================================
// Device VJP helpers (called by the __global__ kernels below)
// ===========================================================================

// Backward: pinhole projection (camera ray → image point, no external distortion).
// Consumes forward scratch [x, y, r2, icD, den] already unpacked by the
// caller. Reconstructs the VJP w.r.t. intrinsics (accumulated into `local`) and
// returns d_uv for the caller to chain back to d_camera_ray (via
// camera_ray_from_project_bwd, which uses inv_z).
// ROI-clip branch (icD outside [kMinRadialDist, kMaxRadialDist]) uses the
// fallback formula image_point = uv * res_len / sqrt(r2) + pp.
__device__ __forceinline__ float2 pinhole_project_bwd(
    const OpenCVPinholeParams& params,
    float x,
    float y,
    float r2,
    float icD,
    float den,
    float2 d_image_point,
    IntrinsicLocalGrads& local) {
    local.cx += d_image_point.x;
    local.cy += d_image_point.y;
    float2 d_uv = make_float2(0.0f, 0.0f);

    if (icD < kMinRadialDist || icD > kMaxRadialDist) {
        float res_len = sqrtf(params.res_x * params.res_x + params.res_y * params.res_y);
        float sqrt_r2 = sqrtf(fmaxf(r2, 1.0e-30f));
        float factor = res_len / sqrt_r2;
        d_uv.x += d_image_point.x * factor;
        d_uv.y += d_image_point.y * factor;
        float d_r2 = (d_image_point.x * x + d_image_point.y * y) * (-0.5f * res_len / (sqrt_r2 * r2));
        compute_distortion_bwd(
            x,
            y,
            r2,
            icD,
            den,
            params,
            make_float4(0.0f, 0.0f, 0.0f, d_r2),
            d_uv,
            local.distortion);
        return d_uv;
    }

    float a1 = 2.0f * x * y;
    float a2 = r2 + 2.0f * x * x;
    float a3 = r2 + 2.0f * y * y;
    float delta_x = params.p[0] * a1 + params.p[1] * a2 + r2 * (params.s[0] + r2 * params.s[1]);
    float delta_y = params.p[0] * a3 + params.p[1] * a1 + r2 * (params.s[2] + r2 * params.s[3]);
    float uv_nd_x = x * icD + delta_x;
    float uv_nd_y = y * icD + delta_y;

    local.fx += d_image_point.x * uv_nd_x;
    local.fy += d_image_point.y * uv_nd_y;

    float d_uv_nd_x = d_image_point.x * params.fx;
    float d_uv_nd_y = d_image_point.y * params.fy;
    d_uv.x += d_uv_nd_x * icD;
    d_uv.y += d_uv_nd_y * icD;
    float d_icD = d_uv_nd_x * x + d_uv_nd_y * y;
    compute_distortion_bwd(
        x,
        y,
        r2,
        icD,
        den,
        params,
        make_float4(d_icD, d_uv_nd_x, d_uv_nd_y, 0.0f),
        d_uv,
        local.distortion);
    return d_uv;
}

// Chains d_uv back through uv = (ray.x, ray.y) * inv_z to produce d_camera_ray.
// d_ray_z = -(x*d_uv.x + y*d_uv.y) * inv_z from d(u/z)/dz = -u/z^2.
__device__ __forceinline__ float3 camera_ray_from_project_bwd(
    float x,
    float y,
    float inv_z,
    float2 d_uv) {
    return make_float3(
        d_uv.x * inv_z,
        d_uv.y * inv_z,
        -(x * d_uv.x + y * d_uv.y) * inv_z);
}

// Backward: backproject (image point → camera ray, no external distortion).
// Consumes forward scratch [xs, ys, r2, icD, den]. Chains d_camera_ray through
// normalize3_bwd, then through the Newton-step Jacobian (M = dF/dxy transposed,
// solved 2x2) to obtain d_xy0, from which intrinsic and distortion-param grads
// are accumulated into `local`. Returns d_image_point.
__device__ __forceinline__ float2 backproject_bwd(
    const OpenCVPinholeParams& params,
    float2 image_point,
    float xs,
    float ys,
    float r2,
    float icD,
    float den,
    float3 d_camera_ray,
    IntrinsicLocalGrads& local) {
    float3 d_ray_pre = normalize3_bwd(make_float3(xs, ys, 1.0f), d_camera_ray);
    float2 d_xy_star = make_float2(d_ray_pre.x, d_ray_pre.y);

    float M[2][2];
    compute_dF_dxy(xs, ys, r2, icD, den, params, M);
    float2 d_xy0 = solve_2x2_transposed(M, d_xy_star);

    float2 d_xy_dummy = make_float2(0.0f, 0.0f);
    float d_icD_outer = -(d_xy0.x * xs + d_xy0.y * ys);
    float d_delta_x_outer = -d_xy0.x;
    float d_delta_y_outer = -d_xy0.y;
    compute_distortion_bwd(
        xs,
        ys,
        r2,
        icD,
        den,
        params,
        make_float4(d_icD_outer, d_delta_x_outer, d_delta_y_outer, 0.0f),
        d_xy_dummy,
        local.distortion);

    local.cx += -d_xy0.x / params.fx;
    local.cy += -d_xy0.y / params.fy;
    local.fx += -d_xy0.x * (image_point.x - params.cx) / (params.fx * params.fx);
    local.fy += -d_xy0.y * (image_point.y - params.cy) / (params.fy * params.fy);
    return make_float2(d_xy0.x / params.fx, d_xy0.y / params.fy);
}

// Block-reduces and atomicAdds pose gradients for the dual-pose (shutter/mean)
// kernels. Writes start/end translation and rotation in wxyz output order.
__device__ __forceinline__ void reduce_pose2_grads_components(
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

// Block-reduces and atomicAdds pose gradients for static-pose kernels.
__device__ __forceinline__ void reduce_static_pose_grads(
    float3 d_trans,
    float4 d_rot_xyzw,
    float* __restrict__ grad_translation,
    float* __restrict__ grad_rotation) {
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
        if (grad_rotation != nullptr) {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
}

// ===========================================================================
// K1 backward — camera_rays_to_image_points (no external distortion)
//
// Backward: pinhole project camera rays → image points (no external distortion).
// Consumes forward scratch [x, y, inv_z, r2, icD, den] (stride 6) and
// reconstructs the VJP w.r.t. intrinsics + camera_rays. Threads where
// inv_z <= 0 (ray behind camera) zero their camera_ray gradient but still
// participate in block-reduction shuffles.
// ===========================================================================
__global__ void camera_rays_to_image_points_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ camera_rays,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};

    if (idx < count) {
        int64_t off = idx * 6;
        float x = scratch[off + 0];
        float y = scratch[off + 1];
        float inv_z = scratch[off + 2];
        float r2 = scratch[off + 3];
        float icD = scratch[off + 4];
        float den = scratch[off + 5];
        float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        if (inv_z > 0.0f) {
            float2 d_uv = pinhole_project_bwd(params, x, y, r2, icD, den, d_img, local);
            d_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
        }
        if (grad_camera_rays != nullptr) {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }

    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K2 backward — image_points_to_camera_rays (no external distortion)
//
// Backward: backproject image points → camera rays (no external distortion).
// Consumes forward scratch [xs, ys, r2, icD, den] (stride 5). Chains
// d_camera_ray through normalize3_bwd and the Newton-step Jacobian into
// d_image_point + intrinsic grads via backproject_bwd.
// ===========================================================================
__global__ void image_points_to_camera_rays_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};

    if (idx < count) {
        int64_t off = idx * 5;
        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float3 d_ray = read_vec3(grad_camera_rays, idx);
        float2 d_img = backproject_bwd(
            params,
            img,
            scratch[off + 0],
            scratch[off + 1],
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_ray,
            local);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K3 backward — project_world_points_mean_pose (no external distortion)
//
// Backward: project world points → image points with mean shutter pose
// (slerp alpha = 0.5, no external distortion). Scratch layout (stride 9):
// [0..2]=p_rel, [3..5]=cam_pt, [6]=r2, [7]=icD, [8]=den,
// [9]=theta, [10]=sin_theta, [11]=dp_signed, [12]=nlerp_len.
// ===========================================================================
__global__ void project_world_points_mean_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ world_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 9;
        float3 p_rel = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        float3 cam_pt = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
        float3 d_world = make_float3(0.0f, 0.0f, 0.0f);
        if (cam_pt.z > 0.0f) {
            float3 camera_ray = normalize3(cam_pt);
            float inv_z = 1.0f / camera_ray.z;
            float x = camera_ray.x * inv_z;
            float y = camera_ray.y * inv_z;
            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float2 d_uv = pinhole_project_bwd(
                params, x, y, scratch[off + 6], scratch[off + 7], scratch[off + 8], d_img, local);
            float3 d_camera_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
            float3 d_cam_pt = normalize3_bwd(cam_pt, d_camera_ray);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, &rx, &ry, &rz, &rw);
            float4 rot = make_float4(rx, ry, rz, rw);
            float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot, p_rel, d_cam_pt, d_rot, d_p_rel);
            d_world = d_p_rel;
            float3 d_trans = scale3(d_p_rel, -1.0f);
            d_trans0 = scale3(d_trans, 0.5f);
            d_trans1 = scale3(d_trans, 0.5f);
            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, rx, ry, rz, rw,
                d_rot.x, d_rot.y, d_rot.z, d_rot.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
            d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);
        }
        if (grad_world_points != nullptr) {
            write_vec3(grad_world_points, idx, d_world);
        }
        (void)world_points;
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K4 backward — project_world_points_shutter_pose (no external distortion)
//
// Backward: project world points → image points with per-ray rolling-shutter
// alpha (no external distortion). Scratch layout (stride 10): same as K3 but
// slot [9]=alpha. d_trans split (1-alpha)/(alpha) between start/end.
// valid_flags == nullptr means all rays are active.
// ===========================================================================
__global__ void project_world_points_shutter_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const bool* __restrict__ valid_flags,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 10;
        float alpha = scratch[off + 9];
        float3 d_world = make_float3(0.0f, 0.0f, 0.0f);
        if (valid_flags == nullptr || valid_flags[idx]) {
            float3 p_rel = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
            float3 cam_pt = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
            float3 camera_ray = normalize3(cam_pt);
            float inv_z = 1.0f / camera_ray.z;
            float x = camera_ray.x * inv_z;
            float y = camera_ray.y * inv_z;
            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float2 d_uv = pinhole_project_bwd(
                params, x, y, scratch[off + 6], scratch[off + 7], scratch[off + 8], d_img, local);
            float3 d_camera_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
            float3 d_cam_pt = normalize3_bwd(cam_pt, d_camera_ray);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &rx, &ry, &rz, &rw);
            float4 rot = make_float4(rx, ry, rz, rw);
            float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot, p_rel, d_cam_pt, d_rot, d_p_rel);
            d_world = d_p_rel;
            float3 d_trans = scale3(d_p_rel, -1.0f);
            d_trans0 = scale3(d_trans, 1.0f - alpha);
            d_trans1 = scale3(d_trans, alpha);
            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, rx, ry, rz, rw,
                d_rot.x, d_rot.y, d_rot.z, d_rot.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
            d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);
        }
        if (grad_world_points != nullptr) {
            write_vec3(grad_world_points, idx, d_world);
        }
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K5 backward — image_points_to_world_rays_static_pose (no external distortion)
//
// Backward: backproject image points → world rays with static pose (no external
// distortion). Scratch layout (stride 5): [0]=xs, [1]=ys, [2]=r2, [3]=icD,
// [4]=den. d_world_ray.origin passes directly to d_trans; d_world_ray.dir
// chains through quat_rotate_bwd_xyzw_geom → d_camera_ray → backproject_bwd.
// ===========================================================================
__global__ void image_points_to_world_rays_static_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_translation,
    float* __restrict__ grad_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        float3 d_origin = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_dir = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);
        d_trans0 = d_origin;
        float4 rot = read_quat_xyzw_from_wxyz(rotation, 0);
        float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        int64_t off = idx * 5;
        float3 camera_ray = normalize3(make_float3(scratch[off + 0], scratch[off + 1], 1.0f));
        quat_rotate_bwd_xyzw_geom(rot, camera_ray, d_dir, d_rot, d_camera_ray);
        d_rot0 = d_rot;

        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float2 d_img = backproject_bwd(
            params,
            img,
            scratch[off + 0],
            scratch[off + 1],
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_camera_ray,
            local);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_static_pose_grads(d_trans0, d_rot0, grad_translation, grad_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K6 backward — image_points_to_world_rays_shutter_pose (no external distortion)
//
// Backward: backproject image points → world rays with rolling-shutter pose
// (no external distortion). Scratch layout (stride 9): [0]=xs, [1]=ys,
// [2]=r2, [3]=icD, [4]=den, [5]=alpha, [6..8]=placeholder (forward writes
// (0.0, 1.0, 0.0); this backward does not consume slots [6..8]).
// d_origin splits (1-alpha)/(alpha) to start/end translation.
// ===========================================================================
__global__ void image_points_to_world_rays_shutter_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* __restrict__ image_points,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    IntrinsicLocalGrads local{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 9;
        float alpha = scratch[off + 5];
        float3 d_origin = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_dir = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);
        d_trans0 = scale3(d_origin, 1.0f - alpha);
        d_trans1 = scale3(d_origin, alpha);

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, &rx, &ry, &rz, &rw);
        float4 rot = make_float4(rx, ry, rz, rw);
        float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        float3 camera_ray = normalize3(make_float3(scratch[off + 0], scratch[off + 1], 1.0f));
        quat_rotate_bwd_xyzw_geom(rot, camera_ray, d_dir, d_rot, d_camera_ray);
        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
        trajectory_cuda::quat_slerp_pair_bwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, rx, ry, rz, rw,
            d_rot.x, d_rot.y, d_rot.z, d_rot.w,
            &gq0x, &gq0y, &gq0z, &gq0w,
            &gq1x, &gq1y, &gq1z, &gq1w,
            &ga_unused);
        d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
        d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);

        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float2 d_img = backproject_bwd(
            params,
            img,
            scratch[off + 0],
            scratch[off + 1],
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_camera_ray,
            local);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
}

// ===========================================================================
// K7 backward — camera_rays_to_image_points (bivariate windshield distortion)
//
// Backward: project camera rays → image points with bivariate distortion.
// Scratch layout (stride 10): [0..2]=distorted_ray (unused here; kept for
// stride consistency), [3]=x, [4]=y, [5]=inv_z, [6]=r2, [7]=icD, [8]=den,
// [9]=ray_forward flag. Chains d_img → pinhole_project_bwd → d_distorted_ray
// → apply_bivariate_distortion_bwd → d_ray + bivariate coeff grads.
// ===========================================================================
__global__ void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ camera_rays,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};

    if (idx < count) {
        int64_t off = idx * 10;
        // scratch[off + 0..2] hold the forward distorted_ray; we don't consume them
        // here (apply_bivariate_distortion_bwd recomputes from the raw camera_ray),
        // but the slots are part of the documented K8 forward layout so the stride
        // is kept at 10.
        float x = scratch[off + 3];
        float y = scratch[off + 4];
        float inv_z = scratch[off + 5];
        float r2 = scratch[off + 6];
        float icD = scratch[off + 7];
        float den = scratch[off + 8];
        bool ray_forward = scratch[off + 9] != 0.0f;
        float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
        if (ray_forward) {
            float2 d_uv = pinhole_project_bwd(params, x, y, r2, icD, den, d_img, local);
            d_distorted_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
        }
        float3 ray = read_vec3(camera_rays, idx);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(ray, bivariate_params, d_distorted_ray, d_ray, local_bivariate);
        if (grad_camera_rays != nullptr) {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }

    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, false, grad_distortion_coeffs);
}

// ===========================================================================
// K8 backward — image_points_to_camera_rays (bivariate windshield distortion)
//
// Backward: backproject image points → camera rays with bivariate distortion.
// Scratch layout (stride 9): [0]=xs, [1]=ys, [2]=r2, [3]=icD, [4]=den,
// [5..7]=camera_ray_pre_norm, [8]=unused. Chains d_camera_ray →
// normalize3_bwd(camera_ray_pre_norm) → apply_bivariate_distortion_bwd →
// d_distorted_ray → backproject_bwd → d_image_point + intrinsic grads.
// ===========================================================================
__global__ void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ image_points,
    const float* __restrict__ grad_camera_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};

    if (idx < count) {
        int64_t off = idx * 9;
        float xs = scratch[off + 0];
        float ys = scratch[off + 1];
        float3 camera_ray_pre_norm = make_float3(scratch[off + 5], scratch[off + 6], scratch[off + 7]);
        float3 d_camera_ray = read_vec3(grad_camera_rays, idx);
        float3 d_pre = normalize3_bwd(camera_ray_pre_norm, d_camera_ray);
        float3 distorted_ray = normalize3(make_float3(xs, ys, 1.0f));
        float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray,
            bivariate_params,
            d_pre,
            d_distorted_ray,
            local_bivariate);
        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float2 d_img = backproject_bwd(
            params,
            img,
            xs,
            ys,
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_distorted_ray,
            local);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, true, grad_distortion_coeffs);
}

// ===========================================================================
// K9 backward — project_world_points_mean_pose (bivariate windshield)
//
// Backward: project world points → image points with mean pose and bivariate
// distortion. Scratch layout (stride 9): same as K3. Inner z-gate on
// distorted_ray.z guards pinhole_project_bwd; outer gate on cam_pt.z guards
// the full SE(3) + bivariate chain so coeff grads are uniformly cam_pt.z based.
// ===========================================================================
__global__ void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 9;
        float3 p_rel = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        float3 cam_pt = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
        float3 d_world = make_float3(0.0f, 0.0f, 0.0f);
        if (cam_pt.z > 0.0f) {
            // Compute d_distorted_ray inside the inner z-positive branch, but always run
            // apply_bivariate_distortion_bwd and the SE(3) chain when cam_pt.z > 0 so
            // coeff-grad gating is uniformly cam_pt.z based.
            float3 camera_ray = normalize3(cam_pt);
            float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
            float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
            if (distorted_ray.z > 0.0f) {
                float inv_z = 1.0f / distorted_ray.z;
                float x = distorted_ray.x * inv_z;
                float y = distorted_ray.y * inv_z;
                float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
                float2 d_uv = pinhole_project_bwd(
                    params, x, y, scratch[off + 6], scratch[off + 7], scratch[off + 8], d_img, local);
                d_distorted_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
            }
            float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(
                camera_ray,
                bivariate_params,
                d_distorted_ray,
                d_camera_ray,
                local_bivariate);
            float3 d_cam_pt = normalize3_bwd(cam_pt, d_camera_ray);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, &rx, &ry, &rz, &rw);
            float4 rot = make_float4(rx, ry, rz, rw);
            float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot, p_rel, d_cam_pt, d_rot, d_p_rel);
            d_world = d_p_rel;
            float3 d_trans = scale3(d_p_rel, -1.0f);
            d_trans0 = scale3(d_trans, 0.5f);
            d_trans1 = scale3(d_trans, 0.5f);
            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                0.5f, rx, ry, rz, rw,
                d_rot.x, d_rot.y, d_rot.z, d_rot.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
            d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);
        }
        if (grad_world_points != nullptr) {
            write_vec3(grad_world_points, idx, d_world);
        }
        (void)world_points;
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, false, grad_distortion_coeffs);
}

// ===========================================================================
// K10 backward — project_world_points_shutter_pose (bivariate windshield)
//
// Backward: project world points → image points with rolling-shutter pose and
// bivariate distortion. Scratch layout (stride 10): same as K4.
// valid_flags == nullptr means all rays are active.
// ===========================================================================
__global__ void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ start_rotation,
    const float* __restrict__ end_rotation,
    const bool* __restrict__ valid_flags,
    const float* __restrict__ grad_image_points,
    float* __restrict__ grad_world_points,
    float* __restrict__ grad_start_translation,
    float* __restrict__ grad_end_translation,
    float* __restrict__ grad_start_rotation,
    float* __restrict__ grad_end_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, false);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 10;
        float alpha = scratch[off + 9];
        float3 d_world = make_float3(0.0f, 0.0f, 0.0f);
        if (valid_flags == nullptr || valid_flags[idx]) {
            float3 p_rel = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
            float3 cam_pt = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
            float3 camera_ray = normalize3(cam_pt);
            float3 distorted_ray = apply_bivariate_distortion(camera_ray, bivariate_params);
            float inv_z = 1.0f / distorted_ray.z;
            float x = distorted_ray.x * inv_z;
            float y = distorted_ray.y * inv_z;
            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float2 d_uv = pinhole_project_bwd(
                params, x, y, scratch[off + 6], scratch[off + 7], scratch[off + 8], d_img, local);
            float3 d_distorted_ray = camera_ray_from_project_bwd(x, y, inv_z, d_uv);
            float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
            apply_bivariate_distortion_bwd(
                camera_ray,
                bivariate_params,
                d_distorted_ray,
                d_camera_ray,
                local_bivariate);
            float3 d_cam_pt = normalize3_bwd(cam_pt, d_camera_ray);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            trajectory_cuda::quat_slerp_pair_fwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, &rx, &ry, &rz, &rw);
            float4 rot = make_float4(rx, ry, rz, rw);
            float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot, p_rel, d_cam_pt, d_rot, d_p_rel);
            d_world = d_p_rel;
            float3 d_trans = scale3(d_p_rel, -1.0f);
            d_trans0 = scale3(d_trans, 1.0f - alpha);
            d_trans1 = scale3(d_trans, alpha);
            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
            trajectory_cuda::quat_slerp_pair_bwd_f(
                rot0.x, rot0.y, rot0.z, rot0.w,
                rot1.x, rot1.y, rot1.z, rot1.w,
                alpha, rx, ry, rz, rw,
                d_rot.x, d_rot.y, d_rot.z, d_rot.w,
                &gq0x, &gq0y, &gq0z, &gq0w,
                &gq1x, &gq1y, &gq1z, &gq1w,
                &ga_unused);
            d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
            d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);
        }
        if (grad_world_points != nullptr) {
            write_vec3(grad_world_points, idx, d_world);
        }
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, false, grad_distortion_coeffs);
}

// ===========================================================================
// K11 backward — image_points_to_world_rays_static_pose (bivariate windshield)
//
// Backward: backproject image points → world rays with static pose and bivariate
// distortion. Scratch layout (stride 9): [0]=xs, [1]=ys, [2]=r2, [3]=icD,
// [4]=den, [5..7]=camera_ray_pre_norm. Chains d_dir through
// quat_rotate_bwd_xyzw_geom → d_camera_ray → normalize3_bwd →
// apply_bivariate_distortion_bwd → d_distorted_ray → backproject_bwd.
// ===========================================================================
__global__ void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* __restrict__ image_points,
    const float* __restrict__ rotation,
    const float* __restrict__ grad_world_rays,
    float* __restrict__ grad_image_points,
    float* __restrict__ grad_translation,
    float* __restrict__ grad_rotation,
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        float3 d_origin = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_dir = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);
        d_trans0 = d_origin;
        float4 rot = read_quat_xyzw_from_wxyz(rotation, 0);
        float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        int64_t off = idx * 9;
        float3 camera_ray_pre_norm = make_float3(scratch[off + 5], scratch[off + 6], scratch[off + 7]);
        float3 camera_ray = normalize3(camera_ray_pre_norm);
        quat_rotate_bwd_xyzw_geom(rot, camera_ray, d_dir, d_rot, d_camera_ray);
        d_rot0 = d_rot;

        float3 d_pre = normalize3_bwd(camera_ray_pre_norm, d_camera_ray);
        float3 distorted_ray = normalize3(make_float3(scratch[off + 0], scratch[off + 1], 1.0f));
        float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray,
            bivariate_params,
            d_pre,
            d_distorted_ray,
            local_bivariate);
        float2 img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float2 d_img = backproject_bwd(
            params,
            img,
            scratch[off + 0],
            scratch[off + 1],
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_distorted_ray,
            local);
        if (grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_static_pose_grads(d_trans0, d_rot0, grad_translation, grad_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, true, grad_distortion_coeffs);
}

// ===========================================================================
// K12 backward — image_points_to_world_rays_shutter_pose (bivariate windshield)
//
// Backward: backproject image points → world rays with rolling-shutter pose
// and bivariate distortion. Scratch layout (stride 12): [0]=xs, [1]=ys,
// [2]=r2, [3]=icD, [4]=den, [5..7]=camera_ray_pre_norm, [8]=alpha, [9..11]=unused.
// image_points == nullptr triggers pixel-center regeneration from (x_idx, y_idx);
// in that case d_image_point writeback is suppressed.
// ===========================================================================
__global__ void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float* __restrict__ grad_focal_length,
    float* __restrict__ grad_principal_point,
    float* __restrict__ grad_radial_coeffs,
    float* __restrict__ grad_tangential_coeffs,
    float* __restrict__ grad_thin_prism_coeffs,
    float* __restrict__ grad_distortion_coeffs,
    const float* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    BivariateWindshieldParams bivariate_params = load_bivariate_windshield_params(distortion, true);
    IntrinsicLocalGrads local{};
    BivariateParamGrads local_bivariate{};
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_rot0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx < count) {
        int64_t off = idx * 12;
        float alpha = scratch[off + 8];
        float3 d_origin = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        float3 d_dir = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);
        d_trans0 = scale3(d_origin, 1.0f - alpha);
        d_trans1 = scale3(d_origin, alpha);

        float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
        float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        trajectory_cuda::quat_slerp_pair_fwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, &rx, &ry, &rz, &rw);
        float4 rot = make_float4(rx, ry, rz, rw);
        float4 d_rot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
        float3 camera_ray_pre_norm = make_float3(scratch[off + 5], scratch[off + 6], scratch[off + 7]);
        float3 camera_ray = normalize3(camera_ray_pre_norm);
        quat_rotate_bwd_xyzw_geom(rot, camera_ray, d_dir, d_rot, d_camera_ray);
        float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w, ga_unused;
        trajectory_cuda::quat_slerp_pair_bwd_f(
            rot0.x, rot0.y, rot0.z, rot0.w,
            rot1.x, rot1.y, rot1.z, rot1.w,
            alpha, rx, ry, rz, rw,
            d_rot.x, d_rot.y, d_rot.z, d_rot.w,
            &gq0x, &gq0y, &gq0z, &gq0w,
            &gq1x, &gq1y, &gq1z, &gq1w,
            &ga_unused);
        d_rot0 = make_float4(gq0x, gq0y, gq0z, gq0w);
        d_rot1 = make_float4(gq1x, gq1y, gq1z, gq1w);

        float3 d_pre = normalize3_bwd(camera_ray_pre_norm, d_camera_ray);
        float3 distorted_ray = normalize3(make_float3(scratch[off + 0], scratch[off + 1], 1.0f));
        float3 d_distorted_ray = make_float3(0.0f, 0.0f, 0.0f);
        apply_bivariate_distortion_bwd(
            distorted_ray,
            bivariate_params,
            d_pre,
            d_distorted_ray,
            local_bivariate);
        // Generated-elements path: regenerate pixel coords from (idx % width, idx / width)
        // and suppress grad_image_points writeback so the caller-allocated tensor (if any)
        // stays zero-initialized.
        float2 img;
        if (image_points != nullptr) {
            img = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        } else {
            int64_t res_x = projection.width;
            int64_t y_idx = idx / res_x;
            int64_t x_idx = idx - y_idx * res_x;
            img = make_float2(0.5f + static_cast<float>(x_idx), 0.5f + static_cast<float>(y_idx));
        }
        float2 d_img = backproject_bwd(
            params,
            img,
            scratch[off + 0],
            scratch[off + 1],
            scratch[off + 2],
            scratch[off + 3],
            scratch[off + 4],
            d_distorted_ray,
            local);
        if (image_points != nullptr && grad_image_points != nullptr) {
            grad_image_points[idx * 2 + 0] = d_img.x;
            grad_image_points[idx * 2 + 1] = d_img.y;
        }
    }

    reduce_pose2_grads_components(
        d_trans0,
        d_trans1,
        d_rot0,
        d_rot1,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation);
    reduce_intrinsic_grads(
        local,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs);
    reduce_bivariate_grads(local_bivariate, distortion, true, grad_distortion_coeffs);
}

} // namespace

// ===========================================================================
// Host wrappers — launch the __global__ kernels above
// ===========================================================================

// Launches K1 backward kernel.
void camera_rays_to_image_points_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K2 backward kernel.
void image_points_to_camera_rays_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K3 backward kernel.
void project_world_points_mean_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_mean_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K4 backward kernel.
void project_world_points_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)world_points;
    (void)shutter_type;
    (void)max_iterations;
    (void)initial_relative_time;
    project_world_points_shutter_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K5 backward kernel.
void image_points_to_world_rays_static_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)translation;
    image_points_to_world_rays_static_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K6 backward kernel.
void image_points_to_world_rays_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)shutter_type;
    image_points_to_world_rays_shutter_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K7 backward kernel.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        distortion,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K8 backward kernel.
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        distortion,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K9 backward kernel.
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K10 backward kernel.
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)world_points;
    (void)shutter_type;
    (void)max_iterations;
    (void)initial_relative_time;
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        distortion,
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K11 backward kernel.
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)translation;
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        distortion,
        image_points,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K12 backward kernel.
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    (void)shutter_type;
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
