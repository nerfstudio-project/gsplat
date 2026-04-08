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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include "pose.cuh"

constexpr int kThreadsPerBlock = 256;

namespace {

// Grid-stride: thread per row; t (N,3), q (N,4) xyzw, p (N,3) → out (N,3) = R(q)p+t.
template <typename scalar_t>
__global__ void se3pose_transform_point_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_transform_point_fwd_device(i, n, translation, rotation, point, out);
}



// Grid-stride: direction (N,3), rotation (N,4) → R(q)d per row.
template <typename scalar_t>
__global__ void se3pose_transform_direction_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ direction,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_transform_direction_fwd_device(i, n, rotation, direction, out);
}



// Grid-stride: inverse point R^T(p−t); q stored xyzw (device uses conjugate for inverse rotation).
template <typename scalar_t>
__global__ void se3pose_inverse_transform_point_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_inverse_transform_point_fwd_device(i, n, translation, rotation, point, out);
}



// Grid-stride: R^T d per row (no translation).
template <typename scalar_t>
__global__ void se3pose_inverse_transform_direction_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ direction,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_inverse_transform_direction_fwd_device(i, n, rotation, direction, out);
}



// Launch the forward SE(3) point transform kernel on the current CUDA stream.
// Inputs: translation (N,3), rotation (N,4) in xyzw order, point (N,3).
// Output: out (N,3) with row i = R(q_i) * point_i + translation_i.
template <typename scalar_t>
void launch_se3pose_transform_point_fwd(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out) {
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    se3pose_transform_point_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        translation.data_ptr<scalar_t>(),
        rotation.data_ptr<scalar_t>(),
        point.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the forward direction-only SE(3) kernel on the current CUDA stream.
// Inputs: rotation (N,4) in xyzw order and direction (N,3).
// Output: out (N,3) with row i = R(q_i) * direction_i.
template <typename scalar_t>
void launch_se3pose_transform_direction_fwd(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out) {
    const int64_t n = direction.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.device().index());
    se3pose_transform_direction_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        rotation.data_ptr<scalar_t>(),
        direction.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the inverse SE(3) point transform kernel on the current CUDA stream.
// Inputs: translation (N,3), rotation (N,4) in xyzw order, point (N,3).
// Output: out (N,3) with row i = R(q_i)^T * (point_i - translation_i).
template <typename scalar_t>
void launch_se3pose_inverse_transform_point_fwd(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out) {
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    se3pose_inverse_transform_point_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        translation.data_ptr<scalar_t>(),
        rotation.data_ptr<scalar_t>(),
        point.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the inverse direction-only SE(3) kernel on the current CUDA stream.
// Inputs: rotation (N,4) in xyzw order and direction (N,3).
// Output: out (N,3) with row i = R(q_i)^T * direction_i.
template <typename scalar_t>
void launch_se3pose_inverse_transform_direction_fwd(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out) {
    const int64_t n = direction.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.device().index());
    se3pose_inverse_transform_direction_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        rotation.data_ptr<scalar_t>(),
        direction.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// Public CUDA entrypoint for batched SE(3) point transforms.
// Inputs: translation (N,3), rotation (N,4) in xyzw order, point (N,3).
// Output: out (N,3) with one transformed point per row.
void se3pose_transform_point_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(point.scalar_type(), "se3pose_transform_point_cuda", ([&] {
        launch_se3pose_transform_point_fwd<scalar_t>(translation, rotation, point, out);
    }));
}

// Public CUDA entrypoint for batched direction-only transforms.
// Inputs: rotation (N,4) in xyzw order and direction (N,3).
// Output: out (N,3) with one rotated direction per row.
void se3pose_transform_direction_cuda(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(direction.scalar_type(), "se3pose_transform_direction_cuda", ([&] {
        launch_se3pose_transform_direction_fwd<scalar_t>(rotation, direction, out);
    }));
}

// Public CUDA entrypoint for inverse batched SE(3) point transforms.
// Inputs: translation (N,3), rotation (N,4) in xyzw order, point (N,3).
// Output: out (N,3) with one inverse-transformed point per row.
void se3pose_inverse_transform_point_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(point.scalar_type(), "se3pose_inverse_transform_point_cuda", ([&] {
        launch_se3pose_inverse_transform_point_fwd<scalar_t>(translation, rotation, point, out);
    }));
}

// Public CUDA entrypoint for inverse batched direction transforms.
// Inputs: rotation (N,4) in xyzw order and direction (N,3).
// Output: out (N,3) with one inverse-rotated direction per row.
void se3pose_inverse_transform_direction_cuda(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(direction.scalar_type(), "se3pose_inverse_transform_direction_cuda", ([&] {
        launch_se3pose_inverse_transform_direction_fwd<scalar_t>(rotation, direction, out);
    }));
}

// -----------------------------------------------------------------------------
// SE3 pose -> 4x4 matrix pack (R from quat_to_matrix_cuda (N,9), row-major 4x4)
// -----------------------------------------------------------------------------

// Grid-stride: R9 (N,9) row-major + translation (N,3) → out16 (N,16) homogeneous matrices.
template <typename scalar_t>
__global__ void se3pose_to_matrix_pack_kernel(
    int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ R9,
    scalar_t* __restrict__ out16) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_to_matrix_pack_device(i, n, translation, R9, out16);
}



// Launch homogeneous matrix packing from translation and row-major rotation blocks.
// Inputs: translation (N,3) and R9 (N,9) storing row-major 3x3 rotations.
// Output: out_flat (N,16) storing row-major homogeneous 4x4 transforms.
template <typename scalar_t>
void launch_se3pose_to_matrix_pack(
    const at::Tensor& translation,
    const at::Tensor& R9,
    at::Tensor& out_flat) {
    const int64_t n = translation.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(translation.device().index());
    se3pose_to_matrix_pack_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        translation.data_ptr<scalar_t>(),
        R9.data_ptr<scalar_t>(),
        out_flat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for SE(3) -> row-major 4x4 matrix packing.
// Inputs: translation (N,3) and R9 (N,9) row-major rotation blocks.
// Output: out_flat (N,16) with one homogeneous matrix per row.
void se3pose_to_matrix_pack_cuda(
    const at::Tensor& translation,
    const at::Tensor& R9,
    at::Tensor& out_flat) {
    AT_DISPATCH_FLOATING_TYPES(
        translation.scalar_type(),
        "se3pose_to_matrix_pack_cuda",
        ([&] { launch_se3pose_to_matrix_pack<scalar_t>(translation, R9, out_flat); }));
}

// Grid-stride: inverse 4×4 per row; wxyz_format selects quaternion component order in rotation buffer.
template <typename scalar_t>
__global__ void se3pose_to_inverse_matrix_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    scalar_t* __restrict__ out16,
    const int wxyz_format) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_to_inverse_matrix_fwd_device(i, n, translation, rotation, out16, wxyz_format);
}



// Launch inverse homogeneous matrix construction for a batch of SE(3) poses.
// Inputs: translation (N,3), rotation (N,4), and wxyz_format selecting quaternion input order.
// Output: out_flat (N,16) storing row-major inverse 4x4 transforms.
template <typename scalar_t>
void launch_se3pose_to_inverse_matrix_fwd(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    at::Tensor& out_flat,
    bool wxyz_format) {
    const int64_t n = translation.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(translation.device().index());
    se3pose_to_inverse_matrix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        translation.data_ptr<scalar_t>(),
        rotation.data_ptr<scalar_t>(),
        out_flat.data_ptr<scalar_t>(),
        wxyz_format ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for inverse SE(3) matrix construction.
// Inputs: translation (N,3), rotation (N,4), and wxyz_format selecting quaternion layout.
// Output: out_flat (N,16) with one row-major inverse matrix per pose.
void se3pose_to_inverse_matrix_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    bool wxyz_format,
    at::Tensor& out_flat) {
    AT_DISPATCH_FLOATING_TYPES(
        translation.scalar_type(),
        "se3pose_to_inverse_matrix_cuda",
        ([&] { launch_se3pose_to_inverse_matrix_fwd<scalar_t>(translation, rotation, out_flat, wxyz_format); }));
}

// Grid-stride: row-major m16 (N,16) → translation (N,3), rotation quaternion (N,4) xyzw.
template <typename scalar_t>
__global__ void se3pose_from_matrix_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ m16,
    scalar_t* __restrict__ translation,
    scalar_t* __restrict__ rotation) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_from_matrix_fwd_device(i, n, m16, translation, rotation);
}



// Grid-stride: VJP for se3pose_from_matrix; per row fills grad_m16 (row-major) from grad_translation and grad_rotation.
template <typename scalar_t>
__global__ void se3pose_from_matrix_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ m16,
    const scalar_t* __restrict__ grad_translation,
    const scalar_t* __restrict__ grad_rotation,
    scalar_t* __restrict__ grad_m16) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    se3pose_from_matrix_bwd_device(i, n, m16, grad_translation, grad_rotation, grad_m16);
}



// Launch matrix -> pose decomposition for a batch of row-major homogeneous transforms.
// Input: matrix (N,16) storing one row-major 4x4 transform per row.
// Outputs: translation (N,3) and rotation (N,4) in xyzw order.
template <typename scalar_t>
void launch_se3pose_from_matrix_fwd(const at::Tensor& matrix, at::Tensor& translation, at::Tensor& rotation) {
    const int64_t n = matrix.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(matrix.device().index());
    se3pose_from_matrix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        matrix.data_ptr<scalar_t>(),
        translation.data_ptr<scalar_t>(),
        rotation.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for matrix -> pose decomposition.
// Inputs: matrix (N,16), grad_translation (N,3), and grad_rotation (N,4).
// Output: grad_matrix (N,16) in the same row-major layout as the forward input.
template <typename scalar_t>
void launch_se3pose_from_matrix_bwd(
    const at::Tensor& matrix,
    const at::Tensor& grad_translation,
    const at::Tensor& grad_rotation,
    at::Tensor& grad_matrix) {
    const int64_t n = matrix.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(matrix.device().index());
    se3pose_from_matrix_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        matrix.data_ptr<scalar_t>(),
        grad_translation.data_ptr<scalar_t>(),
        grad_rotation.data_ptr<scalar_t>(),
        grad_matrix.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for matrix -> pose decomposition.
// Input: matrix (N,16) storing row-major homogeneous transforms.
// Outputs: translation (N,3) and rotation (N,4) in xyzw order.
void se3pose_from_matrix_cuda(
    const at::Tensor& matrix,
    at::Tensor& translation,
    at::Tensor& rotation) {
    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "se3pose_from_matrix_cuda", ([&] {
        launch_se3pose_from_matrix_fwd<scalar_t>(matrix, translation, rotation);
    }));
}

// Public CUDA entrypoint for the VJP of matrix -> pose decomposition.
// Inputs: matrix (N,16), grad_translation (N,3), and grad_rotation (N,4).
// Output: grad_matrix (N,16) in row-major layout.
void se3pose_from_matrix_bwd_cuda(
    const at::Tensor& matrix,
    const at::Tensor& grad_translation,
    const at::Tensor& grad_rotation,
    at::Tensor& grad_matrix) {
    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "se3pose_from_matrix_bwd_cuda", ([&] {
        launch_se3pose_from_matrix_bwd<scalar_t>(matrix, grad_translation, grad_rotation, grad_matrix);
    }));
}

// -----------------------------------------------------------------------------
// 2-pose trajectories, float32 only.
// -----------------------------------------------------------------------------

namespace trajectory_cuda {
// Grid-stride float: 2-keyframe trajectory point + OOB; strides st0,st1,sqt index time/query_time rows (see device).
__global__ void trajectory_transform_point_2poses_fwd_kernel(
    int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ trans0,
    const float* __restrict__ rot0,
    const float* __restrict__ time0,
    const float* __restrict__ trans1,
    const float* __restrict__ rot1,
    const float* __restrict__ time1,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    float* __restrict__ result_point,
    float* __restrict__ result_oob) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_transform_point_2poses_fwd_device(i, n, st0, st1, sqt, trans0, rot0, time0, trans1, rot1, time1, point, query_time, result_point, result_oob);
}



// Grid-stride float: VJP 2-keyframe point transform; all grad_* buffers length N rows with same layouts as fwd.
__global__ void trajectory_transform_point_2poses_bwd_kernel(
    int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ trans0,
    const float* __restrict__ rot0,
    const float* __restrict__ time0,
    const float* __restrict__ trans1,
    const float* __restrict__ rot1,
    const float* __restrict__ time1,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    const float* __restrict__ grad_result_point,
    float* __restrict__ grad_trans0,
    float* __restrict__ grad_rot0,
    float* __restrict__ grad_time0,
    float* __restrict__ grad_trans1,
    float* __restrict__ grad_rot1,
    float* __restrict__ grad_time1,
    float* __restrict__ grad_point,
    float* __restrict__ grad_query_time) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_transform_point_2poses_bwd_device(i, n, st0, st1, sqt, trans0, rot0, time0, trans1, rot1, time1, point, query_time, grad_result_point, grad_trans0, grad_rot0, grad_time0, grad_trans1, grad_rot1, grad_time1, grad_point, grad_query_time);
}



// Grid-stride float: SLERP quaternion only + OOB flag per row.
__global__ void trajectory_get_rotation_2poses_fwd_kernel(
    int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ time0,
    const float* __restrict__ time1,
    const float* __restrict__ query_time,
    const float* __restrict__ rot0,
    const float* __restrict__ rot1,
    float* __restrict__ result_quat,
    float* __restrict__ result_oob) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_get_rotation_2poses_fwd_device(i, n, st0, st1, sqt, time0, time1, query_time, rot0, rot1, result_quat, result_oob);
}



// Grid-stride float: VJP rotation query → grad_rot0/1, grad_time0/1, grad_query_time.
__global__ void trajectory_get_rotation_2poses_bwd_kernel(
    int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ time0,
    const float* __restrict__ time1,
    const float* __restrict__ query_time,
    const float* __restrict__ rot0,
    const float* __restrict__ rot1,
    const float* __restrict__ grad_result_quat,
    float* __restrict__ grad_rot0,
    float* __restrict__ grad_rot1,
    float* __restrict__ grad_time0,
    float* __restrict__ grad_time1,
    float* __restrict__ grad_query_time) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_get_rotation_2poses_bwd_device(i, n, st0, st1, sqt, time0, time1, query_time, rot0, rot1, grad_result_quat, grad_rot0, grad_rot1, grad_time0, grad_time1, grad_query_time);
}




// Grid-stride float: 1 pose, st/sqt are batch strides for time and query_time tensors.
__global__ void trajectory_transform_point_1pose_fwd_kernel(
    int64_t n,
    int64_t st,
    int64_t sqt,
    const float* __restrict__ trans,
    const float* __restrict__ rot,
    const float* __restrict__ time,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    float* __restrict__ result_point,
    float* __restrict__ result_oob) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_transform_point_1pose_fwd_device(i, n, st, sqt, trans, rot, time, point, query_time, result_point, result_oob);
}



// Grid-stride float: VJP 1-pose transform; kernel silences grad_time/grad_query_time (see (void) in body).
__global__ void trajectory_transform_point_1pose_bwd_kernel(
    int64_t n,
    const float* __restrict__ trans,
    const float* __restrict__ rot,
    const float* __restrict__ time,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    const float* __restrict__ grad_result_point,
    float* __restrict__ grad_trans,
    float* __restrict__ grad_rot,
    float* __restrict__ grad_time,
    float* __restrict__ grad_point,
    float* __restrict__ grad_query_time) {
    (void)grad_time;
    (void)grad_query_time;
    (void)time;
    (void)query_time;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    trajectory_transform_point_1pose_bwd_device(i, n, trans, rot, time, point, query_time, grad_result_point, grad_trans, grad_rot, grad_time, grad_point, grad_query_time);
}



// Launch single-pose trajectory point evaluation on the current CUDA stream.
// Inputs: trans (N,3), rot (N,4), time/query_time with explicit row strides, and point (N,3).
// Outputs: result_point (N,3) and result_oob (N,1) float flags.
void launch_trajectory_transform_point_1pose_fwd(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time,
    at::Tensor& result_point,
    at::Tensor& result_oob) {
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    const int64_t st = time.stride(0);
    const int64_t sqt = query_time.stride(0);
    trajectory_transform_point_1pose_fwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        st,
        sqt,
        trans.data_ptr<float>(),
        rot.data_ptr<float>(),
        time.data_ptr<float>(),
        point.data_ptr<float>(),
        query_time.data_ptr<float>(),
        result_point.data_ptr<float>(),
        result_oob.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for the single-pose trajectory point path.
// Inputs: forward tensors plus grad_result_point (N,3).
// Outputs: grad_trans (N,3), grad_rot (N,4), grad_time, grad_point (N,3), and grad_query_time.
void launch_trajectory_transform_point_1pose_bwd(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time,
    const at::Tensor& grad_result_point,
    at::Tensor& grad_trans,
    at::Tensor& grad_rot,
    at::Tensor& grad_time,
    at::Tensor& grad_point,
    at::Tensor& grad_query_time) {
    (void)trans;
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    trajectory_transform_point_1pose_bwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        trans.data_ptr<float>(),
        rot.data_ptr<float>(),
        time.data_ptr<float>(),
        point.data_ptr<float>(),
        query_time.data_ptr<float>(),
        grad_result_point.data_ptr<float>(),
        grad_trans.data_ptr<float>(),
        grad_rot.data_ptr<float>(),
        grad_time.data_ptr<float>(),
        grad_point.data_ptr<float>(),
        grad_query_time.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Grid-stride float: one thread per pose row; shared frame quat (xyzw), translation ft, scale; in/out (N,7).
__global__ void frame_transform_poses_tquat_fwd_kernel(
    int64_t n,
    float frx,
    float fry,
    float frz,
    float frw,
    float ftx,
    float fty,
    float ftz,
    float scale,
    const float* __restrict__ input_poses,
    float* __restrict__ output_poses) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    frame_transform_poses_tquat_fwd_device(i, n, frx, fry, frz, frw, ftx, fty, ftz, scale, input_poses, output_poses);
}



// Launch rigid frame application to pose rows stored as (tx,ty,tz,qx,qy,qz,qw).
// Inputs: input_poses (N,7), shared frame quaternion/translation, and scalar scale.
// Output: output_poses (N,7) in the same translation-plus-xyzw layout.
void launch_frame_transform_poses_tquat_fwd(
    const at::Tensor& input_poses,
    float frx,
    float fry,
    float frz,
    float frw,
    float ftx,
    float fty,
    float ftz,
    float scale,
    at::Tensor& output_poses) {
    const int64_t n = input_poses.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_poses.device().index());
    frame_transform_poses_tquat_fwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        frx,
        fry,
        frz,
        frw,
        ftx,
        fty,
        ftz,
        scale,
        input_poses.data_ptr<float>(),
        output_poses.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch 2-keyframe trajectory point interpolation on the current CUDA stream.
// Inputs: keyframe translations/rotations/times, point (N,3), and query_time.
// Outputs: result_point (N,3) and result_oob (N,1) float flags.
void launch_trajectory_transform_point_2poses_fwd(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& point,
    const at::Tensor& query_time,
    at::Tensor& result_point,
    at::Tensor& result_oob) {
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    const int64_t st0 = time0.stride(0);
    const int64_t st1 = time1.stride(0);
    const int64_t sqt = query_time.stride(0);
    trajectory_transform_point_2poses_fwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        st0,
        st1,
        sqt,
        trans0.data_ptr<float>(),
        rot0.data_ptr<float>(),
        time0.data_ptr<float>(),
        trans1.data_ptr<float>(),
        rot1.data_ptr<float>(),
        time1.data_ptr<float>(),
        point.data_ptr<float>(),
        query_time.data_ptr<float>(),
        result_point.data_ptr<float>(),
        result_oob.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for the 2-keyframe trajectory point path.
// Inputs: forward tensors plus grad_result_point (N,3).
// Outputs: per-keyframe translation/rotation/time grads, grad_point, and grad_query_time.
void launch_trajectory_transform_point_2poses_bwd(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& point,
    const at::Tensor& query_time,
    const at::Tensor& grad_result_point,
    at::Tensor& grad_trans0,
    at::Tensor& grad_rot0,
    at::Tensor& grad_time0,
    at::Tensor& grad_trans1,
    at::Tensor& grad_rot1,
    at::Tensor& grad_time1,
    at::Tensor& grad_point,
    at::Tensor& grad_query_time) {
    const int64_t n = point.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.device().index());
    const int64_t st0 = time0.stride(0);
    const int64_t st1 = time1.stride(0);
    const int64_t sqt = query_time.stride(0);
    trajectory_transform_point_2poses_bwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        st0,
        st1,
        sqt,
        trans0.data_ptr<float>(),
        rot0.data_ptr<float>(),
        time0.data_ptr<float>(),
        trans1.data_ptr<float>(),
        rot1.data_ptr<float>(),
        time1.data_ptr<float>(),
        point.data_ptr<float>(),
        query_time.data_ptr<float>(),
        grad_result_point.data_ptr<float>(),
        grad_trans0.data_ptr<float>(),
        grad_rot0.data_ptr<float>(),
        grad_time0.data_ptr<float>(),
        grad_trans1.data_ptr<float>(),
        grad_rot1.data_ptr<float>(),
        grad_time1.data_ptr<float>(),
        grad_point.data_ptr<float>(),
        grad_query_time.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch 2-keyframe trajectory quaternion interpolation on the current CUDA stream.
// Inputs: rot0/rot1 (N,4), time0/time1/query_time; translation tensors are accepted for API symmetry.
// Outputs: result_quat (N,4) in xyzw order and result_oob (N,1) float flags.
void launch_trajectory_get_rotation_2poses_fwd(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    at::Tensor& result_quat,
    at::Tensor& result_oob) {
    (void)trans0;
    (void)trans1;
    const int64_t n = rot0.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rot0.device().index());
    const int64_t st0 = time0.stride(0);
    const int64_t st1 = time1.stride(0);
    const int64_t sqt = query_time.stride(0);
    trajectory_get_rotation_2poses_fwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        st0,
        st1,
        sqt,
        time0.data_ptr<float>(),
        time1.data_ptr<float>(),
        query_time.data_ptr<float>(),
        rot0.data_ptr<float>(),
        rot1.data_ptr<float>(),
        result_quat.data_ptr<float>(),
        result_oob.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for 2-keyframe trajectory quaternion interpolation.
// Inputs: rot0/rot1, time0/time1/query_time, and grad_result_quat (N,4); translation tensors are unused.
// Outputs: grad_rot0, grad_rot1, grad_time0, grad_time1, and grad_query_time.
void launch_trajectory_get_rotation_2poses_bwd(
    const at::Tensor& time0,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    const at::Tensor& rot0,
    const at::Tensor& rot1,
    const at::Tensor& grad_result_quat,
    at::Tensor& grad_rot0,
    at::Tensor& grad_rot1,
    at::Tensor& grad_time0,
    at::Tensor& grad_time1,
    at::Tensor& grad_query_time) {
    const int64_t n = rot0.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rot0.device().index());
    const int64_t st0 = time0.stride(0);
    const int64_t st1 = time1.stride(0);
    const int64_t sqt = query_time.stride(0);
    trajectory_get_rotation_2poses_bwd_kernel<<<blocks, threads, 0, stream>>>(
        n,
        st0,
        st1,
        sqt,
        time0.data_ptr<float>(),
        time1.data_ptr<float>(),
        query_time.data_ptr<float>(),
        rot0.data_ptr<float>(),
        rot1.data_ptr<float>(),
        grad_result_quat.data_ptr<float>(),
        grad_rot0.data_ptr<float>(),
        grad_rot1.data_ptr<float>(),
        grad_time0.data_ptr<float>(),
        grad_time1.data_ptr<float>(),
        grad_query_time.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace trajectory_cuda

// Public CUDA entrypoint for 2-keyframe trajectory point interpolation.
// Inputs: keyframe translations/rotations/times, point (N,3), and query_time.
// Outputs: result_point (N,3) and result_out_of_bounds (N,1).
void trajectory_transform_point_2poses_cuda(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& point,
    const at::Tensor& query_time,
    at::Tensor& result_point,
    at::Tensor& result_out_of_bounds) {
    TORCH_CHECK(trans0.scalar_type() == at::kFloat, "trajectory_transform_point_2poses_cuda: float32 only");
    trajectory_cuda::launch_trajectory_transform_point_2poses_fwd(
        trans0, rot0, time0, trans1, rot1, time1, point, query_time, result_point, result_out_of_bounds);
}

// Public CUDA entrypoint for the VJP of 2-keyframe trajectory point interpolation.
// Inputs: forward tensors plus grad_result_point (N,3).
// Outputs: per-keyframe translation/rotation/time grads, grad_point, and grad_query_time.
void trajectory_transform_point_2poses_bwd_cuda(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& point,
    const at::Tensor& query_time,
    const at::Tensor& grad_result_point,
    at::Tensor& grad_trans0,
    at::Tensor& grad_rot0,
    at::Tensor& grad_time0,
    at::Tensor& grad_trans1,
    at::Tensor& grad_rot1,
    at::Tensor& grad_time1,
    at::Tensor& grad_point,
    at::Tensor& grad_query_time) {
    TORCH_CHECK(trans0.scalar_type() == at::kFloat, "trajectory_transform_point_2poses_bwd_cuda: float32 only");
    trajectory_cuda::launch_trajectory_transform_point_2poses_bwd(
        trans0,
        rot0,
        time0,
        trans1,
        rot1,
        time1,
        point,
        query_time,
        grad_result_point,
        grad_trans0,
        grad_rot0,
        grad_time0,
        grad_trans1,
        grad_rot1,
        grad_time1,
        grad_point,
        grad_query_time);
}

// Public CUDA entrypoint for 2-keyframe trajectory quaternion interpolation.
// Inputs: keyframe rotations/times and query_time; translation tensors are accepted for API symmetry.
// Outputs: result_quat (N,4) and result_out_of_bounds (N,1).
void trajectory_get_rotation_2poses_cuda(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    at::Tensor& result_quat,
    at::Tensor& result_out_of_bounds) {
    TORCH_CHECK(rot0.scalar_type() == at::kFloat, "trajectory_get_rotation_2poses_cuda: float32 only");
    trajectory_cuda::launch_trajectory_get_rotation_2poses_fwd(
        trans0, rot0, time0, trans1, rot1, time1, query_time, result_quat, result_out_of_bounds);
}

// Public CUDA entrypoint for the VJP of 2-keyframe trajectory quaternion interpolation.
// Inputs: keyframe rotations/times, query_time, and grad_result_quat (N,4); translation tensors are unused.
// Outputs: grad_rot0, grad_rot1, grad_time0, grad_time1, and grad_query_time.
void trajectory_get_rotation_2poses_bwd_cuda(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    const at::Tensor& grad_result_quat,
    at::Tensor& grad_trans0,
    at::Tensor& grad_rot0,
    at::Tensor& grad_time0,
    at::Tensor& grad_trans1,
    at::Tensor& grad_rot1,
    at::Tensor& grad_time1,
    at::Tensor& grad_query_time) {
    (void)trans0;
    (void)trans1;
    TORCH_CHECK(rot0.scalar_type() == at::kFloat, "trajectory_get_rotation_2poses_bwd_cuda: float32 only");
    trajectory_cuda::launch_trajectory_get_rotation_2poses_bwd(
        time0,
        time1,
        query_time,
        rot0,
        rot1,
        grad_result_quat,
        grad_rot0,
        grad_rot1,
        grad_time0,
        grad_time1,
        grad_query_time);
}

// Public CUDA entrypoint for single-pose trajectory point evaluation.
// Inputs: trans (N,3), rot (N,4), time/query_time, and point (N,3).
// Outputs: result_point (N,3) and result_out_of_bounds (N,1).
void trajectory_transform_point_1pose_cuda(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time,
    at::Tensor& result_point,
    at::Tensor& result_out_of_bounds) {
    TORCH_CHECK(trans.scalar_type() == at::kFloat, "trajectory_transform_point_1pose_cuda: float32 only");
    trajectory_cuda::launch_trajectory_transform_point_1pose_fwd(
        trans, rot, time, point, query_time, result_point, result_out_of_bounds);
}

// Public CUDA entrypoint for the VJP of single-pose trajectory point evaluation.
// Inputs: forward tensors plus grad_result_point (N,3).
// Outputs: grad_trans, grad_rot, grad_time, grad_point, and grad_query_time.
void trajectory_transform_point_1pose_bwd_cuda(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time,
    const at::Tensor& grad_result_point,
    at::Tensor& grad_trans,
    at::Tensor& grad_rot,
    at::Tensor& grad_time,
    at::Tensor& grad_point,
    at::Tensor& grad_query_time) {
    TORCH_CHECK(rot.scalar_type() == at::kFloat, "trajectory_transform_point_1pose_bwd_cuda: float32 only");
    trajectory_cuda::launch_trajectory_transform_point_1pose_bwd(
        trans,
        rot,
        time,
        point,
        query_time,
        grad_result_point,
        grad_trans,
        grad_rot,
        grad_time,
        grad_point,
        grad_query_time);
}

// Public CUDA entrypoint for applying a shared frame transform to pose rows.
// Inputs: input_poses (N,7), shared frame quaternion/translation, and scalar scale.
// Output: output_poses (N,7) in the same translation-plus-xyzw layout.
void frame_transform_poses_tquat_cuda(
    const at::Tensor& input_poses,
    float frx,
    float fry,
    float frz,
    float frw,
    float ftx,
    float fty,
    float ftz,
    float scale,
    at::Tensor& output_poses) {
    TORCH_CHECK(input_poses.scalar_type() == at::kFloat, "frame_transform_poses_tquat_cuda: float32 only");
    trajectory_cuda::launch_frame_transform_poses_tquat_fwd(
        input_poses, frx, fry, frz, frw, ftx, fty, ftz, scale, output_poses);
}
