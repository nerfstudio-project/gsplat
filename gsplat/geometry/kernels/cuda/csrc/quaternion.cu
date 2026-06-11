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

#include "quaternion.cuh"

constexpr int kThreadsPerBlock = 256;

namespace {

// Grid-stride launch: one thread per batch row i in [0,n). Computes q1_i ⊗ q2_i → out_i (xyzw, 4*n layout).
template <typename scalar_t>
__global__ void quat_multiply_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_multiply_fwd_device(i, n, q1, q2, out);
}



// Grid-stride: VJP for Hamilton product; grad_out layout matches q1/q2 (4 per row).
template <typename scalar_t>
__global__ void quat_multiply_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_multiply_bwd_device(i, n, q1, q2, grad_out, grad_q1, grad_q2);
}



// Launch batched Hamilton products on the current CUDA stream.
// Inputs: q1 (N,4) and q2 (N,4) in xyzw order.
// Output: out (N,4) with row i = q1_i ⊗ q2_i.
template <typename scalar_t>
void launch_quat_multiply_fwd(const at::Tensor& q1, const at::Tensor& q2, at::Tensor& out) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_multiply_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for batched Hamilton products.
// Inputs: q1 (N,4), q2 (N,4), and grad_out (N,4).
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
template <typename scalar_t>
void launch_quat_multiply_bwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_multiply_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_q1.data_ptr<scalar_t>(),
        grad_q2.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// Public CUDA entrypoint for batched quaternion multiplication.
// Inputs: q1 (N,4) and q2 (N,4) in xyzw order.
// Output: out (N,4) with one Hamilton product per row.
void quat_multiply_cuda(const at::Tensor& q1, const at::Tensor& q2, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(
        q1.scalar_type(), "quat_multiply_cuda", ([&] { launch_quat_multiply_fwd<scalar_t>(q1, q2, out); }));
}

// Public CUDA entrypoint for the VJP of quaternion multiplication.
// Inputs: q1 (N,4), q2 (N,4), and grad_out (N,4).
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
void quat_multiply_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_multiply_bwd_cuda", ([&] {
        launch_quat_multiply_bwd<scalar_t>(q1, q2, grad_out, grad_q1, grad_q2);
    }));
}

// -----------------------------------------------------------------------------
// quat_rotate_vector in xyzw convention for 3D vectors.
// -----------------------------------------------------------------------------

// Grid-stride: quat (N,4) xyzw, vec (N,3) → out (N,3), row i = R(q_i) v_i.
template <typename scalar_t>
__global__ void quat_rotate_vector_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ vec,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_rotate_vector_fwd_device(i, n, quat, vec, out);
}



// Grid-stride: grad_out (N,3) → grad_quat (N,4), grad_vec (N,3) per row.
template <typename scalar_t>
__global__ void quat_rotate_vector_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ vec,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat,
    scalar_t* __restrict__ grad_vec) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_rotate_vector_bwd_device(i, n, quat, vec, grad_out, grad_quat, grad_vec);
}



// Launch quaternion-vector rotation on the current CUDA stream.
// Inputs: quat (N,4) in xyzw order and vec (N,3).
// Output: out (N,3) with row i = R(q_i) * vec_i.
template <typename scalar_t>
void launch_quat_rotate_vector_fwd(const at::Tensor& quat, const at::Tensor& vec, at::Tensor& out) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_rotate_vector_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        quat.data_ptr<scalar_t>(),
        vec.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for quaternion-vector rotation.
// Inputs: quat (N,4), vec (N,3), and grad_out (N,3).
// Outputs: grad_quat (N,4) and grad_vec (N,3).
template <typename scalar_t>
void launch_quat_rotate_vector_bwd(
    const at::Tensor& quat,
    const at::Tensor& vec,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat,
    at::Tensor& grad_vec) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_rotate_vector_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        quat.data_ptr<scalar_t>(),
        vec.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_quat.data_ptr<scalar_t>(),
        grad_vec.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for quaternion-vector rotation.
// Inputs: quat (N,4) in xyzw order and vec (N,3).
// Output: out (N,3) with one rotated vector per row.
void quat_rotate_vector_cuda(const at::Tensor& quat, const at::Tensor& vec, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(
        quat.scalar_type(),
        "quat_rotate_vector_cuda",
        ([&] { launch_quat_rotate_vector_fwd<scalar_t>(quat, vec, out); }));
}

// Public CUDA entrypoint for the VJP of quaternion-vector rotation.
// Inputs: quat (N,4), vec (N,3), and grad_out (N,3).
// Outputs: grad_quat (N,4) and grad_vec (N,3).
void quat_rotate_vector_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& vec,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat,
    at::Tensor& grad_vec) {
    AT_DISPATCH_FLOATING_TYPES(quat.scalar_type(), "quat_rotate_vector_bwd_cuda", ([&] {
        launch_quat_rotate_vector_bwd<scalar_t>(quat, vec, grad_out, grad_quat, grad_vec);
    }));
}

// Grid-stride: quat (N,4) → out_flat (N*9) row-major 3×3 per batch row (normalize-safe).
template <typename scalar_t>
__global__ void quat_to_matrix_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out_flat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_to_matrix_fwd_device(i, n, quat, out_flat);
}



// Grid-stride: grad_flat (N*9) ∂L/∂R row-major → grad_quat (N,4).
template <typename scalar_t>
__global__ void quat_to_matrix_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ grad_flat,
    scalar_t* __restrict__ grad_quat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_to_matrix_bwd_device(i, n, quat, grad_flat, grad_quat);
}



// Launch quaternion -> matrix conversion on the current CUDA stream.
// Input: quat (N,4) in xyzw order.
// Output: out_flat (N,9) storing one row-major 3x3 matrix per row.
template <typename scalar_t>
void launch_quat_to_matrix_fwd(const at::Tensor& quat, at::Tensor& out_flat) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_to_matrix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, quat.data_ptr<scalar_t>(), out_flat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for quaternion -> matrix conversion.
// Inputs: quat (N,4) and grad_flat (N,9) in row-major layout.
// Output: grad_quat (N,4).
template <typename scalar_t>
void launch_quat_to_matrix_bwd(
    const at::Tensor& quat,
    const at::Tensor& grad_flat,
    at::Tensor& grad_quat) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_to_matrix_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        quat.data_ptr<scalar_t>(),
        grad_flat.data_ptr<scalar_t>(),
        grad_quat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for quaternion -> row-major matrix conversion.
// Input: quat (N,4) in xyzw order.
// Output: out_flat (N,9) with one row-major 3x3 matrix per row.
void quat_to_matrix_cuda(const at::Tensor& quat, at::Tensor& out_flat) {
    AT_DISPATCH_FLOATING_TYPES(
        quat.scalar_type(), "quat_to_matrix_cuda", ([&] { launch_quat_to_matrix_fwd<scalar_t>(quat, out_flat); }));
}

// Public CUDA entrypoint for the VJP of quaternion -> matrix conversion.
// Inputs: quat (N,4) and grad_matrix_flat (N,9) in row-major layout.
// Output: grad_quat (N,4).
void quat_to_matrix_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_matrix_flat,
    at::Tensor& grad_quat) {
    AT_DISPATCH_FLOATING_TYPES(quat.scalar_type(), "quat_to_matrix_bwd_cuda", ([&] {
        launch_quat_to_matrix_bwd<scalar_t>(quat, grad_matrix_flat, grad_quat);
    }));
}

// -----------------------------------------------------------------------------
// quat_normalize_safe with the native CUDA epsilon handling.
// -----------------------------------------------------------------------------

// Grid-stride: safe normalize; degenerate rows → identity quaternion in out.
template <typename scalar_t>
__global__ void quat_normalize_safe_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_normalize_safe_fwd_device(i, n, quat, out);
}



// Grid-stride: VJP through safe normalize (grad_out → grad_quat).
template <typename scalar_t>
__global__ void quat_normalize_safe_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_normalize_safe_bwd_device(i, n, quat, grad_out, grad_quat);
}



// Launch safe quaternion normalization on the current CUDA stream.
// Input: quat (N,4) in xyzw order.
// Output: out (N,4), with near-zero rows mapped to the identity quaternion.
template <typename scalar_t>
void launch_quat_normalize_safe_fwd(const at::Tensor& quat, at::Tensor& out) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_normalize_safe_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, quat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for safe quaternion normalization.
// Inputs: quat (N,4) and grad_out (N,4).
// Output: grad_quat (N,4).
template <typename scalar_t>
void launch_quat_normalize_safe_bwd(
    const at::Tensor& quat,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_normalize_safe_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        quat.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_quat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for safe quaternion normalization.
// Input: quat (N,4) in xyzw order.
// Output: out (N,4), with degenerate rows replaced by identity.
void quat_normalize_safe_cuda(const at::Tensor& quat, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(
        quat.scalar_type(),
        "quat_normalize_safe_cuda",
        ([&] { launch_quat_normalize_safe_fwd<scalar_t>(quat, out); }));
}

// Public CUDA entrypoint for the VJP of safe quaternion normalization.
// Inputs: quat (N,4) and grad_out (N,4).
// Output: grad_quat (N,4).
void quat_normalize_safe_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat) {
    AT_DISPATCH_FLOATING_TYPES(quat.scalar_type(), "quat_normalize_safe_bwd_cuda", ([&] {
        launch_quat_normalize_safe_bwd<scalar_t>(quat, grad_out, grad_quat);
    }));
}

// -----------------------------------------------------------------------------
// quat_conjugate — forward (-x,-y,-z,w); backward grad = (-gx,-gy,-gz,gw)
// -----------------------------------------------------------------------------

// Grid-stride: quaternion conjugate per row (xyzw in/out).
template <typename scalar_t>
__global__ void quat_conjugate_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_conjugate_fwd_device(i, n, quat, out);
}



// Grid-stride: VJP for conjugate; grad_out layout matches quaternion.
template <typename scalar_t>
__global__ void quat_conjugate_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_conjugate_bwd_device(i, n, grad_out, grad_quat);
}



// Launch quaternion conjugation on the current CUDA stream.
// Input: quat (N,4) in xyzw order.
// Output: out (N,4) with row i = (-x,-y,-z,w).
template <typename scalar_t>
void launch_quat_conjugate_fwd(const at::Tensor& quat, at::Tensor& out) {
    const int64_t n = quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(quat.device().index());
    quat_conjugate_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, quat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for quaternion conjugation.
// Input: grad_out (N,4).
// Output: grad_quat (N,4) with the conjugate sign pattern applied.
template <typename scalar_t>
void launch_quat_conjugate_bwd(const at::Tensor& grad_out, at::Tensor& grad_quat) {
    const int64_t n = grad_quat.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_quat.device().index());
    quat_conjugate_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, grad_out.data_ptr<scalar_t>(), grad_quat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for quaternion conjugation.
// Input: quat (N,4) in xyzw order.
// Output: out (N,4) with one conjugated quaternion per row.
void quat_conjugate_cuda(const at::Tensor& quat, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(
        quat.scalar_type(), "quat_conjugate_cuda", ([&] { launch_quat_conjugate_fwd<scalar_t>(quat, out); }));
}

// Public CUDA entrypoint for the VJP of quaternion conjugation.
// Inputs: quat (N,4) and grad_out (N,4).
// Output: grad_quat (N,4).
void quat_conjugate_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat) {
    AT_DISPATCH_FLOATING_TYPES(quat.scalar_type(), "quat_conjugate_bwd_cuda", ([&] {
        launch_quat_conjugate_bwd<scalar_t>(grad_out, grad_quat);
    }));
}

// -----------------------------------------------------------------------------
// quat_from_axis_angle from normalized axis-angle inputs.
// q.xyz = axis * sin(angle/2), q.w = cos(angle/2); axis (N,3), angle (N,1)
// -----------------------------------------------------------------------------

// Grid-stride: axis (N,3), angle scalar per row → quat (N,4) xyzw.
template <typename scalar_t>
__global__ void quat_from_axis_angle_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ axis,
    const scalar_t* __restrict__ angle,
    scalar_t* __restrict__ quat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_from_axis_angle_fwd_device(i, n, axis, angle, quat);
}



// Grid-stride: grad_quat (N,4) → grad_axis (N,3), grad_angle (N).
template <typename scalar_t>
__global__ void quat_from_axis_angle_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ axis,
    const scalar_t* __restrict__ angle,
    const scalar_t* __restrict__ grad_quat,
    scalar_t* __restrict__ grad_axis,
    scalar_t* __restrict__ grad_angle) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_from_axis_angle_bwd_device(i, n, axis, angle, grad_quat, grad_axis, grad_angle);
}



// Launch axis-angle -> quaternion conversion on the current CUDA stream.
// Inputs: axis (N,3) and angle (N,1 or equivalent row-wise scalar storage).
// Output: quat (N,4) in xyzw order.
template <typename scalar_t>
void launch_quat_from_axis_angle_fwd(
    const at::Tensor& axis,
    const at::Tensor& angle,
    at::Tensor& quat) {
    const int64_t n = axis.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(axis.device().index());
    quat_from_axis_angle_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        axis.data_ptr<scalar_t>(),
        angle.data_ptr<scalar_t>(),
        quat.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for axis-angle -> quaternion conversion.
// Inputs: axis (N,3), angle row scalars, and grad_quat (N,4).
// Outputs: grad_axis (N,3) and grad_angle (N).
template <typename scalar_t>
void launch_quat_from_axis_angle_bwd(
    const at::Tensor& axis,
    const at::Tensor& angle,
    const at::Tensor& grad_quat,
    at::Tensor& grad_axis,
    at::Tensor& grad_angle) {
    const int64_t n = axis.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(axis.device().index());
    quat_from_axis_angle_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        axis.data_ptr<scalar_t>(),
        angle.data_ptr<scalar_t>(),
        grad_quat.data_ptr<scalar_t>(),
        grad_axis.data_ptr<scalar_t>(),
        grad_angle.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for axis-angle -> quaternion conversion.
// Inputs: axis (N,3) and angle row scalars.
// Output: quat (N,4) in xyzw order.
void quat_from_axis_angle_cuda(const at::Tensor& axis, const at::Tensor& angle, at::Tensor& quat) {
    AT_DISPATCH_FLOATING_TYPES(
        axis.scalar_type(), "quat_from_axis_angle_cuda", ([&] {
            launch_quat_from_axis_angle_fwd<scalar_t>(axis, angle, quat);
        }));
}

// Public CUDA entrypoint for the VJP of axis-angle -> quaternion conversion.
// Inputs: axis (N,3), angle row scalars, and grad_quat (N,4).
// Outputs: grad_axis (N,3) and grad_angle (N).
void quat_from_axis_angle_bwd_cuda(
    const at::Tensor& axis,
    const at::Tensor& angle,
    const at::Tensor& grad_quat,
    at::Tensor& grad_axis,
    at::Tensor& grad_angle) {
    AT_DISPATCH_FLOATING_TYPES(axis.scalar_type(), "quat_from_axis_angle_bwd_cuda", ([&] {
        launch_quat_from_axis_angle_bwd<scalar_t>(axis, angle, grad_quat, grad_axis, grad_angle);
    }));
}

// -----------------------------------------------------------------------------
// quat_lerp with hemisphere flip and normalize-after-blend semantics.
// -----------------------------------------------------------------------------

// Grid-stride: normalized LERP with same scalar t for all rows; q1,q2,out (N,4) xyzw.
template <typename scalar_t>
__global__ void quat_lerp_fwd_kernel(
    int64_t n,
    scalar_t t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_lerp_fwd_device(i, n, t, q1, q2, out);
}



// Grid-stride: VJP for LERP; requires forward `result` buffer (N,4).
template <typename scalar_t>
__global__ void quat_lerp_bwd_kernel(
    int64_t n,
    scalar_t t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ result,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_lerp_bwd_device(i, n, t, q1, q2, result, grad_out, grad_q1, grad_q2);
}



// Launch normalized quaternion LERP on the current CUDA stream.
// Inputs: q1 (N,4), q2 (N,4), and shared scalar t.
// Output: out (N,4) with one normalized blend per row.
template <typename scalar_t>
void launch_quat_lerp_fwd(const at::Tensor& q1, const at::Tensor& q2, double t, at::Tensor& out) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    const scalar_t t_s = static_cast<scalar_t>(t);
    quat_lerp_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, t_s, q1.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for normalized quaternion LERP.
// Inputs: q1 (N,4), q2 (N,4), forward result (N,4), grad_out (N,4), and shared scalar t.
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
template <typename scalar_t>
void launch_quat_lerp_bwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    double t,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    const scalar_t t_s = static_cast<scalar_t>(t);
    quat_lerp_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        t_s,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_q1.data_ptr<scalar_t>(),
        grad_q2.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for normalized quaternion LERP.
// Inputs: q1 (N,4), q2 (N,4), and shared scalar t.
// Output: out (N,4) with one normalized blend per row.
void quat_lerp_cuda(const at::Tensor& q1, const at::Tensor& q2, double t, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(
        q1.scalar_type(), "quat_lerp_cuda", ([&] { launch_quat_lerp_fwd<scalar_t>(q1, q2, t, out); }));
}

// Public CUDA entrypoint for the VJP of normalized quaternion LERP.
// Inputs: q1 (N,4), q2 (N,4), forward result (N,4), grad_out (N,4), and shared scalar t.
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
void quat_lerp_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    double t,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_lerp_bwd_cuda", ([&] {
        launch_quat_lerp_bwd<scalar_t>(q1, q2, result, grad_out, t, grad_q1, grad_q2);
    }));
}

// Grid-stride: SLERP per row; t[i] scalar interpolation parameter; q1,q2,out (N,4).
template <typename scalar_t>
__global__ void quat_slerp_batched_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ t,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_slerp_batched_fwd_device(i, n, q1, q2, t, out);
}



// Grid-stride: VJP batched SLERP; writes grad_t[i] per row; needs forward `result`.
template <typename scalar_t>
__global__ void quat_slerp_batched_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ result,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2,
    scalar_t* __restrict__ grad_t) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_slerp_batched_bwd_device(i, n, q1, q2, t, result, grad_out, grad_q1, grad_q2, grad_t);
}



// Launch batched SLERP on the current CUDA stream.
// Inputs: q1 (N,4), q2 (N,4), and per-row t values.
// Output: out (N,4) with one spherical interpolation result per row.
template <typename scalar_t>
void launch_quat_slerp_batched_fwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    at::Tensor& out) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_slerp_batched_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        t.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for batched SLERP.
// Inputs: q1 (N,4), q2 (N,4), t (N), forward result (N,4), and grad_out (N,4).
// Outputs: grad_q1 (N,4), grad_q2 (N,4), and grad_t (N).
template <typename scalar_t>
void launch_quat_slerp_batched_bwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_slerp_batched_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        t.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_q1.data_ptr<scalar_t>(),
        grad_q2.data_ptr<scalar_t>(),
        grad_t.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for batched SLERP.
// Inputs: q1 (N,4), q2 (N,4), and per-row interpolation parameters t.
// Output: out (N,4) with one SLERP result per row.
void quat_slerp_batched_cuda(const at::Tensor& q1, const at::Tensor& q2, const at::Tensor& t, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_slerp_batched_cuda", ([&] {
        launch_quat_slerp_batched_fwd<scalar_t>(q1, q2, t, out);
    }));
}

// Public CUDA entrypoint for the VJP of batched SLERP.
// Inputs: q1 (N,4), q2 (N,4), t (N), forward result (N,4), and grad_out (N,4).
// Outputs: grad_q1 (N,4), grad_q2 (N,4), and grad_t (N).
void quat_slerp_batched_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_slerp_batched_bwd_cuda", ([&] {
        launch_quat_slerp_batched_bwd<scalar_t>(q1, q2, t, result, grad_out, grad_q1, grad_q2, grad_t);
    }));
}

// Grid-stride: scalar angular distance per row → dist_out[i] (radians); q1,q2 (N,4) xyzw.
template <typename scalar_t>
__global__ void quat_angular_distance_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ dist_out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_angular_distance_fwd_device(i, n, q1, q2, dist_out);
}



// Grid-stride: grad_dist (N,) → grad_q1, grad_q2 (N,4).
template <typename scalar_t>
__global__ void quat_angular_distance_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_dist,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_angular_distance_bwd_device(i, n, q1, q2, grad_dist, grad_q1, grad_q2);
}



// Launch angular distance evaluation on the current CUDA stream.
// Inputs: q1 (N,4) and q2 (N,4) in xyzw order.
// Output: out (N) with one angular distance in radians per row.
template <typename scalar_t>
void launch_quat_angular_distance_fwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    at::Tensor& out) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_angular_distance_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for angular distance evaluation.
// Inputs: q1 (N,4), q2 (N,4), and grad_dist (N).
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
template <typename scalar_t>
void launch_quat_angular_distance_bwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_dist,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_angular_distance_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        grad_dist.data_ptr<scalar_t>(),
        grad_q1.data_ptr<scalar_t>(),
        grad_q2.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Public CUDA entrypoint for quaternion angular distance.
// Inputs: q1 (N,4) and q2 (N,4) in xyzw order.
// Output: out (N) with one angular distance in radians per row.
void quat_angular_distance_cuda(const at::Tensor& q1, const at::Tensor& q2, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_angular_distance_cuda", ([&] {
        launch_quat_angular_distance_fwd<scalar_t>(q1, q2, out);
    }));
}

// Public CUDA entrypoint for the VJP of quaternion angular distance.
// Inputs: q1 (N,4), q2 (N,4), and grad_dist (N).
// Outputs: grad_q1 (N,4) and grad_q2 (N,4).
void quat_angular_distance_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_dist,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_angular_distance_bwd_cuda", ([&] {
        launch_quat_angular_distance_bwd<scalar_t>(q1, q2, grad_dist, grad_q1, grad_q2);
    }));
}

// -----------------------------------------------------------------------------
// quat_manifold_interp:
// q1 * so3Exp(t * so3Log(inverse(q1) * q2)), inverse = conjugate (unit quat)
// -----------------------------------------------------------------------------

namespace {

// Grid-stride: SO(3) geodesic interp; per-row t[i]; q1,q2,out (N,4); inner double precision in device body.
template <typename scalar_t>
__global__ void quat_manifold_interp_fwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_manifold_interp_fwd_device(i, n, t, q1, q2, out);
}



// Grid-stride: VJP manifold interp; grad_out (N,4) → grad_q1, grad_q2, grad_t.
template <typename scalar_t>
__global__ void quat_manifold_interp_bwd_kernel(
    int64_t n,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2,
    scalar_t* __restrict__ grad_t) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    quat_manifold_interp_bwd_device(i, n, t, q1, q2, grad_out, grad_q1, grad_q2, grad_t);
}



// Launch manifold quaternion interpolation on the current CUDA stream.
// Inputs: q1 (N,4), q2 (N,4), and per-row t (N,1).
// Output: out (N,4) with one geodesic interpolation result per row.
template <typename scalar_t>
void launch_quat_manifold_interp_fwd(
    const at::Tensor& q1, const at::Tensor& q2, const at::Tensor& t, at::Tensor& out) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_manifold_interp_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        t.data_ptr<scalar_t>(),
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for manifold quaternion interpolation.
// Inputs: q1 (N,4), q2 (N,4), t (N,1), and grad_out (N,4).
// Outputs: grad_q1 (N,4), grad_q2 (N,4), grad_t (N,1).
template <typename scalar_t>
void launch_quat_manifold_interp_bwd(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t) {
    const int64_t n = q1.size(0);
    if (n == 0) {
        return;
    }
    const int threads = kThreadsPerBlock;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q1.device().index());
    quat_manifold_interp_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n,
        t.data_ptr<scalar_t>(),
        q1.data_ptr<scalar_t>(),
        q2.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        grad_q1.data_ptr<scalar_t>(),
        grad_q2.data_ptr<scalar_t>(),
        grad_t.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// Public CUDA entrypoint for manifold quaternion interpolation.
// Inputs: q1 (N,4), q2 (N,4), and per-row t (N,1).
// Output: out (N,4) with one geodesic interpolation result per row.
void quat_manifold_interp_cuda(const at::Tensor& q1, const at::Tensor& q2, const at::Tensor& t, at::Tensor& out) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_manifold_interp_cuda", ([&] {
        launch_quat_manifold_interp_fwd<scalar_t>(q1, q2, t, out);
    }));
}

// Public CUDA entrypoint for the VJP of manifold quaternion interpolation.
// Inputs: q1 (N,4), q2 (N,4), t (N,1), and grad_out (N,4).
// Outputs: grad_q1 (N,4), grad_q2 (N,4), grad_t (N,1).
void quat_manifold_interp_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t) {
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "quat_manifold_interp_bwd_cuda", ([&] {
        launch_quat_manifold_interp_bwd<scalar_t>(q1, q2, t, grad_out, grad_q1, grad_q2, grad_t);
    }));
}
