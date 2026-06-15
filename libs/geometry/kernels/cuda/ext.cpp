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

#include <torch/extension.h>

void quat_normalize_safe_cuda(const at::Tensor& quat, at::Tensor& out);
void quat_normalize_safe_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat);

void quat_multiply_cuda(const at::Tensor& q1, const at::Tensor& q2, at::Tensor& out);
void quat_multiply_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2);

void quat_rotate_vector_cuda(const at::Tensor& quat, const at::Tensor& vec, at::Tensor& out);
void quat_rotate_vector_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& vec,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat,
    at::Tensor& grad_vec);

void quat_to_matrix_cuda(const at::Tensor& quat, at::Tensor& out_flat);
void quat_to_matrix_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_matrix_flat,
    at::Tensor& grad_quat);

void quat_conjugate_cuda(const at::Tensor& quat, at::Tensor& out);
void quat_conjugate_bwd_cuda(
    const at::Tensor& quat,
    const at::Tensor& grad_out,
    at::Tensor& grad_quat);

void quat_lerp_cuda(const at::Tensor& q1, const at::Tensor& q2, double t, at::Tensor& out);
void quat_lerp_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    double t,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2);

void quat_slerp_batched_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    at::Tensor& out);
void quat_slerp_batched_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& result,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t);

void quat_from_axis_angle_cuda(const at::Tensor& axis, const at::Tensor& angle, at::Tensor& quat);
void quat_from_axis_angle_bwd_cuda(
    const at::Tensor& axis,
    const at::Tensor& angle,
    const at::Tensor& grad_quat,
    at::Tensor& grad_axis,
    at::Tensor& grad_angle);

void quat_angular_distance_cuda(const at::Tensor& q1, const at::Tensor& q2, at::Tensor& out);
void quat_angular_distance_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& grad_dist,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2);

void quat_manifold_interp_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    at::Tensor& out);
void quat_manifold_interp_bwd_cuda(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t,
    const at::Tensor& grad_out,
    at::Tensor& grad_q1,
    at::Tensor& grad_q2,
    at::Tensor& grad_t);

void se3pose_transform_point_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out);
void se3pose_transform_direction_cuda(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out);
void se3pose_inverse_transform_point_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& point,
    at::Tensor& out);
void se3pose_inverse_transform_direction_cuda(
    const at::Tensor& rotation,
    const at::Tensor& direction,
    at::Tensor& out);

void se3pose_to_matrix_pack_cuda(
    const at::Tensor& translation,
    const at::Tensor& R9,
    at::Tensor& out_flat);
void se3pose_to_inverse_matrix_cuda(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    bool wxyz_format,
    at::Tensor& out_flat);

void se3pose_from_matrix_cuda(
    const at::Tensor& matrix,
    at::Tensor& translation,
    at::Tensor& rotation);
void se3pose_from_matrix_bwd_cuda(
    const at::Tensor& matrix,
    const at::Tensor& grad_translation,
    const at::Tensor& grad_rotation,
    at::Tensor& grad_matrix);

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
    at::Tensor& result_out_of_bounds);

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
    at::Tensor& grad_query_time);

void trajectory_get_rotation_2poses_cuda(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    at::Tensor& result_quat,
    at::Tensor& result_out_of_bounds);

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
    at::Tensor& grad_query_time);

void trajectory_transform_point_1pose_cuda(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time,
    at::Tensor& result_point,
    at::Tensor& result_out_of_bounds);

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
    at::Tensor& grad_query_time);

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
    at::Tensor& output_poses);

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK((x).dtype() == torch::kFloat32 || (x).dtype() == torch::kFloat64, #x " must be float32 or float64")

static void check_quat_normalize_safe_inputs(const at::Tensor& quat) {
    CHECK_CUDA_TENSOR(quat);
    CHECK_CONTIGUOUS(quat);
    CHECK_FLOATING(quat);
    TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4, "quat must be (N, 4)");
}

torch::Tensor quat_normalize_safe(const torch::Tensor& quat) {
    check_quat_normalize_safe_inputs(quat);
    auto out = torch::empty_like(quat);
    quat_normalize_safe_cuda(quat, out);
    return out;
}

torch::Tensor quat_normalize_safe_bwd(const torch::Tensor& quat, const torch::Tensor& grad_out) {
    check_quat_normalize_safe_inputs(quat);
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == quat.device(), "grad_out must be on the same device as quat");
    TORCH_CHECK(grad_out.dtype() == quat.dtype(), "grad_out dtype must match quat");
    TORCH_CHECK(grad_out.sizes() == quat.sizes(), "grad_out shape must match quat");
    auto grad_quat = torch::empty_like(quat);
    quat_normalize_safe_bwd_cuda(quat, grad_out, grad_quat);
    return grad_quat;
}

static void check_quat_multiply_inputs(const at::Tensor& q1, const at::Tensor& q2) {
    CHECK_CUDA_TENSOR(q1);
    CHECK_CUDA_TENSOR(q2);
    CHECK_CONTIGUOUS(q1);
    CHECK_CONTIGUOUS(q2);
    CHECK_FLOATING(q1);
    TORCH_CHECK(q1.device() == q2.device(), "q1 and q2 must be on the same CUDA device");
    TORCH_CHECK(q2.dtype() == q1.dtype(), "q1 and q2 must have the same dtype");
    TORCH_CHECK(q1.dim() == 2 && q1.size(1) == 4, "q1 must be (N, 4)");
    TORCH_CHECK(q2.sizes() == q1.sizes(), "q1 and q2 must have the same shape");
}

torch::Tensor quat_multiply(const torch::Tensor& q1, const torch::Tensor& q2) {
    check_quat_multiply_inputs(q1, q2);
    auto out = torch::empty_like(q1);
    quat_multiply_cuda(q1, q2, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
quat_multiply_bwd(const torch::Tensor& q1, const torch::Tensor& q2, const torch::Tensor& grad_out) {
    check_quat_multiply_inputs(q1, q2);
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == q1.device(), "grad_out must be on the same device as q1");
    TORCH_CHECK(grad_out.dtype() == q1.dtype(), "grad_out dtype must match q1");
    TORCH_CHECK(grad_out.sizes() == q1.sizes(), "grad_out shape must match q1");
    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    quat_multiply_bwd_cuda(q1, q2, grad_out, grad_q1, grad_q2);
    return std::make_tuple(grad_q1, grad_q2);
}

static void check_quat_rotate_vector_inputs(const at::Tensor& quat, const at::Tensor& vec) {
    CHECK_CUDA_TENSOR(quat);
    CHECK_CUDA_TENSOR(vec);
    CHECK_CONTIGUOUS(quat);
    CHECK_CONTIGUOUS(vec);
    CHECK_FLOATING(quat);
    TORCH_CHECK(quat.device() == vec.device(), "quat and vec must be on the same CUDA device");
    TORCH_CHECK(vec.dtype() == quat.dtype(), "quat and vec must have the same dtype");
    TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4, "quat must be (N, 4)");
    TORCH_CHECK(vec.dim() == 2 && vec.size(1) == 3, "vec must be (N, 3)");
    TORCH_CHECK(vec.size(0) == quat.size(0), "quat and vec must have the same batch size N");
}

torch::Tensor quat_rotate_vector(const torch::Tensor& quat, const torch::Tensor& vec) {
    check_quat_rotate_vector_inputs(quat, vec);
    auto out = torch::empty_like(vec);
    quat_rotate_vector_cuda(quat, vec, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
quat_rotate_vector_bwd(
    const torch::Tensor& quat,
    const torch::Tensor& vec,
    const torch::Tensor& grad_out) {
    check_quat_rotate_vector_inputs(quat, vec);
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == quat.device(), "grad_out must be on the same device as quat");
    TORCH_CHECK(grad_out.dtype() == quat.dtype(), "grad_out dtype must match quat");
    TORCH_CHECK(grad_out.sizes() == vec.sizes(), "grad_out shape must match vec");
    auto grad_quat = torch::empty_like(quat);
    auto grad_vec = torch::empty_like(vec);
    quat_rotate_vector_bwd_cuda(quat, vec, grad_out, grad_quat, grad_vec);
    return std::make_tuple(grad_quat, grad_vec);
}

static void check_quat_to_matrix_inputs(const at::Tensor& quat) {
    CHECK_CUDA_TENSOR(quat);
    CHECK_CONTIGUOUS(quat);
    CHECK_FLOATING(quat);
    TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4, "quat must be (N, 4)");
}

torch::Tensor quat_to_matrix(const torch::Tensor& quat) {
    check_quat_to_matrix_inputs(quat);
    const int64_t n = quat.size(0);
    auto opts = quat.options();
    auto out = torch::empty({n, 9}, opts);
    quat_to_matrix_cuda(quat, out);
    return out;
}

torch::Tensor quat_to_matrix_bwd(const torch::Tensor& quat, const torch::Tensor& grad_matrix_flat) {
    check_quat_to_matrix_inputs(quat);
    CHECK_CUDA_TENSOR(grad_matrix_flat);
    CHECK_CONTIGUOUS(grad_matrix_flat);
    TORCH_CHECK(grad_matrix_flat.device() == quat.device(), "grad_matrix_flat must be on the same device as quat");
    TORCH_CHECK(grad_matrix_flat.dtype() == quat.dtype(), "grad_matrix_flat dtype must match quat");
    TORCH_CHECK(
        grad_matrix_flat.dim() == 2 && grad_matrix_flat.size(0) == quat.size(0) && grad_matrix_flat.size(1) == 9,
        "grad_matrix_flat must be (N, 9) with same N as quat");
    auto grad_quat = torch::empty_like(quat);
    quat_to_matrix_bwd_cuda(quat, grad_matrix_flat, grad_quat);
    return grad_quat;
}

static void check_quat_conjugate_inputs(const at::Tensor& quat) {
    CHECK_CUDA_TENSOR(quat);
    CHECK_CONTIGUOUS(quat);
    CHECK_FLOATING(quat);
    TORCH_CHECK(quat.dim() == 2 && quat.size(1) == 4, "quat must be (N, 4)");
}

torch::Tensor quat_conjugate(const torch::Tensor& quat) {
    check_quat_conjugate_inputs(quat);
    auto out = torch::empty_like(quat);
    quat_conjugate_cuda(quat, out);
    return out;
}

torch::Tensor quat_conjugate_bwd(const torch::Tensor& quat, const torch::Tensor& grad_out) {
    check_quat_conjugate_inputs(quat);
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == quat.device(), "grad_out must be on the same device as quat");
    TORCH_CHECK(grad_out.dtype() == quat.dtype(), "grad_out dtype must match quat");
    TORCH_CHECK(grad_out.sizes() == quat.sizes(), "grad_out shape must match quat");
    auto grad_quat = torch::empty_like(quat);
    quat_conjugate_bwd_cuda(quat, grad_out, grad_quat);
    return grad_quat;
}

static void check_quat_lerp_inputs(const at::Tensor& q1, const at::Tensor& q2) {
    check_quat_multiply_inputs(q1, q2);
}

torch::Tensor quat_lerp(const torch::Tensor& q1, const torch::Tensor& q2, double t) {
    check_quat_lerp_inputs(q1, q2);
    auto out = torch::empty_like(q1);
    quat_lerp_cuda(q1, q2, t, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> quat_lerp_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& result,
    const torch::Tensor& grad_out,
    double t) {
    check_quat_lerp_inputs(q1, q2);
    CHECK_CUDA_TENSOR(result);
    CHECK_CONTIGUOUS(result);
    TORCH_CHECK(result.sizes() == q1.sizes(), "result shape must match q1");
    TORCH_CHECK(result.dtype() == q1.dtype(), "result dtype must match q1");
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == q1.device(), "grad_out must be on the same device as q1");
    TORCH_CHECK(grad_out.dtype() == q1.dtype(), "grad_out dtype must match q1");
    TORCH_CHECK(grad_out.sizes() == q1.sizes(), "grad_out shape must match q1");
    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    quat_lerp_bwd_cuda(q1, q2, result, grad_out, t, grad_q1, grad_q2);
    return std::make_tuple(grad_q1, grad_q2);
}

static void check_quat_slerp_batched_inputs(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t) {
    check_quat_multiply_inputs(q1, q2);
    CHECK_CUDA_TENSOR(t);
    CHECK_CONTIGUOUS(t);
    TORCH_CHECK(t.dtype() == q1.dtype(), "t must have the same dtype as q1");
    TORCH_CHECK(t.device() == q1.device(), "t must be on the same device as q1");
    TORCH_CHECK(t.dim() == 2 && t.size(0) == q1.size(0) && t.size(1) == 1, "t must be (N, 1) with same N as q1");
}

torch::Tensor quat_slerp_batched(const torch::Tensor& q1, const torch::Tensor& q2, const torch::Tensor& t) {
    check_quat_slerp_batched_inputs(q1, q2, t);
    auto out = torch::empty_like(q1);
    quat_slerp_batched_cuda(q1, q2, t, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quat_slerp_batched_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& t,
    const torch::Tensor& result,
    const torch::Tensor& grad_out) {
    check_quat_slerp_batched_inputs(q1, q2, t);
    CHECK_CUDA_TENSOR(result);
    CHECK_CONTIGUOUS(result);
    TORCH_CHECK(result.sizes() == q1.sizes(), "result shape must match q1");
    TORCH_CHECK(result.dtype() == q1.dtype(), "result dtype must match q1");
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == q1.device(), "grad_out must be on the same device as q1");
    TORCH_CHECK(grad_out.dtype() == q1.dtype(), "grad_out dtype must match q1");
    TORCH_CHECK(grad_out.sizes() == q1.sizes(), "grad_out shape must match q1");
    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    auto grad_t = torch::empty_like(t);
    quat_slerp_batched_bwd_cuda(q1, q2, t, result, grad_out, grad_q1, grad_q2, grad_t);
    return std::make_tuple(grad_q1, grad_q2, grad_t);
}

static void check_quat_from_axis_angle_inputs(const at::Tensor& axis, const at::Tensor& angle) {
    CHECK_CUDA_TENSOR(axis);
    CHECK_CUDA_TENSOR(angle);
    CHECK_CONTIGUOUS(axis);
    CHECK_CONTIGUOUS(angle);
    CHECK_FLOATING(axis);
    TORCH_CHECK(axis.device() == angle.device(), "axis and angle must be on the same CUDA device");
    TORCH_CHECK(angle.dtype() == axis.dtype(), "axis and angle must have the same dtype");
    TORCH_CHECK(axis.dim() == 2 && axis.size(1) == 3, "axis must be (N, 3)");
    TORCH_CHECK(angle.dim() == 2 && angle.size(0) == axis.size(0) && angle.size(1) == 1, "angle must be (N, 1) with same N as axis");
}

torch::Tensor quat_from_axis_angle(const torch::Tensor& axis, const torch::Tensor& angle) {
    check_quat_from_axis_angle_inputs(axis, angle);
    const int64_t n = axis.size(0);
    auto opts = axis.options();
    auto quat = torch::empty({n, 4}, opts);
    quat_from_axis_angle_cuda(axis, angle, quat);
    return quat;
}

std::tuple<torch::Tensor, torch::Tensor> quat_from_axis_angle_bwd(
    const torch::Tensor& axis,
    const torch::Tensor& angle,
    const torch::Tensor& grad_quat) {
    check_quat_from_axis_angle_inputs(axis, angle);
    CHECK_CUDA_TENSOR(grad_quat);
    CHECK_CONTIGUOUS(grad_quat);
    TORCH_CHECK(grad_quat.device() == axis.device(), "grad_quat must be on the same device as axis");
    TORCH_CHECK(grad_quat.dtype() == axis.dtype(), "grad_quat dtype must match axis");
    TORCH_CHECK(grad_quat.dim() == 2 && grad_quat.size(0) == axis.size(0) && grad_quat.size(1) == 4, "grad_quat must be (N, 4) with same N as axis");
    auto grad_axis = torch::empty_like(axis);
    auto grad_angle = torch::empty_like(angle);
    quat_from_axis_angle_bwd_cuda(axis, angle, grad_quat, grad_axis, grad_angle);
    return std::make_tuple(grad_axis, grad_angle);
}

torch::Tensor quat_angular_distance(const torch::Tensor& q1, const torch::Tensor& q2) {
    // Same (N, 4) constraints as quat_multiply.
    check_quat_multiply_inputs(q1, q2);
    const int64_t n = q1.size(0);
    auto out = torch::empty({n, 1}, q1.options());
    if (n > 0) {
        quat_angular_distance_cuda(q1, q2, out);
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> quat_angular_distance_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& grad_dist) {
    // Same (N, 4) constraints as quat_multiply.
    check_quat_multiply_inputs(q1, q2);
    CHECK_CUDA_TENSOR(grad_dist);
    CHECK_CONTIGUOUS(grad_dist);
    TORCH_CHECK(grad_dist.device() == q1.device(), "grad_dist must be on the same device as q1");
    TORCH_CHECK(grad_dist.dtype() == q1.dtype(), "grad_dist dtype must match q1");
    TORCH_CHECK(
        grad_dist.dim() == 2 && grad_dist.size(0) == q1.size(0) && grad_dist.size(1) == 1,
        "grad_dist must be (N, 1) with same N as q1");
    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    if (q1.size(0) > 0) {
        quat_angular_distance_bwd_cuda(q1, q2, grad_dist, grad_q1, grad_q2);
    }
    return std::make_tuple(grad_q1, grad_q2);
}

static void check_quat_manifold_interp_inputs(
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t) {
    check_quat_slerp_batched_inputs(q1, q2, t);
}

torch::Tensor quat_manifold_interp(const torch::Tensor& q1, const torch::Tensor& q2, const torch::Tensor& t) {
    check_quat_manifold_interp_inputs(q1, q2, t);
    auto out = torch::empty_like(q1);
    if (q1.size(0) > 0) {
        quat_manifold_interp_cuda(q1, q2, t, out);
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quat_manifold_interp_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& t,
    const torch::Tensor& grad_out) {
    check_quat_manifold_interp_inputs(q1, q2, t);
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == q1.device(), "grad_out must be on the same device as q1");
    TORCH_CHECK(grad_out.dtype() == q1.dtype(), "grad_out dtype must match q1");
    TORCH_CHECK(grad_out.sizes() == q1.sizes(), "grad_out shape must match q1");
    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    auto grad_t = torch::empty_like(t);
    if (q1.size(0) > 0) {
        quat_manifold_interp_bwd_cuda(q1, q2, t, grad_out, grad_q1, grad_q2, grad_t);
    }
    return std::make_tuple(grad_q1, grad_q2, grad_t);
}

// -----------------------------------------------------------------------------
// SE3 pose transforms (translation (N,3), rotation quaternion xyzw (N,4), vec (N,3))
// -----------------------------------------------------------------------------

static void check_se3_pose_vec3_inputs(
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& vec) {
    CHECK_CUDA_TENSOR(translation);
    CHECK_CUDA_TENSOR(rotation);
    CHECK_CUDA_TENSOR(vec);
    CHECK_CONTIGUOUS(translation);
    CHECK_CONTIGUOUS(rotation);
    CHECK_CONTIGUOUS(vec);
    CHECK_FLOATING(translation);
    TORCH_CHECK(
        translation.device() == rotation.device() && translation.device() == vec.device(),
        "translation, rotation, and vec must be on the same CUDA device");
    TORCH_CHECK(
        rotation.dtype() == translation.dtype() && vec.dtype() == translation.dtype(),
        "translation, rotation, and vec must have the same dtype");
    TORCH_CHECK(
        translation.dim() == 2 && translation.size(1) == 3, "translation must be (N, 3)");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(1) == 4, "rotation must be (N, 4) xyzw");
    TORCH_CHECK(vec.dim() == 2 && vec.size(1) == 3, "vec must be (N, 3)");
    TORCH_CHECK(
        rotation.size(0) == translation.size(0) && vec.size(0) == translation.size(0),
        "translation, rotation, and vec must have the same batch size N");
}

static void check_se3_pose_vec3_grad_out(const at::Tensor& vec, const at::Tensor& grad_out) {
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == vec.device(), "grad_out must be on the same device as vec");
    TORCH_CHECK(grad_out.dtype() == vec.dtype(), "grad_out dtype must match vec");
    TORCH_CHECK(grad_out.sizes() == vec.sizes(), "grad_out shape must match vec");
}

torch::Tensor se3pose_transform_point(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& point) {
    check_se3_pose_vec3_inputs(translation, rotation, point);
    auto out = torch::empty_like(point);
    se3pose_transform_point_cuda(translation, rotation, point, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3pose_transform_point_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& point,
    const torch::Tensor& grad_out) {
    check_se3_pose_vec3_inputs(translation, rotation, point);
    check_se3_pose_vec3_grad_out(point, grad_out);
    const int64_t n = point.size(0);
    auto grad_translation = grad_out.clone();
    auto grad_rotation = torch::empty_like(rotation);
    auto grad_point = torch::empty_like(point);
    if (n > 0) {
        quat_rotate_vector_bwd_cuda(rotation, point, grad_out, grad_rotation, grad_point);
    }
    return std::make_tuple(grad_translation, grad_rotation, grad_point);
}

torch::Tensor se3pose_transform_direction(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& direction) {
    check_se3_pose_vec3_inputs(translation, rotation, direction);
    auto out = torch::empty_like(direction);
    se3pose_transform_direction_cuda(rotation, direction, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3pose_transform_direction_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& direction,
    const torch::Tensor& grad_out) {
    check_se3_pose_vec3_inputs(translation, rotation, direction);
    check_se3_pose_vec3_grad_out(direction, grad_out);
    const int64_t n = direction.size(0);
    auto grad_translation = torch::zeros_like(translation);
    auto grad_rotation = torch::empty_like(rotation);
    auto grad_direction = torch::empty_like(direction);
    if (n > 0) {
        quat_rotate_vector_bwd_cuda(rotation, direction, grad_out, grad_rotation, grad_direction);
    }
    return std::make_tuple(grad_translation, grad_rotation, grad_direction);
}

torch::Tensor se3pose_inverse_transform_point(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& point) {
    check_se3_pose_vec3_inputs(translation, rotation, point);
    auto out = torch::empty_like(point);
    se3pose_inverse_transform_point_cuda(translation, rotation, point, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3pose_inverse_transform_point_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& point,
    const torch::Tensor& grad_out) {
    check_se3_pose_vec3_inputs(translation, rotation, point);
    check_se3_pose_vec3_grad_out(point, grad_out);
    const int64_t n = point.size(0);
    auto grad_rotation = torch::empty_like(rotation);
    auto grad_point = torch::empty_like(point);
    auto grad_translation = torch::empty_like(translation);
    if (n > 0) {
        auto diff = torch::sub(point, translation);
        auto q_conj = torch::empty_like(rotation);
        quat_conjugate_cuda(rotation, q_conj);
        auto grad_q_conj = torch::empty_like(rotation);
        auto grad_diff = torch::empty_like(point);
        quat_rotate_vector_bwd_cuda(q_conj, diff, grad_out, grad_q_conj, grad_diff);
        grad_point.copy_(grad_diff);
        grad_translation.copy_(torch::neg(grad_diff));
        quat_conjugate_bwd_cuda(rotation, grad_q_conj, grad_rotation);
    } else {
        grad_translation.zero_();
        grad_rotation.zero_();
        grad_point.zero_();
    }
    return std::make_tuple(grad_translation, grad_rotation, grad_point);
}

torch::Tensor se3pose_inverse_transform_direction(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& direction) {
    check_se3_pose_vec3_inputs(translation, rotation, direction);
    auto out = torch::empty_like(direction);
    se3pose_inverse_transform_direction_cuda(rotation, direction, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3pose_inverse_transform_direction_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& direction,
    const torch::Tensor& grad_out) {
    check_se3_pose_vec3_inputs(translation, rotation, direction);
    check_se3_pose_vec3_grad_out(direction, grad_out);
    const int64_t n = direction.size(0);
    auto grad_translation = torch::zeros_like(translation);
    auto grad_rotation = torch::empty_like(rotation);
    auto grad_direction = torch::empty_like(direction);
    if (n > 0) {
        auto q_conj = torch::empty_like(rotation);
        quat_conjugate_cuda(rotation, q_conj);
        auto grad_q_conj = torch::empty_like(rotation);
        quat_rotate_vector_bwd_cuda(q_conj, direction, grad_out, grad_q_conj, grad_direction);
        quat_conjugate_bwd_cuda(rotation, grad_q_conj, grad_rotation);
    }
    return std::make_tuple(grad_translation, grad_rotation, grad_direction);
}

// -----------------------------------------------------------------------------
// SE3 pose -> 4x4 matrix and inverse (translation (N,3), rotation (N,4))
// -----------------------------------------------------------------------------

static void check_se3_pose_pair_inputs(const at::Tensor& translation, const at::Tensor& rotation) {
    CHECK_CUDA_TENSOR(translation);
    CHECK_CUDA_TENSOR(rotation);
    CHECK_CONTIGUOUS(translation);
    CHECK_CONTIGUOUS(rotation);
    CHECK_FLOATING(translation);
    TORCH_CHECK(
        translation.device() == rotation.device(),
        "translation and rotation must be on the same CUDA device");
    TORCH_CHECK(rotation.dtype() == translation.dtype(), "translation and rotation must have the same dtype");
    TORCH_CHECK(
        translation.dim() == 2 && translation.size(1) == 3, "translation must be (N, 3)");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(1) == 4, "rotation must be (N, 4)");
    TORCH_CHECK(rotation.size(0) == translation.size(0), "translation and rotation must have the same batch size N");
}

static void check_se3_pose_matrix_grad_out(const at::Tensor& translation, const at::Tensor& grad_out) {
    CHECK_CUDA_TENSOR(grad_out);
    CHECK_CONTIGUOUS(grad_out);
    TORCH_CHECK(grad_out.device() == translation.device(), "grad_out must be on the same device as translation");
    TORCH_CHECK(grad_out.dtype() == translation.dtype(), "grad_out dtype must match translation");
    const int64_t n = translation.size(0);
    TORCH_CHECK(
        grad_out.dim() == 2 && grad_out.size(0) == n && grad_out.size(1) == 16,
        "grad_out must be (N, 16) row-major 4x4 with same N as translation");
}

torch::Tensor se3pose_to_matrix(const torch::Tensor& translation, const torch::Tensor& rotation) {
    check_se3_pose_pair_inputs(translation, rotation);
    const int64_t n = translation.size(0);
    auto opts = translation.options();
    auto out = torch::empty({n, 16}, opts);
    if (n == 0) {
        return out;
    }
    auto R9 = torch::empty({n, 9}, opts);
    quat_to_matrix_cuda(rotation, R9);
    se3pose_to_matrix_pack_cuda(translation, R9, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> se3pose_to_matrix_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    const torch::Tensor& grad_out) {
    check_se3_pose_pair_inputs(translation, rotation);
    check_se3_pose_matrix_grad_out(translation, grad_out);
    const int64_t n = translation.size(0);
    auto grad_translation = torch::empty_like(translation);
    auto grad_rotation = torch::empty_like(rotation);
    if (n == 0) {
        return std::make_tuple(grad_translation, grad_rotation);
    }
    auto g = grad_out;
    grad_translation.select(1, 0) = g.select(1, 3);
    grad_translation.select(1, 1) = g.select(1, 7);
    grad_translation.select(1, 2) = g.select(1, 11);
    auto grad_R9 = torch::empty({n, 9}, rotation.options());
    grad_R9.select(1, 0) = g.select(1, 0);
    grad_R9.select(1, 1) = g.select(1, 1);
    grad_R9.select(1, 2) = g.select(1, 2);
    grad_R9.select(1, 3) = g.select(1, 4);
    grad_R9.select(1, 4) = g.select(1, 5);
    grad_R9.select(1, 5) = g.select(1, 6);
    grad_R9.select(1, 6) = g.select(1, 8);
    grad_R9.select(1, 7) = g.select(1, 9);
    grad_R9.select(1, 8) = g.select(1, 10);
    quat_to_matrix_bwd_cuda(rotation, grad_R9, grad_rotation);
    return std::make_tuple(grad_translation, grad_rotation);
}

torch::Tensor se3pose_to_inverse_matrix(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    bool wxyz_format) {
    check_se3_pose_pair_inputs(translation, rotation);
    const int64_t n = translation.size(0);
    auto out = torch::empty({n, 16}, translation.options());
    if (n > 0) {
        se3pose_to_inverse_matrix_cuda(translation, rotation, wxyz_format, out);
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> se3pose_to_inverse_matrix_bwd(
    const torch::Tensor& translation,
    const torch::Tensor& rotation,
    bool wxyz_format,
    const torch::Tensor& grad_out) {
    check_se3_pose_pair_inputs(translation, rotation);
    check_se3_pose_matrix_grad_out(translation, grad_out);
    const int64_t n = translation.size(0);
    auto grad_translation = torch::empty_like(translation);
    auto grad_rotation = torch::empty_like(rotation);
    if (n == 0) {
        return std::make_tuple(grad_translation, grad_rotation);
    }
    auto rot_xyzw = wxyz_format ? torch::cat({rotation.narrow(1, 1, 3), rotation.narrow(1, 0, 1)}, 1) : rotation;
    auto g = grad_out;
    auto R9 = torch::empty({n, 9}, rotation.options());
    quat_to_matrix_cuda(rot_xyzw, R9);
    auto R = R9.view({n, 3, 3});
    auto gu = torch::stack({g.select(1, 3), g.select(1, 7), g.select(1, 11)}, /*dim=*/1).unsqueeze(-1);
    grad_translation.copy_(torch::bmm(-R, gu).squeeze(-1));
    auto grad_R9 = torch::empty({n, 9}, rotation.options());
    grad_R9.select(1, 0) = g.select(1, 0);
    grad_R9.select(1, 1) = g.select(1, 4);
    grad_R9.select(1, 2) = g.select(1, 8);
    grad_R9.select(1, 3) = g.select(1, 1);
    grad_R9.select(1, 4) = g.select(1, 5);
    grad_R9.select(1, 5) = g.select(1, 9);
    grad_R9.select(1, 6) = g.select(1, 2);
    grad_R9.select(1, 7) = g.select(1, 6);
    grad_R9.select(1, 8) = g.select(1, 10);
    // u = -R^T t also depends on R: dL/dR_{ab} += -gu_b * t_a (row a, col b of R).
    auto gu_st = torch::stack({g.select(1, 3), g.select(1, 7), g.select(1, 11)}, /*dim=*/1);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            grad_R9.select(1, a * 3 + b).sub_(translation.select(1, a) * gu_st.select(1, b));
        }
    }
    auto grad_rot_xyzw = torch::empty_like(rot_xyzw);
    quat_to_matrix_bwd_cuda(rot_xyzw, grad_R9, grad_rot_xyzw);
    if (wxyz_format) {
        grad_rotation.select(1, 0).copy_(grad_rot_xyzw.select(1, 3));
        grad_rotation.select(1, 1).copy_(grad_rot_xyzw.select(1, 0));
        grad_rotation.select(1, 2).copy_(grad_rot_xyzw.select(1, 1));
        grad_rotation.select(1, 3).copy_(grad_rot_xyzw.select(1, 2));
    } else {
        grad_rotation.copy_(grad_rot_xyzw);
    }
    return std::make_tuple(grad_translation, grad_rotation);
}

// -----------------------------------------------------------------------------
// SE3 4x4 matrix -> translation (N,3) + rotation quaternion xyzw (N,4)
// -----------------------------------------------------------------------------

static void check_se3_pose_from_matrix_inputs(const at::Tensor& matrix) {
    CHECK_CUDA_TENSOR(matrix);
    CHECK_CONTIGUOUS(matrix);
    CHECK_FLOATING(matrix);
    TORCH_CHECK(
        matrix.dim() == 2 && matrix.size(1) == 16,
        "matrix must be (N, 16) row-major flattened 4x4");
}

static void check_se3_pose_from_matrix_grad_outputs(
    const at::Tensor& matrix,
    const at::Tensor& grad_translation,
    const at::Tensor& grad_rotation) {
    CHECK_CUDA_TENSOR(grad_translation);
    CHECK_CUDA_TENSOR(grad_rotation);
    CHECK_CONTIGUOUS(grad_translation);
    CHECK_CONTIGUOUS(grad_rotation);
    TORCH_CHECK(grad_translation.device() == matrix.device(), "grad_translation must match matrix device");
    TORCH_CHECK(grad_rotation.device() == matrix.device(), "grad_rotation must match matrix device");
    TORCH_CHECK(grad_translation.dtype() == matrix.dtype(), "grad_translation dtype must match matrix");
    TORCH_CHECK(grad_rotation.dtype() == matrix.dtype(), "grad_rotation dtype must match matrix");
    const int64_t n = matrix.size(0);
    TORCH_CHECK(
        grad_translation.dim() == 2 && grad_translation.size(0) == n && grad_translation.size(1) == 3,
        "grad_translation must be (N, 3) with same N as matrix");
    TORCH_CHECK(
        grad_rotation.dim() == 2 && grad_rotation.size(0) == n && grad_rotation.size(1) == 4,
        "grad_rotation must be (N, 4) with same N as matrix");
}

std::tuple<torch::Tensor, torch::Tensor> se3pose_from_matrix(const torch::Tensor& matrix) {
    check_se3_pose_from_matrix_inputs(matrix);
    const int64_t n = matrix.size(0);
    auto opts = matrix.options();
    auto translation = torch::empty({n, 3}, opts);
    auto rotation = torch::empty({n, 4}, opts);
    if (n > 0) {
        se3pose_from_matrix_cuda(matrix, translation, rotation);
    }
    return std::make_tuple(translation, rotation);
}

torch::Tensor se3pose_from_matrix_bwd(
    const torch::Tensor& matrix,
    const torch::Tensor& grad_translation,
    const torch::Tensor& grad_rotation) {
    check_se3_pose_from_matrix_inputs(matrix);
    check_se3_pose_from_matrix_grad_outputs(matrix, grad_translation, grad_rotation);
    const int64_t n = matrix.size(0);
    auto grad_matrix = torch::empty_like(matrix);
    if (n > 0) {
        se3pose_from_matrix_bwd_cuda(matrix, grad_translation, grad_rotation, grad_matrix);
    }
    return grad_matrix;
}

// -----------------------------------------------------------------------------
// 2-pose trajectory bindings.
// CUDA path is float32-only; out_of_bounds and get_rotation quat outputs are float32.
// -----------------------------------------------------------------------------

static void check_trajectory_2poses_time_query(
    const at::Tensor& ref_device,
    const at::Tensor& time0,
    const at::Tensor& time1,
    const at::Tensor& query_time,
    int64_t n) {
    auto check_one = [&](const char* name, const at::Tensor& t) {
        CHECK_CUDA_TENSOR(t);
        CHECK_CONTIGUOUS(t);
        TORCH_CHECK(t.device() == ref_device.device(), name, " must be on the same CUDA device as poses");
        TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32 for trajectory CUDA");
        TORCH_CHECK(
            (t.dim() == 1 && t.size(0) == n) || (t.dim() == 2 && t.size(0) == n && t.size(1) == 1),
            name,
            " must be (N,) or (N, 1) with same N as poses");
    };
    check_one("time0", time0);
    check_one("time1", time1);
    check_one("query_time", query_time);
}

static void check_trajectory_2poses_transform_inputs(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& point,
    const at::Tensor& query_time) {
    CHECK_CUDA_TENSOR(trans0);
    CHECK_CUDA_TENSOR(rot0);
    CHECK_CUDA_TENSOR(trans1);
    CHECK_CUDA_TENSOR(rot1);
    CHECK_CUDA_TENSOR(point);
    CHECK_CONTIGUOUS(trans0);
    CHECK_CONTIGUOUS(rot0);
    CHECK_CONTIGUOUS(trans1);
    CHECK_CONTIGUOUS(rot1);
    CHECK_CONTIGUOUS(point);
    TORCH_CHECK(trans0.dtype() == torch::kFloat32, "trajectory CUDA expects float32 trans0");
    TORCH_CHECK(
        rot0.dtype() == torch::kFloat32 && trans1.dtype() == torch::kFloat32 &&
            rot1.dtype() == torch::kFloat32 && point.dtype() == torch::kFloat32,
        "trajectory CUDA expects float32 rot0, trans1, rot1, point");
    const int64_t n = point.size(0);
    TORCH_CHECK(trans0.dim() == 2 && trans0.size(1) == 3 && trans0.size(0) == n, "trans0 must be (N, 3)");
    TORCH_CHECK(rot0.dim() == 2 && rot0.size(1) == 4 && rot0.size(0) == n, "rot0 must be (N, 4)");
    TORCH_CHECK(trans1.dim() == 2 && trans1.size(1) == 3 && trans1.size(0) == n, "trans1 must be (N, 3)");
    TORCH_CHECK(rot1.dim() == 2 && rot1.size(1) == 4 && rot1.size(0) == n, "rot1 must be (N, 4)");
    TORCH_CHECK(point.dim() == 2 && point.size(1) == 3, "point must be (N, 3)");
    TORCH_CHECK(
        trans0.device() == rot0.device() && trans0.device() == trans1.device() &&
            trans0.device() == rot1.device() && trans0.device() == point.device(),
        "trajectory pose tensors must share one CUDA device");
    check_trajectory_2poses_time_query(trans0, time0, time1, query_time, n);
}

static void check_trajectory_2poses_rotation_inputs(
    const at::Tensor& trans0,
    const at::Tensor& rot0,
    const at::Tensor& time0,
    const at::Tensor& trans1,
    const at::Tensor& rot1,
    const at::Tensor& time1,
    const at::Tensor& query_time) {
    CHECK_CUDA_TENSOR(trans0);
    CHECK_CUDA_TENSOR(rot0);
    CHECK_CUDA_TENSOR(trans1);
    CHECK_CUDA_TENSOR(rot1);
    CHECK_CONTIGUOUS(trans0);
    CHECK_CONTIGUOUS(rot0);
    CHECK_CONTIGUOUS(trans1);
    CHECK_CONTIGUOUS(rot1);
    TORCH_CHECK(
        trans0.dtype() == torch::kFloat32 && rot0.dtype() == torch::kFloat32 &&
            trans1.dtype() == torch::kFloat32 && rot1.dtype() == torch::kFloat32,
        "trajectory_get_rotation CUDA expects float32 pose tensors");
    const int64_t n = trans0.size(0);
    TORCH_CHECK(trans0.dim() == 2 && trans0.size(1) == 3, "trans0 must be (N, 3)");
    TORCH_CHECK(rot0.dim() == 2 && rot0.size(1) == 4 && rot0.size(0) == n, "rot0 must be (N, 4)");
    TORCH_CHECK(trans1.dim() == 2 && trans1.size(1) == 3 && trans1.size(0) == n, "trans1 must be (N, 3)");
    TORCH_CHECK(rot1.dim() == 2 && rot1.size(1) == 4 && rot1.size(0) == n, "rot1 must be (N, 4)");
    TORCH_CHECK(
        trans0.device() == rot0.device() && trans0.device() == trans1.device() &&
            trans0.device() == rot1.device(),
        "trajectory pose tensors must share one CUDA device");
    check_trajectory_2poses_time_query(trans0, time0, time1, query_time, n);
}

std::tuple<torch::Tensor, torch::Tensor> trajectory_transform_point_2poses(
    const torch::Tensor& trans0,
    const torch::Tensor& rot0,
    const torch::Tensor& time0,
    const torch::Tensor& trans1,
    const torch::Tensor& rot1,
    const torch::Tensor& time1,
    const torch::Tensor& point,
    const torch::Tensor& query_time) {
    check_trajectory_2poses_transform_inputs(trans0, rot0, time0, trans1, rot1, time1, point, query_time);
    const int64_t n = point.size(0);
    auto opts_f = trans0.options().dtype(torch::kFloat32);
    auto result_point = torch::empty_like(point);
    auto result_oob = torch::empty({n}, opts_f);
    if (n > 0) {
        trajectory_transform_point_2poses_cuda(
            trans0, rot0, time0, trans1, rot1, time1, point, query_time, result_point, result_oob);
    }
    return std::make_tuple(result_point, result_oob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trajectory_transform_point_2poses_bwd(
    const torch::Tensor& trans0,
    const torch::Tensor& rot0,
    const torch::Tensor& time0,
    const torch::Tensor& trans1,
    const torch::Tensor& rot1,
    const torch::Tensor& time1,
    const torch::Tensor& point,
    const torch::Tensor& query_time,
    const torch::Tensor& grad_result_point) {
    check_trajectory_2poses_transform_inputs(trans0, rot0, time0, trans1, rot1, time1, point, query_time);
    CHECK_CUDA_TENSOR(grad_result_point);
    CHECK_CONTIGUOUS(grad_result_point);
    TORCH_CHECK(grad_result_point.sizes() == point.sizes(), "grad_result_point shape must match point");
    TORCH_CHECK(grad_result_point.dtype() == point.dtype(), "grad_result_point dtype must match point");
    TORCH_CHECK(grad_result_point.device() == point.device(), "grad_result_point device must match point");
    const int64_t n = point.size(0);
    auto grad_trans0 = torch::zeros_like(trans0);
    auto grad_rot0 = torch::zeros_like(rot0);
    auto grad_time0 = torch::zeros_like(time0);
    auto grad_trans1 = torch::zeros_like(trans1);
    auto grad_rot1 = torch::zeros_like(rot1);
    auto grad_time1 = torch::zeros_like(time1);
    auto grad_point = torch::zeros_like(point);
    auto grad_query_time = torch::zeros_like(query_time);
    if (n > 0) {
        trajectory_transform_point_2poses_bwd_cuda(
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
    return std::make_tuple(
        grad_trans0,
        grad_rot0,
        grad_time0,
        grad_trans1,
        grad_rot1,
        grad_time1,
        grad_point,
        grad_query_time);
}

std::tuple<torch::Tensor, torch::Tensor> trajectory_get_rotation_2poses(
    const torch::Tensor& trans0,
    const torch::Tensor& rot0,
    const torch::Tensor& time0,
    const torch::Tensor& trans1,
    const torch::Tensor& rot1,
    const torch::Tensor& time1,
    const torch::Tensor& query_time) {
    check_trajectory_2poses_rotation_inputs(trans0, rot0, time0, trans1, rot1, time1, query_time);
    const int64_t n = trans0.size(0);
    auto opts_f = trans0.options().dtype(torch::kFloat32);
    auto result_quat = torch::empty({n, 4}, opts_f);
    auto result_oob = torch::empty({n}, opts_f);
    if (n > 0) {
        trajectory_get_rotation_2poses_cuda(
            trans0, rot0, time0, trans1, rot1, time1, query_time, result_quat, result_oob);
    }
    return std::make_tuple(result_quat, result_oob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trajectory_get_rotation_2poses_bwd(
    const torch::Tensor& trans0,
    const torch::Tensor& rot0,
    const torch::Tensor& time0,
    const torch::Tensor& trans1,
    const torch::Tensor& rot1,
    const torch::Tensor& time1,
    const torch::Tensor& query_time,
    const torch::Tensor& grad_result_quat) {
    check_trajectory_2poses_rotation_inputs(trans0, rot0, time0, trans1, rot1, time1, query_time);
    CHECK_CUDA_TENSOR(grad_result_quat);
    CHECK_CONTIGUOUS(grad_result_quat);
    const int64_t n = trans0.size(0);
    TORCH_CHECK(
        grad_result_quat.dim() == 2 && grad_result_quat.size(0) == n && grad_result_quat.size(1) == 4,
        "grad_result_quat must be (N, 4)");
    TORCH_CHECK(grad_result_quat.dtype() == torch::kFloat32, "grad_result_quat must be float32");
    TORCH_CHECK(grad_result_quat.device() == trans0.device(), "grad_result_quat device must match poses");
    auto grad_trans0 = torch::zeros_like(trans0);
    auto grad_rot0 = torch::zeros_like(rot0);
    auto grad_time0 = torch::zeros_like(time0);
    auto grad_trans1 = torch::zeros_like(trans1);
    auto grad_rot1 = torch::zeros_like(rot1);
    auto grad_time1 = torch::zeros_like(time1);
    auto grad_query_time = torch::zeros_like(query_time);
    if (n > 0) {
        trajectory_get_rotation_2poses_bwd_cuda(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            query_time,
            grad_result_quat,
            grad_trans0,
            grad_rot0,
            grad_time0,
            grad_trans1,
            grad_rot1,
            grad_time1,
            grad_query_time);
    }
    return std::make_tuple(
        grad_trans0, grad_rot0, grad_time0, grad_trans1, grad_rot1, grad_time1, grad_query_time);
}

// -----------------------------------------------------------------------------
// 1-pose trajectory + frame_transform_poses_tquat bindings, float32 CUDA.
// -----------------------------------------------------------------------------

static void check_trajectory_1pose_transform_inputs(
    const at::Tensor& trans,
    const at::Tensor& rot,
    const at::Tensor& time,
    const at::Tensor& point,
    const at::Tensor& query_time) {
    CHECK_CUDA_TENSOR(trans);
    CHECK_CUDA_TENSOR(rot);
    CHECK_CUDA_TENSOR(point);
    CHECK_CONTIGUOUS(trans);
    CHECK_CONTIGUOUS(rot);
    CHECK_CONTIGUOUS(point);
    TORCH_CHECK(
        trans.dtype() == torch::kFloat32 && rot.dtype() == torch::kFloat32 &&
            point.dtype() == torch::kFloat32,
        "trajectory_transform_point_1pose CUDA expects float32 trans, rot, point");
    const int64_t n = point.size(0);
    TORCH_CHECK(trans.dim() == 2 && trans.size(1) == 3 && trans.size(0) == n, "trans must be (N, 3)");
    TORCH_CHECK(rot.dim() == 2 && rot.size(1) == 4 && rot.size(0) == n, "rot must be (N, 4)");
    TORCH_CHECK(point.dim() == 2 && point.size(1) == 3, "point must be (N, 3)");
    TORCH_CHECK(
        trans.device() == rot.device() && trans.device() == point.device(),
        "trans, rot, point must share one CUDA device");
    auto check_tq = [&](const char* name, const at::Tensor& t) {
        CHECK_CUDA_TENSOR(t);
        CHECK_CONTIGUOUS(t);
        TORCH_CHECK(t.device() == trans.device(), name, " must be on the same CUDA device as poses");
        TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32 for trajectory CUDA");
        TORCH_CHECK(
            (t.dim() == 1 && t.size(0) == n) || (t.dim() == 2 && t.size(0) == n && t.size(1) == 1),
            name,
            " must be (N,) or (N, 1) with same N as point batch");
    };
    check_tq("time", time);
    check_tq("query_time", query_time);
}

std::tuple<torch::Tensor, torch::Tensor> trajectory_transform_point_1pose(
    const torch::Tensor& trans,
    const torch::Tensor& rot,
    const torch::Tensor& time,
    const torch::Tensor& point,
    const torch::Tensor& query_time) {
    check_trajectory_1pose_transform_inputs(trans, rot, time, point, query_time);
    const int64_t n = point.size(0);
    auto opts_f = trans.options().dtype(torch::kFloat32);
    auto result_point = torch::empty_like(point);
    auto result_oob = torch::empty({n}, opts_f);
    if (n > 0) {
        trajectory_transform_point_1pose_cuda(
            trans, rot, time, point, query_time, result_point, result_oob);
    }
    return std::make_tuple(result_point, result_oob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trajectory_transform_point_1pose_bwd(
    const torch::Tensor& trans,
    const torch::Tensor& rot,
    const torch::Tensor& time,
    const torch::Tensor& point,
    const torch::Tensor& query_time,
    const torch::Tensor& grad_result_point) {
    check_trajectory_1pose_transform_inputs(trans, rot, time, point, query_time);
    CHECK_CUDA_TENSOR(grad_result_point);
    CHECK_CONTIGUOUS(grad_result_point);
    TORCH_CHECK(grad_result_point.sizes() == point.sizes(), "grad_result_point shape must match point");
    TORCH_CHECK(grad_result_point.dtype() == point.dtype(), "grad_result_point dtype must match point");
    TORCH_CHECK(grad_result_point.device() == point.device(), "grad_result_point device must match point");
    const int64_t n = point.size(0);
    auto grad_trans = torch::zeros_like(trans);
    auto grad_rot = torch::zeros_like(rot);
    auto grad_time = torch::zeros_like(time);
    auto grad_point = torch::zeros_like(point);
    auto grad_query_time = torch::zeros_like(query_time);
    if (n > 0) {
        trajectory_transform_point_1pose_bwd_cuda(
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
    return std::make_tuple(grad_trans, grad_rot, grad_time, grad_point, grad_query_time);
}

static void check_frame_transform_poses_tquat_inputs(const at::Tensor& input_poses) {
    CHECK_CUDA_TENSOR(input_poses);
    CHECK_CONTIGUOUS(input_poses);
    TORCH_CHECK(
        input_poses.dtype() == torch::kFloat32, "frame_transform_poses_tquat CUDA expects float32 input");
    TORCH_CHECK(
        input_poses.dim() == 2 && input_poses.size(1) == 7, "input_poses must be (N, 7) tquat rows");
}

torch::Tensor frame_transform_poses_tquat(
    const torch::Tensor& input_poses,
    double rotation_x,
    double rotation_y,
    double rotation_z,
    double rotation_w,
    double translation_x,
    double translation_y,
    double translation_z,
    double scale) {
    check_frame_transform_poses_tquat_inputs(input_poses);
    auto out = torch::empty_like(input_poses);
    const int64_t n = input_poses.size(0);
    if (n > 0) {
        frame_transform_poses_tquat_cuda(
            input_poses,
            static_cast<float>(rotation_x),
            static_cast<float>(rotation_y),
            static_cast<float>(rotation_z),
            static_cast<float>(rotation_w),
            static_cast<float>(translation_x),
            static_cast<float>(translation_y),
            static_cast<float>(translation_z),
            static_cast<float>(scale),
            out);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quat_normalize_safe", &quat_normalize_safe, "Normalize quaternion xyzw with near-zero guard, CUDA");
    m.def("quat_normalize_safe_bwd", &quat_normalize_safe_bwd, "Backward for quat_normalize_safe, CUDA");
    m.def("quat_multiply", &quat_multiply, "Hamilton product q1*q2, xyzw, CUDA");
    m.def("quat_multiply_bwd", &quat_multiply_bwd, "Backward for quat_multiply, CUDA");
    m.def("quat_rotate_vector", &quat_rotate_vector, "Rotate 3-vectors by quaternions xyzw, CUDA");
    m.def("quat_rotate_vector_bwd", &quat_rotate_vector_bwd, "Backward for quat_rotate_vector, CUDA");
    m.def("quat_to_matrix", &quat_to_matrix, "Quaternion xyzw to row-major 3x3 flattened (N,9), CUDA");
    m.def("quat_to_matrix_bwd", &quat_to_matrix_bwd, "Backward for quat_to_matrix, CUDA");
    m.def("quat_conjugate", &quat_conjugate, "Quaternion conjugate xyzw -> (-x,-y,-z,w), CUDA");
    m.def("quat_conjugate_bwd", &quat_conjugate_bwd, "Backward for quat_conjugate, CUDA");
    m.def("quat_lerp", &quat_lerp, "Quaternion lerp with hemisphere fix and normalize, xyzw, CUDA");
    m.def("quat_lerp_bwd", &quat_lerp_bwd, "Backward for quat_lerp, CUDA");
    m.def(
        "quat_slerp_batched",
        &quat_slerp_batched,
        "Batched quaternion slerp (N,4), t (N,1), hemisphere + small-angle lerp fallback, CUDA");
    m.def(
        "quat_slerp_batched_bwd",
        &quat_slerp_batched_bwd,
        "Backward for quat_slerp_batched, CUDA");
    m.def(
        "quat_from_axis_angle",
        &quat_from_axis_angle,
        "Quaternion xyzw from axis (N,3) and angle radians (N,1); matches Slang convertAngleToQuat, CUDA");
    m.def(
        "quat_from_axis_angle_bwd",
        &quat_from_axis_angle_bwd,
        "Backward for quat_from_axis_angle -> (grad_axis, grad_angle), CUDA");
    m.def(
        "quat_angular_distance",
        &quat_angular_distance,
        "Angular distance 2*acos(clamp(|dot(n1,n2)|)) with n = normalize(q), xyzw, CUDA; out (N,1)");
    m.def(
        "quat_angular_distance_bwd",
        &quat_angular_distance_bwd,
        "Backward for quat_angular_distance -> (grad_q1, grad_q2), CUDA");
    m.def(
        "quat_manifold_interp",
        &quat_manifold_interp,
        "SO(3) manifold interp q1*exp(t*log(q1^-1*q2)), t (N,1) per row, xyzw, CUDA.");
    m.def(
        "quat_manifold_interp_bwd",
        &quat_manifold_interp_bwd,
        "Backward for quat_manifold_interp -> (grad_q1, grad_q2, grad_t), CUDA");
    m.def(
        "se3pose_transform_point",
        &se3pose_transform_point,
        "SE3 transform point: R(q)*p + t, translation (N,3), rotation xyzw (N,4), point (N,3), CUDA");
    m.def(
        "se3pose_transform_point_bwd",
        &se3pose_transform_point_bwd,
        "Backward for se3pose_transform_point -> (grad_translation, grad_rotation, grad_point), CUDA");
    m.def(
        "se3pose_transform_direction",
        &se3pose_transform_direction,
        "SE3 transform direction: R(q)*d (translation unused), CUDA");
    m.def(
        "se3pose_transform_direction_bwd",
        &se3pose_transform_direction_bwd,
        "Backward for se3pose_transform_direction -> (grad_translation, grad_rotation, grad_direction), CUDA");
    m.def(
        "se3pose_inverse_transform_point",
        &se3pose_inverse_transform_point,
        "SE3 inverse transform point: R(q)^T*(p-t), CUDA");
    m.def(
        "se3pose_inverse_transform_point_bwd",
        &se3pose_inverse_transform_point_bwd,
        "Backward for se3pose_inverse_transform_point, CUDA");
    m.def(
        "se3pose_inverse_transform_direction",
        &se3pose_inverse_transform_direction,
        "SE3 inverse transform direction: R(q)^T*d, CUDA");
    m.def(
        "se3pose_inverse_transform_direction_bwd",
        &se3pose_inverse_transform_direction_bwd,
        "Backward for se3pose_inverse_transform_direction, CUDA");
    m.def(
        "se3pose_to_matrix",
        &se3pose_to_matrix,
        "SE3 to 4x4 row-major (N,16): R(q) upper-left, t in column 3, CUDA");
    m.def(
        "se3pose_to_matrix_bwd",
        &se3pose_to_matrix_bwd,
        "Backward for se3pose_to_matrix -> (grad_translation, grad_rotation), grad_out (N,16), CUDA");
    m.def(
        "se3pose_to_inverse_matrix",
        &se3pose_to_inverse_matrix,
        "SE3 inverse 4x4 row-major (N,16); wxyz_format swaps quaternion layout, CUDA");
    m.def(
        "se3pose_to_inverse_matrix_bwd",
        &se3pose_to_inverse_matrix_bwd,
        "Backward for se3pose_to_inverse_matrix -> (grad_translation, grad_rotation), CUDA");
    m.def(
        "se3pose_from_matrix",
        &se3pose_from_matrix,
        "SE3 from 4x4 row-major flat (N,16): translation (N,3), rotation xyzw (N,4), CUDA");
    m.def(
        "se3pose_from_matrix_bwd",
        &se3pose_from_matrix_bwd,
        "Backward for se3pose_from_matrix -> grad_matrix (N,16), CUDA");
    m.def(
        "trajectory_transform_point_2poses",
        &trajectory_transform_point_2poses,
        "2-pose trajectory transform point; out_of_bounds float32 (N,), CUDA float32 only");
    m.def(
        "trajectory_transform_point_2poses_bwd",
        &trajectory_transform_point_2poses_bwd,
        "Backward for trajectory_transform_point_2poses -> 8 grads, CUDA");
    m.def(
        "trajectory_get_rotation_2poses",
        &trajectory_get_rotation_2poses,
        "2-pose trajectory rotation slerp; quat and oob float32 (N,4)/(N,), CUDA float32 only");
    m.def(
        "trajectory_get_rotation_2poses_bwd",
        &trajectory_get_rotation_2poses_bwd,
        "Backward for trajectory_get_rotation_2poses -> 7 grads (translations zero), CUDA");
    m.def(
        "trajectory_transform_point_1pose",
        &trajectory_transform_point_1pose,
        "1-pose trajectory transform point; oob float32 (N,) from exact query_time != time, CUDA float32");
    m.def(
        "trajectory_transform_point_1pose_bwd",
        &trajectory_transform_point_1pose_bwd,
        "Backward for trajectory_transform_point_1pose -> 5 grads (time/query_time zero), CUDA");
    m.def(
        "frame_transform_poses_tquat",
        &frame_transform_poses_tquat,
        "Frame transform (N,7) tquat poses: q_frame*pose_q, t_out = scale*(R_frame*t_in + t_frame), float32 CUDA");
}
