/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// lidar_torch.cpp — C++ wrapper layer between TORCH_LIBRARY (ext.cpp) and the
// LiDAR CUDA _launch helpers (lidar_params.h / lidar_kernel.cu).

#include "lidar_torch.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <utility>

namespace gsplat_sensors {
namespace {

// Construction-time table validator: shape, float32 dtype, contiguity only.
// Device is NOT checked here so the projection (and the owning LidarModel) is
// CPU-constructible and movable via .to('cpu'); the is_cuda + device-match guard
// lives in check_lidar_projection_for_device and runs at op launch.
void check_lidar_table(const at::Tensor& tensor, int64_t expected, const char* name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == expected, name, " must be (", expected, ",)");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_finite(double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

void check_same_device(
    const at::Tensor& tensor,
    const at::Tensor& ref,
    const char* name,
    const char* ref_name) {
    TORCH_CHECK(tensor.device() == ref.device(), name, " device must match ", ref_name);
}

void check_inverse_solver_scalars(
    int64_t max_iterations,
    double stop_mean_relative_time_error,
    double stop_delta_mean_relative_time_error,
    double initial_relative_time) {
    TORCH_CHECK(max_iterations >= 1, "max_iterations must be at least 1");
    TORCH_CHECK(
        max_iterations <= kMaxSpinningLidarIterations,
        "max_iterations must be <= ",
        kMaxSpinningLidarIterations);
    TORCH_CHECK(
        std::isfinite(stop_mean_relative_time_error) && stop_mean_relative_time_error >= 0.0,
        "stop_mean_relative_time_error must be finite and nonnegative");
    TORCH_CHECK(
        std::isfinite(stop_delta_mean_relative_time_error) && stop_delta_mean_relative_time_error >= 0.0,
        "stop_delta_mean_relative_time_error must be finite and nonnegative");
    TORCH_CHECK(
        std::isfinite(initial_relative_time) && initial_relative_time >= 0.0 &&
            initial_relative_time <= 1.0,
        "initial_relative_time must be finite and in [0, 1]");
}

cudaStream_t current_stream(const at::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

// Floating-point matrix (N, cols) on CUDA, contiguous; float32 OR float64 so
// the ray<->angle ops stay dtype-polymorphic and fp64 gradcheck is exact.
void check_float_matrix(const at::Tensor& tensor, int64_t cols, const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == cols, name, " must be (N, ", cols, ")");
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble,
        name,
        " must be float32 or float64");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// (N, cols) int32 element-index matrix on CUDA, contiguous.
void check_int_matrix(const at::Tensor& tensor, int64_t cols, const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == cols, name, " must be (N, ", cols, ")");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// (n,) floating-point angle table on CUDA, contiguous, dtype matching `ref`.
void check_angle_table(
    const at::Tensor& tensor,
    int64_t expected,
    const at::Tensor& ref,
    const char* name,
    const char* ref_name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == expected, name, " must be (", expected, ",)");
    TORCH_CHECK(tensor.scalar_type() == ref.scalar_type(), name, " dtype must match ", ref_name);
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    check_same_device(tensor, ref, name, ref_name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

// ===========================================================================
// RowOffsetStructuredSpinningLidarProjection — constructor and accessors
// ===========================================================================

RowOffsetStructuredSpinningLidarProjection::RowOffsetStructuredSpinningLidarProjection(
    at::Tensor row_elevations_rad_,
    at::Tensor column_azimuths_rad_,
    at::Tensor row_azimuth_offsets_rad_,
    double fov_vert_start_rad_,
    double fov_vert_span_rad_,
    double fov_horiz_start_rad_,
    double fov_horiz_span_rad_,
    int64_t spinning_direction_,
    bool has_row_offsets_)
    : row_elevations_rad(std::move(row_elevations_rad_)),
      column_azimuths_rad(std::move(column_azimuths_rad_)),
      row_azimuth_offsets_rad(std::move(row_azimuth_offsets_rad_)),
      fov_vert_start_rad(fov_vert_start_rad_),
      fov_vert_span_rad(fov_vert_span_rad_),
      fov_horiz_start_rad(fov_horiz_start_rad_),
      fov_horiz_span_rad(fov_horiz_span_rad_),
      spinning_direction(spinning_direction_),
      has_row_offsets(has_row_offsets_) {}

RowOffsetStructuredSpinningLidarProjection_KernelParameters
RowOffsetStructuredSpinningLidarProjection::to_kernel_params() const {
    RowOffsetStructuredSpinningLidarProjection_KernelParameters params{};
    params.row_elevations_rad = row_elevations_rad.const_data_ptr<float>();
    params.column_azimuths_rad = column_azimuths_rad.const_data_ptr<float>();
    params.row_azimuth_offsets_rad =
        has_row_offsets ? row_azimuth_offsets_rad.const_data_ptr<float>() : nullptr;
    params.n_rows = row_elevations_rad.size(0);
    params.n_columns = column_azimuths_rad.size(0);
    params.fov_vert_start_rad = static_cast<float>(fov_vert_start_rad);
    params.fov_vert_span_rad = static_cast<float>(fov_vert_span_rad);
    params.fov_horiz_start_rad = static_cast<float>(fov_horiz_start_rad);
    params.fov_horiz_span_rad = static_cast<float>(fov_horiz_span_rad);
    params.spinning_direction = static_cast<SpinningDirection>(spinning_direction);
    params.has_row_offsets = has_row_offsets ? 1 : 0;
    return params;
}

void check_lidar_projection(const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");

    const int64_t n_rows = projection->row_elevations_rad.dim() == 1
        ? projection->row_elevations_rad.size(0)
        : -1;
    const int64_t n_columns = projection->column_azimuths_rad.dim() == 1
        ? projection->column_azimuths_rad.size(0)
        : -1;
    TORCH_CHECK(n_rows > 0, "row_elevations_rad must be a non-empty 1-D (n_rows,) tensor");
    TORCH_CHECK(n_columns > 0, "column_azimuths_rad must be a non-empty 1-D (n_columns,) tensor");

    check_lidar_table(projection->row_elevations_rad, n_rows, "row_elevations_rad");
    check_lidar_table(projection->column_azimuths_rad, n_columns, "column_azimuths_rad");

    if (projection->has_row_offsets) {
        check_lidar_table(projection->row_azimuth_offsets_rad, n_rows, "row_azimuth_offsets_rad");
    } else {
        TORCH_CHECK(
            projection->row_azimuth_offsets_rad.numel() == 0,
            "row_azimuth_offsets_rad must be empty when has_row_offsets is false");
    }

    check_finite(projection->fov_vert_start_rad, "fov_vert_start_rad");
    check_finite(projection->fov_vert_span_rad, "fov_vert_span_rad");
    check_finite(projection->fov_horiz_start_rad, "fov_horiz_start_rad");
    check_finite(projection->fov_horiz_span_rad, "fov_horiz_span_rad");

    TORCH_CHECK(
        projection->spinning_direction == static_cast<int64_t>(SpinningDirection::CLOCKWISE) ||
            projection->spinning_direction == static_cast<int64_t>(SpinningDirection::COUNTERCLOCKWISE),
        "spinning_direction must be 0 (CLOCKWISE) or 1 (COUNTERCLOCKWISE)");
}

// Launch-time device guard (mirrors camera check_projection_for_device): the
// angle tables must be CUDA float32 contiguous AND share the op's input device.
void check_lidar_projection_for_device(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Device& device) {
    check_lidar_projection(projection);
    TORCH_CHECK(projection->row_elevations_rad.is_cuda(), "row_elevations_rad must be a CUDA tensor");
    TORCH_CHECK(projection->column_azimuths_rad.is_cuda(), "column_azimuths_rad must be a CUDA tensor");
    TORCH_CHECK(
        projection->row_elevations_rad.device() == device,
        "row_elevations_rad device must match inputs");
    TORCH_CHECK(
        projection->column_azimuths_rad.device() == device,
        "column_azimuths_rad device must match inputs");
    if (projection->has_row_offsets) {
        TORCH_CHECK(projection->row_azimuth_offsets_rad.is_cuda(), "row_azimuth_offsets_rad must be a CUDA tensor");
        TORCH_CHECK(
            projection->row_azimuth_offsets_rad.device() == device,
            "row_azimuth_offsets_rad device must match inputs");
    }
}

// ===========================================================================
// sensor_rays_to_sensor_angles — forward + backward (dtype-polymorphic)
// ===========================================================================

at::Tensor sensor_rays_to_sensor_angles(const at::Tensor& sensor_rays) {
    check_float_matrix(sensor_rays, 3, "sensor_rays");
    auto guard = c10::cuda::CUDAGuard(sensor_rays.device());
    auto sensor_angles = at::empty({sensor_rays.size(0), 2}, sensor_rays.options());
    AT_DISPATCH_FLOATING_TYPES(
        sensor_rays.scalar_type(), "sensor_rays_to_sensor_angles", [&] {
            sensor_rays_to_sensor_angles_launch<scalar_t>(
                sensor_rays.size(0),
                sensor_rays.const_data_ptr<scalar_t>(),
                sensor_angles.data_ptr<scalar_t>(),
                current_stream(sensor_rays));
        });
    return sensor_angles;
}

at::Tensor sensor_rays_to_sensor_angles_backward(
    const at::Tensor& sensor_rays, const at::Tensor& grad_sensor_angles) {
    check_float_matrix(sensor_rays, 3, "sensor_rays");
    check_float_matrix(grad_sensor_angles, 2, "grad_sensor_angles");
    TORCH_CHECK(
        grad_sensor_angles.size(0) == sensor_rays.size(0),
        "grad_sensor_angles batch must match sensor_rays");
    TORCH_CHECK(
        grad_sensor_angles.scalar_type() == sensor_rays.scalar_type(),
        "grad_sensor_angles dtype must match sensor_rays");
    check_same_device(grad_sensor_angles, sensor_rays, "grad_sensor_angles", "sensor_rays");
    auto guard = c10::cuda::CUDAGuard(sensor_rays.device());
    auto grad_sensor_rays = at::empty_like(sensor_rays);
    AT_DISPATCH_FLOATING_TYPES(
        sensor_rays.scalar_type(), "sensor_rays_to_sensor_angles_backward", [&] {
            sensor_rays_to_sensor_angles_backward_launch<scalar_t>(
                sensor_rays.size(0),
                sensor_rays.const_data_ptr<scalar_t>(),
                grad_sensor_angles.const_data_ptr<scalar_t>(),
                grad_sensor_rays.data_ptr<scalar_t>(),
                current_stream(sensor_rays));
        });
    return grad_sensor_rays;
}

// ===========================================================================
// sensor_angles_to_sensor_rays — forward + backward (dtype-polymorphic)
// ===========================================================================

at::Tensor sensor_angles_to_sensor_rays(const at::Tensor& sensor_angles) {
    check_float_matrix(sensor_angles, 2, "sensor_angles");
    auto guard = c10::cuda::CUDAGuard(sensor_angles.device());
    auto sensor_rays = at::empty({sensor_angles.size(0), 3}, sensor_angles.options());
    AT_DISPATCH_FLOATING_TYPES(
        sensor_angles.scalar_type(), "sensor_angles_to_sensor_rays", [&] {
            sensor_angles_to_sensor_rays_launch<scalar_t>(
                sensor_angles.size(0),
                sensor_angles.const_data_ptr<scalar_t>(),
                sensor_rays.data_ptr<scalar_t>(),
                current_stream(sensor_angles));
        });
    return sensor_rays;
}

at::Tensor sensor_angles_to_sensor_rays_backward(
    const at::Tensor& sensor_angles, const at::Tensor& grad_sensor_rays) {
    check_float_matrix(sensor_angles, 2, "sensor_angles");
    check_float_matrix(grad_sensor_rays, 3, "grad_sensor_rays");
    TORCH_CHECK(
        grad_sensor_rays.size(0) == sensor_angles.size(0),
        "grad_sensor_rays batch must match sensor_angles");
    TORCH_CHECK(
        grad_sensor_rays.scalar_type() == sensor_angles.scalar_type(),
        "grad_sensor_rays dtype must match sensor_angles");
    check_same_device(grad_sensor_rays, sensor_angles, "grad_sensor_rays", "sensor_angles");
    auto guard = c10::cuda::CUDAGuard(sensor_angles.device());
    auto grad_sensor_angles = at::empty_like(sensor_angles);
    AT_DISPATCH_FLOATING_TYPES(
        sensor_angles.scalar_type(), "sensor_angles_to_sensor_rays_backward", [&] {
            sensor_angles_to_sensor_rays_backward_launch<scalar_t>(
                sensor_angles.size(0),
                sensor_angles.const_data_ptr<scalar_t>(),
                grad_sensor_rays.const_data_ptr<scalar_t>(),
                grad_sensor_angles.data_ptr<scalar_t>(),
                current_stream(sensor_angles));
        });
    return grad_sensor_angles;
}

// ===========================================================================
// elements_to_sensor_angles — forward + backward
//
// The angle tables are threaded explicitly (not read off the projection) so
// the kernel can run under fp64 gradcheck while the public binding still forces
// float32.  n_rows / n_columns / has_row_offsets ride on the projection POD;
// only the differentiable tables cross the autograd boundary.
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> elements_to_sensor_angles(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Tensor& elements,
    const at::Tensor& row_elevations_rad,
    const at::Tensor& column_azimuths_rad,
    const at::Tensor& row_azimuth_offsets_rad) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");
    check_int_matrix(elements, 2, "elements");
    TORCH_CHECK(
        row_elevations_rad.dim() == 1 && row_elevations_rad.size(0) > 0,
        "row_elevations_rad must be a non-empty 1-D tensor");
    TORCH_CHECK(
        column_azimuths_rad.dim() == 1 && column_azimuths_rad.size(0) > 0,
        "column_azimuths_rad must be a non-empty 1-D tensor");
    const int64_t n_rows = row_elevations_rad.size(0);
    const int64_t n_columns = column_azimuths_rad.size(0);
    check_angle_table(
        row_elevations_rad,
        n_rows,
        row_elevations_rad,
        "row_elevations_rad",
        "row_elevations_rad");
    check_angle_table(
        column_azimuths_rad,
        n_columns,
        row_elevations_rad,
        "column_azimuths_rad",
        "row_elevations_rad");
    if (projection->has_row_offsets) {
        check_angle_table(
            row_azimuth_offsets_rad,
            n_rows,
            row_elevations_rad,
            "row_azimuth_offsets_rad",
            "row_elevations_rad");
    }
    check_lidar_projection_for_device(projection, elements.device());
    check_same_device(row_elevations_rad, elements, "row_elevations_rad", "elements");
    auto guard = c10::cuda::CUDAGuard(elements.device());

    auto sensor_angles = at::empty({elements.size(0), 2}, row_elevations_rad.options());
    auto valid_flags = at::empty({elements.size(0)}, elements.options().dtype(at::kBool));
    const int has_row_offsets = projection->has_row_offsets ? 1 : 0;
    AT_DISPATCH_FLOATING_TYPES(
        row_elevations_rad.scalar_type(), "elements_to_sensor_angles", [&] {
            elements_to_sensor_angles_launch<scalar_t>(
                elements.size(0),
                n_rows,
                n_columns,
                has_row_offsets,
                elements.const_data_ptr<int32_t>(),
                row_elevations_rad.const_data_ptr<scalar_t>(),
                column_azimuths_rad.const_data_ptr<scalar_t>(),
                has_row_offsets ? row_azimuth_offsets_rad.const_data_ptr<scalar_t>() : nullptr,
                sensor_angles.data_ptr<scalar_t>(),
                valid_flags.data_ptr<bool>(),
                current_stream(elements));
        });
    return {sensor_angles, valid_flags};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> elements_to_sensor_angles_backward(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Tensor& elements,
    const at::Tensor& row_elevations_rad,
    const at::Tensor& column_azimuths_rad,
    const at::Tensor& row_azimuth_offsets_rad,
    const at::Tensor& grad_sensor_angles,
    bool need_row_elevations_grad,
    bool need_column_azimuths_grad,
    bool need_row_offsets_grad) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");
    check_int_matrix(elements, 2, "elements");
    check_float_matrix(grad_sensor_angles, 2, "grad_sensor_angles");
    TORCH_CHECK(
        grad_sensor_angles.size(0) == elements.size(0),
        "grad_sensor_angles batch must match elements");
    const int64_t n_rows = row_elevations_rad.size(0);
    const int64_t n_columns = column_azimuths_rad.size(0);
    check_angle_table(
        row_elevations_rad,
        n_rows,
        row_elevations_rad,
        "row_elevations_rad",
        "row_elevations_rad");
    check_angle_table(
        column_azimuths_rad,
        n_columns,
        row_elevations_rad,
        "column_azimuths_rad",
        "row_elevations_rad");
    if (projection->has_row_offsets) {
        check_angle_table(
            row_azimuth_offsets_rad,
            n_rows,
            row_elevations_rad,
            "row_azimuth_offsets_rad",
            "row_elevations_rad");
    }
    TORCH_CHECK(
        grad_sensor_angles.scalar_type() == row_elevations_rad.scalar_type(),
        "grad_sensor_angles dtype must match angle tables");
    check_lidar_projection_for_device(projection, elements.device());
    check_same_device(row_elevations_rad, elements, "row_elevations_rad", "elements");
    check_same_device(grad_sensor_angles, elements, "grad_sensor_angles", "elements");
    auto guard = c10::cuda::CUDAGuard(elements.device());

    const int has_row_offsets = projection->has_row_offsets ? 1 : 0;
    auto grad_row_elevations = need_row_elevations_grad
        ? at::zeros({n_rows}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    auto grad_column_azimuths = need_column_azimuths_grad
        ? at::zeros({n_columns}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    const bool emit_offsets_grad = has_row_offsets != 0 && need_row_offsets_grad;
    auto grad_row_offsets = emit_offsets_grad
        ? at::zeros({n_rows}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());

    AT_DISPATCH_FLOATING_TYPES(
        row_elevations_rad.scalar_type(), "elements_to_sensor_angles_backward", [&] {
            elements_to_sensor_angles_backward_launch<scalar_t>(
                elements.size(0),
                n_rows,
                n_columns,
                has_row_offsets,
                elements.const_data_ptr<int32_t>(),
                grad_sensor_angles.const_data_ptr<scalar_t>(),
                need_row_elevations_grad ? grad_row_elevations.data_ptr<scalar_t>() : nullptr,
                need_column_azimuths_grad ? grad_column_azimuths.data_ptr<scalar_t>() : nullptr,
                emit_offsets_grad ? grad_row_offsets.data_ptr<scalar_t>() : nullptr,
                current_stream(elements));
        });
    return {grad_row_elevations, grad_column_azimuths, grad_row_offsets};
}

// ===========================================================================
// generate_spinning_lidar_rays — forward + backward
//
// world_rays are FORCED float32 in the public binding; the autograd Function
// permits float64 (gradcheck).  The angle tables / control poses cross the
// autograd boundary as typed tensors; n_rows / n_columns / has_row_offsets ride
// on the projection POD.  timestamps_us / poses are non-differentiable outputs.
// ===========================================================================

namespace {

void check_control_poses(
    const at::Tensor& control_translations,
    const at::Tensor& control_rotations,
    const at::Tensor& ref,
    const char* ref_name) {
    TORCH_CHECK(
        control_translations.dim() == 2 && control_translations.size(0) == 2 &&
            control_translations.size(1) == 3,
        "control_translations must be (2, 3)");
    TORCH_CHECK(
        control_rotations.dim() == 2 && control_rotations.size(0) == 2 &&
            control_rotations.size(1) == 4,
        "control_rotations must be (2, 4) wxyz");
    TORCH_CHECK(
        control_translations.scalar_type() == ref.scalar_type() &&
            control_rotations.scalar_type() == ref.scalar_type(),
        "control poses dtype must match angle tables");
    TORCH_CHECK(
        control_translations.is_cuda() && control_rotations.is_cuda(),
        "control poses must be CUDA tensors");
    check_same_device(control_translations, ref, "control_translations", ref_name);
    check_same_device(control_rotations, ref, "control_rotations", ref_name);
    TORCH_CHECK(
        control_translations.is_contiguous() && control_rotations.is_contiguous(),
        "control poses must be contiguous");
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> generate_spinning_lidar_rays(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Tensor& elements,
    const at::Tensor& row_elevations_rad,
    const at::Tensor& column_azimuths_rad,
    const at::Tensor& row_azimuth_offsets_rad,
    const at::Tensor& control_translations,
    const at::Tensor& control_rotations,
    bool does_generate_elements,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");
    const int64_t n_rows = row_elevations_rad.size(0);
    const int64_t n_columns = column_azimuths_rad.size(0);
    TORCH_CHECK(n_rows > 0 && n_columns > 0, "angle tables must be non-empty");
    check_angle_table(
        row_elevations_rad,
        n_rows,
        row_elevations_rad,
        "row_elevations_rad",
        "row_elevations_rad");
    check_angle_table(
        column_azimuths_rad,
        n_columns,
        row_elevations_rad,
        "column_azimuths_rad",
        "row_elevations_rad");
    if (projection->has_row_offsets) {
        check_angle_table(
            row_azimuth_offsets_rad,
            n_rows,
            row_elevations_rad,
            "row_azimuth_offsets_rad",
            "row_elevations_rad");
    }
    check_control_poses(control_translations, control_rotations, row_elevations_rad, "row_elevations_rad");
    check_lidar_projection_for_device(projection, row_elevations_rad.device());

    const int has_row_offsets = projection->has_row_offsets ? 1 : 0;
    const int generate = does_generate_elements ? 1 : 0;

    int64_t count;
    const int32_t* elements_ptr = nullptr;
    if (does_generate_elements) {
        count = n_rows * n_columns;
    } else {
        check_int_matrix(elements, 2, "elements");
        check_same_device(elements, row_elevations_rad, "elements", "row_elevations_rad");
        count = elements.size(0);
        elements_ptr = elements.const_data_ptr<int32_t>();
    }

    auto guard = c10::cuda::CUDAGuard(row_elevations_rad.device());
    auto world_rays = at::empty({count, 6}, row_elevations_rad.options());
    auto timestamps_us =
        at::empty({count}, row_elevations_rad.options().dtype(at::kLong));
    auto poses_translation = at::empty({count, 3}, row_elevations_rad.options());
    auto poses_rotation = at::empty({count, 4}, row_elevations_rad.options());

    AT_DISPATCH_FLOATING_TYPES(
        row_elevations_rad.scalar_type(), "generate_spinning_lidar_rays", [&] {
            generate_spinning_lidar_rays_launch<scalar_t>(
                count,
                n_rows,
                n_columns,
                has_row_offsets,
                generate,
                start_timestamp_us,
                end_timestamp_us,
                elements_ptr,
                row_elevations_rad.const_data_ptr<scalar_t>(),
                column_azimuths_rad.const_data_ptr<scalar_t>(),
                has_row_offsets ? row_azimuth_offsets_rad.const_data_ptr<scalar_t>() : nullptr,
                control_translations.const_data_ptr<scalar_t>(),
                control_rotations.const_data_ptr<scalar_t>(),
                world_rays.data_ptr<scalar_t>(),
                timestamps_us.data_ptr<int64_t>(),
                poses_translation.data_ptr<scalar_t>(),
                poses_rotation.data_ptr<scalar_t>(),
                current_stream(row_elevations_rad));
        });
    return {world_rays, timestamps_us, poses_translation, poses_rotation};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
generate_spinning_lidar_rays_backward(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Tensor& elements,
    const at::Tensor& row_elevations_rad,
    const at::Tensor& column_azimuths_rad,
    const at::Tensor& row_azimuth_offsets_rad,
    const at::Tensor& control_translations,
    const at::Tensor& control_rotations,
    bool does_generate_elements,
    const at::Tensor& grad_world_rays,
    bool need_row_elevations_grad,
    bool need_column_azimuths_grad,
    bool need_row_offsets_grad,
    bool need_control_translations_grad,
    bool need_control_rotations_grad) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");
    const int64_t n_rows = row_elevations_rad.size(0);
    const int64_t n_columns = column_azimuths_rad.size(0);
    check_angle_table(
        row_elevations_rad,
        n_rows,
        row_elevations_rad,
        "row_elevations_rad",
        "row_elevations_rad");
    check_angle_table(
        column_azimuths_rad,
        n_columns,
        row_elevations_rad,
        "column_azimuths_rad",
        "row_elevations_rad");
    if (projection->has_row_offsets) {
        check_angle_table(
            row_azimuth_offsets_rad,
            n_rows,
            row_elevations_rad,
            "row_azimuth_offsets_rad",
            "row_elevations_rad");
    }
    check_control_poses(control_translations, control_rotations, row_elevations_rad, "row_elevations_rad");
    check_float_matrix(grad_world_rays, 6, "grad_world_rays");
    TORCH_CHECK(
        grad_world_rays.scalar_type() == row_elevations_rad.scalar_type(),
        "grad_world_rays dtype must match angle tables");
    check_lidar_projection_for_device(projection, row_elevations_rad.device());
    check_same_device(grad_world_rays, row_elevations_rad, "grad_world_rays", "row_elevations_rad");

    const int has_row_offsets = projection->has_row_offsets ? 1 : 0;
    const int generate = does_generate_elements ? 1 : 0;

    int64_t count;
    const int32_t* elements_ptr = nullptr;
    if (does_generate_elements) {
        count = n_rows * n_columns;
    } else {
        check_int_matrix(elements, 2, "elements");
        check_same_device(elements, row_elevations_rad, "elements", "row_elevations_rad");
        count = elements.size(0);
        elements_ptr = elements.const_data_ptr<int32_t>();
    }
    TORCH_CHECK(grad_world_rays.size(0) == count, "grad_world_rays batch must match element count");

    auto guard = c10::cuda::CUDAGuard(row_elevations_rad.device());

    auto grad_row_elevations = need_row_elevations_grad
        ? at::zeros({n_rows}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    auto grad_column_azimuths = need_column_azimuths_grad
        ? at::zeros({n_columns}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    const bool emit_offsets_grad = has_row_offsets != 0 && need_row_offsets_grad;
    auto grad_row_offsets = emit_offsets_grad
        ? at::zeros({n_rows}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    auto grad_control_translations = need_control_translations_grad
        ? at::zeros({2, 3}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());
    auto grad_control_rotations = need_control_rotations_grad
        ? at::zeros({2, 4}, row_elevations_rad.options())
        : at::empty({0}, row_elevations_rad.options());

    AT_DISPATCH_FLOATING_TYPES(
        row_elevations_rad.scalar_type(), "generate_spinning_lidar_rays_backward", [&] {
            generate_spinning_lidar_rays_backward_launch<scalar_t>(
                count,
                n_rows,
                n_columns,
                has_row_offsets,
                generate,
                elements_ptr,
                row_elevations_rad.const_data_ptr<scalar_t>(),
                column_azimuths_rad.const_data_ptr<scalar_t>(),
                has_row_offsets ? row_azimuth_offsets_rad.const_data_ptr<scalar_t>() : nullptr,
                control_translations.const_data_ptr<scalar_t>(),
                control_rotations.const_data_ptr<scalar_t>(),
                grad_world_rays.const_data_ptr<scalar_t>(),
                need_row_elevations_grad ? grad_row_elevations.data_ptr<scalar_t>() : nullptr,
                need_column_azimuths_grad ? grad_column_azimuths.data_ptr<scalar_t>() : nullptr,
                emit_offsets_grad ? grad_row_offsets.data_ptr<scalar_t>() : nullptr,
                need_control_translations_grad ? grad_control_translations.data_ptr<scalar_t>() : nullptr,
                need_control_rotations_grad ? grad_control_rotations.data_ptr<scalar_t>() : nullptr,
                current_stream(row_elevations_rad));
        });
    return {
        grad_row_elevations,
        grad_column_azimuths,
        grad_row_offsets,
        grad_control_translations,
        grad_control_rotations};
}

// ===========================================================================
// inverse_project_spinning_lidar — forward + backward (IFT)
//
// world_points keep their dtype (no force-float32); the angle tables and the
// two control poses are threaded explicitly so the op stays dtype-templated.
// FOV / spinning-direction scalars are read off the projection.  The converged
// interpolation alpha is returned as the scratch tuple element for the backward
// IFT step.
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
inverse_project_spinning_lidar(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& projection,
    const at::Tensor& world_points,
    const at::Tensor& row_elevations_rad,
    const at::Tensor& column_azimuths_rad,
    const at::Tensor& row_azimuth_offsets_rad,
    const at::Tensor& control_translations,
    const at::Tensor& control_rotations,
    int64_t max_iterations,
    double stop_mean_relative_time_error,
    double stop_delta_mean_relative_time_error,
    double initial_relative_time,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us) {
    TORCH_CHECK(projection != nullptr, "projection must be RowOffsetStructuredSpinningLidarProjection");
    check_float_matrix(world_points, 3, "world_points");
    const int64_t n_rows = row_elevations_rad.size(0);
    const int64_t n_columns = column_azimuths_rad.size(0);
    TORCH_CHECK(n_rows > 0 && n_columns > 0, "angle tables must be non-empty");
    check_angle_table(
        row_elevations_rad,
        n_rows,
        world_points,
        "row_elevations_rad",
        "world_points");
    check_angle_table(
        column_azimuths_rad,
        n_columns,
        world_points,
        "column_azimuths_rad",
        "world_points");
    if (projection->has_row_offsets) {
        check_angle_table(
            row_azimuth_offsets_rad,
            n_rows,
            world_points,
            "row_azimuth_offsets_rad",
            "world_points");
    }
    check_control_poses(control_translations, control_rotations, world_points, "world_points");
    check_inverse_solver_scalars(
        max_iterations,
        stop_mean_relative_time_error,
        stop_delta_mean_relative_time_error,
        initial_relative_time);
    check_lidar_projection_for_device(projection, world_points.device());

    const int64_t count = world_points.size(0);
    const int has_row_offsets = projection->has_row_offsets ? 1 : 0;
    const int spinning_dir = static_cast<int>(projection->spinning_direction);

    auto guard = c10::cuda::CUDAGuard(world_points.device());
    auto sensor_angles = at::empty({count, 2}, world_points.options());
    auto valid_flags = at::empty({count}, world_points.options().dtype(at::kBool));
    auto timestamps_us = at::empty({count}, world_points.options().dtype(at::kLong));
    auto poses_translation = at::empty({count, 3}, world_points.options());
    auto poses_rotation = at::empty({count, 4}, world_points.options());
    auto scratch = at::empty({count}, world_points.options());

    AT_DISPATCH_FLOATING_TYPES(
        world_points.scalar_type(), "inverse_project_spinning_lidar", [&] {
            inverse_project_spinning_lidar_launch<scalar_t>(
                count,
                n_rows,
                n_columns,
                has_row_offsets,
                spinning_dir,
                static_cast<scalar_t>(projection->fov_vert_start_rad),
                static_cast<scalar_t>(projection->fov_vert_span_rad),
                static_cast<scalar_t>(projection->fov_horiz_start_rad),
                static_cast<scalar_t>(projection->fov_horiz_span_rad),
                max_iterations,
                static_cast<scalar_t>(stop_mean_relative_time_error),
                static_cast<scalar_t>(stop_delta_mean_relative_time_error),
                static_cast<scalar_t>(initial_relative_time),
                start_timestamp_us,
                end_timestamp_us,
                world_points.const_data_ptr<scalar_t>(),
                row_elevations_rad.const_data_ptr<scalar_t>(),
                column_azimuths_rad.const_data_ptr<scalar_t>(),
                has_row_offsets ? row_azimuth_offsets_rad.const_data_ptr<scalar_t>() : nullptr,
                control_translations.const_data_ptr<scalar_t>(),
                control_rotations.const_data_ptr<scalar_t>(),
                sensor_angles.data_ptr<scalar_t>(),
                valid_flags.data_ptr<bool>(),
                timestamps_us.data_ptr<int64_t>(),
                poses_translation.data_ptr<scalar_t>(),
                poses_rotation.data_ptr<scalar_t>(),
                scratch.data_ptr<scalar_t>(),
                current_stream(world_points));
        });
    return {sensor_angles, valid_flags, timestamps_us, poses_translation, poses_rotation, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> inverse_project_spinning_lidar_backward(
    const at::Tensor& world_points,
    const at::Tensor& control_translations,
    const at::Tensor& control_rotations,
    const at::Tensor& valid_flags,
    const at::Tensor& scratch,
    const at::Tensor& grad_sensor_angles,
    bool need_world_points_grad,
    bool need_control_translations_grad,
    bool need_control_rotations_grad) {
    check_float_matrix(world_points, 3, "world_points");
    check_float_matrix(grad_sensor_angles, 2, "grad_sensor_angles");
    check_control_poses(control_translations, control_rotations, world_points, "world_points");
    check_same_device(grad_sensor_angles, world_points, "grad_sensor_angles", "world_points");
    TORCH_CHECK(
        grad_sensor_angles.scalar_type() == world_points.scalar_type(),
        "grad_sensor_angles dtype must match world_points");
    const int64_t count = world_points.size(0);
    TORCH_CHECK(grad_sensor_angles.size(0) == count, "grad_sensor_angles batch must match world_points");
    TORCH_CHECK(valid_flags.dim() == 1, "valid_flags must be (N,)");
    TORCH_CHECK(valid_flags.numel() == count, "valid_flags size must match world_points");
    TORCH_CHECK(valid_flags.is_cuda(), "valid_flags must be a CUDA tensor");
    check_same_device(valid_flags, world_points, "valid_flags", "world_points");
    TORCH_CHECK(valid_flags.scalar_type() == at::kBool, "valid_flags must be bool");
    TORCH_CHECK(valid_flags.is_contiguous(), "valid_flags must be contiguous");
    TORCH_CHECK(scratch.numel() == count, "scratch size must match world_points");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be a CUDA tensor");
    check_same_device(scratch, world_points, "scratch", "world_points");
    TORCH_CHECK(
        scratch.scalar_type() == world_points.scalar_type(),
        "scratch dtype must match world_points");
    TORCH_CHECK(scratch.is_contiguous(), "scratch must be contiguous");

    auto guard = c10::cuda::CUDAGuard(world_points.device());
    auto grad_world_points = need_world_points_grad
        ? at::empty_like(world_points)
        : at::empty({0}, world_points.options());
    auto grad_control_translations = need_control_translations_grad
        ? at::zeros({2, 3}, world_points.options())
        : at::empty({0}, world_points.options());
    auto grad_control_rotations = need_control_rotations_grad
        ? at::zeros({2, 4}, world_points.options())
        : at::empty({0}, world_points.options());

    AT_DISPATCH_FLOATING_TYPES(
        world_points.scalar_type(), "inverse_project_spinning_lidar_backward", [&] {
            inverse_project_spinning_lidar_backward_launch<scalar_t>(
                count,
                world_points.const_data_ptr<scalar_t>(),
                control_translations.const_data_ptr<scalar_t>(),
                control_rotations.const_data_ptr<scalar_t>(),
                valid_flags.const_data_ptr<bool>(),
                grad_sensor_angles.const_data_ptr<scalar_t>(),
                scratch.const_data_ptr<scalar_t>(),
                need_world_points_grad ? grad_world_points.data_ptr<scalar_t>() : nullptr,
                need_control_translations_grad ? grad_control_translations.data_ptr<scalar_t>() : nullptr,
                need_control_rotations_grad ? grad_control_rotations.data_ptr<scalar_t>() : nullptr,
                current_stream(world_points));
        });
    return {grad_world_points, grad_control_translations, grad_control_rotations};
}

} // namespace gsplat_sensors
