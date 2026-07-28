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

// lidar_torch.h — C++ host wrapper exposed to ext.cpp via TORCH_LIBRARY.
//
// Declares the RowOffsetStructuredSpinningLidarProjection CustomClassHolder
// that carries the spinning-LiDAR intrinsics as per-component float32 tensors
// plus scalar FOV / direction fields, and a to_kernel_params() that collects
// raw device pointers into the Tier-1 POD (lidar_params.h).
//
// Mirrors camera_torch.h structure (OpenCVPinholeProjection).

#pragma once

#include "lidar_params.h"

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/custom_class.h>

#include <tuple>

namespace gsplat_sensors
{
// TorchScript custom class carrying the structured spinning-LiDAR intrinsics
// as per-component float32 tensors plus scalar FOV / direction fields.
// Shape contract:
//   row_elevations_rad      (n_rows,)    — per-row elevation in radians
//   column_azimuths_rad     (n_columns,) — per-column azimuth in radians
//   row_azimuth_offsets_rad (n_rows,)    — per-row azimuth offset, OR empty (0,)
//                                          when has_row_offsets is false
struct RowOffsetStructuredSpinningLidarProjection : public torch::CustomClassHolder
{
    at::Tensor row_elevations_rad;
    at::Tensor column_azimuths_rad;
    at::Tensor row_azimuth_offsets_rad;
    double fov_vert_start_rad;
    double fov_vert_span_rad;
    double fov_horiz_start_rad;
    double fov_horiz_span_rad;
    int64_t spinning_direction;
    bool has_row_offsets;

    RowOffsetStructuredSpinningLidarProjection(
        at::Tensor row_elevations_rad,
        at::Tensor column_azimuths_rad,
        at::Tensor row_azimuth_offsets_rad,
        double fov_vert_start_rad,
        double fov_vert_span_rad,
        double fov_horiz_start_rad,
        double fov_horiz_span_rad,
        int64_t spinning_direction,
        bool has_row_offsets
    );

    RowOffsetStructuredSpinningLidarProjection_KernelParameters to_kernel_params() const;
};

// Validates projection shape/dtype/scalar fields (CPU-constructible); throws
// TORCH_CHECK on failure.
void check_lidar_projection(const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection);

// Launch-time guard: asserts the projection's angle tables are CUDA and share
// `device`; throws TORCH_CHECK on failure.
void check_lidar_projection_for_device(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection, const at::Device &device
);

// ===========================================================================
// LIGHT-op wrappers
// ===========================================================================

at::Tensor sensor_rays_to_sensor_angles(const at::Tensor &sensor_rays);
at::Tensor sensor_rays_to_sensor_angles_backward(const at::Tensor &sensor_rays, const at::Tensor &grad_sensor_angles);

at::Tensor sensor_angles_to_sensor_rays(const at::Tensor &sensor_angles);
at::Tensor sensor_angles_to_sensor_rays_backward(const at::Tensor &sensor_angles, const at::Tensor &grad_sensor_rays);

std::tuple<at::Tensor, at::Tensor> elements_to_sensor_angles(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection,
    const at::Tensor &elements,
    const at::Tensor &row_elevations_rad,
    const at::Tensor &column_azimuths_rad,
    const at::Tensor &row_azimuth_offsets_rad
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> elements_to_sensor_angles_backward(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection,
    const at::Tensor &elements,
    const at::Tensor &row_elevations_rad,
    const at::Tensor &column_azimuths_rad,
    const at::Tensor &row_azimuth_offsets_rad,
    const at::Tensor &grad_sensor_angles,
    bool need_row_elevations_grad,
    bool need_column_azimuths_grad,
    bool need_row_offsets_grad
);

// ===========================================================================
// HEAVY-op wrappers: generate_spinning_lidar_rays
// ===========================================================================
//
// Angle tables + control poses are threaded explicitly (dtype-templated; the
// public binding forces float32, fp64 gradcheck passes double straight
// through).  Control rotations are wxyz (2 poses).  `elements` is int32
// (row, col); in generate-mode (does_generate_elements true) it is ignored and
// the full (n_rows x n_columns) grid is iterated.

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> generate_spinning_lidar_rays(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection,
    const at::Tensor &elements,
    const at::Tensor &row_elevations_rad,
    const at::Tensor &column_azimuths_rad,
    const at::Tensor &row_azimuth_offsets_rad,
    const at::Tensor &control_translations,
    const at::Tensor &control_rotations,
    bool does_generate_elements,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> generate_spinning_lidar_rays_backward(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection,
    const at::Tensor &elements,
    const at::Tensor &row_elevations_rad,
    const at::Tensor &column_azimuths_rad,
    const at::Tensor &row_azimuth_offsets_rad,
    const at::Tensor &control_translations,
    const at::Tensor &control_rotations,
    bool does_generate_elements,
    const at::Tensor &grad_world_rays,
    bool need_row_elevations_grad,
    bool need_column_azimuths_grad,
    bool need_row_offsets_grad,
    bool need_control_translations_grad,
    bool need_control_rotations_grad
);

// ===========================================================================
// HEAVY-op wrappers: inverse_project_spinning_lidar
// ===========================================================================
//
// Angle tables + control poses are threaded explicitly (dtype-templated; the
// public binding keeps world_points.dtype, fp64 gradcheck passes double).
// Control rotations are wxyz (2 poses).  FOV / spinning-direction scalars ride
// on the projection POD.  The forward returns the converged-alpha scratch as the
// 6th tuple element so autograd can carry it into the IFT backward.

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> inverse_project_spinning_lidar(
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> &projection,
    const at::Tensor &world_points,
    const at::Tensor &row_elevations_rad,
    const at::Tensor &column_azimuths_rad,
    const at::Tensor &row_azimuth_offsets_rad,
    const at::Tensor &control_translations,
    const at::Tensor &control_rotations,
    int64_t max_iterations,
    double stop_mean_relative_time_error,
    double stop_delta_mean_relative_time_error,
    double initial_relative_time,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> inverse_project_spinning_lidar_backward(
    const at::Tensor &world_points,
    const at::Tensor &control_translations,
    const at::Tensor &control_rotations,
    const at::Tensor &valid_flags,
    const at::Tensor &scratch,
    const at::Tensor &grad_sensor_angles,
    bool need_world_points_grad,
    bool need_control_translations_grad,
    bool need_control_rotations_grad
);
} // namespace gsplat_sensors
