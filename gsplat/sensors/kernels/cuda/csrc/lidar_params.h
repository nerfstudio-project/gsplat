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

// lidar_params.h — kernel parameter-pack struct and CUDA launch prototypes for
// the spinning-LiDAR projection.
//
// Defines the plain-old-data struct passed by value into every LiDAR CUDA
// kernel (lidar_kernel.cu / lidar_kernel_backward.cu), and is the Tier-1 bridge
// that crosses the Torch/CUDA boundary.  No torch/ATen includes here; only
// <cstdint> + a forward-declared cudaStream_t.  All pointers carry __restrict__
// so the compiler may assume no aliasing between them.

#pragma once

#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace gsplat_sensors
{
// Source of truth for spinning-LiDAR azimuth sweep direction. Bound to Python
// in `ext.cpp` and re-exported as a Python IntEnum from
// `libs/sensors/kernels/lidars/types.py`. The Python module verifies at import
// time that its integer values match this enum.
//
// CLOCKWISE / COUNTERCLOCKWISE only affect azimuth; vertical FOV handling is
// direction-independent.
enum class SpinningDirection : int
{
    CLOCKWISE        = 0,
    COUNTERCLOCKWISE = 1,
};
} // namespace gsplat_sensors

// Hard upper bound on the inverse-projection rolling-shutter iterative solve.
// The host guard (check_inverse_solver_scalars in lidar_torch.cpp) rejects
// user-supplied max_iterations values above it and the device loop is bounded
// by the same constant, so host and device share one source of truth; raising
// it requires recompilation.
constexpr int64_t kMaxSpinningLidarIterations = 32;

// Parameter pack consumed by every RowOffsetStructuredSpinningLidar kernel.
// Passed by value so the struct lands in registers / constant memory.
//
// Angle tables are raw float32 device pointers:
//   row_elevations_rad      (n_rows,)    — per-row elevation in radians
//   column_azimuths_rad     (n_columns,) — per-column azimuth in radians
//   row_azimuth_offsets_rad (n_rows,)    — per-row azimuth offset; only valid
//                                          when has_row_offsets != 0
struct RowOffsetStructuredSpinningLidarProjection_KernelParameters
{
    const float *__restrict__ row_elevations_rad;      // (n_rows,) float32
    const float *__restrict__ column_azimuths_rad;     // (n_columns,) float32
    const float *__restrict__ row_azimuth_offsets_rad; // (n_rows,) float32 or nullptr
    int64_t n_rows;
    int64_t n_columns;
    float fov_vert_start_rad;
    float fov_vert_span_rad;
    float fov_horiz_start_rad;
    float fov_horiz_span_rad;
    gsplat_sensors::SpinningDirection spinning_direction;
    int has_row_offsets; // 0 / 1
};

// ===========================================================================
// Forward launch prototypes
// ===========================================================================
//
// Covers the 3 LIGHT ops (sensor_rays_to_sensor_angles,
// sensor_angles_to_sensor_rays, elements_to_sensor_angles) and the 2 HEAVY ops
// (generate_spinning_lidar_rays, inverse_project_spinning_lidar), forward and
// backward.
//
// The ray<->angle conversions are dtype-polymorphic: the launchers are
// templated on the scalar type T (float / double) and explicitly instantiated
// in lidar_kernel.cu / lidar_kernel_backward.cu.  The element->angle op runs
// its conversion in T as well so fp64 gradcheck on the angle tables is exact;
// the angle tables / grads are passed as T pointers (the public binding forces
// float32, the autograd.Function permits double).

template<typename T>
void sensor_rays_to_sensor_angles_launch(int64_t count, const T *sensor_rays, T *sensor_angles, cudaStream_t stream);

template<typename T>
void sensor_angles_to_sensor_rays_launch(int64_t count, const T *sensor_angles, T *sensor_rays, cudaStream_t stream);

// Element->angle table lookup.  Angle tables are typed T pointers; element
// indices are int32 (row, col).  has_row_offsets / n_rows / n_columns ride in
// the projection POD but the tables themselves are passed explicitly as T* so
// the op can run under fp64 gradcheck.
template<typename T>
void elements_to_sensor_angles_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t *elements,
    const T *row_elevations_rad,
    const T *column_azimuths_rad,
    const T *row_azimuth_offsets_rad,
    T *sensor_angles,
    bool *valid_flags,
    cudaStream_t stream
);

// generate_spinning_lidar_rays forward.  Angle tables and control poses are
// typed T pointers so the kernel runs under fp64 gradcheck while the public
// binding forces float32.  Control rotations are wxyz storage (2 poses); the
// kernel reorders to xyzw internally.  `elements` is int32 (row, col) and may be
// nullptr in generate-mode (does_generate_elements != 0).
template<typename T>
void generate_spinning_lidar_rays_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    const int32_t *elements,
    const T *row_elevations_rad,
    const T *column_azimuths_rad,
    const T *row_azimuth_offsets_rad,
    const T *control_translations,
    const T *control_rotations,
    T *world_rays,
    int64_t *timestamps_us,
    T *poses_translation,
    T *poses_rotation,
    cudaStream_t stream
);

// generate_spinning_lidar_rays backward.  Scatters angle-table grads per index
// (atomicAdd) and block-reduces the shared control-pose grads to a single
// atomicAdd per slot.  grad pointers may be nullptr when not required.
template<typename T>
void generate_spinning_lidar_rays_backward_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    const int32_t *elements,
    const T *row_elevations_rad,
    const T *column_azimuths_rad,
    const T *row_azimuth_offsets_rad,
    const T *control_translations,
    const T *control_rotations,
    const T *grad_world_rays,
    T *grad_row_elevations_rad,
    T *grad_column_azimuths_rad,
    T *grad_row_azimuth_offsets_rad,
    T *grad_control_translations,
    T *grad_control_rotations,
    cudaStream_t stream
);

// inverse_project_spinning_lidar forward.  Dtype-templated (the public binding
// keeps world_points.dtype, fp64 gradcheck passes double).  Angle tables and
// control poses (wxyz, 2 poses) are typed T pointers.  The rolling-shutter loop
// runs max_iterations passes (bounded by kMaxSpinningLidarIterations, enforced
// host-side) with the two convergence breaks.  `scratch`
// (N,) stores the converged interpolation alpha for the IFT backward; may be
// nullptr when no grad is required.
template<typename T>
void inverse_project_spinning_lidar_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int spinning_dir,
    T fov_vert_start_rad,
    T fov_vert_span_rad,
    T fov_horiz_start_rad,
    T fov_horiz_span_rad,
    int64_t max_iterations,
    T stop_mean_relative_time_error,
    T stop_delta_mean_relative_time_error,
    T initial_relative_time,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    const T *world_points,
    const T *row_elevations_rad,
    const T *column_azimuths_rad,
    const T *row_azimuth_offsets_rad,
    const T *control_translations,
    const T *control_rotations,
    T *sensor_angles,
    bool *valid_flags,
    int64_t *timestamps_us,
    T *poses_translation,
    T *poses_rotation,
    T *scratch,
    cudaStream_t stream
);

// ===========================================================================
// Backward launch prototypes
// ===========================================================================

template<typename T>
void sensor_rays_to_sensor_angles_backward_launch(
    int64_t count, const T *sensor_rays, const T *grad_sensor_angles, T *grad_sensor_rays, cudaStream_t stream
);

template<typename T>
void sensor_angles_to_sensor_rays_backward_launch(
    int64_t count, const T *sensor_angles, const T *grad_sensor_rays, T *grad_sensor_angles, cudaStream_t stream
);

// Element->angle backward: scatter-accumulate (atomicAdd) into the three angle
// tables for the (row, col) each thread read.  grad pointers may be nullptr
// when the corresponding table does not require grad.
template<typename T>
void elements_to_sensor_angles_backward_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t *elements,
    const T *grad_sensor_angles,
    T *grad_row_elevations_rad,
    T *grad_column_azimuths_rad,
    T *grad_row_azimuth_offsets_rad,
    cudaStream_t stream
);

// inverse_project_spinning_lidar backward (IFT).  Re-loads the converged alpha
// from `scratch` and takes one differentiable step; grad routes to world_points
// (direct write) and the two control poses (block-reduced atomicAdd, xyzw->wxyz
// reorder).  grad pointers may be nullptr when not required.  Angle-table grads
// are identically zero (piecewise-constant relative-time map) and are not
// emitted here.
template<typename T>
void inverse_project_spinning_lidar_backward_launch(
    int64_t count,
    const T *world_points,
    const T *control_translations,
    const T *control_rotations,
    const bool *valid_flags,
    const T *grad_sensor_angles,
    const T *scratch,
    T *grad_world_points,
    T *grad_control_translations,
    T *grad_control_rotations,
    cudaStream_t stream
);
