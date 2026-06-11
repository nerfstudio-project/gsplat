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

// Forward CUDA kernels for the RowOffsetStructuredSpinningLidar projection.
//
// LIGHT ops:
//   sensor_rays_to_sensor_angles   (N,3) -> (N,2)  dtype-polymorphic
//   sensor_angles_to_sensor_rays   (N,2) -> (N,3)  dtype-polymorphic
//   elements_to_sensor_angles      (N,2) int + tables -> (N,2) angles + valid
// HEAVY ops:
//   generate_spinning_lidar_rays   rolling-shutter SLERP ray generation
//   inverse_project_spinning_lidar iterative rolling-shutter inverse projection
// Backwards live in lidar_kernel_backward.cu.

#include "lidar_kernel.cuh"

#include <c10/cuda/CUDAException.h>

using gsplat_sensors::kLidarThreads;
using gsplat_sensors::normalize_angle;
using gsplat_sensors::pi_const;
using gsplat_sensors::slerp_pair_fwd;

namespace {

dim3 grid_for_count(int64_t count) {
    return dim3(static_cast<unsigned int>((count + kLidarThreads - 1) / kLidarThreads));
}

// Inverse-projection device helpers used by the inverse kernel.  The
// relative-time path is the LINEAR fallback (find_nearest_row -> az - row_offset
// -> find_nearest_column -> col/(n_cols-1)).

// find_nearest_row — argmin |elevation - row_elev|, strict < keeps the first
// (lowest-index) row on ties.  No angle wrap
// (elevation never wraps).
template <typename T>
__device__ __forceinline__ int find_nearest_row(
    T elevation, const T* __restrict__ row_elevations_rad, int64_t n_rows) {
    T min_diff = T(1e10);
    int best_row = 0;
    for (int64_t row = 0; row < n_rows; ++row) {
        T diff = fabs(elevation - row_elevations_rad[row]);
        if (diff < min_diff) {
            min_diff = diff;
            best_row = row;
        }
    }
    return best_row;
}

// find_nearest_column — argmin |wrap(azimuth - col_az)| (wrapped angular
// distance, +-pi seam handling), strict < keeps the first column.
// The wrapped distance is computed with a
// branchless rint-based reduction (d - 2pi*rint(d/2pi)) rather than
// normalize_angle's fmod + sign branch: the per-column distance metric is the
// same |wrapped angle| in [0, pi], but rint is a single hardware instruction
// whereas fmod is a costly library call run on every one of the up-to-3600
// columns each fixed-point iteration.
template <typename T>
__device__ __forceinline__ int find_nearest_column(
    T azimuth, const T* __restrict__ column_azimuths_rad, int64_t n_columns) {
    const T two_pi = T(2) * pi_const<T>();
    const T inv_two_pi = T(1) / two_pi;
    T min_diff = T(1e10);
    int best_col = 0;
    for (int64_t col = 0; col < n_columns; ++col) {
        const T d = azimuth - column_azimuths_rad[col];
        const T wrapped = d - two_pi * rint(d * inv_two_pi);
        T diff = fabs(wrapped);
        if (diff < min_diff) {
            min_diff = diff;
            best_col = col;
        }
    }
    return best_col;
}

// sensor_angles_to_relative_time — LINEAR fallback (no +-1-rel CCW flip).
template <typename T>
__device__ __forceinline__ T sensor_angles_to_relative_time(
    T elevation,
    T azimuth,
    const T* __restrict__ row_elevations_rad,
    const T* __restrict__ column_azimuths_rad,
    const T* __restrict__ row_azimuth_offsets_rad,
    int has_row_offsets,
    int64_t n_rows,
    int64_t n_columns) {
    int row = find_nearest_row<T>(elevation, row_elevations_rad, n_rows);
    T adjusted_azimuth = azimuth;
    if (has_row_offsets != 0) {
        adjusted_azimuth -= row_azimuth_offsets_rad[row];
    }
    int col = find_nearest_column<T>(adjusted_azimuth, column_azimuths_rad, n_columns);
    return (n_columns > 1) ? static_cast<T>(col) / static_cast<T>(n_columns - 1) : T(0);
}

// is_in_fov — top-down vertical [start-span, start], direction-dependent
// horizontal sign.  No epsilon (the fov_eps slack is
// Python-model-layer only).  spinning_dir: 0 = CW, 1 = CCW.
template <typename T>
__device__ __forceinline__ bool is_in_fov(
    T elevation,
    T azimuth,
    T fov_vert_start_rad,
    T fov_vert_span_rad,
    T fov_horiz_start_rad,
    T fov_horiz_span_rad,
    int spinning_dir) {
    if (elevation > fov_vert_start_rad ||
        elevation < fov_vert_start_rad - fov_vert_span_rad) {
        return false;
    }
    const T two_pi = T(2) * pi_const<T>();
    T rel_azimuth = (spinning_dir == 1)
        ? azimuth - fov_horiz_start_rad
        : fov_horiz_start_rad - azimuth;
    rel_azimuth = fmod(rel_azimuth, two_pi);
    if (rel_azimuth < T(0)) {
        rel_azimuth += two_pi;
    }
    return rel_azimuth <= fov_horiz_span_rad;
}

template <typename T>
__global__ void sensor_rays_to_sensor_angles_kernel(
    int64_t count,
    const T* __restrict__ sensor_rays,
    T* __restrict__ sensor_angles) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    T x = sensor_rays[idx * 3 + 0];
    T y = sensor_rays[idx * 3 + 1];
    T z = sensor_rays[idx * 3 + 2];
    T elevation;
    T azimuth;
    cartesian_to_spherical<T>(x, y, z, elevation, azimuth);
    sensor_angles[idx * 2 + 0] = elevation;
    sensor_angles[idx * 2 + 1] = azimuth;
}

template <typename T>
__global__ void sensor_angles_to_sensor_rays_kernel(
    int64_t count,
    const T* __restrict__ sensor_angles,
    T* __restrict__ sensor_rays) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    T elevation = sensor_angles[idx * 2 + 0];
    T azimuth = sensor_angles[idx * 2 + 1];
    T rx;
    T ry;
    T rz;
    spherical_to_cartesian<T>(elevation, azimuth, rx, ry, rz);
    sensor_rays[idx * 3 + 0] = rx;
    sensor_rays[idx * 3 + 1] = ry;
    sensor_rays[idx * 3 + 2] = rz;
}

template <typename T>
__global__ void elements_to_sensor_angles_kernel(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t* __restrict__ elements,
    const T* __restrict__ row_elevations_rad,
    const T* __restrict__ column_azimuths_rad,
    const T* __restrict__ row_azimuth_offsets_rad,
    T* __restrict__ sensor_angles,
    bool* __restrict__ valid_flags) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    int32_t row = elements[idx * 2 + 0];
    int32_t col = elements[idx * 2 + 1];

    if (row < 0 || row >= n_rows || col < 0 || col >= n_columns) {
        valid_flags[idx] = false;
        sensor_angles[idx * 2 + 0] = T(0);
        sensor_angles[idx * 2 + 1] = T(0);
        return;
    }

    valid_flags[idx] = true;
    T elevation = row_elevations_rad[row];
    T azimuth = column_azimuths_rad[col];
    if (has_row_offsets != 0) {
        azimuth += row_azimuth_offsets_rad[row];
        azimuth = normalize_angle<T>(azimuth);
    }
    sensor_angles[idx * 2 + 0] = elevation;
    sensor_angles[idx * 2 + 1] = azimuth;
}

// Forward world-ray generation.  Two-pose DynamicPose with
// control_times == [0, 1], so alpha = clamp(relative_time).
// relative_time derives from the COLUMN index (not azimuth).  world ray =
// [origin = interpolated translation, dir = R(q_interp) * sensor_ray].
template <typename T>
__global__ void generate_spinning_lidar_rays_kernel(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    const int32_t* __restrict__ elements,
    const T* __restrict__ row_elevations_rad,
    const T* __restrict__ column_azimuths_rad,
    const T* __restrict__ row_azimuth_offsets_rad,
    const T* __restrict__ control_translations, // (2, 3)
    const T* __restrict__ control_rotations,    // (2, 4) wxyz
    T* __restrict__ world_rays,
    int64_t* __restrict__ timestamps_us,
    T* __restrict__ poses_translation,
    T* __restrict__ poses_rotation) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int32_t row;
    int32_t col;
    if (does_generate_elements != 0) {
        row = static_cast<int32_t>(idx / n_columns);
        col = static_cast<int32_t>(idx - static_cast<int64_t>(row) * n_columns);
    } else {
        row = elements[idx * 2 + 0];
        col = elements[idx * 2 + 1];
    }

    bool valid = (row >= 0 && row < n_rows && col >= 0 && col < n_columns);

    T elevation = T(0);
    T azimuth = T(0);
    if (valid) {
        elevation = row_elevations_rad[row];
        azimuth = column_azimuths_rad[col];
        if (has_row_offsets != 0) {
            azimuth += row_azimuth_offsets_rad[row];
            azimuth = normalize_angle<T>(azimuth);
        }
    }

    // Column index is the exact time ordering of the scan.
    T relative_time = (n_columns > 1)
        ? static_cast<T>(col) / static_cast<T>(n_columns - 1)
        : T(0);
    T alpha = relative_time < T(0) ? T(0) : (relative_time > T(1) ? T(1) : relative_time);

    // Control poses (wxyz storage -> xyzw registers).
    const T t0x = control_translations[0], t0y = control_translations[1], t0z = control_translations[2];
    const T t1x = control_translations[3], t1y = control_translations[4], t1z = control_translations[5];
    const T q0x = control_rotations[1], q0y = control_rotations[2], q0z = control_rotations[3], q0w = control_rotations[0];
    const T q1x = control_rotations[5], q1y = control_rotations[6], q1z = control_rotations[7], q1w = control_rotations[4];

    const T om = T(1) - alpha;
    T trans_x = om * t0x + alpha * t1x;
    T trans_y = om * t0y + alpha * t1y;
    T trans_z = om * t0z + alpha * t1z;

    T qix, qiy, qiz, qiw;
    slerp_pair_fwd<T>(q0x, q0y, q0z, q0w, q1x, q1y, q1z, q1w, alpha, &qix, &qiy, &qiz, &qiw);

    T sx, sy, sz;
    spherical_to_cartesian<T>(elevation, azimuth, sx, sy, sz);
    T wx, wy, wz;
    quat_rotate_vector_fwd_impl<T>(qix, qiy, qiz, qiw, sx, sy, sz, &wx, &wy, &wz);

    if (!valid) {
        trans_x = trans_y = trans_z = T(0);
        wx = wy = wz = T(0);
    }

    world_rays[idx * 6 + 0] = trans_x;
    world_rays[idx * 6 + 1] = trans_y;
    world_rays[idx * 6 + 2] = trans_z;
    world_rays[idx * 6 + 3] = wx;
    world_rays[idx * 6 + 4] = wy;
    world_rays[idx * 6 + 5] = wz;

    int64_t timestamp = valid
        ? start_timestamp_us +
            static_cast<int64_t>(static_cast<double>(relative_time) *
                static_cast<double>(end_timestamp_us - start_timestamp_us))
        : start_timestamp_us;
    timestamps_us[idx] = timestamp;

    poses_translation[idx * 3 + 0] = valid ? trans_x : T(0);
    poses_translation[idx * 3 + 1] = valid ? trans_y : T(0);
    poses_translation[idx * 3 + 2] = valid ? trans_z : T(0);
    // wxyz writeback (identity (1,0,0,0) when invalid).
    poses_rotation[idx * 4 + 0] = valid ? qiw : T(1);
    poses_rotation[idx * 4 + 1] = valid ? qix : T(0);
    poses_rotation[idx * 4 + 2] = valid ? qiy : T(0);
    poses_rotation[idx * 4 + 3] = valid ? qiz : T(0);
}

// Inverse projection.  Per world point, runs the
// azimuth->relative_time fixed-point loop: interpolate the 2-pose trajectory at
// relative_time, R^T(p - t), normalize, ray->angles, FOV reject, then a fresh
// relative_time from the LINEAR angle->time fallback.  Differentiable output is
// sensor_angles (N,2); valid_flags / timestamps_us / poses are non-diff.  The
// converged interpolation alpha is saved to scratch (stride 1) for the IFT
// backward, which freezes that alpha and takes one differentiable step (the
// LINEAR relative_time map is piecewise-constant, so the implicit-function
// correction term vanishes and the frozen-alpha step is the exact VJP).
template <typename T>
__global__ void __launch_bounds__(kLidarThreads, 5)
inverse_project_spinning_lidar_kernel(
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
    const T* __restrict__ world_points,
    const T* __restrict__ row_elevations_rad,
    const T* __restrict__ column_azimuths_rad,
    const T* __restrict__ row_azimuth_offsets_rad,
    const T* __restrict__ control_translations, // (2, 3)
    const T* __restrict__ control_rotations,    // (2, 4) wxyz
    T* __restrict__ sensor_angles,
    bool* __restrict__ valid_flags,
    int64_t* __restrict__ timestamps_us,
    T* __restrict__ poses_translation,
    T* __restrict__ poses_rotation,
    T* __restrict__ scratch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const T px = world_points[idx * 3 + 0];
    const T py = world_points[idx * 3 + 1];
    const T pz = world_points[idx * 3 + 2];

    const T t0x = control_translations[0], t0y = control_translations[1], t0z = control_translations[2];
    const T t1x = control_translations[3], t1y = control_translations[4], t1z = control_translations[5];
    const T q0x = control_rotations[1], q0y = control_rotations[2], q0z = control_rotations[3], q0w = control_rotations[0];
    const T q1x = control_rotations[5], q1y = control_rotations[6], q1z = control_rotations[7], q1w = control_rotations[4];

    T relative_time = initial_relative_time;
    T prev_time_diff = T(1);

    T result_elev = T(0);
    T result_az = T(0);
    bool result_valid = false;

    T final_alpha = T(0);
    T final_trans_x = T(0), final_trans_y = T(0), final_trans_z = T(0);
    T final_qx = T(0), final_qy = T(0), final_qz = T(0), final_qw = T(1);

    for (int64_t iter = 0; iter < max_iterations; ++iter) {
        T alpha = relative_time < T(0) ? T(0) : (relative_time > T(1) ? T(1) : relative_time);
        const T om = T(1) - alpha;
        T trans_x = om * t0x + alpha * t1x;
        T trans_y = om * t0y + alpha * t1y;
        T trans_z = om * t0z + alpha * t1z;
        T qix, qiy, qiz, qiw;
        slerp_pair_fwd<T>(q0x, q0y, q0z, q0w, q1x, q1y, q1z, q1w, alpha, &qix, &qiy, &qiz, &qiw);

        final_alpha = alpha;
        final_trans_x = trans_x;
        final_trans_y = trans_y;
        final_trans_z = trans_z;
        final_qx = qix;
        final_qy = qiy;
        final_qz = qiz;
        final_qw = qiw;

        T sx, sy, sz;
        se3_inverse_transform_point<T>(qix, qiy, qiz, qiw, trans_x, trans_y, trans_z, px, py, pz, &sx, &sy, &sz);
        const T inv_n = T(1) / sqrt(fmax(sx * sx + sy * sy + sz * sz, gsplat_sensors::normalize_normsq_floor<T>()));
        sx *= inv_n;
        sy *= inv_n;
        sz *= inv_n;

        T elevation, azimuth;
        cartesian_to_spherical<T>(sx, sy, sz, elevation, azimuth);

        if (!is_in_fov<T>(elevation, azimuth, fov_vert_start_rad, fov_vert_span_rad,
                          fov_horiz_start_rad, fov_horiz_span_rad, spinning_dir)) {
            result_valid = false;
            break;
        }

        result_elev = elevation;
        result_az = azimuth;
        result_valid = true;

        T new_relative_time = sensor_angles_to_relative_time<T>(
            elevation, azimuth, row_elevations_rad, column_azimuths_rad,
            row_azimuth_offsets_rad, has_row_offsets, n_rows, n_columns);

        T time_diff = fabs(new_relative_time - relative_time);
        if (time_diff < stop_mean_relative_time_error) {
            break;
        }
        T delta_time_diff = fabs(time_diff - prev_time_diff);
        if (iter > 0 && delta_time_diff < stop_delta_mean_relative_time_error) {
            break;
        }
        prev_time_diff = time_diff;
        relative_time = new_relative_time;
    }

    sensor_angles[idx * 2 + 0] = result_elev;
    sensor_angles[idx * 2 + 1] = result_az;
    valid_flags[idx] = result_valid;
    timestamps_us[idx] = start_timestamp_us +
        static_cast<int64_t>(static_cast<double>(relative_time) *
            static_cast<double>(end_timestamp_us - start_timestamp_us));
    poses_translation[idx * 3 + 0] = final_trans_x;
    poses_translation[idx * 3 + 1] = final_trans_y;
    poses_translation[idx * 3 + 2] = final_trans_z;
    // wxyz writeback.
    poses_rotation[idx * 4 + 0] = final_qw;
    poses_rotation[idx * 4 + 1] = final_qx;
    poses_rotation[idx * 4 + 2] = final_qy;
    poses_rotation[idx * 4 + 3] = final_qz;
    if (scratch != nullptr) {
        scratch[idx] = final_alpha;
    }
}

} // namespace

// ===========================================================================
// Launchers
// ===========================================================================

template <typename T>
void sensor_rays_to_sensor_angles_launch(
    int64_t count,
    const T* sensor_rays,
    T* sensor_angles,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    sensor_rays_to_sensor_angles_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count, sensor_rays, sensor_angles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void sensor_angles_to_sensor_rays_launch(
    int64_t count,
    const T* sensor_angles,
    T* sensor_rays,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    sensor_angles_to_sensor_rays_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count, sensor_angles, sensor_rays);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void elements_to_sensor_angles_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t* elements,
    const T* row_elevations_rad,
    const T* column_azimuths_rad,
    const T* row_azimuth_offsets_rad,
    T* sensor_angles,
    bool* valid_flags,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    elements_to_sensor_angles_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        n_rows,
        n_columns,
        has_row_offsets,
        elements,
        row_elevations_rad,
        column_azimuths_rad,
        row_azimuth_offsets_rad,
        sensor_angles,
        valid_flags);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void sensor_rays_to_sensor_angles_launch<float>(
    int64_t, const float*, float*, cudaStream_t);
template void sensor_rays_to_sensor_angles_launch<double>(
    int64_t, const double*, double*, cudaStream_t);
template void sensor_angles_to_sensor_rays_launch<float>(
    int64_t, const float*, float*, cudaStream_t);
template void sensor_angles_to_sensor_rays_launch<double>(
    int64_t, const double*, double*, cudaStream_t);
template void elements_to_sensor_angles_launch<float>(
    int64_t, int64_t, int64_t, int, const int32_t*, const float*, const float*,
    const float*, float*, bool*, cudaStream_t);
template void elements_to_sensor_angles_launch<double>(
    int64_t, int64_t, int64_t, int, const int32_t*, const double*, const double*,
    const double*, double*, bool*, cudaStream_t);

template <typename T>
void generate_spinning_lidar_rays_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    const int32_t* elements,
    const T* row_elevations_rad,
    const T* column_azimuths_rad,
    const T* row_azimuth_offsets_rad,
    const T* control_translations,
    const T* control_rotations,
    T* world_rays,
    int64_t* timestamps_us,
    T* poses_translation,
    T* poses_rotation,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    generate_spinning_lidar_rays_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        n_rows,
        n_columns,
        has_row_offsets,
        does_generate_elements,
        start_timestamp_us,
        end_timestamp_us,
        elements,
        row_elevations_rad,
        column_azimuths_rad,
        row_azimuth_offsets_rad,
        control_translations,
        control_rotations,
        world_rays,
        timestamps_us,
        poses_translation,
        poses_rotation);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void generate_spinning_lidar_rays_launch<float>(
    int64_t, int64_t, int64_t, int, int, int64_t, int64_t, const int32_t*,
    const float*, const float*, const float*, const float*, const float*,
    float*, int64_t*, float*, float*, cudaStream_t);
template void generate_spinning_lidar_rays_launch<double>(
    int64_t, int64_t, int64_t, int, int, int64_t, int64_t, const int32_t*,
    const double*, const double*, const double*, const double*, const double*,
    double*, int64_t*, double*, double*, cudaStream_t);

template <typename T>
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
    const T* world_points,
    const T* row_elevations_rad,
    const T* column_azimuths_rad,
    const T* row_azimuth_offsets_rad,
    const T* control_translations,
    const T* control_rotations,
    T* sensor_angles,
    bool* valid_flags,
    int64_t* timestamps_us,
    T* poses_translation,
    T* poses_rotation,
    T* scratch,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    inverse_project_spinning_lidar_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        n_rows,
        n_columns,
        has_row_offsets,
        spinning_dir,
        fov_vert_start_rad,
        fov_vert_span_rad,
        fov_horiz_start_rad,
        fov_horiz_span_rad,
        max_iterations,
        stop_mean_relative_time_error,
        stop_delta_mean_relative_time_error,
        initial_relative_time,
        start_timestamp_us,
        end_timestamp_us,
        world_points,
        row_elevations_rad,
        column_azimuths_rad,
        row_azimuth_offsets_rad,
        control_translations,
        control_rotations,
        sensor_angles,
        valid_flags,
        timestamps_us,
        poses_translation,
        poses_rotation,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void inverse_project_spinning_lidar_launch<float>(
    int64_t, int64_t, int64_t, int, int, float, float, float, float, int64_t,
    float, float, float, int64_t, int64_t, const float*, const float*, const float*,
    const float*, const float*, const float*, float*, bool*, int64_t*, float*, float*,
    float*, cudaStream_t);
template void inverse_project_spinning_lidar_launch<double>(
    int64_t, int64_t, int64_t, int, int, double, double, double, double, int64_t,
    double, double, double, int64_t, int64_t, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, bool*, int64_t*, double*, double*,
    double*, cudaStream_t);
