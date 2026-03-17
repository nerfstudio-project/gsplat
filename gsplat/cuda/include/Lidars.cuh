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
#pragma once

#include "Cameras.cuh" // For BaseCameraModel
#include "Lidars.h"

namespace gsplat
{
    struct RowOffsetStructuredSpinningLidarModelParametersExt;
}

// Lidar camera parameters struct (device-side)
// TODO: Create a poper LidarModelParamters hierarchy.
struct RowOffsetStructuredSpinningLidarModelParametersExtDevice
{
    RowOffsetStructuredSpinningLidarModelParametersExtDevice(const gsplat::RowOffsetStructuredSpinningLidarModelParametersExt &params);

    // Sensor dimensions
    int n_rows;
    int n_columns;

    gsplat::FOVDevice fov_vert_rad;
    gsplat::FOVDevice fov_horiz_rad;
    float fov_eps_rad;

    // Not adding row_elevations_rad/column_azimuth_rad/... because
    // angles_to_columns_map has everything we need from them.

    // Spinning direction and frequency of the lidar
    gsplat::SpinningDirection spinning_direction;
    float spinning_frequency_hz;

    // Precomputed acceleration structure for rolling shutter timing
    // Maps (.x=azimuth, .y=elevation) angles to actual column indices.
    const int32_t* angles_to_columns_map = nullptr;
    int2 map_dim;              // Dimensions of the map grid
    float2 map_resolution_rad; // Grid cell size in radians [.x=azimuth, .y=elevation]

    // Angle to pixel scaling factor
    static constexpr int ANGLE_TO_PIXEL_SCALING_FACTOR = 1024;

    // Tiling info
    int n_bins_azimuth;
    int n_bins_elevation;
    int cdf_resolution_elevation;
    int cdf_resolution_azimuth;
    const int32_t *cdf_elevation;
    const int32_t *cdf_dense_ray_mask;
    const glm::ivec2 *tiles_pack_info;
    const glm::ivec2 *tiles_to_elements_map;
};


// Lidar camera model for spinning lidar sensors (e.g., Hesai Pandar128, AT128)
// TODO: Create a proper LidarModel hierarchy.
struct RowOffsetStructuredSpinningLidarModel : BaseCameraModel<RowOffsetStructuredSpinningLidarModel>
{
    using Base = BaseCameraModel<RowOffsetStructuredSpinningLidarModel>;

    struct Parameters : Base::Parameters
    {
        __device__
        explicit Parameters(RowOffsetStructuredSpinningLidarModelParametersExtDevice lidar_params)
            // TODO: We're mapping n_rows->width and n_columns->height because
            // the current pipeline uses image_point = [x, y] = [row, column]
            // = [elevation, azimuth]. This should be reverted once the pipeline
            // switches to the standard [x, y] = [column, row] = [azimuth,
            // elevation].
            : Base::Parameters{
                {static_cast<unsigned int>(lidar_params.n_rows), static_cast<unsigned int>(lidar_params.n_columns)},
                ShutterType::ROLLING_LEFT_TO_RIGHT
              }
            , lidar(std::move(lidar_params))
        {
        }

        RowOffsetStructuredSpinningLidarModelParametersExtDevice lidar;
    };

    __device__
    explicit RowOffsetStructuredSpinningLidarModel(RowOffsetStructuredSpinningLidarModelParametersExtDevice params)
        : parameters(std::move(params))
    {
    }

    Parameters parameters;

public:
    // Override shutter_relative_frame_time for lidar-specific rolling shutter timing
    __device__
    float shutter_relative_frame_time(const glm::fvec2 &image_point) const
    {
        // Unlike cameras with simple linear rolling shutter (top-to-bottom, left-to-right),
        // spinning lidars have complex timing patterns:
        // - The sensor spins (clockwise or counter-clockwise)
        // - Each column corresponds to a specific capture time
        // - The mapping from (elevation, azimuth) angles to column (time) is non-linear
        //   due to mechanical characteristics and row-specific azimuth offsets
        //
        // This function uses the angleToColumnMap lookup table
        // to determine the correct capture time for a given image point.

        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        assert(lidar.angles_to_columns_map != nullptr);

        // Convert image point (in scaled pixel space) back to angles
        // TODO: transposing here because the current pipeline uses image_point =
        // [x, y] = [elevation, azimuth], this needs to be reverted
        // once the pipeline switches to the standard [x, y] = [azimuth, elevation].
        constexpr float kToAngle = 1.f / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float elevation = image_point.x * kToAngle;
        const float azimuth = image_point.y * kToAngle;

        // Compute relative angles (within sensor FOV)
        const auto [rel_elevation, rel_azimuth] = this->relative_sensor_angles(elevation, azimuth);

        // Compute the index into the map and clamp it to valid range
        const int ihoriz = static_cast<int>(std::clamp<float>(
                rel_azimuth/lidar.map_resolution_rad.x + 0.5f,
                0,
                lidar.map_dim.x-1));

        const int ivert = static_cast<int>(std::clamp<float>(
                rel_elevation/lidar.map_resolution_rad.y + 0.5f,
                0,
                lidar.map_dim.y-1));

        assert(0 <= ihoriz && ihoriz < lidar.map_dim.x);
        assert(0 <= ivert && ivert < lidar.map_dim.y);

        // Lookup column index from the angle-to-column map
        // Note: map is on row-major layout
        const int column_idx = lidar.angles_to_columns_map[ivert*lidar.map_dim.x + ihoriz];

        // Normalize to [0, 1] range (relative frame time)
        return static_cast<float>(column_idx)/(lidar.n_columns - 1);
    }

    __device__
    auto camera_ray_to_image_point(glm::fvec3 const &cam_ray, float margin_factor) const
        -> typename Base::ImagePointReturn
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // Normalize the camera ray (lidar uses normalized rays)
        const float ray_length = glm::length(cam_ray);
        if (ray_length < 1e-6f)
        {
            return {{0.f, 0.f}, false};
        }

        glm::fvec3 const ray_normalized = cam_ray / ray_length;

        // Convert 3D ray to spherical coordinates (elevation, azimuth)
        const float elevation = std::asin(ray_normalized.z);
        const float azimuth = std::atan2(ray_normalized.y, ray_normalized.x);

        // Image point: [row, column] (in scaled angle space)
        // TODO: the current pipeline uses image_point = [x, y] = [row, column]
        // = [elevation, azimuth]. Must undo this swap once the pipeline switches
        // to the standard [x, y] = [column, row] = [azimuth, elevation].
        const float row = elevation * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float column = azimuth * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const glm::fvec2 image_point = glm::fvec2{row, column};

        // For validation, compute relative angles
        auto [rel_elevation, rel_azimuth] = this->relative_sensor_angles(elevation, azimuth);

        // Check FOV bounds with margin
        // The margin allows rays slightly outside FOV that might round to valid pixels
        const float tol_azimuth = margin_factor * lidar.fov_horiz_rad.span;
        const float tol_elevation = margin_factor * lidar.fov_vert_rad.span;

        bool valid = rel_elevation >= -tol_elevation &&
                     rel_azimuth >= -tol_azimuth &&
                     rel_elevation <= lidar.fov_vert_rad.span + tol_elevation &&
                     rel_azimuth <= lidar.fov_horiz_rad.span + tol_azimuth;

        return {image_point, valid};
    }

    __device__
    CameraRay image_point_to_camera_ray(glm::fvec2 image_point) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // TODO: transposing because the current pipeline uses image_point =
        // [x, y] = [row, column] = [elevation, azimuth], but this must be reverted
        // once the pipeline switches to the standard [x, y] = [column, row] =
        // [azimuth, elevation].
        const float row = image_point.x;
        const float col = image_point.y;

        // Inverse of forward projection: elevation = row / scale
        const float elevation = row / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float azimuth = col / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;

        // Convert spherical coordinates (elevation, azimuth) to 3D ray
        const float sin_elevation = std::sin(elevation);
        const float cos_elevation = std::cos(elevation);
        const float sin_azimuth = std::sin(azimuth);
        const float cos_azimuth = std::cos(azimuth);

        const glm::fvec3 camera_ray = glm::fvec3
        {
            cos_azimuth * cos_elevation,  // x component
            sin_azimuth * cos_elevation,  // y component
            sin_elevation                 // z component
        };

        return {
            camera_ray / glm::length(camera_ray),
            this->valid_sensor_angles(elevation, azimuth)
        };
    }

    __device__
    float relative_clock_rotation(float begin, float end, gsplat::SpinningDirection direction) const
    {
        return (direction == gsplat::SpinningDirection::CLOCKWISE) ? (begin - end) : (end - begin);
    }

    __device__
    float relative_angle(float angle_start, float angle_end, gsplat::SpinningDirection direction, float scale=1) const
    {
        const float period = scale*2*PI;

        float rel_angle = this->relative_clock_rotation(angle_start, angle_end, direction);
        // Avoid expensive fmod if the angle is already in a reasonable range
        // Similar optimization as in normalize_angle

        if (-period < rel_angle && rel_angle < 2*period)
        {
            // Fast path: handle common cases without fmod
            rel_angle = (rel_angle >= period) ? rel_angle - period : rel_angle;
            rel_angle = (rel_angle < 0.f) ? rel_angle + period : rel_angle;
            return rel_angle;
        }
        else
        {
            // Slow path: use fmod for values outside the fast range
            rel_angle = std::fmod(rel_angle, period);
            // output range [0, period)
            return (rel_angle < 0.f) ? (rel_angle + period) : rel_angle;
        }
    }

    __device__
    std::tuple<float,float> relative_sensor_angles(float elevation, float azimuth, float scale=1) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = this->parameters.lidar;

        const float rel_elevation = this->relative_clock_rotation(lidar.fov_vert_rad.start*scale, elevation, gsplat::SpinningDirection::CLOCKWISE);
        const float rel_azimuth = this->relative_angle(lidar.fov_horiz_rad.start*scale, azimuth, lidar.spinning_direction, scale);

        return {rel_elevation, rel_azimuth};
    }

    // Checks if sensor angles are within the FOV of the sensor.
    __device__
    bool valid_sensor_angles(float elevation, float azimuth, float scale=1) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
        const float fov_vert_start_adj = lidar.fov_vert_rad.start + lidar.fov_eps_rad;
        const float fov_horiz_start_adj = lidar.spinning_direction == gsplat::SpinningDirection::CLOCKWISE
                                            ? lidar.fov_horiz_rad.start + lidar.fov_eps_rad
                                            : lidar.fov_horiz_rad.start - lidar.fov_eps_rad;

        const float rel_elevation = this->relative_clock_rotation(fov_vert_start_adj*scale, elevation, gsplat::SpinningDirection::CLOCKWISE);
        const float rel_azimuth = this->relative_angle(fov_horiz_start_adj*scale, azimuth, lidar.spinning_direction, scale);

        // Also account for accumulated numerical errors via some epsilon in FOV check
        // (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
        //  so effectively this checks 1x eps on the end of the FOV)
        return (rel_elevation <= scale*(lidar.fov_vert_rad.span + lidar.fov_eps_rad*2)) &&
               (rel_azimuth <= scale*(lidar.fov_horiz_rad.span + lidar.fov_eps_rad*2));
    }
};
