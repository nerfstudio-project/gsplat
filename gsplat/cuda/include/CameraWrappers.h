/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

/**
 * @file CameraWrappers.h
 * @brief C++ camera model wrappers exported in main gsplat API
 *
 * These are the PRIMARY camera model implementations. PyTorch versions
 * in _torch_cameras.py serve as reference implementations for testing.
 */

#pragma once
#include <torch/extension.h>
#include <memory>
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Ops.h"

namespace gsplat {


/**
 * @brief Camera pose for rolling shutter
 *
 * Represents camera pose with translation and rotation (quaternion).
 */

template <class CameraModel=void>
class PyBaseCameraModel;

/**
 * @brief Base class for camera model wrappers (Python-compatible)
 */
template <>
class PyBaseCameraModel<> : public torch::CustomClassHolder
{
public:
    virtual ~PyBaseCameraModel() = default;

    // ========== Camera Projection Methods ==========

    /**
     * @brief Project camera rays to image points
     * @param camera_ray Camera ray with direction [..., 3]
     * @param margin_factor Boundary margin factor (default 0.0)
     * @return (image_points [..., 2], valid [...])
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> camera_ray_to_image_point(
        const torch::Tensor& camera_ray,
        float margin_factor) const = 0;

    /**
     * @brief Unproject image points to camera rays
     * @param image_points Image coordinates [..., 2]
     * @return (Camera ray with direction [..., 3], valid [...])
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> image_point_to_camera_ray(
        const torch::Tensor& image_points) const = 0;

    /**
     * @brief Compute relative frame time for rolling shutter (non-virtual, implemented in base)
     * @param image_points Image coordinates [..., 2]
     * @return Relative times [...] in range [0, 1]
     */
    virtual torch::Tensor shutter_relative_frame_time(
        const torch::Tensor& image_points) const = 0;

    /**
     * @brief Unproject image point to world ray with rolling shutter (non-virtual, implemented in base)
     * @param image_points Image coordinates [..., 2]
     * @param pose_start Camera pose at frame start
     * @param pose_end Camera pose at frame end
     * @return (World ray origin [..., 3], World ray direction [..., 3], valid [...])
     */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> image_point_to_world_ray_shutter_pose(
        const torch::Tensor& image_points,
        const torch::Tensor& pose_start,
        const torch::Tensor& pose_end) const = 0;

    /**
     * @brief Project world points to image with rolling shutter (non-virtual, implemented in base)
     * @param world_points World coordinates [..., M, 3]
     * @param pose_start Camera pose at frame start
     * @param pose_end Camera pose at frame end
     * @param margin_factor Boundary margin factor (default 0.0)
     * @return (image_points [..., M, 2], valid [..., M])
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> world_point_to_image_point_shutter_pose(
        const torch::Tensor& world_points,
        const torch::Tensor& pose_start,
        const torch::Tensor& pose_end,
        float margin_factor) const = 0;

    // ========== Properties ==========

    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual int num_cameras() const = 0;
    virtual ShutterType rs_type() const = 0;

    virtual torch::Tensor principal_points() const = 0;
    virtual torch::Tensor focal_lengths() const = 0; // Might be an approximation!

    // ========== Factory Method ==========

    /**
     * @brief Factory method to create camera from model type string
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param camera_model Camera model type ("pinhole", "fisheye", "ftheta")
     * @param principal_points Principal points [..., 2] (cx, cy) - required for all models
     * @param focal_lengths Focal lengths [..., 2] (fx, fy) - required for pinhole and fisheye
     * @param radial_coeffs Optional radial distortion coefficients
     * @param tangential_coeffs Optional tangential distortion coefficients
     * @param thin_prism_coeffs Optional thin prism distortion coefficients
     * @param ftheta_coeffs Optional ftheta distortion parameters
     * @param rs_type Rolling shutter type (default: GLOBAL)
     * @return Shared pointer to created camera model
     * @note For ftheta model, focal_lengths can be empty/nullopt as focal length is embedded in polynomial
     */
    static c10::intrusive_ptr<PyBaseCameraModel> create(
        int width,
        int height,
        const std::string& camera_model,
        const torch::Tensor& principal_points,
        const std::optional<torch::Tensor>& radial_coeffs,
        const std::optional<torch::Tensor>& tangential_coeffs,
        const std::optional<torch::Tensor>& focal_lengths,
        const std::optional<torch::Tensor>& thin_prism_coeffs,
        const std::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>>& ftheta_coeffs,
        ShutterType rs_type
    );
};

template <class CameraModel>
class PyBaseCameraModel : public PyBaseCameraModel<>
{
public:
    std::tuple<torch::Tensor, torch::Tensor> camera_ray_to_image_point(
        const torch::Tensor& camera_ray,
        float margin_factor) const override;

    std::tuple<torch::Tensor, torch::Tensor> image_point_to_camera_ray(
        const torch::Tensor& image_points) const override;

    torch::Tensor shutter_relative_frame_time(
        const torch::Tensor& image_points) const override;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> image_point_to_world_ray_shutter_pose(
        const torch::Tensor& image_points,
        const torch::Tensor& pose_start,
        const torch::Tensor& pose_end) const override;

    std::tuple<torch::Tensor, torch::Tensor> world_point_to_image_point_shutter_pose(
        const torch::Tensor& world_points,
        const torch::Tensor& pose_start,
        const torch::Tensor& pose_end,
        float margin_factor) const override;

    int num_cameras() const override { return m_num_cameras; }
    int width() const override { return m_width; }
    int height() const override { return m_height; }
    ShutterType rs_type() const override { return m_rs_type; }

    torch::Tensor principal_points() const override { return m_principal_points; }
    torch::Tensor focal_lengths() const override { return m_focal_lengths; }

private:
    /**
     * @brief Deleter for CUDA memory
     */
    struct CudaDeleter
    {
        void operator()(void* ptr) const;
    };

protected:
    int m_num_cameras;
    int m_width;
    int m_height;
    ShutterType m_rs_type;
    PyBaseCameraModel(int num_cameras, int width, int height, ShutterType rs_type,
                      const torch::Tensor &focal_lengths, const torch::Tensor &principal_points);

    CameraModel* dev_cameras() { return m_dev_cameras.get(); }
    const CameraModel* dev_cameras() const { return m_dev_cameras.get(); }
    const CameraModel* const_dev_cameras() const { return m_dev_cameras.get(); }

private:
    std::unique_ptr<CameraModel, CudaDeleter> m_dev_cameras;

    torch::Tensor m_focal_lengths;
    torch::Tensor m_principal_points;
};

/**
 * @brief Perfect pinhole camera model (no distortion)
 */
class PyPerfectPinholeCameraModel : public PyBaseCameraModel<PerfectPinholeCameraModel>
{
public:
    /**
     * @brief Constructor with focal lengths and principal points
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_lengths Focal lengths [..., 2] (fx, fy)
     * @param principal_points Principal points [..., 2] (cx, cy)
     * @param shutter_type Rolling shutter type
     */
    PyPerfectPinholeCameraModel(
        int width,
        int height,
        const torch::Tensor& focal_lengths,
        const torch::Tensor& principal_points,
        ShutterType rs_type
    );
};

/**
 * @brief OpenCV pinhole camera model with distortion
 */
class PyOpenCVPinholeCameraModel : public PyBaseCameraModel<OpenCVPinholeCameraModel<>>
{
public:
    /**
     * @brief Constructor with focal lengths and principal points
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_lengths Focal lengths [..., 2] (fx, fy)
     * @param principal_points Principal points [..., 2] (cx, cy)
     * @param radial_coeffs Optional radial distortion coefficients [..., 4] or [..., 6]
     * @param tangential_coeffs Optional tangential distortion coefficients [..., 2]
     * @param thin_prism_coeffs Optional thin prism distortion coefficients [..., 4]
     * @param shutter_type Rolling shutter type
     */
    PyOpenCVPinholeCameraModel(
        int width,
        int height,
        const torch::Tensor& focal_lengths,
        const torch::Tensor& principal_points,
        const std::optional<torch::Tensor>& radial_coeffs,
        const std::optional<torch::Tensor>& tangential_coeffs,
        const std::optional<torch::Tensor>& thin_prism_coeffs,
        ShutterType rs_type
    );
};

/**
 * @brief OpenCV fisheye camera model with distortion
 */
class PyOpenCVFisheyeCameraModel : public PyBaseCameraModel<OpenCVFisheyeCameraModel<>>
{
public:
    /**
     * @brief Constructor with focal lengths and principal points
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_lengths Focal lengths [..., 2] (fx, fy)
     * @param principal_points Principal points [..., 2] (cx, cy)
     * @param radial_coeffs Optional radial distortion coefficients [..., 4]
     * @param shutter_type Rolling shutter type
     */
    PyOpenCVFisheyeCameraModel(
        int width,
        int height,
        const torch::Tensor& focal_lengths,
        const torch::Tensor& principal_points,
        const std::optional<torch::Tensor>& radial_coeffs,  // [..., 4]
        ShutterType rs_type
    );
};

/**
 * @brief F-Theta camera model with polynomial distortion
 */
class PyFThetaCameraModel : public PyBaseCameraModel<FThetaCameraModel<>>
{
public:
    /**
     * @brief Constructor for F-Theta camera (no K matrix - uses principal points directly)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param principal_points Principal points [..., 2] (cx, cy)
     * @param pixeldist_to_angle_poly Pixel distance to angle polynomial [..., 6]
     * @param angle_to_pixeldist_poly Angle to pixel distance polynomial [..., 6]
     * @param linear_cde Linear correction coefficients [..., 3]
     * @param reference_poly Reference polynomial type (FThetaPolynomialType enum value)
     * @param max_angle Maximum angle for FOV clamping (radians)
     * @param shutter_type Rolling shutter type
     */
    PyFThetaCameraModel(
        int width,
        int height,
        const torch::Tensor& principal_points,         // [..., 2]
        const torch::Tensor& pixeldist_to_angle_poly,  // [..., 6]
        const torch::Tensor& angle_to_pixeldist_poly,  // [..., 6]
        const torch::Tensor& linear_cde,               // [..., 3]
        FThetaCameraDistortionParameters::PolynomialType reference_poly,
        const torch::Tensor &max_angle,                // [...]
        ShutterType rs_type
    );
};

/**
 * @brief Lidar camera model for spinning lidar sensors
 */
class PyRowOffsetStructuredSpinningLidarModel : public PyBaseCameraModel<RowOffsetStructuredSpinningLidarModel>
{
public:
    explicit PyRowOffsetStructuredSpinningLidarModel(RowOffsetStructuredSpinningLidarModelParametersExt params);

private:
    // Store the parameters to keep its tensors.
    RowOffsetStructuredSpinningLidarModelParametersExt m_params;
};

} // namespace gsplat

