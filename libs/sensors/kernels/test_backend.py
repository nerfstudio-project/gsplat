# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from gsplat_sensors.kernels._backend import _C


def test_extension_loads():
    """Verify that the native extension object is not None after import."""
    assert _C is not None


def test_required_torch_ops_registered():
    """Check that every expected torch.ops.gsplat_sensors entry point is present."""
    required = [
        "generate_image_points",
        "camera_rays_to_image_points_opencv_pinhole_no_external",
        "camera_rays_to_image_points_opencv_pinhole_bivariate_windshield",
        "image_points_to_camera_rays_opencv_pinhole_no_external",
        "image_points_to_camera_rays_opencv_pinhole_bivariate_windshield",
        "project_world_points_mean_pose_opencv_pinhole_no_external",
        "project_world_points_mean_pose_opencv_pinhole_bivariate_windshield",
        "project_world_points_shutter_pose_opencv_pinhole_no_external",
        "project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield",
        "image_points_to_world_rays_static_pose_opencv_pinhole_no_external",
        "image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield",
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external",
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield",
        "camera_rays_to_image_points_ftheta_no_external",
        "camera_rays_to_image_points_ftheta_bivariate_windshield",
        "image_points_to_camera_rays_ftheta_no_external",
        "image_points_to_camera_rays_ftheta_bivariate_windshield",
        "project_world_points_mean_pose_ftheta_no_external",
        "project_world_points_mean_pose_ftheta_bivariate_windshield",
        "project_world_points_shutter_pose_ftheta_no_external",
        "project_world_points_shutter_pose_ftheta_bivariate_windshield",
        "image_points_to_world_rays_static_pose_ftheta_no_external",
        "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield",
        "image_points_to_world_rays_shutter_pose_ftheta_no_external",
        "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield",
        "sensor_rays_to_sensor_angles",
        "sensor_rays_to_sensor_angles_backward",
        "sensor_angles_to_sensor_rays",
        "sensor_angles_to_sensor_rays_backward",
        "elements_to_sensor_angles",
        "elements_to_sensor_angles_backward",
        "generate_spinning_lidar_rays",
        "generate_spinning_lidar_rays_backward",
        "inverse_project_spinning_lidar",
        "inverse_project_spinning_lidar_backward",
    ]
    for name in required:
        assert hasattr(torch.ops.gsplat_sensors, name)


def test_bivariate_windshield_class_registered():
    """Confirm that BivariateWindshieldDistortion is registered as a TorchScript custom class."""
    assert hasattr(torch.classes.gsplat_sensors, "BivariateWindshieldDistortion")


def test_ftheta_projection_class_registered():
    assert hasattr(torch.classes.gsplat_sensors, "FThetaProjection")


def test_row_offset_spinning_lidar_projection_class_registered():
    """Confirm RowOffsetStructuredSpinningLidarProjection is a registered TorchScript class."""
    assert hasattr(
        torch.classes.gsplat_sensors, "RowOffsetStructuredSpinningLidarProjection"
    )


def test_spinning_direction_enum_registered():
    """Confirm SpinningDirection is exposed by the native extension with matching values."""
    assert hasattr(_C, "SpinningDirection")
    assert int(_C.SpinningDirection.CLOCKWISE) == 0
    assert int(_C.SpinningDirection.COUNTERCLOCKWISE) == 1
