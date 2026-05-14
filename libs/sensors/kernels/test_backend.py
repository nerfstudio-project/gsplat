# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    ]
    for name in required:
        assert hasattr(torch.ops.gsplat_sensors, name)


def test_bivariate_windshield_class_registered():
    """Confirm that BivariateWindshieldDistortion is registered as a TorchScript custom class."""
    assert hasattr(torch.classes.gsplat_sensors, "BivariateWindshieldDistortion")
