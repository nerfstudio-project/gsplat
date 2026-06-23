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

import io

import pytest
import torch
from torch import nn

from gsplat.sensors.functional.return_types import ImagePointsReturn
from gsplat.sensors.kernels.cameras import ShutterType
from gsplat.sensors.models import CameraModel, ImageFrame

FISHEYE_BASIC_ATOL = 1e-4
FISHEYE_IMAGE_ROUNDTRIP_ATOL = 1e-2
FISHEYE_TRANSFORM_STRICT_ATOL = 1e-5


@pytest.fixture
def fisheye_model(fisheye_projection, no_external):
    """Global-shutter CameraModel backed by the mild equidistant fisheye projection."""
    return CameraModel(
        projection=fisheye_projection,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


@pytest.fixture
def fisheye_model_with_windshield(fisheye_projection, windshield_distortion):
    """Fisheye camera model layered with the identity-equivalent windshield distortion."""
    return CameraModel(
        projection=fisheye_projection,
        external_distortion=windshield_distortion,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


def test_fisheye_camera_model_initialization(
    fisheye_model, fisheye_projection, no_external
):
    """Assert CameraModel stores projection, external distortion, resolution, and shutter type correctly."""
    assert fisheye_model.projection is fisheye_projection
    assert fisheye_model.external_distortion is no_external
    assert fisheye_model.resolution == (100, 80)
    assert fisheye_model.shutter_type == ShutterType.GLOBAL


def test_fisheye_camera_rays_to_image_points_basic(fisheye_model):
    """Assert an optical-axis ray projects to the principal point with a valid flag."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=fisheye_model.projection.principal_point.device,
    )
    result = fisheye_model.camera_rays_to_image_points(rays)
    assert isinstance(result, ImagePointsReturn)
    assert torch.allclose(
        result.image_points,
        fisheye_model.projection.principal_point.reshape(1, 2),
        atol=FISHEYE_BASIC_ATOL,
    )
    assert result.valid_flag.item()


def test_fisheye_image_points_to_camera_rays_roundtrip(fisheye_model):
    """Assert image_points -> camera_rays -> image_points round-trips within 1e-2 tolerance."""
    pp = fisheye_model.projection.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor(
        [[0.0, 0.0], [2.0, 1.0], [-3.0, 4.0]],
        device=pp.device,
    )
    rays = fisheye_model.image_points_to_camera_rays(image_points)
    roundtrip = fisheye_model.camera_rays_to_image_points(rays).image_points
    assert torch.allclose(roundtrip, image_points, atol=FISHEYE_IMAGE_ROUNDTRIP_ATOL)


def test_fisheye_world_points_to_image_points_static_pose(fisheye_model, static_pose):
    """Assert static-pose projection filters invalid points and returns optional T_sensor_world and timestamps."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=fisheye_model.projection.principal_point.device,
    )
    result = fisheye_model.world_points_to_image_points_static_pose(
        world_points,
        static_pose,
        timestamp_us=11,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    assert result.image_points.shape == (1, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]
    assert result.timestamps_us.tolist() == [11]
    assert result.T_sensor_world.shape == (1, 4, 4)


def test_fisheye_world_points_to_image_points_shutter_pose(
    fisheye_projection, no_external, dynamic_pose
):
    """Assert rolling-shutter projection produces per-point image_points, valid_flag, and timestamps."""
    model = CameraModel(
        projection=fisheye_projection,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
    )
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=fisheye_projection.principal_point.device,
    )
    result = model.world_points_to_image_points_shutter_pose(
        world_points,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_valid_flag=True,
        return_timestamps=True,
    )
    assert result.image_points.shape == (1, 2)
    assert result.valid_flag.shape == (1,)
    assert result.timestamps_us.shape == (1,)


@pytest.mark.parametrize(
    "shutter_type",
    [
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        ShutterType.ROLLING_LEFT_TO_RIGHT,
        ShutterType.ROLLING_BOTTOM_TO_TOP,
        ShutterType.ROLLING_RIGHT_TO_LEFT,
        ShutterType.GLOBAL,
    ],
)
def test_fisheye_image_points_relative_frame_times(
    fisheye_projection, no_external, shutter_type
):
    """Assert relative frame times are in [0, 1] for all supported shutter types."""
    model = CameraModel(
        projection=fisheye_projection,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=shutter_type,
    )
    image_points = torch.tensor(
        [[0.5, 0.5], [99.5, 79.5]],
        device=fisheye_projection.principal_point.device,
    )
    times = model.image_points_relative_frame_times(image_points)
    assert times.shape == (2,)
    assert torch.all((times >= 0) & (times <= 1))


def test_fisheye_pixels_image_points_conversions(fisheye_model):
    """Assert pixels_to_image_points adds 0.5 and image_points_to_pixels floors to int32."""
    pixels = torch.tensor(
        [[0, 1]],
        device=fisheye_model.projection.principal_point.device,
    )
    image_points = fisheye_model.pixels_to_image_points(pixels)
    assert torch.equal(
        image_points, torch.tensor([[0.5, 1.5]], device=image_points.device)
    )
    assert torch.equal(
        fisheye_model.image_points_to_pixels(image_points), pixels.to(torch.int32)
    )


def test_fisheye_transform_scale_offset(fisheye_model):
    """Assert transform(scale=0.5) scales focal_length by scale, applies the
    ``(pp + 0.5) * s - 0.5`` pixel-center convention to principal_point, leaves
    the equidistant forward_poly unchanged, and updates resolution to (50, 40)."""
    scaled = fisheye_model.transform(0.5)
    assert torch.allclose(
        scaled.projection.focal_length,
        fisheye_model.projection.focal_length * 0.5,
        atol=FISHEYE_TRANSFORM_STRICT_ATOL,
    )
    expected_pp = (fisheye_model.projection.principal_point + 0.5) * 0.5 - 0.5
    assert torch.allclose(
        scaled.projection.principal_point,
        expected_pp,
        atol=FISHEYE_TRANSFORM_STRICT_ATOL,
    )
    assert torch.allclose(
        scaled.projection.forward_poly,
        fisheye_model.projection.forward_poly,
        atol=FISHEYE_TRANSFORM_STRICT_ATOL,
    )
    assert scaled.resolution == (50, 40)


def test_fisheye_transform_preserves_scalar_fields(fisheye_model):
    """Assert max_angle, newton_iterations, and min_2d_norm survive transform unchanged."""
    scaled = fisheye_model.transform(0.5)
    assert float(scaled.projection.max_angle) == pytest.approx(
        float(fisheye_model.projection.max_angle)
    )
    assert int(scaled.projection.newton_iterations) == int(
        fisheye_model.projection.newton_iterations
    )
    assert float(scaled.projection.min_2d_norm) == pytest.approx(
        float(fisheye_model.projection.min_2d_norm)
    )


def test_fisheye_transform_returns_new_object_no_in_place_mutation(fisheye_model):
    """Assert transform returns a new CameraModel without mutating the original focal_length."""
    original_focal = fisheye_model.projection.focal_length.clone()
    transformed = fisheye_model.transform(0.5)
    assert transformed is not fisheye_model
    assert torch.equal(fisheye_model.projection.focal_length, original_focal)


def test_fisheye_camera_is_nn_module_and_to_device(fisheye_model):
    """Assert CameraModel is an nn.Module and .to(device) moves all projection tensors."""
    assert isinstance(fisheye_model, nn.Module)
    moved = fisheye_model.to(fisheye_model.projection.principal_point.device)
    assert isinstance(moved, CameraModel)
    device = fisheye_model.projection.principal_point.device
    # Every component tensor must be threaded by _move_projection.
    assert moved.projection.principal_point.device == device
    assert moved.projection.focal_length.device == device
    assert moved.projection.forward_poly.device == device
    assert moved.projection.approx_backward_factor.device == device


def test_fisheye_camera_to_device_preserves_scalar_fields(fisheye_model):
    """Assert the scalar config fields round-trip through ``.to(device)`` without mutation."""
    moved = fisheye_model.to(fisheye_model.projection.principal_point.device)
    src = fisheye_model.projection
    dst = moved.projection
    assert int(dst.newton_iterations) == int(src.newton_iterations)
    assert float(dst.max_angle) == pytest.approx(float(src.max_angle))
    assert float(dst.min_2d_norm) == pytest.approx(float(src.min_2d_norm))


def test_fisheye_camera_model_state_dict_and_pickle(fisheye_model):
    """Assert state_dict is a dict and the model round-trips through torch.save/load."""
    assert isinstance(fisheye_model.state_dict(), dict)
    buf = io.BytesIO()
    torch.save(fisheye_model, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert loaded.projection.resolution == fisheye_model.projection.resolution
    assert torch.allclose(
        loaded.projection.principal_point, fisheye_model.projection.principal_point
    )
    assert torch.allclose(
        loaded.projection.focal_length, fisheye_model.projection.focal_length
    )
    assert torch.allclose(
        loaded.projection.forward_poly, fisheye_model.projection.forward_poly
    )
    assert float(loaded.projection.max_angle) == pytest.approx(
        float(fisheye_model.projection.max_angle)
    )


def test_fisheye_with_windshield_model_static_pose(
    fisheye_model_with_windshield, static_pose
):
    """Assert the windshield-augmented fisheye model projects an on-axis point as
    valid and a behind-camera point as invalid, yielding ``valid_flag == [True, False]``."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=fisheye_model_with_windshield.projection.principal_point.device,
    )
    result = fisheye_model_with_windshield.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
    )
    assert result.valid_flag.tolist() == [True, False]


def test_fisheye_image_frame_image_buffer_registered(fisheye_model, static_pose):
    """Assert ImageFrame registers the image tensor as a buffer."""
    frame = ImageFrame("cam", fisheye_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    assert "image" in dict(frame.named_buffers())


def test_fisheye_image_frame_to_device_moves_image_and_pose(fisheye_model, static_pose):
    """Assert ImageFrame.to(device) moves the image, pose, and camera projection together."""
    frame = ImageFrame("cam", fisheye_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    frame = frame.to(frame.image.device)
    assert frame.pose.translation.device == frame.image.device
    assert frame.camera_model.projection.principal_point.device == frame.image.device


def test_fisheye_world_points_to_pixels_static_pose_filters_invalid(
    fisheye_model, static_pose
):
    """Assert world_points_to_pixels_static_pose drops invalid points and returns valid_indices."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=fisheye_model.projection.principal_point.device,
    )
    result = fisheye_model.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
        return_valid_indices=True,
    )
    assert result.pixels.shape == (1, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]


def test_real_fisheye_camera_rays_to_image_points(real_fisheye_projection, no_external):
    """Assert a real-camera fisheye projection maps an on-axis ray to its principal point."""
    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    pp = real_fisheye_projection.principal_point
    rays = torch.tensor([[0.0, 0.0, 1.0]], device=pp.device, dtype=torch.float32)
    image_points, valid = camera_rays_to_image_points(
        rays, real_fisheye_projection, no_external
    )
    assert valid.tolist() == [True]
    assert torch.allclose(image_points[0], pp, atol=FISHEYE_BASIC_ATOL)


def test_real_fisheye_matches_cv2_oracle(
    real_fisheye_projection, no_external, reference_opencv_fisheye_camera
):
    """Assert the real fisheye projection matches cv2.fisheye.projectPoints within 0.01 px
    on in-FOV in-bounds rays via the stateless ``ReferenceOpenCVFisheyeCamera`` oracle."""
    pytest.importorskip("cv2")
    import numpy as np

    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    proj = real_fisheye_projection
    oracle = reference_opencv_fisheye_camera(
        focal_length=proj.focal_length.detach().cpu().numpy(),
        principal_point=proj.principal_point.detach().cpu().numpy(),
        radial_coeffs=proj.forward_poly.detach().cpu().numpy(),
        max_angle=float(proj.max_angle),
        resolution=tuple(proj.resolution),
    )
    rays = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.05, -0.02, 1.0],
            [-0.08, 0.04, 1.0],
            [0.1, 0.03, 1.0],
        ],
        device=proj.principal_point.device,
        dtype=torch.float32,
    )
    image_points, valid = camera_rays_to_image_points(rays, proj, no_external)
    rays_np = rays.detach().cpu().numpy()
    compared = 0
    for i in range(rays.shape[0]):
        cv2_point, cv2_valid = oracle.camera_ray_to_image_point_opencv(rays_np[i])
        if not (bool(valid[i]) and cv2_valid):
            continue
        compared += 1
        deviation = float(
            np.max(np.abs(image_points[i].detach().cpu().numpy() - cv2_point))
        )
        assert deviation <= 0.01, (i, deviation)
    assert compared > 0


def test_real_fisheye_image_points_to_camera_rays_round_trip(
    real_fisheye_projection, no_external
):
    """Assert image -> ray -> image round-trips on a real fisheye camera within 1e-2 px."""
    from gsplat.sensors.kernels.cameras import (
        camera_rays_to_image_points,
        image_points_to_camera_rays,
    )

    pp = real_fisheye_projection.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor(
        [[0.0, 0.0], [40.0, 25.0], [-30.0, 18.0]],
        device=pp.device,
        dtype=torch.float32,
    )
    rays = image_points_to_camera_rays(
        image_points, real_fisheye_projection, no_external
    )
    image_points_back, _ = camera_rays_to_image_points(
        rays, real_fisheye_projection, no_external
    )
    assert torch.allclose(
        image_points_back, image_points, atol=FISHEYE_IMAGE_ROUNDTRIP_ATOL
    )


def test_real_fisheye_matches_cv2_oracle_swept_grid(
    real_fisheye_projection, no_external, reference_opencv_fisheye_camera
):
    """Assert forward projection matches cv2.fisheye across a swept angular grid (not just
    near-axis), within 0.01 px on every ray where both gsplat and cv2 report valid."""
    pytest.importorskip("cv2")
    import math

    import numpy as np

    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    proj = real_fisheye_projection
    oracle = reference_opencv_fisheye_camera(
        focal_length=proj.focal_length.detach().cpu().numpy(),
        principal_point=proj.principal_point.detach().cpu().numpy(),
        radial_coeffs=proj.forward_poly.detach().cpu().numpy(),
        max_angle=float(proj.max_angle),
        resolution=tuple(proj.resolution),
    )
    thetas = torch.linspace(0.0, 0.9 * float(proj.max_angle), 8)
    phis = torch.linspace(0.0, 2.0 * math.pi, 12)[:-1]
    rays = [
        [
            math.sin(float(theta)) * math.cos(float(phi)),
            math.sin(float(theta)) * math.sin(float(phi)),
            math.cos(float(theta)),
        ]
        for theta in thetas
        for phi in phis
    ]
    rays_t = torch.tensor(rays, device=proj.principal_point.device, dtype=torch.float32)
    image_points, valid = camera_rays_to_image_points(rays_t, proj, no_external)
    rays_np = rays_t.detach().cpu().numpy()
    compared = 0
    for i in range(rays_t.shape[0]):
        cv2_point, cv2_valid = oracle.camera_ray_to_image_point_opencv(rays_np[i])
        if not (bool(valid[i]) and cv2_valid):
            continue
        compared += 1
        deviation = float(
            np.max(np.abs(image_points[i].detach().cpu().numpy() - cv2_point))
        )
        assert deviation <= 0.01, (i, deviation)
    assert compared > 0


def test_real_fisheye_windshield_through_camera_model(
    real_fisheye_projection_with_windshield, real_fisheye_windshield_distortion
):
    """Assert real-windshield distortion routed through ``CameraModel`` matches the kernel-direct
    forward output, guarding the model-layer dispatch against silently dropping the external
    distortion on the fisheye path."""
    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    proj = real_fisheye_projection_with_windshield
    distortion = real_fisheye_windshield_distortion
    model = CameraModel(
        projection=proj,
        external_distortion=distortion,
        resolution=proj.resolution,
        shutter_type=ShutterType.GLOBAL,
    )
    pp = proj.principal_point
    rays = torch.tensor(
        [
            [0.05, 0.0, 1.0],
            [-0.05, 0.04, 1.0],
            [0.04, -0.03, 1.0],
        ],
        device=pp.device,
        dtype=torch.float32,
    )
    direct_pts, direct_valid = camera_rays_to_image_points(rays, proj, distortion)
    via_model = model.camera_rays_to_image_points(rays)
    assert direct_valid.all()
    assert via_model.valid_flag.all()
    assert torch.allclose(
        via_model.image_points, direct_pts, atol=FISHEYE_TRANSFORM_STRICT_ATOL
    )
