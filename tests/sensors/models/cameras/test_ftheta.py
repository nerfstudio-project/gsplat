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
import math

import pytest
import torch
from torch import nn

from gsplat._helper import assert_grad_reference_close
from gsplat.sensors.functional.return_types import ImagePointsReturn
from gsplat.sensors.kernels.cameras import ShutterType
from gsplat.sensors.models import CameraModel, ImageFrame

FTHETA_BASIC_ATOL = 1e-4
FTHETA_IMAGE_ROUNDTRIP_ATOL = 1e-3
FTHETA_TRANSFORM_STRICT_ATOL = 1e-5
FTHETA_TRANSFORM_ATOL = 1e-4
FTHETA_BACKWARD_REF_ATOL = 1e-2
FTHETA_BACKWARD_REF_RTOL = 1e-4


def test_ftheta_camera_model_initialization(
    ftheta_model, ftheta_projection_forward_ref, no_external
):
    """Assert CameraModel stores projection, external distortion, resolution, and shutter type correctly."""
    assert ftheta_model.projection is ftheta_projection_forward_ref
    assert ftheta_model.external_distortion is no_external
    assert ftheta_model.resolution == (100, 80)
    assert ftheta_model.shutter_type == ShutterType.GLOBAL


def test_ftheta_camera_rays_to_image_points_basic(ftheta_model):
    """Assert an optical-axis ray projects to the principal point with a valid flag."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=ftheta_model.projection.principal_point.device,
    )
    result = ftheta_model.camera_rays_to_image_points(rays, return_jacobians=True)
    assert isinstance(result, ImagePointsReturn)
    assert torch.allclose(
        result.image_points,
        ftheta_model.projection.principal_point.reshape(1, 2),
        atol=FTHETA_BASIC_ATOL,
    )
    assert result.valid_flag.item()
    assert result.jacobians.shape == (1, 2, 3)


def test_ftheta_image_points_to_camera_rays_roundtrip(ftheta_model):
    """Assert image_points -> camera_rays -> image_points round-trips within 1e-3 tolerance."""
    pp = ftheta_model.projection.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor(
        [[0.0, 0.0], [2.0, 1.0], [-3.0, 4.0]],
        device=pp.device,
    )
    rays = ftheta_model.image_points_to_camera_rays(image_points)
    roundtrip = ftheta_model.camera_rays_to_image_points(rays).image_points
    assert torch.allclose(roundtrip, image_points, atol=FTHETA_IMAGE_ROUNDTRIP_ATOL)


def test_ftheta_world_points_to_image_points_static_pose(ftheta_model, static_pose):
    """Assert static-pose projection filters invalid points and returns optional T_sensor_world and timestamps."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=ftheta_model.projection.principal_point.device,
    )
    result = ftheta_model.world_points_to_image_points_static_pose(
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


def test_ftheta_world_points_to_image_points_shutter_pose(
    ftheta_projection_forward_ref, no_external, dynamic_pose
):
    """Assert rolling-shutter projection produces per-point image_points, valid_flag, and timestamps."""
    model = CameraModel(
        projection=ftheta_projection_forward_ref,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
    )
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=ftheta_projection_forward_ref.principal_point.device,
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
def test_ftheta_image_points_relative_frame_times(
    ftheta_projection_forward_ref, no_external, shutter_type
):
    """Assert relative frame times are in [0, 1] for all supported shutter types."""
    model = CameraModel(
        projection=ftheta_projection_forward_ref,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=shutter_type,
    )
    image_points = torch.tensor(
        [[0.5, 0.5], [99.5, 79.5]],
        device=ftheta_projection_forward_ref.principal_point.device,
    )
    times = model.image_points_relative_frame_times(image_points)
    assert times.shape == (2,)
    assert torch.all((times >= 0) & (times <= 1))


def test_ftheta_pixels_image_points_conversions(ftheta_model):
    """Assert pixels_to_image_points adds 0.5 and image_points_to_pixels floors to int32."""
    pixels = torch.tensor(
        [[0, 1]],
        device=ftheta_model.projection.principal_point.device,
    )
    image_points = ftheta_model.pixels_to_image_points(pixels)
    assert torch.equal(
        image_points, torch.tensor([[0.5, 1.5]], device=image_points.device)
    )
    assert torch.equal(
        ftheta_model.image_points_to_pixels(image_points), pixels.to(torch.int32)
    )


def test_ftheta_transform_scale_offset(ftheta_model):
    """Assert transform(scale=0.5) scales fw_poly by scale_v, applies the
    ``(pp + 0.5) * s - 0.5`` pixel-center convention to principal_point, and
    updates resolution to (50, 40)."""
    scaled = ftheta_model.transform(0.5)
    expected_fw_poly = ftheta_model.projection.fw_poly * 0.5
    assert torch.allclose(
        scaled.projection.fw_poly, expected_fw_poly, atol=FTHETA_TRANSFORM_STRICT_ATOL
    )
    expected_pp = (ftheta_model.projection.principal_point + 0.5) * 0.5 - 0.5
    assert torch.allclose(
        scaled.projection.principal_point,
        expected_pp,
        atol=FTHETA_TRANSFORM_STRICT_ATOL,
    )
    assert scaled.resolution == (50, 40)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_ftheta_transform_isotropic_scale_bw_poly_and_A_invariants(ftheta_model, scale):
    """For isotropic scale s, bw_poly[i] scales by (1/s)^i and A is unchanged
    (since scale_u/scale_v == 1). Ainv must satisfy A @ Ainv ~= I."""
    transformed = ftheta_model.transform(scale)
    inv = 1.0 / scale
    powers = torch.arange(
        ftheta_model.projection.bw_poly.shape[0],
        device=ftheta_model.projection.bw_poly.device,
        dtype=ftheta_model.projection.bw_poly.dtype,
    )
    expected_bw = ftheta_model.projection.bw_poly * torch.pow(
        torch.tensor(inv, device=powers.device, dtype=powers.dtype), powers
    )
    assert torch.allclose(
        transformed.projection.bw_poly,
        expected_bw,
        atol=FTHETA_TRANSFORM_STRICT_ATOL,
    )
    # Isotropic scale leaves A unchanged (top row scales by scale_u/scale_v == 1).
    assert torch.allclose(
        transformed.projection.A,
        ftheta_model.projection.A,
        atol=FTHETA_TRANSFORM_STRICT_ATOL,
    )
    a = transformed.projection.A.reshape(2, 2)
    ainv = transformed.projection.Ainv.reshape(2, 2)
    eye = torch.eye(2, device=a.device, dtype=a.dtype)
    assert torch.allclose(a @ ainv, eye, atol=FTHETA_TRANSFORM_ATOL)


def test_ftheta_transform_anisotropic_scale_changes_A_top_row(ftheta_model):
    """Assert anisotropic (1.0, 0.5) scales the top row of A by scale_u/scale_v = 2."""
    transformed = ftheta_model.transform((1.0, 0.5))
    expected_A = ftheta_model.projection.A.clone()
    expected_A[0] = expected_A[0] * 2.0
    expected_A[1] = expected_A[1] * 2.0
    assert torch.allclose(
        transformed.projection.A, expected_A, atol=FTHETA_TRANSFORM_STRICT_ATOL
    )
    a = transformed.projection.A.reshape(2, 2)
    ainv = transformed.projection.Ainv.reshape(2, 2)
    eye = torch.eye(2, device=a.device, dtype=a.dtype)
    assert torch.allclose(a @ ainv, eye, atol=FTHETA_TRANSFORM_ATOL)


def test_ftheta_transform_round_trip_restores_original(ftheta_model):
    """Assert scaling by s then 1/s reproduces the original projection."""
    forward = ftheta_model.transform(2.0)
    back = forward.transform(0.5)
    orig = ftheta_model.projection
    assert torch.allclose(
        back.projection.principal_point,
        orig.principal_point,
        atol=FTHETA_TRANSFORM_ATOL,
    )
    assert torch.allclose(
        back.projection.fw_poly, orig.fw_poly, atol=FTHETA_TRANSFORM_ATOL
    )
    assert torch.allclose(
        back.projection.bw_poly, orig.bw_poly, atol=FTHETA_TRANSFORM_ATOL
    )
    assert torch.allclose(back.projection.A, orig.A, atol=FTHETA_TRANSFORM_ATOL)
    assert torch.allclose(back.projection.Ainv, orig.Ainv, atol=FTHETA_TRANSFORM_ATOL)


def test_ftheta_transform_returns_new_object_no_in_place_mutation(ftheta_model):
    """Assert transform returns a new CameraModel without mutating the original fw_poly."""
    original_fw_poly = ftheta_model.projection.fw_poly.clone()
    transformed = ftheta_model.transform(0.5)
    assert transformed is not ftheta_model
    assert torch.equal(ftheta_model.projection.fw_poly, original_fw_poly)


def test_ftheta_transform_preserves_grad(ftheta_model):
    """Assert gradients flow back through transform to the original fw_poly with correct scale."""
    ftheta_model.projection.fw_poly.requires_grad_(True)
    transformed = ftheta_model.transform(0.5)
    transformed.projection.fw_poly.sum().backward()
    assert ftheta_model.projection.fw_poly.grad is not None
    assert_grad_reference_close(
        ftheta_model.projection.fw_poly.grad,
        torch.full_like(ftheta_model.projection.fw_poly, 0.5),
        rtol=0.0,
        atol=0.0,
        max_rel_l2=0.0,
        max_rel_l1=0.0,
        min_cosine=1.0 - 1e-12,
        max_signed_bias=0.0,
        msg="ftheta transform fw_poly.grad",
    )


def test_ftheta_camera_is_nn_module_and_to_device(ftheta_model):
    """Assert CameraModel is an nn.Module and .to(device) moves all projection tensors."""
    assert isinstance(ftheta_model, nn.Module)
    moved = ftheta_model.to(ftheta_model.projection.principal_point.device)
    assert isinstance(moved, CameraModel)
    assert (
        moved.projection.principal_point.device
        == ftheta_model.projection.principal_point.device
    )
    # Every component tensor must be threaded by _move_projection.
    assert (
        moved.projection.fw_poly.device
        == ftheta_model.projection.principal_point.device
    )
    assert moved.projection.A.device == ftheta_model.projection.principal_point.device


def test_ftheta_camera_model_state_dict_and_pickle(ftheta_model):
    """Assert state_dict is a dict and the model round-trips through torch.save/load."""
    assert isinstance(ftheta_model.state_dict(), dict)
    buf = io.BytesIO()
    torch.save(ftheta_model, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert loaded.projection.resolution == ftheta_model.projection.resolution
    assert torch.allclose(loaded.projection.fw_poly, ftheta_model.projection.fw_poly)
    assert torch.allclose(loaded.projection.bw_poly, ftheta_model.projection.bw_poly)
    assert torch.allclose(loaded.projection.A, ftheta_model.projection.A)
    assert torch.allclose(loaded.projection.Ainv, ftheta_model.projection.Ainv)
    assert int(loaded.projection.reference_polynomial) == int(
        ftheta_model.projection.reference_polynomial
    )


def test_ftheta_camera_to_device_preserves_scalar_fields(ftheta_model):
    """Assert the six scalar config fields round-trip through ``.to(device)`` without mutation."""
    moved = ftheta_model.to(ftheta_model.projection.principal_point.device)
    src = ftheta_model.projection
    dst = moved.projection
    assert int(dst.reference_polynomial) == int(src.reference_polynomial)
    assert int(dst.fw_poly_degree) == int(src.fw_poly_degree)
    assert int(dst.bw_poly_degree) == int(src.bw_poly_degree)
    assert int(dst.newton_iterations) == int(src.newton_iterations)
    assert float(dst.max_angle) == pytest.approx(float(src.max_angle))
    assert float(dst.min_2d_norm) == pytest.approx(float(src.min_2d_norm))


def test_ftheta_camera_to_device_preserves_backward_reference(
    ftheta_projection_backward_ref, no_external
):
    """Assert BACKWARD-ref projection survives ``.to(device)`` with reference_polynomial intact;
    otherwise the kernel silently runs the Newton path on the wrong polynomial."""
    from gsplat.sensors.kernels.cameras import ShutterType

    model = CameraModel(
        projection=ftheta_projection_backward_ref,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )
    moved = model.to(ftheta_projection_backward_ref.principal_point.device)
    assert int(moved.projection.reference_polynomial) == int(
        ftheta_projection_backward_ref.reference_polynomial
    )


def test_ftheta_backward_ref_projection_round_trip(
    ftheta_projection_backward_ref, no_external
):
    """Assert image -> ray -> image round-trips on BACKWARD-ref intrinsics, where
    backprojection is a direct ``bw_poly`` evaluation and the forward (ray->image)
    direction is the one that exercises the BACKWARD-ref Newton path."""
    from gsplat.sensors.kernels.cameras import (
        camera_rays_to_image_points,
        image_points_to_camera_rays,
    )

    pp = ftheta_projection_backward_ref.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor(
        [[2.0, 1.5]], device=pp.device, dtype=torch.float32
    )
    rays = image_points_to_camera_rays(
        image_points, ftheta_projection_backward_ref, no_external
    )
    image_points_back, _ = camera_rays_to_image_points(
        rays, ftheta_projection_backward_ref, no_external
    )
    assert torch.allclose(
        image_points_back,
        image_points,
        atol=FTHETA_BACKWARD_REF_ATOL,
        rtol=FTHETA_BACKWARD_REF_RTOL,
    )


def test_ftheta_fov_clamp_marks_invalid_beyond_max_angle(
    ftheta_projection_forward_ref, no_external
):
    """Assert a camera ray with theta > max_angle is marked invalid."""
    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    max_angle = float(ftheta_projection_forward_ref.max_angle)
    # Clamp to <= 89 deg so z stays positive.
    over = min(max_angle + 0.1, math.radians(89.0))
    device = ftheta_projection_forward_ref.principal_point.device
    rays = torch.tensor(
        [[math.sin(over), 0.0, math.cos(over)]],
        device=device,
        dtype=torch.float32,
    )
    _, valid = camera_rays_to_image_points(
        rays, ftheta_projection_forward_ref, no_external
    )
    assert valid.tolist() == [False]


def test_ftheta_min_2d_norm_clamp_returns_principal_point(
    ftheta_projection_forward_ref, no_external
):
    """Assert an exactly-on-axis ray lands at the principal point via the min_2d short-circuit
    and the ``<=`` predicate does not produce NaNs."""
    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    device = ftheta_projection_forward_ref.principal_point.device
    rays = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    image_points, valid = camera_rays_to_image_points(
        rays, ftheta_projection_forward_ref, no_external
    )
    assert torch.isfinite(image_points).all()
    assert valid.tolist() == [True]
    assert torch.allclose(
        image_points[0],
        ftheta_projection_forward_ref.principal_point,
        atol=FTHETA_TRANSFORM_STRICT_ATOL,
    )


def test_ftheta_with_windshield_model_static_pose(
    ftheta_model_with_windshield, static_pose
):
    """Assert the windshield-augmented FTheta model projects an on-axis point as
    valid and a behind-camera point as invalid, yielding ``valid_flag == [True, False]``."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=ftheta_model_with_windshield.projection.principal_point.device,
    )
    result = ftheta_model_with_windshield.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
    )
    assert result.valid_flag.tolist() == [True, False]


def test_ftheta_image_frame_image_buffer_registered(ftheta_model, static_pose):
    """Assert ImageFrame registers the image tensor as a buffer."""
    frame = ImageFrame("cam", ftheta_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    assert "image" in dict(frame.named_buffers())


def test_ftheta_image_frame_to_device_moves_image_and_pose(ftheta_model, static_pose):
    """Assert ImageFrame.to(device) moves the image, pose, and camera projection together."""
    frame = ImageFrame("cam", ftheta_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    frame = frame.to(frame.image.device)
    assert frame.pose.translation.device == frame.image.device
    assert frame.camera_model.projection.principal_point.device == frame.image.device


def test_ftheta_world_points_to_pixels_static_pose_filters_invalid(
    ftheta_model, static_pose
):
    """Assert world_points_to_pixels_static_pose drops invalid points and returns valid_indices."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=ftheta_model.projection.principal_point.device,
    )
    result = ftheta_model.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
        return_valid_indices=True,
    )
    assert result.pixels.shape == (1, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]


def test_ftheta_world_points_to_pixels_static_pose_return_all_keeps_input_alignment(
    ftheta_model, static_pose
):
    """Assert return_all_projections preserves input alignment alongside valid_flag/valid_indices."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=ftheta_model.projection.principal_point.device,
    )
    result = ftheta_model.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
        return_valid_indices=True,
        return_all_projections=True,
    )
    assert result.pixels.shape == (2, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]


def test_real_ftheta_windshield_through_camera_model(
    real_ftheta_projection_with_windshield, real_ftheta_windshield_distortion
):
    """Assert real-windshield distortion routed through ``CameraModel`` matches the kernel-direct
    forward output, guarding the model-layer dispatch and ``_check_pair`` fence against silently
    dropping the external distortion."""
    from gsplat.sensors.kernels.cameras import camera_rays_to_image_points

    proj = real_ftheta_projection_with_windshield
    distortion = real_ftheta_windshield_distortion
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
        via_model.image_points, direct_pts, atol=FTHETA_TRANSFORM_STRICT_ATOL
    )
