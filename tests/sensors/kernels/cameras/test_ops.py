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

"""Tests for camera kernel ops covering forward/backward correctness and validity logic.

Exercises camera_rays_to_image_points, image_points_to_camera_rays,
project_world_points_*, image_points_to_world_rays_*, parameter-class
construction/pickling, and autograd.gradcheck for all public verbs with both
NoExternalDistortion and BivariateWindshield.
"""

# Row mapping: design-tests-opencvpinhole.md §6 rows 3-21 are covered here.

import io

import numpy as np
import pytest
import torch
from torch import Tensor

from gsplat._helper import assert_grad_reference_close
from gsplat.sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
    FThetaProjection,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
    camera_rays_to_image_points,
    from_components,
    generate_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_shutter_pose,
    image_points_to_world_rays_static_pose,
    pixel_grid_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
    relative_frame_times,
    unpack_dynamic_pose_components,
)
from gsplat.sensors.kernels.cameras.ops import (
    _quat_slerp_wxyz,
    _unpack_static_pose,
)
from gsplat.sensors.kernels.common import DynamicPose, Pose


# Public camera ops cast inputs to float32 before launching CUDA, so use
# finite-difference settings that are stable for float32 kernels.
GRADCHECK_KWARGS = {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2}
# Looser absolute tolerance for ops whose forward chain runs through a
# slerp + lerp interpolation (mean-pose, shutter-pose). With the build's
# `-use_fast_math` (geometry parity) the interpolated cam_pt picks up a few
# ULPs of float32 noise; the analytical Jacobian is correct (diagonals match
# at <0.01% relative error) but the cross-coupled near-zero off-diagonals are
# below the finite-difference noise floor at atol=1e-2.
GRADCHECK_KWARGS_DYNAMIC = {"eps": 1e-3, "atol": 5e-2, "rtol": 1e-2}
# FTheta projection runs through acos/sin/cos in float32. The main diagonal
# derivatives match tightly, but small cross terms sit close to the finite
# difference noise floor under the public float32 CUDA dispatch.
GRADCHECK_KWARGS_FTHETA_PROJECT = {"eps": 1e-3, "atol": 8e-2, "rtol": 1e-2}


def torch_pinhole_project(
    camera_rays: Tensor, focal_length: Tensor, principal_point: Tensor
) -> Tensor:
    """Reference pinhole projection: divide by z, scale by focal, shift by principal point."""
    xy = camera_rays[:, :2] / camera_rays[:, 2:3]
    return xy * focal_length + principal_point


def torch_pinhole_backproject(
    image_points: Tensor, focal_length: Tensor, principal_point: Tensor
) -> Tensor:
    """Reference pinhole back-projection: undo principal-point shift, undo focal scale, normalize."""
    xy = (image_points - principal_point) / focal_length
    rays = torch.cat(
        [xy, torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)], dim=-1
    )
    return torch.nn.functional.normalize(rays, dim=-1)


def make_grid_image_points(device: torch.device) -> Tensor:
    """Return three fixed image points (near principal point) for reuse across tests."""
    return torch.tensor([[50.0, 40.0], [60.0, 40.0], [50.0, 52.0]], device=device)


def _ftheta_eval_poly(coeffs: Tensor, x: Tensor, degree: int) -> Tensor:
    """Standard power-basis polynomial r = sum_i c_i * x^i over i in [0, degree].

    Mirrors the device-side ``ftheta_poly_eval`` in ``ftheta_kernel.cuh``
    (constant term INCLUDED at i=0).
    """
    result = torch.zeros_like(x)
    for i in range(degree, -1, -1):
        result = result * x + coeffs[i]
    return result


def _ftheta_eval_poly_derivative(coeffs: Tensor, x: Tensor, degree: int) -> Tensor:
    """Derivative of the standard power-basis polynomial."""
    result = torch.zeros_like(x)
    for i in range(degree, 0, -1):
        result = result * x + coeffs[i] * float(i)
    return result


def _ftheta_invert_bw_newton(
    bw_poly: Tensor,
    bw_degree: int,
    target_theta: Tensor,
    iterations: int,
) -> Tensor:
    """Newton's method to invert a power-basis polynomial (standard powers)."""
    r = target_theta.clone()
    for _ in range(int(iterations)):
        f = _ftheta_eval_poly(bw_poly, r, bw_degree) - target_theta
        df = _ftheta_eval_poly_derivative(bw_poly, r, bw_degree)
        r = torch.where(torch.abs(df) > 1e-10, r - f / df, r)
    return r


def torch_ftheta_project(
    camera_rays: Tensor,
    principal_point: Tensor,
    fw_poly: Tensor,
    bw_poly: Tensor,
    A: Tensor,
    reference_polynomial: int,
    fw_poly_degree: int,
    bw_poly_degree: int,
    newton_iterations: int,
    max_angle: float,
    min_2d_norm: float,
) -> Tensor:
    """Pure-torch reference for the FTheta forward projection.

    Mirrors ``ftheta_project_ray`` in
    ``gsplat/sensors/kernels/cuda/csrc/ftheta_kernel.cuh`` (BACKWARD-ref
    branch invokes Newton inversion). Operates on float64 inputs with
    high-precision intermediates so it can serve as a gradient/forward
    oracle for the float32 CUDA kernel.

    The polynomials use the standard power basis ``sum_i c_i * x^i``
    (constant term included). A is a flat ``(4,)`` row-major 2x2 matrix.
    """
    rays_norm = torch.linalg.norm(camera_rays, dim=-1, keepdim=True)
    rays_unit = camera_rays / rays_norm.clamp_min(1e-30)
    z = rays_unit[:, 2]
    xy = rays_unit[:, :2]
    xy_norm = torch.linalg.norm(xy, dim=-1)
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))

    if reference_polynomial == 0:  # FORWARD-ref
        r = _ftheta_eval_poly(fw_poly, theta, fw_poly_degree)
    else:  # BACKWARD-ref
        r = _ftheta_invert_bw_newton(bw_poly, bw_poly_degree, theta, newton_iterations)

    safe_norm = torch.where(xy_norm < min_2d_norm, torch.ones_like(xy_norm), xy_norm)
    rxy = xy * (r / safe_norm).unsqueeze(-1)

    A_mat = A.reshape(2, 2)
    warped = rxy @ A_mat.T

    image_points = warped + principal_point.reshape(1, 2)
    invalid = (z <= 0.0) | (theta > float(max_angle))
    pp_broadcast = principal_point.reshape(1, 2).expand_as(image_points)
    image_points = torch.where(
        invalid.unsqueeze(-1).expand_as(image_points),
        pp_broadcast,
        image_points,
    )
    return image_points


def torch_ftheta_backproject(
    image_points: Tensor,
    principal_point: Tensor,
    fw_poly: Tensor,
    bw_poly: Tensor,
    Ainv: Tensor,
    reference_polynomial: int,
    fw_poly_degree: int,
    bw_poly_degree: int,
    newton_iterations: int,
    min_2d_norm: float,
) -> Tensor:
    """Pure-torch reference for the FTheta inverse projection.

    Mirrors ``ftheta_backproject_image_point``: subtract principal point,
    apply Ainv to the (x, y) offset, compute the radial distance,
    evaluate the polynomial (or its inverse via Newton) to recover the
    polar angle, then construct the unit ray.
    """
    centered = image_points - principal_point.reshape(1, 2)
    Ainv_mat = Ainv.reshape(2, 2)
    transformed = centered @ Ainv_mat.T
    rdist = torch.linalg.norm(transformed, dim=-1)

    if reference_polynomial == 0:  # FORWARD-ref
        theta = _ftheta_invert_bw_newton(
            fw_poly, fw_poly_degree, rdist, newton_iterations
        )
    else:  # BACKWARD-ref
        theta = _ftheta_eval_poly(bw_poly, rdist, bw_poly_degree)

    safe_rdist = torch.where(rdist < min_2d_norm, torch.ones_like(rdist), rdist)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    xy_unit = transformed * (sin_t / safe_rdist).unsqueeze(-1)
    rays = torch.cat([xy_unit, cos_t.unsqueeze(-1)], dim=-1)

    fallback = torch.tensor(
        [0.0, 0.0, 1.0], device=image_points.device, dtype=image_points.dtype
    )
    rays = torch.where(
        (rdist < min_2d_norm).unsqueeze(-1),
        fallback.expand_as(rays),
        rays,
    )
    return torch.nn.functional.normalize(rays, dim=-1)


def make_ideal_ftheta_components(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> dict:
    """Synthetic linear-polynomial F-Theta intrinsics for unit tests.

    The polynomials reduce to ``r = k * theta`` and ``theta = r / k`` so
    Newton inversion converges in one step and the round-trip is exact
    up to float32 rounding. The 2x2 warp ``A`` and ``Ainv`` are identity.
    """
    k = 100.0
    fw_poly = torch.zeros(6, device=device, dtype=dtype)
    fw_poly[1] = k
    bw_poly = torch.zeros(6, device=device, dtype=dtype)
    bw_poly[1] = 1.0 / k
    A = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    Ainv = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    principal_point = torch.tensor([50.0, 40.0], device=device, dtype=dtype)
    return {
        "principal_point": principal_point,
        "fw_poly": fw_poly,
        "bw_poly": bw_poly,
        "A": A,
        "Ainv": Ainv,
        "resolution": (100, 80),
        "fw_poly_degree": 1,
        "bw_poly_degree": 1,
        "newton_iterations": 10,
        "max_angle": 1.4,
        "min_2d_norm": 1e-6,
    }


def _opencv_project_reference(camera_rays, projection):
    """Compute the expected OpenCV rational+tangential+thin-prism projection in pure PyTorch."""
    xy = camera_rays[:, :2] / camera_rays[:, 2:3]
    x = xy[:, 0]
    y = xy[:, 1]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    k = projection.radial_coeffs
    p = projection.tangential_coeffs
    s = projection.thin_prism_coeffs
    radial = (1 + k[0] * r2 + k[1] * r4 + k[2] * r6) / (
        1 + k[3] * r2 + k[4] * r4 + k[5] * r6
    )
    xy_prod = x * y
    delta_x = 2 * p[0] * xy_prod + p[1] * (r2 + 2 * x * x) + s[0] * r2 + s[1] * r4
    delta_y = p[0] * (r2 + 2 * y * y) + 2 * p[1] * xy_prod + s[2] * r2 + s[3] * r4
    distorted = torch.stack((x * radial + delta_x, y * radial + delta_y), dim=-1)
    return distorted * projection.focal_length + projection.principal_point


def test_construction_with_zero_distortion_succeeds(ideal_projection, no_external):
    """Verify that constructing an OpenCVPinholeProjection with all-zero distortion coeffs succeeds."""
    assert ideal_projection.resolution == (100, 80)
    assert no_external is not None


def test_construction_with_full_distortion_succeeds(distorted_projection):
    """Verify that constructing an OpenCVPinholeProjection with non-zero distortion coeffs succeeds."""
    assert distorted_projection.radial_coeffs.shape == (6,)
    assert distorted_projection.tangential_coeffs.shape == (2,)


def test_real_camera_fixture_loads_expected_camera_count(test_camera_params):
    """Verify the test JSON file contains exactly 5 camera entries."""
    assert len(test_camera_params) == 5


def test_real_camera_fixture_loads_json_intrinsics(
    real_camera_record, real_camera_projection
):
    """Verify that focal_length, principal_point, and resolution are loaded from JSON correctly."""
    intrinsics = real_camera_record["intrinsics"]
    assert real_camera_projection.resolution == tuple(real_camera_record["resolution"])
    assert torch.allclose(
        real_camera_projection.focal_length.cpu(),
        torch.tensor(intrinsics["focal_length"], dtype=torch.float32),
    )
    assert torch.allclose(
        real_camera_projection.principal_point.cpu(),
        torch.tensor(intrinsics["principal_point"], dtype=torch.float32),
    )
    assert real_camera_projection.resolution[0] > 1000


def test_real_camera_projection_matches_opencv_reference(
    real_camera_projection, no_external
):
    """Verify camera_rays_to_image_points matches the pure-PyTorch OpenCV reference on real intrinsics."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0], [0.05, -0.03, 1.0], [-0.08, 0.04, 1.0]],
        device=real_camera_projection.focal_length.device,
    )
    image_points, valid = camera_rays_to_image_points(
        rays, real_camera_projection, no_external, allow_device_transfer=True
    )
    expected = _opencv_project_reference(rays, real_camera_projection)
    assert torch.allclose(image_points, expected, atol=1e-3, rtol=1e-5)
    assert valid.all()


def test_real_camera_distorted_round_trip(real_camera_projection, no_external):
    """Verify back-project then re-project recovers the original image points on real intrinsics."""
    offsets = torch.tensor(
        [[0.0, 0.0], [80.0, -60.0], [-120.0, 40.0]],
        device=real_camera_projection.focal_length.device,
    )
    image_points = real_camera_projection.principal_point.reshape(1, 2) + offsets
    rays = image_points_to_camera_rays(
        image_points, real_camera_projection, no_external, allow_device_transfer=True
    )
    roundtrip, valid = camera_rays_to_image_points(
        rays, real_camera_projection, no_external, allow_device_transfer=True
    )
    assert torch.allclose(roundtrip, image_points, atol=1e-2, rtol=1e-5)
    assert valid.all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focal_length": torch.ones(1)},
        {"principal_point": torch.ones(1)},
        {"radial_coeffs": torch.ones(5)},
        {"tangential_coeffs": torch.ones(1)},
        {"thin_prism_coeffs": torch.ones(3)},
    ],
)
def test_component_shape_rejection(kwargs):
    """Verify that OpenCVPinholeProjection raises RuntimeError when any intrinsic tensor has wrong shape."""
    base = {
        "focal_length": torch.ones(2),
        "principal_point": torch.ones(2),
        "radial_coeffs": torch.zeros(6),
        "tangential_coeffs": torch.zeros(2),
        "thin_prism_coeffs": torch.zeros(4),
        "resolution": (100, 80),
    }
    base.update(kwargs)
    with pytest.raises(RuntimeError):
        OpenCVPinholeProjection(**base)


def test_pickle_roundtrip(ideal_projection):
    """Verify that an OpenCVPinholeProjection can be saved and loaded via torch.save/load."""
    buf = io.BytesIO()
    torch.save(ideal_projection, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert torch.equal(loaded.focal_length, ideal_projection.focal_length)
    assert loaded.resolution == ideal_projection.resolution


def test_bivariate_windshield_constructor_validation():
    """Verify that BivariateWindshieldDistortion raises on invalid coeffs length, polynomial enum, or degree bounds."""
    coeffs = torch.zeros(42)
    BivariateWindshieldDistortion(coeffs, int(ReferencePolynomial.FORWARD), 1, 1)
    with pytest.raises(RuntimeError, match="distortion_coeffs"):
        BivariateWindshieldDistortion(torch.zeros(41), 0, 1, 1)
    with pytest.raises(RuntimeError, match="reference_polynomial"):
        BivariateWindshieldDistortion(coeffs, 2, 1, 1)
    with pytest.raises(RuntimeError, match="h_poly_degree"):
        BivariateWindshieldDistortion(coeffs, 0, 3, 1)
    with pytest.raises(RuntimeError, match="v_poly_degree"):
        BivariateWindshieldDistortion(coeffs, 0, 1, 5)
    with pytest.raises(ValueError, match="triangular"):
        from_components(
            torch.ones(2),
            torch.ones(3),
            torch.ones(2),
            torch.ones(3),
            ReferencePolynomial.FORWARD,
        )


def test_bivariate_windshield_pickle_roundtrip(windshield_distortion):
    """Verify that a BivariateWindshieldDistortion survives a torch.save/load round-trip."""
    buf = io.BytesIO()
    torch.save(windshield_distortion, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert torch.equal(
        loaded.distortion_coeffs, windshield_distortion.distortion_coeffs
    )
    assert loaded.reference_polynomial == int(ReferencePolynomial.FORWARD)
    assert loaded.h_poly_degree == windshield_distortion.h_poly_degree
    assert loaded.v_poly_degree == windshield_distortion.v_poly_degree


def test_bivariate_windshield_pickle_validates_on_load(windshield_distortion):
    """Verify that the deserializer rejects corrupt state with the same errors as the constructor."""
    # The pickle deserializer registered by `def_pickle` at ext.cpp:165 calls
    # `check_bivariate_windshield_distortion(ptr)` after constructing the
    # `BivariateWindshieldDistortion` from the unpickled state tuple. That
    # check is the same one the public constructor runs, so corrupt-pickle
    # rejection is provable by exercising the constructor with the same bad
    # arguments. (We cannot exercise the deserializer via `__setstate__`
    # directly: torch ScriptObjects use a different pickle protocol where the
    # raw `__setstate__` entry point is a no-op and the validator only fires
    # via the `pickle.loads(pickle.dumps(...))` round-trip path.)
    coeffs = windshield_distortion.distortion_coeffs
    with pytest.raises(RuntimeError, match="distortion_coeffs"):
        BivariateWindshieldDistortion(
            torch.zeros(41, device=coeffs.device, dtype=coeffs.dtype),
            int(ReferencePolynomial.FORWARD),
            int(windshield_distortion.h_poly_degree),
            int(windshield_distortion.v_poly_degree),
        )
    with pytest.raises(RuntimeError, match="reference_polynomial"):
        BivariateWindshieldDistortion(
            torch.zeros(42, device=coeffs.device, dtype=coeffs.dtype),
            2,
            int(windshield_distortion.h_poly_degree),
            int(windshield_distortion.v_poly_degree),
        )


@pytest.mark.parametrize(
    "h_terms, v_terms, expected_h_degree, expected_v_degree",
    [
        (1, 1, 0, 0),
        (3, 3, 1, 1),
        (3, 6, 1, 2),
        (3, 15, 1, 4),
        (6, 15, 2, 4),
    ],
)
def test_bivariate_windshield_from_components_polynomial_bounds(
    h_terms, v_terms, expected_h_degree, expected_v_degree
):
    """Verify that from_components infers the correct h/v polynomial degrees from term counts."""
    h_poly = torch.zeros(h_terms)
    v_poly = torch.zeros(v_terms)
    distortion = from_components(
        h_poly,
        v_poly,
        h_poly.clone(),
        v_poly.clone(),
        ReferencePolynomial.FORWARD,
    )
    assert distortion.distortion_coeffs.numel() == 42
    assert distortion.h_poly_degree == expected_h_degree
    assert distortion.v_poly_degree == expected_v_degree


def test_bivariate_windshield_from_components_rejects_mismatched_h_inv_degree():
    """Verify that from_components raises ValueError when h_poly and h_poly_inv have different degrees."""
    with pytest.raises(ValueError, match="h_poly"):
        from_components(
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(6),
            torch.zeros(3),
            ReferencePolynomial.FORWARD,
        )


def test_bivariate_windshield_from_components_rejects_mismatched_v_inv_degree():
    """Verify that from_components raises ValueError when v_poly and v_poly_inv have different degrees."""
    with pytest.raises(ValueError, match="v_poly"):
        from_components(
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(6),
            ReferencePolynomial.FORWARD,
        )


def test_bivariate_windshield_from_components_rejects_dtype_mismatch():
    """Verify that from_components raises ValueError when polynomial tensors have mismatched dtypes."""
    with pytest.raises(ValueError, match="dtype"):
        from_components(
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
            ReferencePolynomial.FORWARD,
        )


def test_bivariate_windshield_from_components_rejects_device_mismatch(sensor_device):
    """Verify that from_components raises ValueError when polynomial tensors are on different devices."""
    with pytest.raises(ValueError, match="device"):
        from_components(
            torch.zeros(3, device=sensor_device),
            torch.zeros(3, device="cpu"),
            torch.zeros(3, device=sensor_device),
            torch.zeros(3, device=sensor_device),
            ReferencePolynomial.FORWARD,
        )


def test_camera_rays_to_image_points_basic(ideal_projection, no_external):
    """Verify camera_rays_to_image_points matches the pinhole reference projection on two sample rays."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]], device=ideal_projection.focal_length.device
    )
    image_points, valid = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    expected = torch_pinhole_project(
        rays, ideal_projection.focal_length, ideal_projection.principal_point
    )
    assert torch.allclose(image_points, expected)
    assert valid.tolist() == [True, True]


def test_identity_bivariate_matches_no_external_all_public_ops(
    ideal_projection, no_external, windshield_distortion, dynamic_pose, static_pose
):
    """Verify that identity windshield coefficients produce the same outputs as NoExternalDistortion across all public ops."""
    device = ideal_projection.focal_length.device
    rays = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]], device=device)
    image_points = make_grid_image_points(device)
    world_points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.5]], device=device)

    no_img, no_valid = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    bi_img, bi_valid = camera_rays_to_image_points(
        rays, ideal_projection, windshield_distortion, allow_device_transfer=True
    )
    assert torch.allclose(bi_img, no_img, atol=1e-4)
    assert torch.equal(bi_valid, no_valid)

    no_rays = image_points_to_camera_rays(
        image_points, ideal_projection, no_external, allow_device_transfer=True
    )
    bi_rays = image_points_to_camera_rays(
        image_points,
        ideal_projection,
        windshield_distortion,
        allow_device_transfer=True,
    )
    assert torch.allclose(bi_rays, no_rays, atol=1e-5)

    no_mean = project_world_points_mean_pose(
        world_points,
        ideal_projection,
        no_external,
        dynamic_pose,
        (100, 80),
        return_valid_flags=True,
        allow_device_transfer=True,
    )
    bi_mean = project_world_points_mean_pose(
        world_points,
        ideal_projection,
        windshield_distortion,
        dynamic_pose,
        (100, 80),
        return_valid_flags=True,
        allow_device_transfer=True,
    )
    assert torch.allclose(bi_mean[0], no_mean[0], atol=1e-4)
    assert torch.equal(bi_mean[1], no_mean[1])

    no_shutter = project_world_points_shutter_pose(
        world_points,
        ideal_projection,
        no_external,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        return_valid_flags=True,
        allow_device_transfer=True,
    )
    bi_shutter = project_world_points_shutter_pose(
        world_points,
        ideal_projection,
        windshield_distortion,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        return_valid_flags=True,
        allow_device_transfer=True,
    )
    assert torch.allclose(bi_shutter[0], no_shutter[0], atol=1e-4)
    assert torch.equal(bi_shutter[1], no_shutter[1])

    no_world_static = image_points_to_world_rays_static_pose(
        image_points,
        ideal_projection,
        no_external,
        static_pose,
        allow_device_transfer=True,
    )[0]
    bi_world_static = image_points_to_world_rays_static_pose(
        image_points,
        ideal_projection,
        windshield_distortion,
        static_pose,
        allow_device_transfer=True,
    )[0]
    assert torch.allclose(bi_world_static, no_world_static, atol=1e-5)

    no_world_shutter = image_points_to_world_rays_shutter_pose(
        image_points,
        ideal_projection,
        no_external,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        allow_device_transfer=True,
    )[0]
    bi_world_shutter = image_points_to_world_rays_shutter_pose(
        image_points,
        ideal_projection,
        windshield_distortion,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        allow_device_transfer=True,
    )[0]
    assert torch.allclose(bi_world_shutter, no_world_shutter, atol=1e-5)


@pytest.mark.parametrize(
    "reference, active_slice",
    [
        (ReferencePolynomial.FORWARD, slice(0, 21)),
        (ReferencePolynomial.BACKWARD, slice(21, 42)),
    ],
)
def test_bivariate_camera_rays_to_image_points_distortion_grad_slice(
    ideal_projection, windshield_distortion, reference, active_slice
):
    """Verify that gradient flows into only the active coefficient slice (forward or backward) for camera_rays_to_image_points."""
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(reference),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = torch.tensor(
        [[0.05, -0.02, 1.0], [0.1, 0.03, 1.0]],
        device=ideal_projection.focal_length.device,
    )
    image_points, valid = camera_rays_to_image_points(
        rays, ideal_projection, distortion, allow_device_transfer=True
    )
    assert valid.all()
    image_points.sum().backward()
    grad = coeffs.grad
    assert grad is not None
    inactive = slice(21, 42) if active_slice.start == 0 else slice(0, 21)
    assert grad[active_slice].abs().sum() > 0
    assert_grad_reference_close(
        grad[inactive],
        torch.zeros_like(grad[inactive]),
        rtol=0.0,
        atol=0.0,
        msg="inactive bivariate coeff gradient slice",
    )


def test_identity_bivariate_backward_smoke_all_public_ops(
    ideal_projection, windshield_distortion, dynamic_pose, static_pose
):
    """Verify that backward() populates the correct half of distortion_coeffs.grad for each public op with FORWARD reference."""
    device = ideal_projection.focal_length.device
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(ReferencePolynomial.FORWARD),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = torch.tensor([[0.05, -0.02, 1.0], [0.1, 0.03, 1.0]], device=device)
    image_points = ideal_projection.principal_point.reshape(1, 2) + torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]], device=device
    )
    world_points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.5]], device=device)

    checks = [
        (
            lambda: camera_rays_to_image_points(
                rays, ideal_projection, distortion, allow_device_transfer=True
            )[0],
            slice(0, 21),
        ),
        (
            lambda: image_points_to_camera_rays(
                image_points,
                ideal_projection,
                distortion,
                allow_device_transfer=True,
            ),
            slice(21, 42),
        ),
        (
            lambda: project_world_points_mean_pose(
                world_points,
                ideal_projection,
                distortion,
                dynamic_pose,
                (100, 80),
                allow_device_transfer=True,
            )[0],
            slice(0, 21),
        ),
        (
            lambda: project_world_points_shutter_pose(
                world_points,
                ideal_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            slice(0, 21),
        ),
        (
            lambda: image_points_to_world_rays_static_pose(
                image_points,
                ideal_projection,
                distortion,
                static_pose,
                allow_device_transfer=True,
            )[0],
            slice(21, 42),
        ),
        (
            lambda: image_points_to_world_rays_shutter_pose(
                image_points,
                ideal_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            slice(21, 42),
        ),
    ]
    for op, active_slice in checks:
        coeffs.grad = None
        op().sum().backward()
        assert coeffs.grad is not None
        assert coeffs.grad[active_slice].abs().sum() > 0
        inactive = slice(0, 21) if active_slice.start == 21 else slice(21, 42)
        assert_grad_reference_close(
            coeffs.grad[inactive],
            torch.zeros_like(coeffs.grad[inactive]),
            rtol=0.0,
            atol=0.0,
            msg="inactive FORWARD-reference bivariate coeff gradients",
        )


def test_identity_bivariate_backward_smoke_all_public_ops_backward_reference(
    ideal_projection, windshield_distortion, dynamic_pose, static_pose
):
    """Verify that the active gradient slice is flipped for all public ops when using BACKWARD reference polynomial."""
    # Plan §7: the inverse-path active slice flips when reference_polynomial
    # changes, so forward-direction verbs accumulate into slice(21, 42) and
    # inverse-direction verbs accumulate into slice(0, 21) for BACKWARD.
    device = ideal_projection.focal_length.device
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(ReferencePolynomial.BACKWARD),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = torch.tensor([[0.05, -0.02, 1.0], [0.1, 0.03, 1.0]], device=device)
    image_points = ideal_projection.principal_point.reshape(1, 2) + torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]], device=device
    )
    world_points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.5]], device=device)

    forward_slice = slice(21, 42)
    inverse_slice = slice(0, 21)
    checks = [
        (
            lambda: camera_rays_to_image_points(
                rays, ideal_projection, distortion, allow_device_transfer=True
            )[0],
            forward_slice,
        ),
        (
            lambda: image_points_to_camera_rays(
                image_points,
                ideal_projection,
                distortion,
                allow_device_transfer=True,
            ),
            inverse_slice,
        ),
        (
            lambda: project_world_points_mean_pose(
                world_points,
                ideal_projection,
                distortion,
                dynamic_pose,
                (100, 80),
                allow_device_transfer=True,
            )[0],
            forward_slice,
        ),
        (
            lambda: project_world_points_shutter_pose(
                world_points,
                ideal_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            forward_slice,
        ),
        (
            lambda: image_points_to_world_rays_static_pose(
                image_points,
                ideal_projection,
                distortion,
                static_pose,
                allow_device_transfer=True,
            )[0],
            inverse_slice,
        ),
        (
            lambda: image_points_to_world_rays_shutter_pose(
                image_points,
                ideal_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            inverse_slice,
        ),
    ]
    for op, active_slice in checks:
        coeffs.grad = None
        op().sum().backward()
        assert coeffs.grad is not None
        assert coeffs.grad[active_slice].abs().sum() > 0
        inactive = slice(0, 21) if active_slice.start == 21 else slice(21, 42)
        assert_grad_reference_close(
            coeffs.grad[inactive],
            torch.zeros_like(coeffs.grad[inactive]),
            rtol=0.0,
            atol=0.0,
            msg="inactive BACKWARD-reference bivariate coeff gradients",
        )


def test_camera_rays_to_image_points_scratch_is_grad_gated(
    ideal_projection, no_external
):
    """Verify that the scratch tensor from camera_rays_to_image_points is empty when no grad is needed and sized correctly when grad is required."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]],
        device=ideal_projection.focal_length.device,
    )
    op = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_pinhole_no_external

    _, _, scratch = op(ideal_projection, no_external, rays)
    assert scratch.numel() == 0

    _, _, ray_grad_scratch = op(
        ideal_projection, no_external, rays.detach().clone().requires_grad_(True)
    )
    assert ray_grad_scratch.shape == (2, 6)

    projection_with_grad = OpenCVPinholeProjection(
        focal_length=ideal_projection.focal_length.detach()
        .clone()
        .requires_grad_(True),
        principal_point=ideal_projection.principal_point.detach().clone(),
        radial_coeffs=ideal_projection.radial_coeffs.detach().clone(),
        tangential_coeffs=ideal_projection.tangential_coeffs.detach().clone(),
        thin_prism_coeffs=ideal_projection.thin_prism_coeffs.detach().clone(),
        resolution=ideal_projection.resolution,
    )
    _, _, projection_grad_scratch = op(projection_with_grad, no_external, rays)
    assert projection_grad_scratch.shape == (2, 6)


def test_camera_rays_to_image_points_rejects_input_dtype_conversion(
    ideal_projection, no_external
):
    """Verify that camera_rays_to_image_points raises RuntimeError when rays have a non-float32 dtype and allow_device_transfer is not set."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
    )
    with pytest.raises(RuntimeError, match="requires transfer"):
        camera_rays_to_image_points(rays, ideal_projection, no_external)


def test_camera_rays_to_image_points_allows_input_dtype_conversion(
    ideal_projection, no_external
):
    """Verify that camera_rays_to_image_points accepts float64 rays and returns float32 when allow_device_transfer=True."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
    )
    image_points, valid = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    assert image_points.dtype == torch.float32
    assert valid.item()


def test_camera_rays_to_image_points_rejects_projection_dtype_conversion(
    ideal_projection, no_external
):
    """Verify that camera_rays_to_image_points raises RuntimeError when projection tensors have a non-float32 dtype."""
    projection = OpenCVPinholeProjection(
        focal_length=ideal_projection.focal_length.to(dtype=torch.float64),
        principal_point=ideal_projection.principal_point,
        radial_coeffs=ideal_projection.radial_coeffs,
        tangential_coeffs=ideal_projection.tangential_coeffs,
        thin_prism_coeffs=ideal_projection.thin_prism_coeffs,
        resolution=ideal_projection.resolution,
    )
    rays = torch.tensor([[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device)
    with pytest.raises(RuntimeError, match="requires transfer"):
        camera_rays_to_image_points(rays, projection, no_external)


def test_image_points_to_camera_rays_basic(ideal_projection, no_external):
    """Verify image_points_to_camera_rays matches the pinhole back-projection reference on grid image points."""
    image_points = make_grid_image_points(ideal_projection.focal_length.device)
    rays = image_points_to_camera_rays(
        image_points, ideal_projection, no_external, allow_device_transfer=True
    )
    expected = torch_pinhole_backproject(
        image_points, ideal_projection.focal_length, ideal_projection.principal_point
    )
    assert torch.allclose(rays, expected)


def test_image_points_to_camera_rays_scratch_is_grad_gated(
    ideal_projection, no_external
):
    """Verify the scratch tensor from image_points_to_camera_rays is empty without grad and sized correctly with grad."""
    image_points = make_grid_image_points(ideal_projection.focal_length.device)
    op = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_pinhole_no_external

    _, scratch = op(ideal_projection, no_external, image_points)
    assert scratch.numel() == 0

    _, grad_scratch = op(
        ideal_projection,
        no_external,
        image_points.detach().clone().requires_grad_(True),
    )
    assert grad_scratch.shape == (image_points.shape[0], 5)

    projection_with_grad = OpenCVPinholeProjection(
        focal_length=ideal_projection.focal_length.detach()
        .clone()
        .requires_grad_(True),
        principal_point=ideal_projection.principal_point.detach().clone(),
        radial_coeffs=ideal_projection.radial_coeffs.detach().clone(),
        tangential_coeffs=ideal_projection.tangential_coeffs.detach().clone(),
        thin_prism_coeffs=ideal_projection.thin_prism_coeffs.detach().clone(),
        resolution=ideal_projection.resolution,
    )
    _, projection_grad_scratch = op(
        projection_with_grad,
        no_external,
        image_points.detach(),
    )
    assert projection_grad_scratch.shape == (image_points.shape[0], 5)


def test_round_trip_projection(ideal_projection, no_external):
    """Verify that back-projecting then re-projecting recovers the original image points with NoExternalDistortion."""
    image_points = make_grid_image_points(ideal_projection.focal_length.device)
    rays = image_points_to_camera_rays(
        image_points, ideal_projection, no_external, allow_device_transfer=True
    )
    roundtrip, valid = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    assert torch.allclose(roundtrip, image_points, atol=1e-4)
    assert valid.all()


def test_round_trip_projection_bivariate(ideal_projection, windshield_distortion):
    """Verify that back-projecting then re-projecting recovers the original image points with identity windshield distortion."""
    image_points = make_grid_image_points(ideal_projection.focal_length.device)
    rays = image_points_to_camera_rays(
        image_points,
        ideal_projection,
        windshield_distortion,
        allow_device_transfer=True,
    )
    roundtrip, valid = camera_rays_to_image_points(
        rays, ideal_projection, windshield_distortion, allow_device_transfer=True
    )
    assert valid.all()
    assert torch.allclose(roundtrip, image_points, atol=1e-3)


def test_bivariate_distortion_grad_accumulates_uniformly(
    ideal_projection, windshield_distortion, sensor_device
):
    """Verify that batched and chunked backward passes accumulate identical distortion coefficient gradients."""
    rays = torch.randn(64, 3, device=sensor_device) + torch.tensor(
        [0.0, 0.0, 2.0], device=sensor_device
    )

    coeffs_full = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion_full = BivariateWindshieldDistortion(
        coeffs_full,
        int(windshield_distortion.reference_polynomial),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    camera_rays_to_image_points(
        rays, ideal_projection, distortion_full, allow_device_transfer=True
    )[0].sum().backward()

    coeffs_split = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion_split = BivariateWindshieldDistortion(
        coeffs_split,
        int(windshield_distortion.reference_polynomial),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    for chunk in rays.split(16):
        camera_rays_to_image_points(
            chunk, ideal_projection, distortion_split, allow_device_transfer=True
        )[0].sum().backward()

    assert coeffs_full.grad is not None
    assert coeffs_split.grad is not None
    assert_grad_reference_close(
        coeffs_full.grad,
        coeffs_split.grad,
        rtol=1e-5,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="bivariate distortion coeff gradients",
    )


def test_bivariate_image_points_to_world_rays_shutter_pose_generated(
    ideal_projection, no_external, windshield_distortion, dynamic_pose
):
    """Verify that pixel_grid_to_world_rays_shutter_pose with identity windshield matches NoExternalDistortion output."""
    no_world, *_ = pixel_grid_to_world_rays_shutter_pose(
        ideal_projection,
        no_external,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        allow_device_transfer=True,
    )
    bi_world, *_ = pixel_grid_to_world_rays_shutter_pose(
        ideal_projection,
        windshield_distortion,
        (100, 80),
        ShutterType.GLOBAL,
        dynamic_pose,
        allow_device_transfer=True,
    )
    assert torch.allclose(bi_world, no_world, atol=1e-5)


def test_camera_model_to_moves_windshield_coeffs(
    pinhole_model_with_windshield,
):
    """Verify that CameraModel.cpu() moves windshield distortion coefficients to CPU and returns a new object."""
    original_distortion = pinhole_model_with_windshield.external_distortion
    moved = pinhole_model_with_windshield.cpu()
    assert (
        moved.external_distortion.distortion_coeffs.device
        == moved.projection.focal_length.device
    )
    assert moved.external_distortion.distortion_coeffs.device.type == "cpu"
    assert moved.external_distortion is not original_distortion


def test_principal_point_projects_to_optical_axis(ideal_projection, no_external):
    """Verify that back-projecting the principal point produces a ray along the optical axis (0,0,1)."""
    ray = image_points_to_camera_rays(
        ideal_projection.principal_point.reshape(1, 2),
        ideal_projection,
        no_external,
        allow_device_transfer=True,
    )
    assert torch.allclose(
        ray, torch.tensor([[0.0, 0.0, 1.0]], device=ray.device), atol=1e-6
    )


def test_behind_camera_invalid(ideal_projection, no_external):
    """Verify that a ray pointing behind the camera (z < 0) is flagged as invalid."""
    rays = torch.tensor([[0.0, 0.0, -1.0]], device=ideal_projection.focal_length.device)
    _, valid = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    assert not valid.item()


def test_generate_image_points(sensor_device):
    """Verify generate_image_points returns the correct shape and pixel-center coordinates."""
    pts = generate_image_points(
        (2, 3), device=sensor_device, allow_device_transfer=True
    )
    assert pts.shape == (3, 2, 2)
    assert torch.allclose(pts[0, 0], torch.tensor([0.5, 0.5], device=pts.device))


def test_project_world_points_mean_pose(ideal_projection, no_external, dynamic_pose):
    """Verify project_world_points_mean_pose returns the correct output shapes and mid-frame timestamp."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    image_points, valid, timestamps, poses_t, poses_r = project_world_points_mean_pose(
        world_points,
        ideal_projection,
        no_external,
        dynamic_pose,
        (100, 80),
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_valid_flags=True,
        return_timestamps=True,
        return_poses=True,
        allow_device_transfer=True,
    )
    assert image_points.shape == (1, 2)
    assert valid.shape == (1,)
    assert timestamps.item() == 5
    assert poses_t.shape == (1, 3)
    assert poses_r.shape == (1, 4)


def test_project_world_points_mean_pose_scratch_is_grad_gated(
    ideal_projection, no_external, dynamic_pose
):
    """Verify the scratch tensor from project_world_points_mean_pose is empty without grad and sized correctly with grad."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose,
        world_points.device,
        torch.float32,
        allow_device_transfer=True,
    )

    op = (
        torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_pinhole_no_external
    )

    *_, scratch = op(
        ideal_projection,
        no_external,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        0,
        10,
    )
    assert scratch.numel() == 0

    *_, grad_scratch = op(
        ideal_projection,
        no_external,
        world_points.requires_grad_(True),
        start_t,
        start_r,
        end_t,
        end_r,
        0,
        10,
    )
    assert grad_scratch.shape == (1, 9)


def test_project_world_points_shutter_pose_scratch_is_grad_gated(
    ideal_projection, no_external, dynamic_pose
):
    """Verify the scratch tensor from project_world_points_shutter_pose is empty without grad and sized correctly with grad."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose,
        world_points.device,
        torch.float32,
        allow_device_transfer=True,
    )
    op = (
        torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_pinhole_no_external
    )

    *_, scratch = op(
        ideal_projection,
        no_external,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
        10,
        0.01,
        0.01,
        0.5,
    )
    assert scratch.numel() == 0

    *_, grad_scratch = op(
        ideal_projection,
        no_external,
        world_points.requires_grad_(True),
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
        10,
        0.01,
        0.01,
        0.5,
    )
    assert grad_scratch.shape == (1, 10)


@pytest.mark.parametrize(
    "shutter_type",
    [
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        ShutterType.ROLLING_LEFT_TO_RIGHT,
        ShutterType.ROLLING_BOTTOM_TO_TOP,
        ShutterType.ROLLING_RIGHT_TO_LEFT,
    ],
)
def test_project_world_points_shutter_pose(
    ideal_projection, no_external, dynamic_pose, shutter_type
):
    """Verify project_world_points_shutter_pose returns correct output shapes for each rolling-shutter direction."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    image_points, valid, timestamps, *_ = project_world_points_shutter_pose(
        world_points,
        ideal_projection,
        no_external,
        (100, 80),
        shutter_type,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_valid_flags=True,
        return_timestamps=True,
        allow_device_transfer=True,
    )
    assert image_points.shape == (1, 2)
    assert valid.shape == (1,)
    assert timestamps.shape == (1,)


def test_image_points_to_world_rays_static_pose(
    ideal_projection, no_external, static_pose
):
    """Verify image_points_to_world_rays_static_pose returns an optical-axis ray for the principal point and correct output shapes."""
    image_points = ideal_projection.principal_point.reshape(1, 2)
    world_rays, timestamps, pose_t, pose_r = image_points_to_world_rays_static_pose(
        image_points,
        ideal_projection,
        no_external,
        static_pose,
        timestamp_us=7,
        return_timestamps=True,
        return_poses=True,
        allow_device_transfer=True,
    )
    assert torch.allclose(
        world_rays[:, 3:], torch.tensor([[0.0, 0.0, 1.0]], device=world_rays.device)
    )
    assert timestamps.item() == 7
    assert pose_t.shape == (1, 3)
    assert pose_r.shape == (1, 4)


def test_image_points_to_world_rays_static_pose_scratch_is_grad_gated(
    ideal_projection, no_external, static_pose
):
    """Verify the scratch tensor from image_points_to_world_rays_static_pose is empty without grad and sized correctly with grad."""
    image_points = ideal_projection.principal_point.reshape(1, 2).detach().clone()
    translations, rotations, _ = _unpack_static_pose(
        static_pose,
        image_points.device,
        torch.float32,
        allow_device_transfer=True,
    )
    op = (
        torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_pinhole_no_external
    )

    *_, scratch = op(
        ideal_projection, no_external, image_points, translations, rotations, 7
    )
    assert scratch.numel() == 0

    *_, grad_scratch = op(
        ideal_projection,
        no_external,
        image_points.requires_grad_(True),
        translations,
        rotations,
        7,
    )
    assert grad_scratch.shape == (1, 5)


def test_image_points_to_world_rays_shutter_pose(
    ideal_projection, no_external, dynamic_pose
):
    """Verify image_points_to_world_rays_shutter_pose returns world rays and timestamps with the correct shapes."""
    image_points = ideal_projection.principal_point.reshape(1, 2)
    world_rays, timestamps, *_ = image_points_to_world_rays_shutter_pose(
        image_points,
        ideal_projection,
        no_external,
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_timestamps=True,
        allow_device_transfer=True,
    )
    assert world_rays.shape == (1, 6)
    assert timestamps.shape == (1,)


def test_image_points_to_world_rays_shutter_pose_scratch_is_grad_gated(
    ideal_projection, no_external, dynamic_pose
):
    """Verify the scratch tensor from image_points_to_world_rays_shutter_pose is empty without grad and sized correctly with grad."""
    image_points = ideal_projection.principal_point.reshape(1, 2).detach().clone()
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose,
        image_points.device,
        torch.float32,
        allow_device_transfer=True,
    )
    op = (
        torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external
    )

    *_, scratch = op(
        ideal_projection,
        no_external,
        image_points,
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
    )
    assert scratch.numel() == 0

    *_, grad_scratch = op(
        ideal_projection,
        no_external,
        image_points.requires_grad_(True),
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
    )
    assert grad_scratch.shape == (1, 9)


def test_camera_rays_to_image_points_gradcheck(ideal_projection, no_external):
    """End-to-end autograd.gradcheck for camera_rays_to_image_points with NoExternalDistortion."""
    rays = torch.tensor(
        [[0.1, 0.2, 1.0], [-0.1, 0.05, 1.2]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(camera_rays):
        return camera_rays_to_image_points(
            camera_rays,
            ideal_projection,
            no_external,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (rays,), **GRADCHECK_KWARGS)


def test_intrinsics_gradient(ideal_projection, no_external):
    """Verify that focal_length gradient from camera_rays_to_image_points equals the ray xy components."""
    ideal_projection.focal_length.requires_grad_(True)
    rays = torch.tensor([[0.1, 0.2, 1.0]], device=ideal_projection.focal_length.device)
    image_points, _ = camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    image_points.sum().backward()
    assert ideal_projection.focal_length.grad is not None
    assert_grad_reference_close(
        ideal_projection.focal_length.grad,
        torch.tensor([0.1, 0.2], device=rays.device),
        rtol=1e-5,
        atol=1e-8,
        max_rel_l2=1e-5,
        max_rel_l1=1e-5,
        min_cosine=1.0 - 1e-10,
        max_signed_bias=1e-5,
        msg="pinhole focal_length.grad",
    )


def test_pose_gradient_is_correct(ideal_projection, no_external, dynamic_pose):
    """Verify that gradients flow back to start and end pose translations through project_world_points_mean_pose."""
    dynamic_pose.start_pose.translation.requires_grad_(True)
    dynamic_pose.end_pose.translation.requires_grad_(True)
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    image_points, *_ = project_world_points_mean_pose(
        world_points,
        ideal_projection,
        no_external,
        dynamic_pose,
        (100, 80),
        allow_device_transfer=True,
    )
    image_points.sum().backward()
    assert dynamic_pose.start_pose.translation.grad is not None
    assert dynamic_pose.end_pose.translation.grad is not None


def test_project_world_points_mean_pose_gradcheck(
    ideal_projection, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for project_world_points_mean_pose with NoExternalDistortion."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, -0.2, 1.5]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(points):
        return project_world_points_mean_pose(
            points,
            ideal_projection,
            no_external,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (world_points,), **GRADCHECK_KWARGS)


def test_project_world_points_shutter_pose_gradcheck(
    ideal_projection, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for project_world_points_shutter_pose with NoExternalDistortion."""
    world_points = torch.tensor(
        [[0.1, 0.123, 2.0], [-0.15, 0.07, 2.5]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(points):
        return project_world_points_shutter_pose(
            points,
            ideal_projection,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (world_points,), **GRADCHECK_KWARGS)


def test_quat_slerp_wxyz_gradcheck_equal_rotation(sensor_device):
    """End-to-end autograd.gradcheck for _quat_slerp_wxyz when both input quaternions are identical."""
    q0 = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=sensor_device,
        requires_grad=True,
    )
    q1 = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=sensor_device,
        requires_grad=True,
    )
    alpha = torch.tensor([0.5], dtype=torch.float64, device=sensor_device)

    def fn(a, b):
        return _quat_slerp_wxyz(a, b, alpha)

    assert torch.autograd.gradcheck(fn, (q0, q1), **GRADCHECK_KWARGS)


def test_quat_slerp_wxyz_no_nan_grad_on_equal_rotation(sensor_device):
    """Verify that _quat_slerp_wxyz produces finite gradients when both input quaternions are the same."""
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sensor_device, requires_grad=True)
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sensor_device, requires_grad=True)
    alpha = torch.tensor([0.5], device=sensor_device)
    out = _quat_slerp_wxyz(q0, q1, alpha)
    out.sum().backward()
    assert torch.isfinite(q0.grad).all()
    assert torch.isfinite(q1.grad).all()


def test_pose_dataclass_unpacked_to_separated_tensors(
    static_pose, dynamic_pose, sensor_device
):
    """Verify that _unpack_static_pose and unpack_dynamic_pose_components produce tensors with the expected shapes."""
    trans, rots, _ = _unpack_static_pose(static_pose, sensor_device, torch.float32)
    assert trans.shape == (1, 3)
    assert rots[0, 0] == 1
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, sensor_device, torch.float32
    )
    assert start_t.shape == (3,)
    assert end_t.shape == (3,)
    assert start_r.shape == (4,)
    assert end_r.shape == (4,)
    assert start_r[0] == 1
    assert end_r[0] == 1


def test_dynamic_pose_from_static_pose_clones_end_pose(static_pose):
    """Verify that DynamicPose.from_static_pose shares start_pose by reference but deep-copies end_pose."""
    dynamic_pose = DynamicPose.from_static_pose(static_pose)
    assert dynamic_pose.start_pose is static_pose
    assert dynamic_pose.end_pose is not static_pose
    assert (
        dynamic_pose.end_pose.translation.data_ptr()
        != static_pose.translation.data_ptr()
    )
    assert dynamic_pose.end_pose.rotation.data_ptr() != static_pose.rotation.data_ptr()


def test_relative_frame_times_raises_on_unknown_shutter(sensor_device):
    """Verify that relative_frame_times raises ValueError for an unrecognised ShutterType integer."""
    pts = torch.tensor([[0.5, 0.5]], device=sensor_device)
    with pytest.raises(ValueError, match="Unsupported ShutterType"):
        relative_frame_times(pts, (100, 80), 99)


def _make_distortion_coeffs_f64(windshield_distortion) -> torch.Tensor:
    """Clone windshield distortion_coeffs as float64 with requires_grad=True for gradcheck inputs."""
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().to(dtype=torch.float64)
    )
    coeffs.requires_grad_(True)
    return coeffs


def _build_distortion(
    windshield_distortion,
    reference_polynomial: ReferencePolynomial,
    distortion_coeffs: torch.Tensor,
):
    """Construct a BivariateWindshieldDistortion reusing h/v degrees from the fixture but with provided coeffs."""
    return BivariateWindshieldDistortion(
        distortion_coeffs,
        int(reference_polynomial),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_camera_rays_to_image_points_gradcheck(
    ideal_projection, windshield_distortion, reference_polynomial
):
    """End-to-end autograd.gradcheck for camera_rays_to_image_points with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    rays = torch.tensor(
        [[0.1, 0.2, 1.0], [-0.1, 0.05, 1.2]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(camera_rays, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return camera_rays_to_image_points(
            camera_rays, ideal_projection, distortion, allow_device_transfer=True
        )[0]

    assert torch.autograd.gradcheck(fn, (rays, distortion_coeffs), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_image_points_to_camera_rays_gradcheck(
    ideal_projection, windshield_distortion, reference_polynomial
):
    """End-to-end autograd.gradcheck for image_points_to_camera_rays with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    image_points = (
        (
            ideal_projection.principal_point.reshape(1, 2).to(dtype=torch.float64)
            + torch.tensor(
                [[1.0, -2.0], [3.0, 4.0]],
                device=ideal_projection.focal_length.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(pts, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return image_points_to_camera_rays(
            pts, ideal_projection, distortion, allow_device_transfer=True
        )

    assert torch.autograd.gradcheck(
        fn, (image_points, distortion_coeffs), **GRADCHECK_KWARGS
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_project_world_points_mean_pose_gradcheck(
    ideal_projection, windshield_distortion, dynamic_pose, reference_polynomial
):
    """End-to-end autograd.gradcheck for project_world_points_mean_pose with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, -0.2, 1.5]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(points, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return project_world_points_mean_pose(
            points,
            ideal_projection,
            distortion,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (world_points, distortion_coeffs), **GRADCHECK_KWARGS_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_project_world_points_shutter_pose_gradcheck(
    ideal_projection, windshield_distortion, dynamic_pose, reference_polynomial
):
    """End-to-end autograd.gradcheck for project_world_points_shutter_pose with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    # Plan §7: use max_iterations=1 to keep finite-difference checks stable
    # on the rolling-shutter inner-loop.
    world_points = torch.tensor(
        [[0.1, 0.123, 2.0], [-0.15, 0.07, 2.5]],
        device=ideal_projection.focal_length.device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(points, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return project_world_points_shutter_pose(
            points,
            ideal_projection,
            distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (world_points, distortion_coeffs), **GRADCHECK_KWARGS_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_image_points_to_world_rays_static_pose_gradcheck(
    ideal_projection, windshield_distortion, static_pose, reference_polynomial
):
    """End-to-end autograd.gradcheck for image_points_to_world_rays_static_pose with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    image_points = (
        (
            ideal_projection.principal_point.reshape(1, 2).to(dtype=torch.float64)
            + torch.tensor(
                [[1.0, -2.0], [3.0, 4.0]],
                device=ideal_projection.focal_length.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(pts, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return image_points_to_world_rays_static_pose(
            pts,
            ideal_projection,
            distortion,
            static_pose,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (image_points, distortion_coeffs), **GRADCHECK_KWARGS
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_image_points_to_world_rays_shutter_pose_gradcheck(
    ideal_projection, windshield_distortion, dynamic_pose, reference_polynomial
):
    # GLOBAL shutter makes the pose-time path independent of image_points, so
    # gradcheck can cover the direct image-point and distortion VJPs.
    image_points = (
        ideal_projection.principal_point.reshape(1, 2).to(dtype=torch.float64)
        + torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]],
            device=ideal_projection.focal_length.device,
            dtype=torch.float64,
        )
    ).detach()
    image_points.requires_grad_(True)
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(points, coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return image_points_to_world_rays_shutter_pose(
            points,
            ideal_projection,
            distortion,
            (100, 80),
            ShutterType.GLOBAL,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (image_points, distortion_coeffs), **GRADCHECK_KWARGS
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_bivariate_image_points_to_world_rays_shutter_pose_image_points_grad_flows(
    ideal_projection, windshield_distortion, dynamic_pose, reference_polynomial
):
    """Verify rolling shutter keeps the direct image-point VJP for bivariate windshield."""
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)
    distortion = _build_distortion(
        windshield_distortion, reference_polynomial, distortion_coeffs
    )
    image_points = (
        ideal_projection.principal_point.reshape(1, 2).to(dtype=torch.float64)
        + torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]],
            device=ideal_projection.focal_length.device,
            dtype=torch.float64,
        )
    ).detach()
    image_points.requires_grad_(True)

    world_rays = image_points_to_world_rays_shutter_pose(
        image_points,
        ideal_projection,
        distortion,
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        allow_device_transfer=True,
    )[0]
    loss = world_rays[:, 3].sum() + 0.25 * world_rays[:, 4].sum()
    loss.backward()

    assert image_points.grad is not None
    assert image_points.grad.abs().sum() > 0


@pytest.mark.gradcheck
def test_bivariate_camera_rays_to_image_points_intrinsics_gradcheck(
    windshield_distortion, sensor_device
):
    """End-to-end autograd.gradcheck for camera_rays_to_image_points covering all intrinsic tensors alongside distortion coefficients."""
    # Covers gradcheck on intrinsics tensors alongside distortion_coeffs for
    # the K3 forward verb. Other verbs share the same OpenCVPinholeProjection
    # intrinsics path so this single coverage is sufficient.
    focal_length = torch.tensor(
        [100.0, 120.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    principal_point = torch.tensor(
        [50.0, 40.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    radial_coeffs = torch.zeros(
        6, device=sensor_device, dtype=torch.float64, requires_grad=True
    )
    tangential_coeffs = torch.zeros(
        2, device=sensor_device, dtype=torch.float64, requires_grad=True
    )
    thin_prism_coeffs = torch.zeros(
        4, device=sensor_device, dtype=torch.float64, requires_grad=True
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)
    rays = torch.tensor(
        [[0.1, 0.2, 1.0], [-0.1, 0.05, 1.2]],
        device=sensor_device,
        dtype=torch.float64,
    )

    def fn(focal, pp, radial, tangential, thin, coeffs):
        projection = OpenCVPinholeProjection(
            focal_length=focal,
            principal_point=pp,
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            thin_prism_coeffs=thin,
            resolution=(100, 80),
        )
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return camera_rays_to_image_points(
            rays, projection, distortion, allow_device_transfer=True
        )[0]

    assert torch.autograd.gradcheck(
        fn,
        (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            distortion_coeffs,
        ),
        **GRADCHECK_KWARGS,
    )


@pytest.mark.gradcheck
def test_bivariate_project_world_points_mean_pose_pose_gradcheck(
    ideal_projection, windshield_distortion, sensor_device
):
    """End-to-end autograd.gradcheck for project_world_points_mean_pose covering dynamic-pose tensors alongside distortion coefficients."""
    # Covers gradcheck on dynamic-pose tensors alongside distortion_coeffs for
    # K7. Other pose-using verbs (K9/K11/K13) share the same WXYZ pose ABI.
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, -0.2, 1.5]],
        device=sensor_device,
        dtype=torch.float64,
    )
    start_translation = torch.zeros(
        3, device=sensor_device, dtype=torch.float64, requires_grad=True
    )
    start_rotation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    end_translation = torch.tensor(
        [0.1, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    end_rotation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(start_t, start_r, end_t, end_r, coeffs):
        from gsplat.sensors.kernels.common import DynamicPose, Pose

        pose = DynamicPose(
            Pose(translation=start_t, rotation=start_r),
            Pose(translation=end_t, rotation=end_r),
        )
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return project_world_points_mean_pose(
            world_points,
            ideal_projection,
            distortion,
            pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn,
        (
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            distortion_coeffs,
        ),
        **GRADCHECK_KWARGS,
    )


# ===========================================================================
# F-Theta-specific test coverage for the
# (FThetaProjection, NoExternalDistortion) and
# (FThetaProjection, BivariateWindshieldDistortion) pairs. Parametrized
# tests cover both reference_polynomial branches (FORWARD-ref direct and
# BACKWARD-ref Newton inversion through the implicit-function adjoint).
# ===========================================================================


def _ftheta_synthetic_rays(device: torch.device, dtype: torch.dtype = torch.float32):
    return torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 1.0],
            [0.0, 0.04, 1.0],
            [-0.05, 0.04, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def test_camera_rays_to_image_points_ftheta_no_external_basic(
    ftheta_projection, no_external
):
    """Verify camera_rays_to_image_points returns finite in-frame points for the synthetic FTheta ray set."""
    rays = _ftheta_synthetic_rays(ftheta_projection.principal_point.device)
    image_points, valid_flags = camera_rays_to_image_points(
        rays, ftheta_projection, no_external
    )
    assert image_points.shape == (4, 2)
    assert valid_flags.dtype == torch.bool
    assert valid_flags.all()
    pp = ftheta_projection.principal_point
    assert torch.allclose(image_points[0], pp, atol=1e-3)


def test_image_points_to_camera_rays_ftheta_round_trip(ftheta_projection, no_external):
    """Verify project -> unproject -> project recovers the original FTheta image points."""
    rays = _ftheta_synthetic_rays(ftheta_projection.principal_point.device)
    image_points, _ = camera_rays_to_image_points(rays, ftheta_projection, no_external)
    rays_back = image_points_to_camera_rays(
        image_points, ftheta_projection, no_external
    )
    image_points_back, _ = camera_rays_to_image_points(
        rays_back, ftheta_projection, no_external
    )
    assert torch.allclose(image_points_back, image_points, atol=1e-3)


def test_camera_rays_to_image_points_ftheta_matches_torch_oracle(
    ftheta_projection, no_external
):
    """Verify camera_rays_to_image_points matches the pure-torch FTheta forward oracle."""
    rays = _ftheta_synthetic_rays(ftheta_projection.principal_point.device)
    image_points, _ = camera_rays_to_image_points(rays, ftheta_projection, no_external)
    expected = torch_ftheta_project(
        rays.to(torch.float64),
        ftheta_projection.principal_point.to(torch.float64),
        ftheta_projection.fw_poly.to(torch.float64),
        ftheta_projection.bw_poly.to(torch.float64),
        ftheta_projection.A.to(torch.float64),
        int(ftheta_projection.reference_polynomial),
        int(ftheta_projection.fw_poly_degree),
        int(ftheta_projection.bw_poly_degree),
        int(ftheta_projection.newton_iterations),
        float(ftheta_projection.max_angle),
        float(ftheta_projection.min_2d_norm),
    )
    assert torch.allclose(
        image_points, expected.to(torch.float32), atol=1e-3, rtol=1e-5
    )


def test_image_points_to_camera_rays_ftheta_matches_torch_oracle(
    ftheta_projection, no_external
):
    """Verify image_points_to_camera_rays matches the pure-torch FTheta back-projection oracle."""
    pp = ftheta_projection.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor(
        [[0.0, 0.0], [3.0, 0.0], [0.0, -2.0], [4.0, -3.0]],
        device=pp.device,
    )
    rays = image_points_to_camera_rays(image_points, ftheta_projection, no_external)
    expected = torch_ftheta_backproject(
        image_points.to(torch.float64),
        ftheta_projection.principal_point.to(torch.float64),
        ftheta_projection.fw_poly.to(torch.float64),
        ftheta_projection.bw_poly.to(torch.float64),
        ftheta_projection.Ainv.to(torch.float64),
        int(ftheta_projection.reference_polynomial),
        int(ftheta_projection.fw_poly_degree),
        int(ftheta_projection.bw_poly_degree),
        int(ftheta_projection.newton_iterations),
        float(ftheta_projection.min_2d_norm),
    )
    assert torch.allclose(rays, expected.to(torch.float32), atol=1e-3, rtol=1e-5)


def test_camera_rays_to_image_points_ftheta_principal_point_invariance(
    ftheta_projection, no_external
):
    """The optical-axis ray (0,0,1) MUST land on the principal point within atol=1e-4."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]],
        device=ftheta_projection.principal_point.device,
    )
    image_points, valid_flags = camera_rays_to_image_points(
        rays, ftheta_projection, no_external
    )
    assert valid_flags.all()
    assert torch.allclose(
        image_points,
        ftheta_projection.principal_point.reshape(1, 2),
        atol=1e-4,
    )


def test_camera_rays_to_image_points_ftheta_behind_camera_invalid(
    ftheta_projection, no_external
):
    """Rays with z<=0 MUST be marked invalid (the kernel falls back to PP)."""
    rays = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # forward, valid
            [0.0, 0.0, -1.0],  # behind, invalid
        ],
        device=ftheta_projection.principal_point.device,
    )
    _, valid_flags = camera_rays_to_image_points(rays, ftheta_projection, no_external)
    assert valid_flags.tolist() == [True, False]


def test_camera_rays_to_image_points_ftheta_out_of_frame_marks_invalid(
    ftheta_projection_forward_ref, no_external
):
    """Verify a ray projecting outside ``[0, W) x [0, H)`` is flagged invalid even
    when the projection itself succeeds (z > 0, theta well below max_angle).
    The bounds check is applied at the kernel write site so ``valid_flags``
    carries in-frame-and-finite semantics for downstream filtering.
    """
    proj = ftheta_projection_forward_ref
    width, height = int(proj.resolution[0]), int(proj.resolution[1])
    pp = proj.principal_point
    device = pp.device
    in_frame_ray = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    # tan(0.78) ~= 0.99; ray (1.0, 0.0, 1.0) gives theta ~= 0.785 rad.
    far_off_axis = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    rays = torch.cat([in_frame_ray, far_off_axis], dim=0)
    image_points, valid_flags = camera_rays_to_image_points(rays, proj, no_external)
    assert bool(valid_flags[0]) is True
    px = image_points[1]
    in_x = (0.0 <= float(px[0])) and (float(px[0]) < width)
    in_y = (0.0 <= float(px[1])) and (float(px[1]) < height)
    assert not (
        in_x and in_y
    ), f"test ray was supposed to land off-frame; got {(float(px[0]), float(px[1]))} for resolution {(width, height)}"
    assert bool(valid_flags[1]) is False, (
        "out-of-frame pixel must be flagged invalid; got "
        f"valid_flags={valid_flags.tolist()}, image_points={image_points.tolist()}"
    )


def test_camera_rays_to_image_points_ftheta_nan_input_marks_invalid(
    ftheta_projection_forward_ref, no_external
):
    """Verify a NaN-bearing camera ray is flagged invalid by the bounds-and-finite
    gate at the kernel write site (the forward kernel does not gracefully
    convert NaN inputs into a defined pixel).
    """
    proj = ftheta_projection_forward_ref
    device = proj.principal_point.device
    rays = torch.tensor(
        [[float("nan"), 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    _, valid_flags = camera_rays_to_image_points(rays, proj, no_external)
    assert (
        bool(valid_flags[0]) is False
    ), f"NaN-bearing ray must be flagged invalid; got valid_flags={valid_flags.tolist()}"


def _ftheta_dynamic_pose_tensors(pose: DynamicPose, device: torch.device):
    return unpack_dynamic_pose_components(
        pose,
        device,
        torch.float32,
        allow_device_transfer=True,
    )


@pytest.mark.parametrize(
    "distortion_fixture, op_suffix, expected_strides",
    [
        ("no_external", "no_external", (8, 8, 10, 11, 8, 9)),
        (
            "windshield_distortion",
            "bivariate_windshield",
            (8, 12, 10, 11, 12, 12),
        ),
    ],
)
def test_ftheta_forward_scratch_abi_all_op_families(
    request,
    distortion_fixture,
    op_suffix,
    expected_strides,
    ftheta_projection_forward_ref,
    dynamic_pose,
    static_pose,
):
    """Every FTheta forward policy variant must preserve grad gating and stride."""
    distortion = request.getfixturevalue(distortion_fixture)
    device = ftheta_projection_forward_ref.principal_point.device
    rays = _ftheta_synthetic_rays(device).detach()
    image_points = (
        ftheta_projection_forward_ref.principal_point.reshape(1, 2)
        + torch.tensor([[2.0, 1.0], [-1.5, 0.7]], device=device)
    ).detach()
    world_points = torch.tensor([[0.1, 0.05, 1.6], [-0.1, 0.08, 2.1]], device=device)
    start_t, start_r, end_t, end_r = _ftheta_dynamic_pose_tensors(dynamic_pose, device)
    translations, rotations, _ = _unpack_static_pose(
        static_pose,
        device,
        torch.float32,
        allow_device_transfer=True,
    )

    cases = [
        ("camera_rays_to_image_points", rays, ()),
        ("image_points_to_camera_rays", image_points, ()),
        (
            "project_world_points_mean_pose",
            world_points,
            (start_t, start_r, end_t, end_r, 0, 10),
        ),
        (
            "project_world_points_shutter_pose",
            world_points,
            (
                start_t,
                start_r,
                end_t,
                end_r,
                100,
                80,
                int(ShutterType.ROLLING_TOP_TO_BOTTOM),
                0,
                10,
                1,
                0.01,
                0.01,
                0.5,
            ),
        ),
        (
            "image_points_to_world_rays_static_pose",
            image_points,
            (translations, rotations, 7),
        ),
        (
            "image_points_to_world_rays_shutter_pose",
            image_points,
            (
                start_t,
                start_r,
                end_t,
                end_r,
                100,
                80,
                int(ShutterType.ROLLING_TOP_TO_BOTTOM),
                0,
                10,
            ),
        ),
    ]

    for (family, primary_input, extra_args), expected_stride in zip(
        cases, expected_strides, strict=True
    ):
        op = getattr(torch.ops.gsplat_sensors, f"{family}_ftheta_{op_suffix}")
        *_, scratch = op(
            ftheta_projection_forward_ref,
            distortion,
            primary_input,
            *extra_args,
        )
        assert scratch.numel() == 0, family

        grad_input = primary_input.detach().clone().requires_grad_(True)
        *_, grad_scratch = op(
            ftheta_projection_forward_ref,
            distortion,
            grad_input,
            *extra_args,
        )
        assert grad_scratch.shape == (primary_input.shape[0], expected_stride), family


@pytest.mark.gradcheck
def test_camera_rays_to_image_points_ftheta_no_external_gradcheck(
    ftheta_projection, no_external
):
    """End-to-end autograd.gradcheck for FTheta no-external camera_rays -> image_points across both reference polynomial branches."""
    # Avoid the exact optical axis: that path intentionally hits the min-2D
    # clamp and returns the principal point, which is not finite-difference
    # smooth around x=y=0.
    rays = torch.tensor(
        [[0.03, 0.02, 1.0], [0.05, -0.01, 1.1], [-0.04, 0.03, 1.2]],
        device=ftheta_projection.principal_point.device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(camera_rays):
        return camera_rays_to_image_points(
            camera_rays,
            ftheta_projection,
            no_external,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (rays,), **GRADCHECK_KWARGS_FTHETA_PROJECT)


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "projection_fixture",
    ["ftheta_projection_forward_ref", "ftheta_projection_backward_ref"],
)
def test_ftheta_bivariate_camera_rays_to_image_points_nonidentity_gradcheck(
    request, projection_fixture, windshield_distortion
):
    """Exercise FTheta bivariate forward scratch replay through public autograd."""
    projection = request.getfixturevalue(projection_fixture)
    rays = torch.tensor(
        [[0.04, 0.03, 1.0], [-0.05, 0.02, 1.1]],
        device=projection.principal_point.device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)
    with torch.no_grad():
        distortion_coeffs[0] += 0.01
        distortion_coeffs[7] += 0.002

    def fn(camera_rays, coeffs):
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return camera_rays_to_image_points(
            camera_rays,
            projection,
            distortion,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn,
        (rays, distortion_coeffs),
        **GRADCHECK_KWARGS_FTHETA_PROJECT,
    )


@pytest.mark.gradcheck
def test_image_points_to_camera_rays_ftheta_no_external_gradcheck(
    ftheta_projection, no_external
):
    """End-to-end autograd.gradcheck for FTheta no-external image_points -> camera_rays across both reference polynomial branches."""
    pp = ftheta_projection.principal_point
    image_points = (
        (
            pp.reshape(1, 2).to(torch.float64)
            + torch.tensor(
                [[0.8, -0.6], [1.3, 0.9]],
                device=pp.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )

    def fn(pts):
        return image_points_to_camera_rays(
            pts,
            ftheta_projection,
            no_external,
            allow_device_transfer=True,
        )

    assert torch.autograd.gradcheck(fn, (image_points,), **GRADCHECK_KWARGS)


def _ftheta_projection_from_grad_tensors(
    base_projection,
    principal_point: Tensor,
    fw_poly: Tensor,
    bw_poly: Tensor,
    A: Tensor,
) -> FThetaProjection:
    return FThetaProjection(
        principal_point=principal_point,
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        A=A,
        reference_polynomial=int(base_projection.reference_polynomial),
        resolution=base_projection.resolution,
        fw_poly_degree=int(base_projection.fw_poly_degree),
        bw_poly_degree=int(base_projection.bw_poly_degree),
        newton_iterations=int(base_projection.newton_iterations),
        max_angle=float(base_projection.max_angle),
        min_2d_norm=float(base_projection.min_2d_norm),
    )


def _ftheta_projection_from_components(
    *,
    principal_point: Tensor,
    fw_poly: Tensor,
    bw_poly: Tensor,
    A: Tensor,
    reference_polynomial: int,
    fw_poly_degree: int,
    bw_poly_degree: int,
) -> FThetaProjection:
    return FThetaProjection(
        principal_point=principal_point,
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        A=A,
        reference_polynomial=int(reference_polynomial),
        resolution=(256, 256),
        fw_poly_degree=int(fw_poly_degree),
        bw_poly_degree=int(bw_poly_degree),
        newton_iterations=8,
        max_angle=1.4,
        min_2d_norm=1e-6,
    )


def _ftheta_intrinsic_grad_inputs(base_projection):
    return tuple(
        tensor.detach().clone().to(torch.float64).requires_grad_(True)
        for tensor in (
            base_projection.principal_point,
            base_projection.fw_poly,
            base_projection.bw_poly,
            base_projection.A,
        )
    )


def test_camera_rays_to_image_points_ftheta_saturated_branch_backward(no_external):
    """Force the BACKWARD-ref safe_df=1 branch and verify bw_poly adjoints."""
    device = torch.device("cuda")
    principal_point = torch.zeros(2, device=device)
    fw_poly = torch.zeros(6, device=device)
    fw_poly[1] = 10.0
    bw_poly = torch.zeros(6, device=device, requires_grad=True)
    A = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
    projection = _ftheta_projection_from_components(
        principal_point=principal_point,
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        A=A,
        reference_polynomial=int(ReferencePolynomial.BACKWARD),
        fw_poly_degree=1,
        bw_poly_degree=5,
    )
    camera_rays = torch.tensor([[0.1, 0.0, 1.0]], device=device, requires_grad=True)

    image_points, valid = camera_rays_to_image_points(
        camera_rays, projection, no_external
    )
    assert valid.item()
    image_points[:, 0].sum().backward()

    theta = torch.acos(
        camera_rays.detach()[0, 2] / torch.linalg.norm(camera_rays.detach()[0])
    )
    r_star = fw_poly.detach()[1] * theta
    expected_bw_grad = -(r_star ** torch.arange(6, device=device))
    assert_grad_reference_close(
        bw_poly.grad,
        expected_bw_grad,
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="bw_poly saturated-branch gradient",
    )
    assert camera_rays.grad is not None
    assert torch.isfinite(camera_rays.grad).all()
    assert camera_rays.grad.abs().sum() > 0


def test_image_points_to_camera_rays_ftheta_saturated_branch_backward(no_external):
    """Force the FORWARD-ref safe_df=1 branch and verify fw_poly adjoints."""
    device = torch.device("cuda")
    principal_point = torch.zeros(2, device=device)
    fw_poly = torch.zeros(6, device=device, requires_grad=True)
    bw_poly = torch.zeros(6, device=device)
    bw_poly[1] = 1.0
    A = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
    projection = _ftheta_projection_from_components(
        principal_point=principal_point,
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        A=A,
        reference_polynomial=int(ReferencePolynomial.FORWARD),
        fw_poly_degree=5,
        bw_poly_degree=1,
    )
    image_points = torch.tensor([[0.25, 0.0]], device=device, requires_grad=True)

    camera_rays = image_points_to_camera_rays(image_points, projection, no_external)
    torch.atan2(camera_rays[:, 0], camera_rays[:, 2]).sum().backward()

    theta_star = image_points.detach()[0, 0]
    expected_fw_grad = -(theta_star ** torch.arange(6, device=device))
    assert_grad_reference_close(
        fw_poly.grad,
        expected_fw_grad,
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="fw_poly saturated-branch gradient",
    )
    assert image_points.grad is not None
    assert torch.isfinite(image_points.grad).all()
    assert image_points.grad.abs().sum() > 0


@pytest.mark.gradcheck
def test_camera_rays_to_image_points_ftheta_intrinsics_gradcheck(
    ftheta_projection, no_external
):
    """FTheta intrinsic gradcheck for camera_rays_to_image_points over (principal_point, fw_poly, bw_poly, A)."""
    rays = _ftheta_synthetic_rays(
        ftheta_projection.principal_point.device, dtype=torch.float64
    )
    pp, fw, bw, A = _ftheta_intrinsic_grad_inputs(ftheta_projection)

    def fn(principal_point, fw_poly, bw_poly, A_tensor):
        projection = _ftheta_projection_from_grad_tensors(
            ftheta_projection,
            principal_point,
            fw_poly,
            bw_poly,
            A_tensor,
        )
        return camera_rays_to_image_points(
            rays,
            projection,
            no_external,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (pp, fw, bw, A), **GRADCHECK_KWARGS_FTHETA_PROJECT
    )


@pytest.mark.gradcheck
def test_image_points_to_camera_rays_ftheta_intrinsics_gradcheck(
    ftheta_projection, no_external
):
    """FTheta intrinsic gradcheck for image_points_to_camera_rays over (principal_point, fw_poly, bw_poly, A)."""
    pp0 = ftheta_projection.principal_point
    image_points = pp0.reshape(1, 2).to(torch.float64) + torch.tensor(
        [[0.8, -0.6], [1.3, 0.9]],
        device=pp0.device,
        dtype=torch.float64,
    )
    pp, fw, bw, A = _ftheta_intrinsic_grad_inputs(ftheta_projection)

    def fn(principal_point, fw_poly, bw_poly, A_tensor):
        projection = _ftheta_projection_from_grad_tensors(
            ftheta_projection,
            principal_point,
            fw_poly,
            bw_poly,
            A_tensor,
        )
        return image_points_to_camera_rays(
            image_points,
            projection,
            no_external,
            allow_device_transfer=True,
        )

    assert torch.autograd.gradcheck(
        fn, (pp, fw, bw, A), **GRADCHECK_KWARGS_FTHETA_PROJECT
    )


def test_ftheta_projection_ainv_structural_invariant(ftheta_projection):
    """Verify Ainv equals the true 2x2 inverse of A, with A @ Ainv == I."""
    A_flat = torch.tensor(
        [2.0, 0.25, -0.5, 1.5],
        device=ftheta_projection.A.device,
        dtype=ftheta_projection.A.dtype,
    )
    projection = _ftheta_projection_from_components(
        principal_point=ftheta_projection.principal_point,
        fw_poly=ftheta_projection.fw_poly,
        bw_poly=ftheta_projection.bw_poly,
        A=A_flat,
        reference_polynomial=int(ftheta_projection.reference_polynomial),
        fw_poly_degree=int(ftheta_projection.fw_poly_degree),
        bw_poly_degree=int(ftheta_projection.bw_poly_degree),
    )
    A = projection.A.view(2, 2)
    Ainv = projection.Ainv.view(2, 2)
    eye = torch.eye(2, device=A.device, dtype=A.dtype)
    assert torch.allclose(Ainv, torch.linalg.inv(A), atol=1e-5)
    assert torch.allclose(Ainv @ A, eye, atol=1e-5)
    assert torch.allclose(A @ Ainv, eye, atol=1e-5)
    assert projection.Ainv.shape == (4,)


def test_ftheta_projection_bindings_readonly(ftheta_projection):
    """Verify FThetaProjection tensor and resolution bindings are read-only.

    Rebinding attributes (e.g. ``proj.A = new_tensor``) must raise.
    """
    for attr in ("principal_point", "fw_poly", "bw_poly", "A"):
        with pytest.raises((AttributeError, RuntimeError)):
            setattr(
                ftheta_projection,
                attr,
                torch.zeros(1, device=ftheta_projection.A.device),
            )
    with pytest.raises((AttributeError, RuntimeError, TypeError)):
        ftheta_projection.resolution = (1, 1)


def _slerp_branch_rotations(branch: str, device: torch.device):
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    if branch == "ordinary":
        q1 = torch.tensor(
            [0.70710678, 0.0, 0.0, 0.70710678], device=device, dtype=torch.float64
        )
    elif branch == "nlerp":
        q1 = torch.tensor(
            [0.9999875, 0.0, 0.0, 0.00499998], device=device, dtype=torch.float64
        )
    elif branch == "hemisphere_flip":
        q1 = torch.tensor(
            [-0.70710678, 0.0, 0.0, -0.70710678], device=device, dtype=torch.float64
        )
    else:
        raise ValueError(f"unknown SLERP branch fixture: {branch}")
    return q0.requires_grad_(True), q1.requires_grad_(True)


def _pose_from_rotations(
    start_rotation: Tensor, end_rotation: Tensor, device: torch.device
) -> DynamicPose:
    zero_t = torch.zeros(3, device=device, dtype=torch.float64)
    end_t = torch.tensor([0.05, -0.02, 0.01], device=device, dtype=torch.float64)
    return DynamicPose(
        Pose(translation=zero_t, rotation=start_rotation),
        Pose(translation=end_t, rotation=end_rotation),
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_ftheta_mean_pose_rotation_slerp_branch_gradcheck(
    ftheta_projection_forward_ref, no_external, branch
):
    """Pose-gradient coverage for SLERP branches in FTheta mean-pose no-external projection."""
    device = ftheta_projection_forward_ref.principal_point.device
    points = _ftheta_world_points_f64(device, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return project_world_points_mean_pose(
            points,
            ftheta_projection_forward_ref,
            no_external,
            _pose_from_rotations(q0, q1, device),
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (start_r, end_r), **GRADCHECK_KWARGS_FTHETA_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_ftheta_shutter_pose_rotation_slerp_branch_gradcheck(
    ftheta_projection_forward_ref, no_external, branch
):
    """Pose-gradient coverage for SLERP branches in FTheta shutter-pose no-external projection."""
    device = ftheta_projection_forward_ref.principal_point.device
    points = _ftheta_world_points_f64(device, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return project_world_points_shutter_pose(
            points,
            ftheta_projection_forward_ref,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _pose_from_rotations(q0, q1, device),
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (start_r, end_r), **GRADCHECK_KWARGS_FTHETA_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_ftheta_world_rays_shutter_pose_rotation_slerp_branch_gradcheck(
    ftheta_projection_forward_ref, no_external, branch
):
    """Pose-gradient coverage for SLERP branches in FTheta image_points_to_world_rays_shutter_pose no-external."""
    device = ftheta_projection_forward_ref.principal_point.device
    pts = _ftheta_image_points_f64(ftheta_projection_forward_ref, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return image_points_to_world_rays_shutter_pose(
            pts,
            ftheta_projection_forward_ref,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _pose_from_rotations(q0, q1, device),
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (start_r, end_r), **GRADCHECK_KWARGS)


def test_ftheta_identity_windshield_matches_no_external(
    ftheta_projection_forward_ref, no_external, windshield_distortion
):
    """Identity windshield should match no_external within atol=1e-4."""
    rays = _ftheta_synthetic_rays(ftheta_projection_forward_ref.principal_point.device)
    no_ext_pts, _ = camera_rays_to_image_points(
        rays, ftheta_projection_forward_ref, no_external
    )
    bw_pts, _ = camera_rays_to_image_points(
        rays, ftheta_projection_forward_ref, windshield_distortion
    )
    assert torch.allclose(no_ext_pts, bw_pts, atol=1e-4, rtol=1e-5)


def test_real_ftheta_camera_round_trip(real_ftheta_projection, no_external):
    """Project -> unproject -> project round-trip on real F-Theta intrinsics.

    Uses 8 sample image points within the camera's valid radial range.
    Tolerance scaled to the F-Theta radial polynomial nonlinearity in float32.
    """
    pp = real_ftheta_projection.principal_point
    res_x, res_y = real_ftheta_projection.resolution
    offset_scale = min(res_x, res_y) * 0.1
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [offset_scale, 0.0],
            [-offset_scale, 0.0],
            [0.0, offset_scale],
            [0.0, -offset_scale],
            [offset_scale, offset_scale],
            [-offset_scale, -offset_scale],
            [0.5 * offset_scale, -0.5 * offset_scale],
        ],
        device=pp.device,
    )
    image_points = pp.reshape(1, 2) + offsets
    rays = image_points_to_camera_rays(
        image_points, real_ftheta_projection, no_external
    )
    image_points_back, _ = camera_rays_to_image_points(
        rays, real_ftheta_projection, no_external
    )
    assert torch.allclose(image_points_back, image_points, atol=1e-2, rtol=1e-4)


def test_real_ftheta_camera_principal_point_invariance(
    real_ftheta_projection, no_external
):
    """Optical-axis ray must land on the principal point on every real camera."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]], device=real_ftheta_projection.principal_point.device
    )
    image_points, valid_flags = camera_rays_to_image_points(
        rays, real_ftheta_projection, no_external
    )
    assert valid_flags.all()
    assert torch.allclose(
        image_points,
        real_ftheta_projection.principal_point.reshape(1, 2),
        atol=1e-3,
    )


def test_real_ftheta_camera_matches_torch_oracle(real_ftheta_projection, no_external):
    """Forward parity vs the pure-torch reference on real intrinsics.

    Tolerances atol=1e-2, rtol=1e-4: the kernel runs in float32 while the
    oracle runs in float64, so a small absolute mismatch in pixel space is
    expected. Anything larger indicates a structural CUDA bug.
    """
    rays = torch.tensor(
        [
            [0.05, 0.0, 1.0],
            [0.0, 0.04, 1.0],
            [-0.05, 0.04, 1.0],
            [0.1, -0.05, 1.0],
        ],
        device=real_ftheta_projection.principal_point.device,
    )
    image_points, valid_flags = camera_rays_to_image_points(
        rays, real_ftheta_projection, no_external
    )
    expected = torch_ftheta_project(
        rays.to(torch.float64),
        real_ftheta_projection.principal_point.to(torch.float64),
        real_ftheta_projection.fw_poly.to(torch.float64),
        real_ftheta_projection.bw_poly.to(torch.float64),
        real_ftheta_projection.A.to(torch.float64),
        int(real_ftheta_projection.reference_polynomial),
        int(real_ftheta_projection.fw_poly_degree),
        int(real_ftheta_projection.bw_poly_degree),
        int(real_ftheta_projection.newton_iterations),
        float(real_ftheta_projection.max_angle),
        float(real_ftheta_projection.min_2d_norm),
    )
    valid_mask = valid_flags
    assert torch.allclose(
        image_points[valid_mask],
        expected.to(torch.float32)[valid_mask],
        atol=1e-2,
        rtol=1e-4,
    )


def test_real_ftheta_windshield_diverges_from_no_external(
    real_ftheta_projection_with_windshield,
    real_ftheta_windshield_distortion,
    no_external,
):
    """Verify real bivariate-windshield distortion produces visibly different
    image points from ``no_external`` for off-axis rays, proving the
    coefficients are wired through dispatch (an identity windshield or
    wiring bug would silently match no_external).
    """
    proj = real_ftheta_projection_with_windshield
    pp = proj.principal_point
    device = pp.device
    rays = torch.tensor(
        [
            [0.05, 0.0, 1.0],
            [0.0, 0.04, 1.0],
            [-0.05, 0.04, 1.0],
            [0.1, -0.05, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    no_ext_pts, no_ext_valid = camera_rays_to_image_points(rays, proj, no_external)
    ws_pts, ws_valid = camera_rays_to_image_points(
        rays, proj, real_ftheta_windshield_distortion
    )
    assert no_ext_valid.all() and ws_valid.all()
    max_pixel_delta = (no_ext_pts - ws_pts).abs().max().item()
    # Real windshield coefficients perturb the camera-frame ray before
    # projection; a 1px floor is comfortably above float32 noise and well
    # below the magnitude of the windshield correction observed on the
    # committed test record.
    assert max_pixel_delta > 1.0, (
        "real windshield distortion produced no observable pixel shift "
        f"vs no_external (max delta = {max_pixel_delta:.4e}); the "
        "external_distortion fields in the JSON are likely either "
        "identity-equivalent or not threaded through dispatch."
    )


def test_real_ftheta_windshield_round_trip(
    real_ftheta_projection_with_windshield, real_ftheta_windshield_distortion
):
    """Verify forward-project then back-project under a real windshield distortion
    recovers the input ray direction within a tolerance derived from the
    float32 polynomial chain. Exercises both halves of the bivariate
    distortion (forward and inverse polynomials).
    """
    proj = real_ftheta_projection_with_windshield
    distortion = real_ftheta_windshield_distortion
    pp = proj.principal_point
    device = pp.device
    rays_in = torch.nn.functional.normalize(
        torch.tensor(
            [
                [0.05, 0.0, 1.0],
                [0.0, 0.04, 1.0],
                [-0.05, 0.04, 1.0],
                [0.04, -0.03, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )
    image_points, valid = camera_rays_to_image_points(rays_in, proj, distortion)
    assert valid.all()
    rays_back = image_points_to_camera_rays(image_points, proj, distortion)
    rays_back = torch.nn.functional.normalize(rays_back, dim=-1)
    # The forward+inverse windshield polynomials are approximate inverses
    # of each other and the FTheta IFT correction is a single Newton step,
    # so a ~1e-2 angular tolerance is realistic for float32. Tighten if
    # the kernel's IFT inversion accuracy improves.
    cos_sim = (rays_in * rays_back).sum(dim=-1)
    assert (
        cos_sim >= 1.0 - 1e-2
    ).all(), (
        f"round-trip cosine similarity below tolerance: min={cos_sim.min().item():.4e}"
    )


@pytest.mark.gradcheck
def test_real_ftheta_image_points_to_camera_rays_gradcheck(
    real_ftheta_projection, no_external
):
    """Inverse-path gradcheck on a real F-Theta camera. The fixture is
    parametrized over both reference_poly variants in the JSON: under the
    FORWARD-ref record the inverse path triggers the IFT-Newton adjoint
    (Newton on ``fw_poly``), while under the BACKWARD-ref record it is a
    direct polynomial evaluation of ``bw_poly``. Real intrinsics keep
    gradient magnitudes well within the standard GRADCHECK_KWARGS tolerance.
    """
    pp = real_ftheta_projection.principal_point
    image_points = (
        (
            pp.reshape(1, 2).to(torch.float64)
            + torch.tensor(
                [[100.0, 50.0], [-200.0, 75.0]],
                device=pp.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )

    def fn(pts):
        return image_points_to_camera_rays(
            pts,
            real_ftheta_projection,
            no_external,
            allow_device_transfer=True,
        )

    assert torch.autograd.gradcheck(fn, (image_points,), **GRADCHECK_KWARGS)


# ----------------------------------------------------------------------------
# FTheta autograd gradchecks for project_world_points_{mean,shutter}_pose
# and image_points_to_world_rays_{static,shutter}_pose, no-external and
# bivariate variants. Inputs finite-difference through the public op entrypoint.
# ----------------------------------------------------------------------------


def _ftheta_world_points_f64(device, n=2):
    # Points are intentionally offset from any coordinate axis and from the
    # mean-pose midpoint translation so float32 cast in the autograd Function
    # doesn't manufacture an exact-zero cam_pt.x or .y, which would inflate
    # numerical-vs-analytical Jacobian off-diagonal noise.
    return torch.tensor(
        [[0.171, 0.097, 1.83], [-0.123, 0.158, 2.27]][:n],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )


# Alias: the FTheta mean-pose / shutter-pose paths see the same float32 noise
# floor as the pinhole equivalents (slerp + lerp on the interpolated pose).
GRADCHECK_KWARGS_FTHETA_DYNAMIC = GRADCHECK_KWARGS_DYNAMIC


def _ftheta_image_points_f64(projection, n=2):
    pp = projection.principal_point
    return (
        (
            pp.reshape(1, 2).to(torch.float64)
            + torch.tensor(
                [[2.0, 1.5], [-1.5, 0.7]][:n],
                device=pp.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )


@pytest.mark.gradcheck
def test_ftheta_project_world_points_mean_pose_gradcheck(
    ftheta_projection_forward_ref, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta project_world_points_mean_pose with NoExternalDistortion."""
    points = _ftheta_world_points_f64(
        ftheta_projection_forward_ref.principal_point.device
    )

    def fn(p):
        return project_world_points_mean_pose(
            p,
            ftheta_projection_forward_ref,
            no_external,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_ftheta_project_world_points_shutter_pose_gradcheck(
    ftheta_projection_forward_ref, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta project_world_points_shutter_pose with NoExternalDistortion."""
    points = _ftheta_world_points_f64(
        ftheta_projection_forward_ref.principal_point.device
    )

    def fn(p):
        return project_world_points_shutter_pose(
            p,
            ftheta_projection_forward_ref,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_ftheta_image_points_to_world_rays_static_pose_gradcheck(
    ftheta_projection_forward_ref, no_external, static_pose
):
    """End-to-end autograd.gradcheck for FTheta image_points_to_world_rays_static_pose with NoExternalDistortion."""
    pts = _ftheta_image_points_f64(ftheta_projection_forward_ref)

    def fn(p):
        return image_points_to_world_rays_static_pose(
            p,
            ftheta_projection_forward_ref,
            no_external,
            static_pose,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_ftheta_bivariate_image_points_to_camera_rays_nonidentity_distortion_gradcheck(
    ftheta_projection_forward_ref, windshield_distortion
):
    """Non-identity bivariate backprojection gradcheck over inverse distortion coefficients."""
    pp = ftheta_projection_forward_ref.principal_point
    image_points = (
        pp.reshape(1, 2).to(torch.float64)
        + torch.tensor(
            [[1.0, -0.5], [1.8, 0.7]],
            device=pp.device,
            dtype=torch.float64,
        )
    ).detach()
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)
    with torch.no_grad():
        distortion_coeffs[21] += 0.01
        distortion_coeffs[22] += 0.002

    def fn(coeffs):
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return image_points_to_camera_rays(
            image_points,
            ftheta_projection_forward_ref,
            distortion,
            allow_device_transfer=True,
        )

    assert torch.autograd.gradcheck(fn, (distortion_coeffs,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_ftheta_image_points_to_world_rays_shutter_pose_gradcheck(
    ftheta_projection_forward_ref, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta image_points_to_world_rays_shutter_pose with NoExternalDistortion."""
    pts = _ftheta_image_points_f64(ftheta_projection_forward_ref)

    def fn(p):
        return image_points_to_world_rays_shutter_pose(
            p,
            ftheta_projection_forward_ref,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_ftheta_bivariate_project_world_points_mean_pose_gradcheck(
    ftheta_projection_forward_ref, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta project_world_points_mean_pose with BivariateWindshieldDistortion."""
    points = _ftheta_world_points_f64(
        ftheta_projection_forward_ref.principal_point.device
    )

    def fn(p):
        return project_world_points_mean_pose(
            p,
            ftheta_projection_forward_ref,
            windshield_distortion,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_ftheta_bivariate_project_world_points_shutter_pose_gradcheck(
    ftheta_projection_forward_ref, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta project_world_points_shutter_pose with BivariateWindshieldDistortion."""
    points = _ftheta_world_points_f64(
        ftheta_projection_forward_ref.principal_point.device
    )

    def fn(p):
        return project_world_points_shutter_pose(
            p,
            ftheta_projection_forward_ref,
            windshield_distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_ftheta_bivariate_image_points_to_world_rays_static_pose_gradcheck(
    ftheta_projection_forward_ref, windshield_distortion, static_pose
):
    """End-to-end autograd.gradcheck for FTheta image_points_to_world_rays_static_pose with BivariateWindshieldDistortion."""
    pts = _ftheta_image_points_f64(ftheta_projection_forward_ref)

    def fn(p):
        return image_points_to_world_rays_static_pose(
            p,
            ftheta_projection_forward_ref,
            windshield_distortion,
            static_pose,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_ftheta_bivariate_image_points_to_world_rays_shutter_pose_gradcheck(
    ftheta_projection_forward_ref, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for FTheta image_points_to_world_rays_shutter_pose with BivariateWindshieldDistortion."""
    pts = _ftheta_image_points_f64(ftheta_projection_forward_ref)

    def fn(p):
        return image_points_to_world_rays_shutter_pose(
            p,
            ftheta_projection_forward_ref,
            windshield_distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


# ===========================================================================
# OpenCV-fisheye tests. Intrinsics are the 4-tuple (principal_point,
# focal_length, forward_poly, approx_backward_factor); approx_backward_factor
# never receives a gradient. Gradchecks reuse the float32 FTheta harness
# tolerances.
# ===========================================================================


def _fisheye_synthetic_rays(device: torch.device, dtype: torch.dtype = torch.float32):
    return torch.tensor(
        [
            [0.05, 0.0, 1.0],
            [0.0, 0.04, 1.0],
            [-0.05, 0.04, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _fisheye_image_points_f64(projection, n=2):
    pp = projection.principal_point
    return (
        (
            pp.reshape(1, 2).to(torch.float64)
            + torch.tensor(
                [[2.0, 1.5], [-1.5, 0.7]][:n],
                device=pp.device,
                dtype=torch.float64,
            )
        )
        .detach()
        .requires_grad_(True)
    )


def _fisheye_projection_from_grad_tensors(
    base_projection,
    principal_point: Tensor,
    focal_length: Tensor,
    forward_poly: Tensor,
    approx_backward_factor: Tensor,
) -> OpenCVFisheyeProjection:
    return OpenCVFisheyeProjection(
        principal_point=principal_point,
        focal_length=focal_length,
        forward_poly=forward_poly,
        approx_backward_factor=approx_backward_factor,
        resolution=base_projection.resolution,
        newton_iterations=int(base_projection.newton_iterations),
        max_angle=float(base_projection.max_angle),
        min_2d_norm=float(base_projection.min_2d_norm),
    )


def _fisheye_intrinsic_grad_inputs(base_projection):
    return tuple(
        tensor.detach().clone().to(torch.float64).requires_grad_(True)
        for tensor in (
            base_projection.principal_point,
            base_projection.focal_length,
            base_projection.forward_poly,
        )
    )


def test_camera_rays_to_image_points_fisheye_basic(fisheye_projection, no_external):
    """Fisheye camera_rays -> image_points: on-axis ray maps to the principal point."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]], device=fisheye_projection.principal_point.device
    )
    image_points, valid_flags = camera_rays_to_image_points(
        rays, fisheye_projection, no_external
    )
    assert valid_flags.all()
    assert torch.allclose(
        image_points[0], fisheye_projection.principal_point, atol=1e-4
    )


def test_camera_rays_to_image_points_fisheye_clamps_over_max_angle(
    fisheye_projection, no_external
):
    """Fisheye rays beyond max_angle clamp theta to max_angle and stay valid.

    The reference device math does not reject an over-FOV ray: it clamps theta to
    max_angle and projects, so the ray lands at the same in-bounds image point as
    a same-azimuth ray sitting exactly at max_angle.
    """
    device = fisheye_projection.principal_point.device
    max_angle = 0.01
    projection = OpenCVFisheyeProjection(
        principal_point=fisheye_projection.principal_point,
        focal_length=fisheye_projection.focal_length,
        forward_poly=fisheye_projection.forward_poly,
        approx_backward_factor=fisheye_projection.approx_backward_factor,
        resolution=fisheye_projection.resolution,
        newton_iterations=int(fisheye_projection.newton_iterations),
        max_angle=max_angle,
        min_2d_norm=float(fisheye_projection.min_2d_norm),
    )
    # theta = atan2(0.1, 1.0) ~= 0.0997 > max_angle -> clamped to max_angle.
    over_fov = torch.tensor([[0.1, 0.0, 1.0]], device=device)
    # Same +x azimuth, but sitting exactly at max_angle (not clamped).
    at_max = torch.tensor(
        [[float(np.sin(max_angle)), 0.0, float(np.cos(max_angle))]], device=device
    )
    over_pts, over_valid = camera_rays_to_image_points(
        over_fov, projection, no_external
    )
    at_pts, at_valid = camera_rays_to_image_points(at_max, projection, no_external)

    assert over_valid.tolist() == [True]
    assert at_valid.tolist() == [True]
    assert torch.allclose(over_pts, at_pts, atol=1e-4)


def test_image_points_to_camera_rays_fisheye_round_trip(
    fisheye_projection, no_external
):
    """Fisheye project then back-project recovers the original normalized ray."""
    rays = _fisheye_synthetic_rays(fisheye_projection.principal_point.device)
    image_points, _ = camera_rays_to_image_points(rays, fisheye_projection, no_external)
    rays_back = image_points_to_camera_rays(
        image_points, fisheye_projection, no_external
    )
    rays_unit = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)
    assert torch.allclose(rays_back, rays_unit, atol=1e-4)


def test_fisheye_all_public_ops_callable(
    fisheye_projection, no_external, windshield_distortion, dynamic_pose, static_pose
):
    """Every public op dispatches to a fisheye Function for both distortions."""
    device = fisheye_projection.principal_point.device
    rays = _fisheye_synthetic_rays(device)
    pts = fisheye_projection.principal_point.reshape(1, 2) + torch.tensor(
        [[2.0, 1.5], [-1.5, 0.7]], device=device
    )
    world = torch.tensor([[0.17, 0.1, 1.8], [-0.12, 0.16, 2.3]], device=device)
    for distortion in (no_external, windshield_distortion):
        camera_rays_to_image_points(rays, fisheye_projection, distortion)
        image_points_to_camera_rays(pts, fisheye_projection, distortion)
        project_world_points_mean_pose(
            world, fisheye_projection, distortion, dynamic_pose, (100, 80)
        )
        project_world_points_shutter_pose(
            world,
            fisheye_projection,
            distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
        )
        image_points_to_world_rays_static_pose(
            pts, fisheye_projection, distortion, static_pose
        )
        image_points_to_world_rays_shutter_pose(
            pts,
            fisheye_projection,
            distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
        )


def test_fisheye_empty_input_returns_shaped_tensors(fisheye_projection, no_external):
    """N==0 fisheye inputs return correctly shaped empty tensors."""
    device = fisheye_projection.principal_point.device
    rays = torch.zeros((0, 3), device=device)
    pts = torch.zeros((0, 2), device=device)
    image_points, valid_flags = camera_rays_to_image_points(
        rays, fisheye_projection, no_external
    )
    assert image_points.shape == (0, 2)
    assert valid_flags.shape == (0,)
    camera_rays = image_points_to_camera_rays(pts, fisheye_projection, no_external)
    assert camera_rays.shape == (0, 3)


@pytest.mark.gradcheck
def test_camera_rays_to_image_points_fisheye_gradcheck(fisheye_projection, no_external):
    """End-to-end autograd.gradcheck for fisheye camera_rays -> image_points."""
    rays = torch.tensor(
        [[0.03, 0.02, 1.0], [0.05, -0.01, 1.1], [-0.04, 0.03, 1.2]],
        device=fisheye_projection.principal_point.device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(camera_rays):
        return camera_rays_to_image_points(
            camera_rays, fisheye_projection, no_external, allow_device_transfer=True
        )[0]

    assert torch.autograd.gradcheck(fn, (rays,), **GRADCHECK_KWARGS_FTHETA_PROJECT)


@pytest.mark.gradcheck
def test_image_points_to_camera_rays_fisheye_gradcheck(fisheye_projection, no_external):
    """End-to-end autograd.gradcheck for fisheye image_points -> camera_rays."""
    image_points = _fisheye_image_points_f64(fisheye_projection)

    def fn(pts):
        return image_points_to_camera_rays(
            pts, fisheye_projection, no_external, allow_device_transfer=True
        )

    assert torch.autograd.gradcheck(fn, (image_points,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_camera_rays_to_image_points_fisheye_intrinsics_gradcheck(
    fisheye_projection, no_external
):
    """Fisheye intrinsic gradcheck over (principal_point, focal_length, forward_poly)."""
    rays = _fisheye_synthetic_rays(
        fisheye_projection.principal_point.device, dtype=torch.float64
    )
    pp, focal, fw = _fisheye_intrinsic_grad_inputs(fisheye_projection)
    ab32 = fisheye_projection.approx_backward_factor.detach().to(torch.float32)

    def fn(principal_point, focal_length, forward_poly):
        # The fisheye projection constructor admits fp32-only intrinsics, so
        # cast the fp64 gradcheck leaves at the construction boundary; autograd
        # flows the cotangent back through the cast.
        projection = _fisheye_projection_from_grad_tensors(
            fisheye_projection,
            principal_point.to(torch.float32),
            focal_length.to(torch.float32),
            forward_poly.to(torch.float32),
            ab32,
        )
        return camera_rays_to_image_points(
            rays, projection, no_external, allow_device_transfer=True
        )[0]

    assert torch.autograd.gradcheck(
        fn, (pp, focal, fw), **GRADCHECK_KWARGS_FTHETA_PROJECT
    )


@pytest.mark.gradcheck
def test_fisheye_project_world_points_mean_pose_gradcheck(
    fisheye_projection, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye project_world_points_mean_pose."""
    points = _ftheta_world_points_f64(fisheye_projection.principal_point.device)

    def fn(p):
        return project_world_points_mean_pose(
            p,
            fisheye_projection,
            no_external,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_fisheye_project_world_points_shutter_pose_gradcheck(
    fisheye_projection, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye project_world_points_shutter_pose."""
    points = _ftheta_world_points_f64(fisheye_projection.principal_point.device)

    def fn(p):
        return project_world_points_shutter_pose(
            p,
            fisheye_projection,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_fisheye_image_points_to_world_rays_static_pose_gradcheck(
    fisheye_projection, no_external, static_pose
):
    """End-to-end autograd.gradcheck for fisheye image_points_to_world_rays_static_pose."""
    pts = _fisheye_image_points_f64(fisheye_projection)

    def fn(p):
        return image_points_to_world_rays_static_pose(
            p, fisheye_projection, no_external, static_pose, allow_device_transfer=True
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_fisheye_image_points_to_world_rays_shutter_pose_gradcheck(
    fisheye_projection, no_external, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye image_points_to_world_rays_shutter_pose."""
    pts = _fisheye_image_points_f64(fisheye_projection)

    def fn(p):
        return image_points_to_world_rays_shutter_pose(
            p,
            fisheye_projection,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_fisheye_bivariate_image_points_to_camera_rays_gradcheck(
    fisheye_projection, windshield_distortion
):
    """Bivariate fisheye backprojection gradcheck over inverse distortion coefficients."""
    image_points = _fisheye_image_points_f64(fisheye_projection)
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(coeffs):
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return image_points_to_camera_rays(
            image_points, fisheye_projection, distortion, allow_device_transfer=True
        )

    assert torch.autograd.gradcheck(fn, (distortion_coeffs,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_fisheye_bivariate_project_world_points_mean_pose_gradcheck(
    fisheye_projection, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye project_world_points_mean_pose bivariate."""
    points = _ftheta_world_points_f64(fisheye_projection.principal_point.device)

    def fn(p):
        return project_world_points_mean_pose(
            p,
            fisheye_projection,
            windshield_distortion,
            dynamic_pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_fisheye_bivariate_project_world_points_shutter_pose_gradcheck(
    fisheye_projection, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye project_world_points_shutter_pose bivariate."""
    points = _ftheta_world_points_f64(fisheye_projection.principal_point.device)

    def fn(p):
        return project_world_points_shutter_pose(
            p,
            fisheye_projection,
            windshield_distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (points,), **GRADCHECK_KWARGS_FTHETA_DYNAMIC)


@pytest.mark.gradcheck
def test_fisheye_bivariate_image_points_to_world_rays_static_pose_gradcheck(
    fisheye_projection, windshield_distortion, static_pose
):
    """End-to-end autograd.gradcheck for fisheye image_points_to_world_rays_static_pose bivariate."""
    pts = _fisheye_image_points_f64(fisheye_projection)

    def fn(p):
        return image_points_to_world_rays_static_pose(
            p,
            fisheye_projection,
            windshield_distortion,
            static_pose,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_fisheye_bivariate_image_points_to_world_rays_shutter_pose_gradcheck(
    fisheye_projection, windshield_distortion, dynamic_pose
):
    """End-to-end autograd.gradcheck for fisheye image_points_to_world_rays_shutter_pose bivariate."""
    pts = _fisheye_image_points_f64(fisheye_projection)

    def fn(p):
        return image_points_to_world_rays_shutter_pose(
            p,
            fisheye_projection,
            windshield_distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (pts,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_fisheye_mean_pose_rotation_slerp_branch_gradcheck(
    fisheye_projection, no_external, branch
):
    """Pose-rotation-gradient coverage (xyzw->wxyz store) for fisheye mean-pose projection."""
    device = fisheye_projection.principal_point.device
    points = _ftheta_world_points_f64(device, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return project_world_points_mean_pose(
            points,
            fisheye_projection,
            no_external,
            _pose_from_rotations(q0, q1, device),
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (start_r, end_r), **GRADCHECK_KWARGS_FTHETA_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_fisheye_shutter_pose_rotation_slerp_branch_gradcheck(
    fisheye_projection, no_external, branch
):
    """Pose-rotation-gradient coverage (xyzw->wxyz store) for fisheye shutter-pose projection."""
    device = fisheye_projection.principal_point.device
    points = _ftheta_world_points_f64(device, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return project_world_points_shutter_pose(
            points,
            fisheye_projection,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _pose_from_rotations(q0, q1, device),
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn, (start_r, end_r), **GRADCHECK_KWARGS_FTHETA_DYNAMIC
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("branch", ["ordinary", "nlerp", "hemisphere_flip"])
def test_fisheye_world_rays_shutter_pose_rotation_slerp_branch_gradcheck(
    fisheye_projection, no_external, branch
):
    """Pose-rotation-gradient coverage (xyzw->wxyz store) for fisheye world-rays shutter-pose."""
    device = fisheye_projection.principal_point.device
    pts = _fisheye_image_points_f64(fisheye_projection, n=1).detach()
    start_r, end_r = _slerp_branch_rotations(branch, device)

    def fn(q0, q1):
        return image_points_to_world_rays_shutter_pose(
            pts,
            fisheye_projection,
            no_external,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _pose_from_rotations(q0, q1, device),
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(fn, (start_r, end_r), **GRADCHECK_KWARGS)


def test_camera_rays_to_image_points_fisheye_behind_camera_invalid(
    fisheye_projection, no_external
):
    """Rays with z<=0 MUST be marked invalid (the kernel falls back to PP)."""
    rays = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # forward, valid
            [0.0, 0.0, -1.0],  # behind, invalid
        ],
        device=fisheye_projection.principal_point.device,
    )
    _, valid_flags = camera_rays_to_image_points(rays, fisheye_projection, no_external)
    assert valid_flags.tolist() == [True, False]


def test_camera_rays_to_image_points_fisheye_out_of_frame_marks_invalid(
    fisheye_projection, no_external
):
    """A ray that projects outside ``[0, W) x [0, H)`` is flagged invalid even when projection succeeds (z > 0)."""
    proj = fisheye_projection
    device = proj.principal_point.device
    in_frame_ray = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    far_off_axis = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    rays = torch.cat([in_frame_ray, far_off_axis], dim=0)
    _, valid_flags = camera_rays_to_image_points(rays, proj, no_external)
    assert bool(valid_flags[0]) is True
    assert bool(valid_flags[1]) is False, (
        "out-of-frame pixel must be flagged invalid; got "
        f"valid_flags={valid_flags.tolist()}"
    )


def test_camera_rays_to_image_points_fisheye_nan_input_propagates_non_finite(
    fisheye_projection, no_external
):
    """A NaN-bearing ray yields a non-finite pixel; the gate checks bounds but not finiteness, so the ray stays flagged valid."""
    proj = fisheye_projection
    device = proj.principal_point.device
    rays = torch.tensor(
        [[float("nan"), 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    image_points, valid_flags = camera_rays_to_image_points(rays, proj, no_external)
    assert not torch.isfinite(
        image_points[0]
    ).all(), (
        f"NaN-bearing ray must yield a non-finite pixel; got {image_points.tolist()}"
    )
    assert (
        bool(valid_flags[0]) is True
    ), f"fisheye gate does not reject NaN rays; got valid_flags={valid_flags.tolist()}"


def test_camera_rays_to_image_points_fisheye_no_external_scratch_gating(
    fisheye_projection, no_external, sensor_device: torch.device
):
    """When no input requires_grad, scratch tensor must be empty; gradient mode
    must allocate a scratch tensor sized for the per-row save layout."""
    rays = _fisheye_synthetic_rays(sensor_device).detach()
    op = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_fisheye_no_external

    _, _, scratch = op(fisheye_projection, no_external, rays)
    assert scratch.numel() == 0

    _, _, ray_grad_scratch = op(
        fisheye_projection,
        no_external,
        rays.detach().clone().requires_grad_(True),
    )
    assert ray_grad_scratch.shape[0] == rays.shape[0]
    assert ray_grad_scratch.numel() > 0


def test_camera_rays_to_image_points_fisheye_no_external_basic_multiray(
    fisheye_projection, no_external
):
    """Verify camera_rays_to_image_points returns finite in-frame points for the synthetic fisheye ray set."""
    rays = _fisheye_synthetic_rays(fisheye_projection.principal_point.device)
    image_points, valid_flags = camera_rays_to_image_points(
        rays, fisheye_projection, no_external
    )
    assert image_points.shape == (3, 2)
    assert valid_flags.dtype == torch.bool
    assert valid_flags.all()


def test_image_points_to_camera_rays_fisheye_scratch_gating(
    fisheye_projection, no_external, sensor_device: torch.device
):
    """Inverse direction gates scratch allocation by requires_grad."""
    pp = fisheye_projection.principal_point
    image_points = pp.reshape(1, 2) + torch.tensor([[2.0, 1.0]], device=sensor_device)
    op = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_fisheye_no_external

    _, scratch = op(fisheye_projection, no_external, image_points)
    assert scratch.numel() == 0

    _, grad_scratch = op(
        fisheye_projection,
        no_external,
        image_points.detach().clone().requires_grad_(True),
    )
    assert grad_scratch.shape[0] == image_points.shape[0]
    assert grad_scratch.numel() > 0


@pytest.mark.gradcheck
def test_image_points_to_camera_rays_fisheye_intrinsics_gradcheck(
    fisheye_projection, no_external
):
    """Fisheye intrinsic gradcheck for image_points -> camera_rays over (principal_point, focal_length, forward_poly)."""
    image_points = _fisheye_image_points_f64(fisheye_projection)
    pp, focal, fw = _fisheye_intrinsic_grad_inputs(fisheye_projection)
    ab32 = fisheye_projection.approx_backward_factor.detach().to(torch.float32)

    def fn(principal_point, focal_length, forward_poly):
        # The fisheye projection constructor admits fp32-only intrinsics, so
        # cast the fp64 gradcheck leaves at the construction boundary; autograd
        # flows the cotangent back through the cast.
        projection = _fisheye_projection_from_grad_tensors(
            fisheye_projection,
            principal_point.to(torch.float32),
            focal_length.to(torch.float32),
            forward_poly.to(torch.float32),
            ab32,
        )
        return image_points_to_camera_rays(
            image_points,
            projection,
            no_external,
            allow_device_transfer=True,
        )

    assert torch.autograd.gradcheck(
        fn, (pp, focal, fw), **GRADCHECK_KWARGS_FTHETA_PROJECT
    )


def test_project_world_points_mean_pose_fisheye_scratch_shapes(
    fisheye_projection, no_external, windshield_distortion, dynamic_pose
):
    """Verify fisheye mean-pose scratch is (N, 14) for both no-external and bivariate paths."""
    device = fisheye_projection.principal_point.device
    world_points = torch.tensor(
        [[0.0, 0.0, 1.5], [0.1, -0.1, 2.0]], device=device
    ).requires_grad_(True)
    start_t, start_r, end_t, end_r = _ftheta_dynamic_pose_tensors(dynamic_pose, device)

    (
        *_,
        scratch_no_external,
    ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_no_external(
        fisheye_projection,
        no_external,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        0,
        10,
    )
    assert scratch_no_external.shape == (world_points.shape[0], 14)

    (
        *_,
        scratch_bivariate,
    ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_bivariate_windshield(
        fisheye_projection,
        windshield_distortion,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        0,
        10,
    )
    assert scratch_bivariate.shape == (world_points.shape[0], 14)


@pytest.mark.gradcheck
def test_fisheye_project_world_points_mean_pose_pose_gradcheck(
    fisheye_projection, windshield_distortion, sensor_device
):
    """Gradcheck fisheye project_world_points_mean_pose over dynamic-pose tensors and distortion coefficients."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, -0.2, 1.5]],
        device=sensor_device,
        dtype=torch.float64,
    )
    start_translation = torch.zeros(
        3, device=sensor_device, dtype=torch.float64, requires_grad=True
    )
    start_rotation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    end_translation = torch.tensor(
        [0.1, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    end_rotation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(start_t, start_r, end_t, end_r, coeffs):
        from gsplat.sensors.kernels.common import DynamicPose, Pose

        pose = DynamicPose(
            Pose(translation=start_t, rotation=start_r),
            Pose(translation=end_t, rotation=end_r),
        )
        distortion = _build_distortion(
            windshield_distortion, ReferencePolynomial.FORWARD, coeffs
        )
        return project_world_points_mean_pose(
            world_points,
            fisheye_projection,
            distortion,
            pose,
            (100, 80),
            allow_device_transfer=True,
        )[0]

    assert torch.autograd.gradcheck(
        fn,
        (
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            distortion_coeffs,
        ),
        **GRADCHECK_KWARGS_FTHETA_DYNAMIC,
    )


def test_project_world_points_shutter_pose_fisheye_scratch_shapes(
    fisheye_projection, no_external, windshield_distortion, dynamic_pose
):
    """Verify OpenCV-fisheye shutter-pose scratch is (N, 16) for no-external and bivariate paths."""
    device = fisheye_projection.principal_point.device
    world_points = torch.tensor(
        [[0.1, 0.05, 1.6], [-0.1, 0.08, 2.1]], device=device
    ).requires_grad_(True)
    start_t, start_r, end_t, end_r = _ftheta_dynamic_pose_tensors(dynamic_pose, device)

    (
        *_,
        scratch_no_external,
    ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_no_external(
        fisheye_projection,
        no_external,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
        1,
        0.01,
        0.01,
        0.5,
    )
    assert scratch_no_external.shape == (world_points.shape[0], 16)

    (
        *_,
        scratch_bivariate,
    ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield(
        fisheye_projection,
        windshield_distortion,
        world_points,
        start_t,
        start_r,
        end_t,
        end_r,
        100,
        80,
        int(ShutterType.ROLLING_TOP_TO_BOTTOM),
        0,
        10,
        1,
        0.01,
        0.01,
        0.5,
    )
    assert scratch_bivariate.shape == (world_points.shape[0], 16)


@pytest.mark.gradcheck
@pytest.mark.parametrize(
    "reference_polynomial",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_fisheye_bivariate_image_points_to_world_rays_shutter_pose_image_points_grad_flows(
    fisheye_projection, windshield_distortion, dynamic_pose, reference_polynomial
):
    """Verify rolling shutter keeps the direct image-point VJP for bivariate fisheye."""
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)
    distortion = _build_distortion(
        windshield_distortion, reference_polynomial, distortion_coeffs
    )
    image_points = _fisheye_image_points_f64(fisheye_projection)

    world_rays = image_points_to_world_rays_shutter_pose(
        image_points,
        fisheye_projection,
        distortion,
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        allow_device_transfer=True,
    )[0]
    loss = world_rays[:, 3].sum() + 0.25 * world_rays[:, 4].sum()
    loss.backward()

    assert image_points.grad is not None
    assert image_points.grad.abs().sum() > 0


def test_fisheye_identity_windshield_matches_no_external(
    fisheye_projection, no_external, windshield_distortion
):
    """Identity windshield should match no_external within atol=1e-4."""
    rays = _fisheye_synthetic_rays(fisheye_projection.principal_point.device)
    no_ext_pts, _ = camera_rays_to_image_points(rays, fisheye_projection, no_external)
    bw_pts, _ = camera_rays_to_image_points(
        rays, fisheye_projection, windshield_distortion
    )
    assert torch.allclose(no_ext_pts, bw_pts, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    "reference, active_slice",
    [
        (ReferencePolynomial.FORWARD, slice(0, 21)),
        (ReferencePolynomial.BACKWARD, slice(21, 42)),
    ],
)
def test_fisheye_camera_rays_to_image_points_distortion_grad_slice(
    fisheye_projection, windshield_distortion, reference, active_slice
):
    """Gradient flows into only the active coefficient half, not the inactive one."""
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(reference),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = _fisheye_synthetic_rays(fisheye_projection.principal_point.device)
    image_points, valid = camera_rays_to_image_points(
        rays, fisheye_projection, distortion, allow_device_transfer=True
    )
    assert valid.all()
    image_points.sum().backward()
    grad = coeffs.grad
    assert grad is not None
    inactive = slice(21, 42) if active_slice.start == 0 else slice(0, 21)
    assert grad[active_slice].abs().sum() > 0
    assert_grad_reference_close(
        grad[inactive],
        torch.zeros_like(grad[inactive]),
        rtol=0.0,
        atol=0.0,
        msg="inactive fisheye bivariate coeff gradient slice",
    )


def test_fisheye_identity_bivariate_backward_smoke_all_public_ops(
    fisheye_projection, windshield_distortion, dynamic_pose, static_pose
):
    """FORWARD reference: each public op accumulates gradient in its expected coefficient half."""
    device = fisheye_projection.principal_point.device
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(ReferencePolynomial.FORWARD),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = _fisheye_synthetic_rays(device)
    image_points = fisheye_projection.principal_point.reshape(1, 2) + torch.tensor(
        [[2.0, 1.5], [-1.5, 0.7]], device=device
    )
    world_points = torch.tensor([[0.17, 0.1, 1.8], [-0.12, 0.16, 2.3]], device=device)

    checks = [
        (
            lambda: camera_rays_to_image_points(
                rays, fisheye_projection, distortion, allow_device_transfer=True
            )[0],
            slice(0, 21),
        ),
        (
            lambda: image_points_to_camera_rays(
                image_points,
                fisheye_projection,
                distortion,
                allow_device_transfer=True,
            ),
            slice(21, 42),
        ),
        (
            lambda: project_world_points_mean_pose(
                world_points,
                fisheye_projection,
                distortion,
                dynamic_pose,
                (100, 80),
                allow_device_transfer=True,
            )[0],
            slice(0, 21),
        ),
        (
            lambda: project_world_points_shutter_pose(
                world_points,
                fisheye_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                max_iterations=1,
                allow_device_transfer=True,
            )[0],
            slice(0, 21),
        ),
        (
            lambda: image_points_to_world_rays_static_pose(
                image_points,
                fisheye_projection,
                distortion,
                static_pose,
                allow_device_transfer=True,
            )[0],
            slice(21, 42),
        ),
        (
            lambda: image_points_to_world_rays_shutter_pose(
                image_points,
                fisheye_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            slice(21, 42),
        ),
    ]
    for op, active_slice in checks:
        coeffs.grad = None
        op().sum().backward()
        assert coeffs.grad is not None
        assert coeffs.grad[active_slice].abs().sum() > 0
        inactive = slice(0, 21) if active_slice.start == 21 else slice(21, 42)
        assert_grad_reference_close(
            coeffs.grad[inactive],
            torch.zeros_like(coeffs.grad[inactive]),
            rtol=0.0,
            atol=0.0,
            msg="inactive FORWARD-reference fisheye coeff gradients",
        )


def test_fisheye_identity_bivariate_backward_smoke_all_public_ops_backward_reference(
    fisheye_projection, windshield_distortion, dynamic_pose, static_pose
):
    """BACKWARD reference: the active gradient half flips, with the other half left at zero."""
    device = fisheye_projection.principal_point.device
    coeffs = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion = BivariateWindshieldDistortion(
        coeffs,
        int(ReferencePolynomial.BACKWARD),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    rays = _fisheye_synthetic_rays(device)
    image_points = fisheye_projection.principal_point.reshape(1, 2) + torch.tensor(
        [[2.0, 1.5], [-1.5, 0.7]], device=device
    )
    world_points = torch.tensor([[0.17, 0.1, 1.8], [-0.12, 0.16, 2.3]], device=device)

    forward_slice = slice(21, 42)
    inverse_slice = slice(0, 21)
    checks = [
        (
            lambda: camera_rays_to_image_points(
                rays, fisheye_projection, distortion, allow_device_transfer=True
            )[0],
            forward_slice,
        ),
        (
            lambda: image_points_to_camera_rays(
                image_points,
                fisheye_projection,
                distortion,
                allow_device_transfer=True,
            ),
            inverse_slice,
        ),
        (
            lambda: project_world_points_mean_pose(
                world_points,
                fisheye_projection,
                distortion,
                dynamic_pose,
                (100, 80),
                allow_device_transfer=True,
            )[0],
            forward_slice,
        ),
        (
            lambda: project_world_points_shutter_pose(
                world_points,
                fisheye_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                max_iterations=1,
                allow_device_transfer=True,
            )[0],
            forward_slice,
        ),
        (
            lambda: image_points_to_world_rays_static_pose(
                image_points,
                fisheye_projection,
                distortion,
                static_pose,
                allow_device_transfer=True,
            )[0],
            inverse_slice,
        ),
        (
            lambda: image_points_to_world_rays_shutter_pose(
                image_points,
                fisheye_projection,
                distortion,
                (100, 80),
                ShutterType.GLOBAL,
                dynamic_pose,
                allow_device_transfer=True,
            )[0],
            inverse_slice,
        ),
    ]
    for op, active_slice in checks:
        coeffs.grad = None
        op().sum().backward()
        assert coeffs.grad is not None
        assert coeffs.grad[active_slice].abs().sum() > 0
        inactive = slice(0, 21) if active_slice.start == 21 else slice(21, 42)
        assert_grad_reference_close(
            coeffs.grad[inactive],
            torch.zeros_like(coeffs.grad[inactive]),
            rtol=0.0,
            atol=0.0,
            msg="inactive BACKWARD-reference fisheye coeff gradients",
        )


def test_fisheye_bivariate_distortion_grad_accumulates_uniformly(
    fisheye_projection, windshield_distortion, sensor_device
):
    """Chunked backward passes accumulate the same coefficient gradient as one batched pass."""
    rays = torch.randn(64, 3, device=sensor_device) + torch.tensor(
        [0.0, 0.0, 2.0], device=sensor_device
    )

    coeffs_full = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion_full = BivariateWindshieldDistortion(
        coeffs_full,
        int(windshield_distortion.reference_polynomial),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    camera_rays_to_image_points(
        rays, fisheye_projection, distortion_full, allow_device_transfer=True
    )[0].sum().backward()

    coeffs_split = (
        windshield_distortion.distortion_coeffs.detach().clone().requires_grad_(True)
    )
    distortion_split = BivariateWindshieldDistortion(
        coeffs_split,
        int(windshield_distortion.reference_polynomial),
        windshield_distortion.h_poly_degree,
        windshield_distortion.v_poly_degree,
    )
    for chunk in rays.split(16):
        camera_rays_to_image_points(
            chunk, fisheye_projection, distortion_split, allow_device_transfer=True
        )[0].sum().backward()

    assert coeffs_full.grad is not None
    assert coeffs_split.grad is not None
    assert_grad_reference_close(
        coeffs_full.grad,
        coeffs_split.grad,
        rtol=1e-5,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="fisheye bivariate distortion coeff gradients",
    )


def test_fisheye_projection_approx_backward_factor_no_grad(
    fisheye_projection, no_external
):
    """The forward fisheye projection never feeds a gradient to approx_backward_factor."""
    device = fisheye_projection.principal_point.device
    pp = fisheye_projection.principal_point.detach().clone().requires_grad_(True)
    focal = fisheye_projection.focal_length.detach().clone().requires_grad_(True)
    fw = fisheye_projection.forward_poly.detach().clone().requires_grad_(True)
    ab = fisheye_projection.approx_backward_factor.detach().clone().requires_grad_(True)
    projection = _fisheye_projection_from_grad_tensors(
        fisheye_projection, pp, focal, fw, ab
    )
    rays = _fisheye_synthetic_rays(device)
    camera_rays_to_image_points(
        rays, projection, no_external, allow_device_transfer=True
    )[0].sum().backward()

    ab_grad = torch.zeros_like(ab) if ab.grad is None else ab.grad
    assert_grad_reference_close(
        ab_grad,
        torch.zeros_like(ab),
        rtol=0.0,
        atol=0.0,
        max_rel_l2=0.0,
        max_rel_l1=0.0,
        min_cosine=1.0,
        max_signed_bias=0.0,
        msg="fisheye forward approx_backward_factor.grad",
    )
    assert fw.grad is not None and fw.grad.abs().sum() > 0
    assert focal.grad is not None and focal.grad.abs().sum() > 0


def test_fisheye_projection_bindings_readonly(fisheye_projection):
    """OpenCVFisheyeProjection tensor and resolution bindings are read-only (rebinding raises)."""
    for attr in (
        "principal_point",
        "focal_length",
        "forward_poly",
        "approx_backward_factor",
    ):
        with pytest.raises((AttributeError, RuntimeError)):
            setattr(
                fisheye_projection,
                attr,
                torch.zeros(1, device=fisheye_projection.principal_point.device),
            )
    with pytest.raises((AttributeError, RuntimeError, TypeError)):
        fisheye_projection.resolution = (1, 1)


# ===========================================================================
# Backward DistortionPolicy regression gates.
# ===========================================================================


_DISTORTION_POLICY_OPERATIONS = (
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
)
_PROJECT_LIKE_OPERATIONS = frozenset(
    {
        "camera_rays_to_image_points",
        "project_world_points_mean_pose",
        "project_world_points_shutter_pose",
    }
)
_DYNAMIC_POSE_OPERATIONS = frozenset(
    {
        "project_world_points_mean_pose",
        "project_world_points_shutter_pose",
        "image_points_to_world_rays_shutter_pose",
    }
)


def _nonidentity_distortion_coeffs(
    windshield_distortion,
    *,
    dtype: torch.dtype,
    requires_grad: bool,
) -> Tensor:
    coeffs = windshield_distortion.distortion_coeffs.detach().clone().to(dtype=dtype)
    with torch.no_grad():
        coeffs[1] += 2e-3
        coeffs[8] -= 1.5e-3
        coeffs[22] -= 1e-3
        coeffs[29] += 2e-3
    return coeffs.requires_grad_(requires_grad)


def _ftheta_projection_with_reference(base_projection, reference_polynomial):
    return FThetaProjection(
        principal_point=base_projection.principal_point,
        fw_poly=base_projection.fw_poly,
        bw_poly=base_projection.bw_poly,
        A=base_projection.A,
        resolution=base_projection.resolution,
        reference_polynomial=int(reference_polynomial),
        fw_poly_degree=int(base_projection.fw_poly_degree),
        bw_poly_degree=int(base_projection.bw_poly_degree),
        newton_iterations=int(base_projection.newton_iterations),
        max_angle=float(base_projection.max_angle),
        min_2d_norm=float(base_projection.min_2d_norm),
    )


def _operation_primary_input(
    operation: str,
    projection,
    *,
    count: int,
    dtype: torch.dtype,
) -> Tensor:
    device = projection.principal_point.device
    if operation == "camera_rays_to_image_points":
        value = torch.tensor([[0.037, -0.021, 1.07]], device=device, dtype=dtype)
    elif operation in _PROJECT_LIKE_OPERATIONS:
        value = torch.tensor([[0.171, 0.097, 1.83]], device=device, dtype=dtype)
    else:
        value = projection.principal_point.detach().reshape(1, 2).to(
            dtype=dtype
        ) + torch.tensor([[1.7, -1.1]], device=device, dtype=dtype)
    return value.expand(count, -1).clone().requires_grad_(True)


def _operation_pose_inputs(
    operation: str,
    device: torch.device,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    start_translation = torch.tensor(
        [0.01, -0.02, 0.03], device=device, dtype=dtype, requires_grad=True
    )
    start_rotation = torch.tensor(
        [0.999825, 0.01, -0.015, 0.005],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if operation == "image_points_to_world_rays_static_pose":
        return start_translation, start_rotation
    if operation in _DYNAMIC_POSE_OPERATIONS:
        end_translation = torch.tensor(
            [0.08, 0.01, -0.02], device=device, dtype=dtype, requires_grad=True
        )
        end_rotation = torch.tensor(
            [0.99975, -0.008, 0.02, -0.006],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        return start_translation, start_rotation, end_translation, end_rotation
    return ()


def _operation_output(
    operation: str,
    primary_input: Tensor,
    projection,
    distortion,
    pose_inputs: tuple[Tensor, ...],
) -> Tensor:
    if operation == "camera_rays_to_image_points":
        return camera_rays_to_image_points(
            primary_input,
            projection,
            distortion,
            allow_device_transfer=True,
        )[0]
    if operation == "image_points_to_camera_rays":
        return image_points_to_camera_rays(
            primary_input,
            projection,
            distortion,
            allow_device_transfer=True,
        )
    if operation == "image_points_to_world_rays_static_pose":
        translation, rotation = pose_inputs
        return image_points_to_world_rays_static_pose(
            primary_input,
            projection,
            distortion,
            Pose(translation=translation, rotation=rotation),
            allow_device_transfer=True,
        )[0]

    start_t, start_r, end_t, end_r = pose_inputs
    pose = DynamicPose(
        Pose(translation=start_t, rotation=start_r),
        Pose(translation=end_t, rotation=end_r),
    )
    if operation == "project_world_points_mean_pose":
        return project_world_points_mean_pose(
            primary_input,
            projection,
            distortion,
            pose,
            (100, 80),
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]
    if operation == "project_world_points_shutter_pose":
        return project_world_points_shutter_pose(
            primary_input,
            projection,
            distortion,
            (100, 80),
            ShutterType.GLOBAL,
            pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            max_iterations=1,
            initial_relative_time=0.5,
            allow_device_transfer=True,
        )[0]
    if operation == "image_points_to_world_rays_shutter_pose":
        return image_points_to_world_rays_shutter_pose(
            primary_input,
            projection,
            distortion,
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            pose,
            start_timestamp_us=0,
            end_timestamp_us=10,
            allow_device_transfer=True,
        )[0]
    raise AssertionError(f"unsupported operation: {operation}")


def _joint_bivariate_gradcheck(
    operation: str,
    projection,
    windshield_distortion,
    external_reference: ReferencePolynomial,
) -> bool:
    primary_input = _operation_primary_input(
        operation, projection, count=1, dtype=torch.float64
    )
    pose_inputs = _operation_pose_inputs(
        operation, projection.principal_point.device, dtype=torch.float64
    )
    distortion_coeffs = _nonidentity_distortion_coeffs(
        windshield_distortion, dtype=torch.float64, requires_grad=True
    )

    def fn(*inputs):
        primary = inputs[0]
        poses = tuple(
            torch.nn.functional.normalize(value, dim=0)
            if value.shape == (4,)
            else value
            for value in inputs[1:-1]
        )
        coeffs = inputs[-1]
        distortion = _build_distortion(
            windshield_distortion, external_reference, coeffs
        )
        return _operation_output(operation, primary, projection, distortion, poses)

    if operation == "camera_rays_to_image_points":
        kwargs = GRADCHECK_KWARGS_FTHETA_PROJECT
    elif operation in _DYNAMIC_POSE_OPERATIONS:
        kwargs = GRADCHECK_KWARGS_FTHETA_DYNAMIC
    else:
        kwargs = GRADCHECK_KWARGS
    return torch.autograd.gradcheck(
        fn,
        (primary_input, *pose_inputs, distortion_coeffs),
        **kwargs,
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("operation", _DISTORTION_POLICY_OPERATIONS)
@pytest.mark.parametrize(
    "external_reference",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_fisheye_bivariate_joint_gradcheck_all_public_operations(
    operation,
    external_reference,
    fisheye_projection,
    windshield_distortion,
):
    """Jointly check each fisheye primary, pose, and coefficient VJP."""
    assert _joint_bivariate_gradcheck(
        operation,
        fisheye_projection,
        windshield_distortion,
        external_reference,
    )


@pytest.mark.gradcheck
@pytest.mark.parametrize("operation", _DISTORTION_POLICY_OPERATIONS)
@pytest.mark.parametrize(
    "projection_reference",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
@pytest.mark.parametrize(
    "external_reference",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_ftheta_bivariate_joint_gradcheck_reference_cross_product(
    operation,
    projection_reference,
    external_reference,
    ftheta_projection_forward_ref,
    windshield_distortion,
):
    """Check every FTheta intrinsic/external reference selector combination."""
    projection = _ftheta_projection_with_reference(
        ftheta_projection_forward_ref, projection_reference
    )
    assert _joint_bivariate_gradcheck(
        operation,
        projection,
        windshield_distortion,
        external_reference,
    )


def _active_distortion_slice(
    operation: str, reference_polynomial: ReferencePolynomial
) -> slice:
    forward_half_is_active = (operation in _PROJECT_LIKE_OPERATIONS) == (
        reference_polynomial == ReferencePolynomial.FORWARD
    )
    return slice(0, 21) if forward_half_is_active else slice(21, 42)


@pytest.mark.parametrize("operation", _DISTORTION_POLICY_OPERATIONS)
@pytest.mark.parametrize(
    "external_reference",
    [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD],
)
def test_ftheta_active_coefficient_half_all_public_operations(
    operation,
    external_reference,
    ftheta_projection_forward_ref,
    windshield_distortion,
):
    """Every FTheta operation must reduce only its selected coefficient half."""
    coeffs = _nonidentity_distortion_coeffs(
        windshield_distortion, dtype=torch.float32, requires_grad=True
    )
    distortion = _build_distortion(windshield_distortion, external_reference, coeffs)
    primary_input = _operation_primary_input(
        operation, ftheta_projection_forward_ref, count=2, dtype=torch.float32
    )
    pose_inputs = _operation_pose_inputs(
        operation,
        ftheta_projection_forward_ref.principal_point.device,
        dtype=torch.float32,
    )
    output = _operation_output(
        operation,
        primary_input,
        ftheta_projection_forward_ref,
        distortion,
        pose_inputs,
    )
    weights = torch.linspace(
        0.3, 1.1, output.shape[-1], device=output.device, dtype=output.dtype
    )
    (output * weights).sum().backward()
    torch.cuda.synchronize(output.device)

    assert coeffs.grad is not None
    active = _active_distortion_slice(operation, external_reference)
    inactive = slice(21, 42) if active.start == 0 else slice(0, 21)
    assert coeffs.grad[active].abs().sum() > 0
    assert torch.count_nonzero(coeffs.grad[inactive]).item() == 0


def _partial_block_projection(sensor: str, base_projection):
    if sensor == "ftheta":
        principal_point, fw_poly, bw_poly, A = (
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                base_projection.principal_point,
                base_projection.fw_poly,
                base_projection.bw_poly,
                base_projection.A,
            )
        )
        projection = _ftheta_projection_from_grad_tensors(
            base_projection, principal_point, fw_poly, bw_poly, A
        )
        return projection, (principal_point, fw_poly, bw_poly, A)

    principal_point, focal_length, forward_poly = (
        tensor.detach().clone().requires_grad_(True)
        for tensor in (
            base_projection.principal_point,
            base_projection.focal_length,
            base_projection.forward_poly,
        )
    )
    projection = _fisheye_projection_from_grad_tensors(
        base_projection,
        principal_point,
        focal_length,
        forward_poly,
        base_projection.approx_backward_factor.detach().clone(),
    )
    return projection, (principal_point, focal_length, forward_poly)


def _partial_block_state(
    *,
    sensor: str,
    distortion_policy: str,
    operation: str,
    count: int,
    base_projection,
    no_external,
    windshield_distortion,
):
    projection, intrinsic_inputs = _partial_block_projection(sensor, base_projection)
    primary_input = _operation_primary_input(
        operation, projection, count=count, dtype=torch.float32
    )
    pose_inputs = _operation_pose_inputs(
        operation, projection.principal_point.device, dtype=torch.float32
    )
    if distortion_policy == "bivariate":
        distortion_coeffs = _nonidentity_distortion_coeffs(
            windshield_distortion, dtype=torch.float32, requires_grad=True
        )
        distortion = _build_distortion(
            windshield_distortion,
            ReferencePolynomial.FORWARD,
            distortion_coeffs,
        )
    else:
        distortion_coeffs = None
        distortion = no_external
    return (
        primary_input,
        projection,
        distortion,
        pose_inputs,
        intrinsic_inputs,
        distortion_coeffs,
    )


def _backward_output_lane(
    operation: str,
    primary_input: Tensor,
    projection,
    distortion,
    pose_inputs: tuple[Tensor, ...],
    output_index: int,
) -> None:
    output = _operation_output(
        operation, primary_input, projection, distortion, pose_inputs
    )
    weights = torch.linspace(
        0.3, 1.1, output.shape[-1], device=output.device, dtype=output.dtype
    )
    (output[output_index] * weights).sum().backward()
    torch.cuda.synchronize(output.device)


def _assert_partial_block_grad_matches(
    actual: Tensor,
    expected: Tensor,
    *,
    label: str,
) -> None:
    assert actual.grad is not None, label
    assert expected.grad is not None, label
    assert torch.isfinite(actual.grad).all(), label
    assert torch.isfinite(expected.grad).all(), label
    assert torch.allclose(actual.grad, expected.grad, atol=3e-4, rtol=2e-3), label


@pytest.mark.parametrize("sensor", ["fisheye", "ftheta"])
@pytest.mark.parametrize("distortion_policy", ["no_external", "bivariate"])
@pytest.mark.parametrize("operation", _DISTORTION_POLICY_OPERATIONS)
@pytest.mark.parametrize("count", [255, 257])
def test_backward_partial_blocks_cover_every_specialization(
    sensor,
    distortion_policy,
    operation,
    count,
    fisheye_projection,
    ftheta_projection_forward_ref,
    no_external,
    windshield_distortion,
):
    """The last lane/block must contribute every applicable backward output."""
    base_projection = (
        fisheye_projection if sensor == "fisheye" else ftheta_projection_forward_ref
    )
    batched = _partial_block_state(
        sensor=sensor,
        distortion_policy=distortion_policy,
        operation=operation,
        count=count,
        base_projection=base_projection,
        no_external=no_external,
        windshield_distortion=windshield_distortion,
    )
    reference = _partial_block_state(
        sensor=sensor,
        distortion_policy=distortion_policy,
        operation=operation,
        count=256,
        base_projection=base_projection,
        no_external=no_external,
        windshield_distortion=windshield_distortion,
    )
    reference_lane = (count - 1) % 256

    _backward_output_lane(operation, *batched[:4], -1)
    _backward_output_lane(operation, *reference[:4], reference_lane)

    batched_primary, _, _, batched_poses, batched_intrinsics, batched_coeffs = batched
    (
        reference_primary,
        _,
        _,
        reference_poses,
        reference_intrinsics,
        reference_coeffs,
    ) = reference
    assert batched_primary.grad is not None
    assert reference_primary.grad is not None
    assert torch.count_nonzero(batched_primary.grad[:-1]).item() == 0
    assert torch.count_nonzero(reference_primary.grad[:reference_lane]).item() == 0
    assert torch.count_nonzero(reference_primary.grad[reference_lane + 1 :]).item() == 0
    assert batched_primary.grad[-1].abs().sum() > 0
    assert reference_primary.grad[reference_lane].abs().sum() > 0
    assert torch.allclose(
        batched_primary.grad[-1],
        reference_primary.grad[reference_lane],
        atol=3e-4,
        rtol=2e-3,
    )

    for index, (actual, expected) in enumerate(
        zip(batched_intrinsics, reference_intrinsics, strict=True)
    ):
        _assert_partial_block_grad_matches(
            actual, expected, label=f"intrinsic[{index}]"
        )
        if sensor == "fisheye" or index != 2:
            assert expected.grad.abs().sum() > 0, f"intrinsic[{index}]"

    for index, (actual, expected) in enumerate(
        zip(batched_poses, reference_poses, strict=True)
    ):
        _assert_partial_block_grad_matches(actual, expected, label=f"pose[{index}]")
        assert expected.grad.abs().sum() > 0, f"pose[{index}]"

    if distortion_policy == "bivariate":
        assert batched_coeffs is not None
        assert reference_coeffs is not None
        _assert_partial_block_grad_matches(
            batched_coeffs, reference_coeffs, label="distortion"
        )
        active = _active_distortion_slice(operation, ReferencePolynomial.FORWARD)
        inactive = slice(21, 42) if active.start == 0 else slice(0, 21)
        assert batched_coeffs.grad[active].abs().sum() > 0
        assert reference_coeffs.grad[active].abs().sum() > 0
        assert torch.count_nonzero(batched_coeffs.grad[inactive]).item() == 0
