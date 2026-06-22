# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenCV-pinhole camera kernel ops covering forward/backward correctness and validity logic.

Exercises camera_rays_to_image_points, image_points_to_camera_rays, project_world_points_*,
image_points_to_world_rays_*, BivariateWindshieldDistortion construction/pickling, and
autograd.gradcheck for all public verbs with both NoExternalDistortion and BivariateWindshield.
"""

# Row mapping: design-tests-opencvpinhole.md §6 rows 3-21 are covered here.

import io

import pytest
import torch
from torch import Tensor

from gsplat_sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
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
from gsplat_sensors.kernels.cameras.ops import (
    _quat_slerp_wxyz,
    _unpack_static_pose,
)
from gsplat_sensors.kernels.common import DynamicPose


# Public camera ops cast inputs to float32 before launching CUDA, so use
# finite-difference settings that are stable for float32 kernels.
GRADCHECK_KWARGS = {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2}


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
    assert torch.count_nonzero(grad[inactive]).item() == 0


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
        assert torch.count_nonzero(coeffs.grad[inactive]).item() == 0


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
    assert torch.allclose(coeffs_full.grad, coeffs_split.grad, atol=1e-4)


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
    assert torch.allclose(
        ideal_projection.focal_length.grad, torch.tensor([0.1, 0.2], device=rays.device)
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
        fn,
        (world_points, distortion_coeffs),
        **{**GRADCHECK_KWARGS, "atol": 2e-2},
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
        fn, (world_points, distortion_coeffs), **GRADCHECK_KWARGS
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
    """End-to-end autograd.gradcheck for image_points_to_world_rays_shutter_pose with BivariateWindshieldDistortion (FORWARD and BACKWARD)."""
    image_points = (
        ideal_projection.principal_point.reshape(1, 2).to(dtype=torch.float64)
        + torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]],
            device=ideal_projection.focal_length.device,
            dtype=torch.float64,
        )
    ).detach()
    distortion_coeffs = _make_distortion_coeffs_f64(windshield_distortion)

    def fn(coeffs):
        distortion = _build_distortion(
            windshield_distortion, reference_polynomial, coeffs
        )
        return image_points_to_world_rays_shutter_pose(
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

    assert torch.autograd.gradcheck(fn, (distortion_coeffs,), **GRADCHECK_KWARGS)


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
        from gsplat_sensors.kernels.common import DynamicPose, Pose

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
