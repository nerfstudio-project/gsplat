# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the gsplat-sensors test suite.

Provides: ``sensor_device``, ``test_camera_params``, ``real_camera_record``,
``real_camera_projection``, ``ideal_projection``, ``distorted_projection``,
``no_external``, ``windshield_distortion``, ``static_pose``, ``dynamic_pose``,
``pinhole_model``, ``pinhole_model_with_windshield``.

Session-scoped CUDA fixtures (``sensor_device``, ``real_camera_projection``) skip
automatically when no GPU is available.  ``_seed_test_rng`` and
``_cleanup_cuda_after_module`` are autouse and require no explicit request.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import torch

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
TEST_CAMERA_PARAMS_PATH = TEST_DATA_DIR / "test_pinhole_camera_params.json"


def _load_test_camera_params() -> dict:
    with TEST_CAMERA_PARAMS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


try:
    TEST_CAMERA_PARAMS = _load_test_camera_params()
except FileNotFoundError:
    TEST_CAMERA_PARAMS = None
TEST_CAMERA_IDS = (
    tuple(sorted(TEST_CAMERA_PARAMS)) if TEST_CAMERA_PARAMS is not None else ()
)


@pytest.fixture(autouse=True)
def _seed_test_rng():
    """Set a fixed random seed (42) before each test for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture(autouse=True, scope="module")
def _cleanup_cuda_after_module():
    """Release CUDA memory after each test module to prevent cross-module leaks."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_sensor_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("gsplat_sensors CUDA tests require CUDA")


@pytest.fixture(scope="session")
def sensor_device() -> torch.device:
    """Return the CUDA device for the session; skips the test suite if no GPU is found."""
    _require_sensor_cuda()
    return torch.device("cuda")


@pytest.fixture(scope="session")
def test_camera_params() -> dict:
    """Return the full parsed JSON dict of test camera parameters; skips if the file is absent."""
    if TEST_CAMERA_PARAMS is None:
        pytest.skip("test data not available")
    return TEST_CAMERA_PARAMS


@pytest.fixture(
    scope="session",
    params=list(TEST_CAMERA_IDS) or [None],
    ids=lambda key: key.split("@", 1)[0] if key is not None else "no-real-camera-json",
)
def real_camera_record(request):
    """Yield one raw camera record dict per camera ID found in the test JSON file."""
    if TEST_CAMERA_PARAMS is None or request.param is None:
        pytest.skip("test data not available")
    return TEST_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="session")
def real_camera_projection(real_camera_record, sensor_device: torch.device):
    """Build an OpenCVPinholeProjection from a real_camera_record on the session CUDA device."""
    if TEST_CAMERA_PARAMS is None:
        pytest.skip("test data not available")
    from gsplat_sensors.kernels.cameras import OpenCVPinholeProjection

    intrinsics = real_camera_record["intrinsics"]
    return OpenCVPinholeProjection(
        focal_length=torch.tensor(
            intrinsics["focal_length"], device=sensor_device, dtype=torch.float32
        ),
        principal_point=torch.tensor(
            intrinsics["principal_point"], device=sensor_device, dtype=torch.float32
        ),
        radial_coeffs=torch.tensor(
            intrinsics["radial_coeffs"], device=sensor_device, dtype=torch.float32
        ),
        tangential_coeffs=torch.tensor(
            intrinsics["tangential_coeffs"], device=sensor_device, dtype=torch.float32
        ),
        thin_prism_coeffs=torch.tensor(
            intrinsics["thin_prism_coeffs"], device=sensor_device, dtype=torch.float32
        ),
        resolution=tuple(real_camera_record["resolution"]),
    )


@pytest.fixture
def ideal_projection(sensor_device: torch.device):
    """Return a distortion-free OpenCVPinholeProjection (100x80, f=[100,120], pp=[50,40])."""
    from gsplat_sensors.kernels.cameras import OpenCVPinholeProjection

    return OpenCVPinholeProjection(
        focal_length=torch.tensor([100.0, 120.0], device=sensor_device),
        principal_point=torch.tensor([50.0, 40.0], device=sensor_device),
        radial_coeffs=torch.zeros(6, device=sensor_device),
        tangential_coeffs=torch.zeros(2, device=sensor_device),
        thin_prism_coeffs=torch.zeros(4, device=sensor_device),
        resolution=(100, 80),
    )


@pytest.fixture
def distorted_projection(sensor_device: torch.device):
    """Return an OpenCVPinholeProjection with small radial, tangential, and thin-prism coefficients."""
    from gsplat_sensors.kernels.cameras import OpenCVPinholeProjection

    return OpenCVPinholeProjection(
        focal_length=torch.tensor([100.0, 120.0], device=sensor_device),
        principal_point=torch.tensor([50.0, 40.0], device=sensor_device),
        radial_coeffs=torch.tensor(
            [0.01, -0.002, 0.0001, 0.0, 0.0, 0.0], device=sensor_device
        ),
        tangential_coeffs=torch.tensor([0.001, -0.002], device=sensor_device),
        thin_prism_coeffs=torch.tensor(
            [0.0001, 0.0, -0.0001, 0.0], device=sensor_device
        ),
        resolution=(100, 80),
    )


@pytest.fixture
def no_external():
    """Return a NoExternalDistortion instance; skips if CUDA is unavailable."""
    _require_sensor_cuda()
    from gsplat_sensors.kernels.cameras import NoExternalDistortion

    return NoExternalDistortion()


@pytest.fixture
def windshield_distortion(sensor_device: torch.device):
    """Identity-equivalent windshield distortion at order 1.

    With ``eval_poly_2d`` order-1 layout ``c0 + c1*phi + c2*theta`` (see
    ``external_distortion_kernel.cuh::eval_poly_2d``), the coefficients
    ``h_poly = [0, 1, 0]`` and ``v_poly = [0, 0, 1]`` evaluate to
    ``adj_phi = phi`` and ``adj_theta = theta``. ``apply_bivariate_distortion``
    then computes ``sin(asin(ray_norm.{x,y})) = ray_norm.{x,y}``, which is an
    identity *up to* the unit-sphere normalization that the subsequent
    pinhole ``(x/z, y/z)`` step cancels. Matched FWD/INV components share the
    same identity so round-trip and forward equivalence both hold.
    """
    from gsplat_sensors.kernels.cameras import ReferencePolynomial, from_components

    h_poly = torch.tensor([0.0, 1.0, 0.0], device=sensor_device)
    v_poly = torch.tensor([0.0, 0.0, 1.0], device=sensor_device)
    return from_components(
        h_poly,
        v_poly,
        h_poly.clone(),
        v_poly.clone(),
        ReferencePolynomial.FORWARD,
    )


@pytest.fixture
def static_pose(sensor_device: torch.device):
    """Return an identity Pose (zero translation, unit quaternion) on the sensor device."""
    from gsplat_sensors.kernels.common import Pose

    return Pose(
        translation=torch.zeros(3, device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )


@pytest.fixture
def dynamic_pose(sensor_device: torch.device):
    """Return a DynamicPose that translates 0.1 m in X from start to end, both with identity rotation."""
    from gsplat_sensors.kernels.common import DynamicPose, Pose

    start = Pose(
        translation=torch.zeros(3, device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )
    end = Pose(
        translation=torch.tensor([0.1, 0.0, 0.0], device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )
    return DynamicPose(start, end)


@pytest.fixture
def pinhole_model(ideal_projection, no_external):
    """Return a global-shutter CameraModel backed by ideal_projection and no_external."""
    from gsplat_sensors.kernels.cameras import ShutterType
    from gsplat_sensors.models import CameraModel

    return CameraModel(
        projection=ideal_projection,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


@pytest.fixture
def pinhole_model_with_windshield(ideal_projection, windshield_distortion):
    """Return a global-shutter CameraModel backed by ideal_projection and windshield_distortion."""
    from gsplat_sensors.kernels.cameras import ShutterType
    from gsplat_sensors.models import CameraModel

    return CameraModel(
        projection=ideal_projection,
        external_distortion=windshield_distortion,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )
