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

"""Shared pytest fixtures for the gsplat-sensors test suite.

Provides: ``sensor_device``, ``test_camera_params``, ``real_camera_record``,
``real_camera_projection``, ``ideal_projection``, ``distorted_projection``,
``no_external``, ``windshield_distortion``, ``static_pose``, ``dynamic_pose``,
``pinhole_model``, ``pinhole_model_with_windshield``, ``ftheta_projection``,
``ftheta_projection_forward_ref``, ``ftheta_projection_backward_ref``,
``ftheta_model``, ``ftheta_model_with_windshield``, ``real_ftheta_camera_record``,
``real_ftheta_projection``, ``real_ftheta_camera_record_with_windshield``,
``real_ftheta_windshield_distortion``, ``real_ftheta_projection_with_windshield``,
``fisheye_projection``, ``real_fisheye_camera_record``, ``real_fisheye_projection``,
``real_fisheye_camera_record_with_windshield``,
``real_fisheye_projection_with_windshield``, ``real_fisheye_windshield_distortion``.

Session-scoped CUDA fixtures (``sensor_device``, ``real_camera_projection``)
skip only when automatic CUDA detection is configured and finds no device.
Forced coverage runs them so an unavailable GPU fails visibly.
``_seed_test_rng`` and ``_cleanup_cuda_after_module`` are autouse and require
no explicit request.
"""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path

import pytest
import torch

from tests._cuda import cuda_is_available

from .._backend_collect import cuda_collect_ignore_glob

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
TEST_CAMERA_PARAMS_PATH = TEST_DATA_DIR / "test_pinhole_camera_params.json"
TEST_FTHETA_CAMERA_PARAMS_PATH = TEST_DATA_DIR / "test_ftheta_camera_params.json"
TEST_FISHEYE_CAMERA_PARAMS_PATH = TEST_DATA_DIR / "test_fisheye_camera_params.json"


def _load_json_or_none(path: Path) -> dict | None:
    """Module-level JSON loader guarded against missing files."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


TEST_CAMERA_PARAMS = _load_json_or_none(TEST_CAMERA_PARAMS_PATH)
TEST_CAMERA_IDS = (
    tuple(sorted(TEST_CAMERA_PARAMS)) if TEST_CAMERA_PARAMS is not None else (None,)
)
TEST_FTHETA_CAMERA_PARAMS = _load_json_or_none(TEST_FTHETA_CAMERA_PARAMS_PATH)
TEST_FTHETA_CAMERA_IDS = (
    tuple(sorted(TEST_FTHETA_CAMERA_PARAMS))
    if TEST_FTHETA_CAMERA_PARAMS is not None
    else (None,)
)
TEST_FISHEYE_CAMERA_PARAMS = _load_json_or_none(TEST_FISHEYE_CAMERA_PARAMS_PATH)
TEST_FISHEYE_CAMERA_IDS = (
    tuple(sorted(TEST_FISHEYE_CAMERA_PARAMS))
    if TEST_FISHEYE_CAMERA_PARAMS is not None
    else (None,)
)


@pytest.fixture(autouse=True)
def _seed_test_rng():
    """Set a fixed random seed (42) before each test for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture(autouse=True, scope="module")
def _cleanup_cuda_after_module():
    """Release CUDA memory after each test module to prevent cross-module leaks."""
    yield
    gc.collect()
    if cuda_is_available():
        torch.cuda.empty_cache()


def _require_sensor_cuda() -> None:
    if not cuda_is_available():
        pytest.skip("gsplat_sensors CUDA tests require CUDA")


# gsplat.sensors binds the native extension at import time: kernels/cameras/
# types.py evaluates ``torch.classes.gsplat_sensors.*`` at module scope, so a
# sensor test cannot be collected unless the extension loads. Automatic CUDA
# detection may de-collect the directory when no GPU is present. Otherwise the
# probe imports the binding module and deliberately propagates an error if the
# extension is unavailable, preventing a broken build from becoming a silent
# coverage gap.
collect_ignore_glob = cuda_collect_ignore_glob(
    probe=lambda: importlib.import_module("gsplat.sensors.kernels.cameras.types")
)


@pytest.fixture(scope="session")
def sensor_device() -> torch.device:
    """Return CUDA, skipping only when automatic detection finds no GPU."""
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
    if request.param is None or TEST_CAMERA_PARAMS is None:
        pytest.skip("test_pinhole_camera_params.json not available")
    return TEST_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="session")
def real_camera_projection(real_camera_record, sensor_device: torch.device):
    """Build an OpenCVPinholeProjection from a real_camera_record on the session CUDA device."""
    if TEST_CAMERA_PARAMS is None:
        pytest.skip("test data not available")
    from gsplat.sensors.kernels.cameras import OpenCVPinholeProjection

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
    from gsplat.sensors.kernels.cameras import OpenCVPinholeProjection

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
    from gsplat.sensors.kernels.cameras import OpenCVPinholeProjection

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
    from gsplat.sensors.kernels.cameras import NoExternalDistortion

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
    from gsplat.sensors.kernels.cameras import ReferencePolynomial, from_components

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
    from gsplat.sensors.kernels.common import Pose

    return Pose(
        translation=torch.zeros(3, device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )


@pytest.fixture
def dynamic_pose(sensor_device: torch.device):
    """Return a DynamicPose that translates 0.1 m in X from start to end, both with identity rotation."""
    from gsplat.sensors.kernels.common import DynamicPose, Pose

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
    from gsplat.sensors.kernels.cameras import ShutterType
    from gsplat.sensors.models import CameraModel

    return CameraModel(
        projection=ideal_projection,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


@pytest.fixture
def pinhole_model_with_windshield(ideal_projection, windshield_distortion):
    """Return a global-shutter CameraModel backed by ideal_projection and windshield_distortion."""
    from gsplat.sensors.kernels.cameras import ShutterType
    from gsplat.sensors.models import CameraModel

    return CameraModel(
        projection=ideal_projection,
        external_distortion=windshield_distortion,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


# ===========================================================================
# F-Theta fixtures
# ===========================================================================


def _make_ideal_ftheta_components(device: torch.device, dtype: torch.dtype):
    """Synthetic linear-polynomial F-Theta intrinsics for unit tests.

    The polynomials reduce to ``r = k * theta`` and ``theta = r / k`` so
    Newton inversion converges in one step and the round-trip is exact
    up to float32 rounding. The 2x2 warp ``A`` is identity.
    """
    k = 100.0
    fw_poly = torch.zeros(6, device=device, dtype=dtype)
    fw_poly[1] = k
    bw_poly = torch.zeros(6, device=device, dtype=dtype)
    bw_poly[1] = 1.0 / k
    A = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    principal_point = torch.tensor([50.0, 40.0], device=device, dtype=dtype)
    return {
        "principal_point": principal_point,
        "fw_poly": fw_poly,
        "bw_poly": bw_poly,
        "A": A,
        "resolution": (100, 80),
        "fw_poly_degree": 1,
        "bw_poly_degree": 1,
        "newton_iterations": 10,
        "max_angle": 1.4,
        "min_2d_norm": 1e-6,
    }


def _make_ftheta_projection(device: torch.device, reference_polynomial: int):
    from gsplat.sensors.kernels.cameras import FThetaProjection

    components = _make_ideal_ftheta_components(device, dtype=torch.float32)
    return FThetaProjection(
        principal_point=components["principal_point"],
        fw_poly=components["fw_poly"],
        bw_poly=components["bw_poly"],
        A=components["A"],
        resolution=components["resolution"],
        reference_polynomial=int(reference_polynomial),
        fw_poly_degree=components["fw_poly_degree"],
        bw_poly_degree=components["bw_poly_degree"],
        newton_iterations=components["newton_iterations"],
        max_angle=components["max_angle"],
        min_2d_norm=components["min_2d_norm"],
    )


@pytest.fixture(
    params=[0, 1],
    ids=["forward_ref", "backward_ref"],
)
def ftheta_projection(request, sensor_device: torch.device):
    """Linear-polynomial F-Theta projection. Parametrized over reference_polynomial."""
    return _make_ftheta_projection(sensor_device, request.param)


def _make_fisheye_projection(device: torch.device, k1: float = 0.0):
    """Synthetic mild OpenCV-fisheye intrinsics for unit tests.

    With ``forward_poly`` zeroed the equidistant map reduces to ``delta = theta``,
    so the Newton inversion is exact and the round-trip is smooth for gradcheck.
    A nonzero ``k1`` exercises the odd-power polynomial adjoints.
    """
    from gsplat.sensors.kernels.cameras import OpenCVFisheyeProjection

    return OpenCVFisheyeProjection(
        principal_point=torch.tensor([50.0, 40.0], device=device),
        focal_length=torch.tensor([100.0, 100.0], device=device),
        forward_poly=torch.tensor([k1, 0.0, 0.0, 0.0], device=device),
        approx_backward_factor=torch.tensor([1.0], device=device),
        resolution=(100, 80),
        newton_iterations=10,
        max_angle=1.8,
        min_2d_norm=1e-6,
    )


@pytest.fixture
def fisheye_projection(sensor_device: torch.device):
    """Mild equidistant OpenCV-fisheye projection (identity distortion)."""
    return _make_fisheye_projection(sensor_device)


@pytest.fixture
def ftheta_projection_forward_ref(sensor_device: torch.device):
    """F-Theta projection with reference_polynomial=FORWARD (no Newton on forward)."""
    return _make_ftheta_projection(sensor_device, 0)


@pytest.fixture
def ftheta_projection_backward_ref(sensor_device: torch.device):
    """F-Theta projection with reference_polynomial=BACKWARD (Newton on forward path)."""
    return _make_ftheta_projection(sensor_device, 1)


@pytest.fixture
def ftheta_model(ftheta_projection_forward_ref, no_external):
    """F-Theta camera model with NoExternalDistortion. FORWARD-ref to keep tests fast."""
    from gsplat.sensors.kernels.cameras import ShutterType
    from gsplat.sensors.models import CameraModel

    return CameraModel(
        projection=ftheta_projection_forward_ref,
        external_distortion=no_external,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


@pytest.fixture
def ftheta_model_with_windshield(ftheta_projection_forward_ref, windshield_distortion):
    """F-Theta camera model layered with the identity-equivalent windshield distortion."""
    from gsplat.sensors.kernels.cameras import ShutterType
    from gsplat.sensors.models import CameraModel

    return CameraModel(
        projection=ftheta_projection_forward_ref,
        external_distortion=windshield_distortion,
        resolution=(100, 80),
        shutter_type=ShutterType.GLOBAL,
    )


# Map the JSON ``reference_poly`` field under ``intrinsics`` onto the gsplat
# ReferencePolynomial enum. Source encoding:
#   "1" = PIXELDIST_TO_ANGLE  -> gsplat BACKWARD (Newton on fw_poly)
#   "2" = ANGLE_TO_PIXELDIST  -> gsplat FORWARD (use fw_poly directly)
_JSON_INTRINSICS_REF_POLY_TO_GSPLAT = {"1": 1, "2": 0}

# Map the JSON ``reference_poly`` field under ``external_distortion`` onto the
# gsplat ReferencePolynomial enum. This is a DIFFERENT encoding from the
# intrinsics-side mapping above: the windshield field uses FORWARD=1 /
# BACKWARD=2, while gsplat's enum is FORWARD=0 / BACKWARD=1.
_JSON_WINDSHIELD_REF_POLY_TO_GSPLAT = {"1": 0, "2": 1}


def _build_real_ftheta_projection(record: dict, device: torch.device):
    """Construct an ``FThetaProjection`` from a real-camera intrinsics record.

    Shared by ``real_ftheta_projection`` and the windshield-augmented
    fixtures so both use exactly the same intrinsic build path.
    """
    from gsplat.sensors.kernels.cameras import (
        FTHETA_MAX_POLYNOMIAL_TERMS,
        FThetaProjection,
    )

    intrinsics = record["intrinsics"]
    fw_poly = list(intrinsics["angle_to_pixeldist_poly"])
    bw_poly = list(intrinsics["pixeldist_to_angle_poly"])
    fw_degree = len(fw_poly) - 1
    bw_degree = len(bw_poly) - 1
    if len(fw_poly) > FTHETA_MAX_POLYNOMIAL_TERMS:
        raise ValueError(
            f"fw_poly degree exceeds kernel max ({FTHETA_MAX_POLYNOMIAL_TERMS}); "
            f"got {len(fw_poly)}"
        )
    if len(bw_poly) > FTHETA_MAX_POLYNOMIAL_TERMS:
        raise ValueError(
            f"bw_poly degree exceeds kernel max ({FTHETA_MAX_POLYNOMIAL_TERMS}); "
            f"got {len(bw_poly)}"
        )
    fw_poly = fw_poly + [0.0] * (FTHETA_MAX_POLYNOMIAL_TERMS - len(fw_poly))
    bw_poly = bw_poly + [0.0] * (FTHETA_MAX_POLYNOMIAL_TERMS - len(bw_poly))
    c, d, e = intrinsics["linear_cde"]
    a_flat = [float(c), float(d), float(e), 1.0]
    return FThetaProjection(
        principal_point=torch.tensor(
            intrinsics["principal_point"], device=device, dtype=torch.float32
        ),
        fw_poly=torch.tensor(fw_poly, device=device, dtype=torch.float32),
        bw_poly=torch.tensor(bw_poly, device=device, dtype=torch.float32),
        A=torch.tensor(a_flat, device=device, dtype=torch.float32),
        resolution=tuple(record["resolution"]),
        reference_polynomial=_JSON_INTRINSICS_REF_POLY_TO_GSPLAT[
            intrinsics["reference_poly"]
        ],
        fw_poly_degree=fw_degree,
        bw_poly_degree=bw_degree,
        newton_iterations=10,
        max_angle=float(intrinsics["max_angle_rad"]),
        min_2d_norm=1e-6,
    )


@pytest.fixture(
    scope="session",
    params=TEST_FTHETA_CAMERA_IDS,
    ids=lambda key: key.split("@", 1)[0] if key is not None else "no-data",
)
def real_ftheta_camera_record(request):
    """Yield one raw F-Theta camera record dict per camera ID found in the test JSON file."""
    if request.param is None or TEST_FTHETA_CAMERA_PARAMS is None:
        pytest.skip("test_ftheta_camera_params.json not available")
    return TEST_FTHETA_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="session")
def real_ftheta_projection(real_ftheta_camera_record, sensor_device: torch.device):
    """Build an FThetaProjection from a real_ftheta_camera_record on the session CUDA device."""
    return _build_real_ftheta_projection(real_ftheta_camera_record, sensor_device)


# Filter the camera records to those carrying a bivariate-windshield
# external_distortion. ``or [None]`` keeps the parametrize list non-empty so
# pytest still reports a single parametrized id (``no-windshield-data``) and
# the fixture body skips cleanly when the JSON has no matching record.
_TEST_FTHETA_WINDSHIELD_IDS = [
    key
    for key in TEST_FTHETA_CAMERA_IDS
    if key is not None
    and TEST_FTHETA_CAMERA_PARAMS is not None
    and (TEST_FTHETA_CAMERA_PARAMS[key].get("external_distortion") or {}).get("type")
    == "bivariate-windshield"
] or [None]


@pytest.fixture(
    scope="module",
    params=_TEST_FTHETA_WINDSHIELD_IDS,
    ids=lambda key: key.split("@", 1)[0] if key is not None else "no-windshield-data",
)
def real_ftheta_camera_record_with_windshield(request):
    """Yield one raw F-Theta camera record dict that carries a bivariate-windshield external_distortion."""
    if request.param is None or TEST_FTHETA_CAMERA_PARAMS is None:
        pytest.skip("no bivariate-windshield record in test_ftheta_camera_params.json")
    return TEST_FTHETA_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="module")
def real_ftheta_windshield_distortion(
    real_ftheta_camera_record_with_windshield, sensor_device: torch.device
):
    """BivariateWindshieldDistortion built via ``from_components(int_code, ...)``.

    The ``reference_polynomial`` argument passed to ``from_components`` is the
    *integer* code from ``_JSON_WINDSHIELD_REF_POLY_TO_GSPLAT`` (0 or 1), not a
    ``ReferencePolynomial`` enum member.

    NOTE: the windshield ``reference_poly`` JSON field uses FORWARD=1 /
    BACKWARD=2, the OPPOSITE direction of the intrinsics ``reference_poly``
    (1=BACKWARD, 2=FORWARD). The distinct
    ``_JSON_WINDSHIELD_REF_POLY_TO_GSPLAT`` table above keeps the two
    from being conflated before the int_code reaches ``from_components``.
    """
    from gsplat.sensors.kernels.cameras import from_components

    ext = real_ftheta_camera_record_with_windshield["external_distortion"]
    if ext.get("type") != "bivariate-windshield":
        raise ValueError(
            f"expected bivariate-windshield external_distortion; got {ext.get('type')!r}"
        )
    h_poly = torch.tensor(
        ext["horizontal_poly"], device=sensor_device, dtype=torch.float32
    )
    v_poly = torch.tensor(
        ext["vertical_poly"], device=sensor_device, dtype=torch.float32
    )
    h_poly_inv = torch.tensor(
        ext["horizontal_poly_inverse"], device=sensor_device, dtype=torch.float32
    )
    v_poly_inv = torch.tensor(
        ext["vertical_poly_inverse"], device=sensor_device, dtype=torch.float32
    )
    return from_components(
        h_poly,
        v_poly,
        h_poly_inv,
        v_poly_inv,
        _JSON_WINDSHIELD_REF_POLY_TO_GSPLAT[ext["reference_poly"]],
    )


@pytest.fixture(scope="module")
def real_ftheta_projection_with_windshield(
    real_ftheta_camera_record_with_windshield, sensor_device: torch.device
):
    """Build an FThetaProjection from a windshield-carrying camera record on the sensor device."""
    return _build_real_ftheta_projection(
        real_ftheta_camera_record_with_windshield, sensor_device
    )


# ---------------------------------------------------------------------------
# Spinning-LiDAR fixtures
# ---------------------------------------------------------------------------

LIDAR_JSON_PATHS = {
    "generic": TEST_DATA_DIR / "row-offset-spinning-lidar-model-parameters.json",
    "waymo": TEST_DATA_DIR / "row-offset-spinning-lidar-model-parameters-waymo.json",
}
# generic carries per-row azimuth offsets; waymo has none -- together they cover
# both branches of the has_row_offsets kernel fork.
LIDAR_PRODUCTION_CONFIGS = ("generic", "waymo")


@pytest.fixture
def lidar_projection_from_json(sensor_device: torch.device):
    """Return a callable ``config -> projection`` built from a reference sensor JSON.

    FOV is derived from the angle tables (vertical from elevations; horizontal as
    a full sweep); spinning direction is read from the JSON.
    """
    import json as _json

    from gsplat.sensors.kernels.lidars.types import (
        RowOffsetStructuredSpinningLidarProjection,
        SpinningDirection,
    )

    def _factory(config: str = "generic"):
        path = LIDAR_JSON_PATHS[config]
        if not path.exists():
            pytest.fail(f"required lidar JSON not available: {path.name}")
        with path.open(encoding="utf-8") as f:
            params = _json.load(f)
        row_elev = torch.tensor(
            params["row_elevations_rad"], device=sensor_device, dtype=torch.float32
        )
        col_az = torch.tensor(
            params["column_azimuths_rad"], device=sensor_device, dtype=torch.float32
        )
        offsets = params.get("row_azimuth_offsets_rad")
        has_offsets = offsets is not None and any(v != 0 for v in offsets)
        row_offsets = (
            torch.tensor(offsets, device=sensor_device, dtype=torch.float32)
            if has_offsets
            else torch.zeros((0,), device=sensor_device, dtype=torch.float32)
        )
        fov_vert_start = float(row_elev.max().item())
        fov_vert_span = abs(fov_vert_start - float(row_elev.min().item()))
        raw_direction = params.get("spinning_direction", 0)
        if isinstance(raw_direction, str):
            text_to_direction = {
                "cw": SpinningDirection.CLOCKWISE,
                "ccw": SpinningDirection.COUNTERCLOCKWISE,
            }
            direction = text_to_direction.get(raw_direction.lower())
            if direction is None:
                pytest.fail(
                    f"invalid spinning_direction {raw_direction!r}; "
                    "expected 'cw' or 'ccw'"
                )
        elif raw_direction in (0, 1):
            direction = SpinningDirection(raw_direction)
        else:
            pytest.fail(
                f"invalid spinning_direction {raw_direction!r}; expected 0 or 1"
            )
        spinning_direction = int(direction)
        return RowOffsetStructuredSpinningLidarProjection(
            row_elev,
            col_az,
            row_offsets,
            fov_vert_start,
            fov_vert_span,
            -3.14159,
            6.28318,
            spinning_direction,
            has_offsets,
        )

    return _factory


@pytest.fixture
def lidar_model(lidar_projection_from_json):
    """Return a LidarModel backed by the generic reference sensor JSON."""
    from gsplat.sensors.models import LidarModel

    return LidarModel(projection=lidar_projection_from_json("generic"))


@pytest.fixture
def lidar_dynamic_pose(sensor_device: torch.device):
    """Return a DynamicPose translating 0.1 m in X, identity rotation, wxyz."""
    from gsplat.sensors.kernels.common import DynamicPose, Pose

    start = Pose(
        translation=torch.zeros(3, device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )
    end = Pose(
        translation=torch.tensor([0.1, 0.0, 0.0], device=sensor_device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=sensor_device),
    )
    return DynamicPose(start, end)


# ===========================================================================
# OpenCV-fisheye fixtures
# ===========================================================================


def _build_real_fisheye_projection(record: dict, device: torch.device):
    """Construct an ``OpenCVFisheyeProjection`` from a real-camera intrinsics record."""
    from gsplat.sensors.kernels.cameras import OpenCVFisheyeProjection

    intrinsics = record["intrinsics"]
    return OpenCVFisheyeProjection(
        principal_point=torch.tensor(
            intrinsics["principal_point"], device=device, dtype=torch.float32
        ),
        focal_length=torch.tensor(
            intrinsics["focal_length"], device=device, dtype=torch.float32
        ),
        forward_poly=torch.tensor(
            intrinsics["forward_poly"], device=device, dtype=torch.float32
        ),
        approx_backward_factor=torch.tensor([1.0], device=device, dtype=torch.float32),
        resolution=tuple(record["resolution"]),
        newton_iterations=10,
        max_angle=float(intrinsics["max_angle_rad"]),
        min_2d_norm=1e-6,
    )


@pytest.fixture(
    scope="session",
    params=TEST_FISHEYE_CAMERA_IDS,
    ids=lambda key: key.split("@", 1)[0] if key is not None else "no-data",
)
def real_fisheye_camera_record(request):
    """Yield one raw OpenCV-fisheye camera record dict per camera ID in the test JSON file."""
    if request.param is None or TEST_FISHEYE_CAMERA_PARAMS is None:
        pytest.skip("test_fisheye_camera_params.json not available")
    return TEST_FISHEYE_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="session")
def real_fisheye_projection(real_fisheye_camera_record, sensor_device: torch.device):
    """Build an OpenCVFisheyeProjection from a real_fisheye_camera_record on the session CUDA device."""
    return _build_real_fisheye_projection(real_fisheye_camera_record, sensor_device)


@pytest.fixture(scope="session")
def reference_opencv_fisheye_camera():
    """Expose the ``ReferenceOpenCVFisheyeCamera`` oracle class to tests."""
    return ReferenceOpenCVFisheyeCamera


# Filter the fisheye records to those carrying a bivariate-windshield
# external_distortion. ``or [None]`` keeps the parametrize list non-empty so
# the fixture still reports a single id and skips cleanly when absent.
_TEST_FISHEYE_WINDSHIELD_IDS = [
    key
    for key in TEST_FISHEYE_CAMERA_IDS
    if key is not None
    and TEST_FISHEYE_CAMERA_PARAMS is not None
    and (TEST_FISHEYE_CAMERA_PARAMS[key].get("external_distortion") or {}).get("type")
    == "bivariate-windshield"
] or [None]


@pytest.fixture(
    scope="module",
    params=_TEST_FISHEYE_WINDSHIELD_IDS,
    ids=lambda key: key.split("@", 1)[0] if key is not None else "no-windshield-data",
)
def real_fisheye_camera_record_with_windshield(request):
    """Yield one raw fisheye camera record dict that carries a bivariate-windshield external_distortion."""
    if request.param is None or TEST_FISHEYE_CAMERA_PARAMS is None:
        pytest.skip("no bivariate-windshield record in test_fisheye_camera_params.json")
    return TEST_FISHEYE_CAMERA_PARAMS[request.param]


@pytest.fixture(scope="module")
def real_fisheye_projection_with_windshield(
    real_fisheye_camera_record_with_windshield, sensor_device: torch.device
):
    """Build an OpenCVFisheyeProjection from a windshield-carrying fisheye record."""
    return _build_real_fisheye_projection(
        real_fisheye_camera_record_with_windshield, sensor_device
    )


@pytest.fixture(scope="module")
def real_fisheye_windshield_distortion(
    real_fisheye_camera_record_with_windshield, sensor_device: torch.device
):
    """BivariateWindshieldDistortion built via ``from_components`` from the fisheye JSON.

    Mirrors ``real_ftheta_windshield_distortion``: the windshield ``reference_poly``
    JSON field uses FORWARD=1 / BACKWARD=2, so it is resolved through
    ``_JSON_WINDSHIELD_REF_POLY_TO_GSPLAT`` before reaching ``from_components``.
    """
    from gsplat.sensors.kernels.cameras import from_components

    ext = real_fisheye_camera_record_with_windshield["external_distortion"]
    if ext.get("type") != "bivariate-windshield":
        raise ValueError(
            f"expected bivariate-windshield external_distortion; got {ext.get('type')!r}"
        )
    h_poly = torch.tensor(
        ext["horizontal_poly"], device=sensor_device, dtype=torch.float32
    )
    v_poly = torch.tensor(
        ext["vertical_poly"], device=sensor_device, dtype=torch.float32
    )
    h_poly_inv = torch.tensor(
        ext["horizontal_poly_inverse"], device=sensor_device, dtype=torch.float32
    )
    v_poly_inv = torch.tensor(
        ext["vertical_poly_inverse"], device=sensor_device, dtype=torch.float32
    )
    return from_components(
        h_poly,
        v_poly,
        h_poly_inv,
        v_poly_inv,
        _JSON_WINDSHIELD_REF_POLY_TO_GSPLAT[ext["reference_poly"]],
    )


class ReferenceOpenCVFisheyeCamera:
    """Reference implementation of OpenCV Fisheye camera model."""

    def __init__(
        self,
        focal_length,
        principal_point,
        radial_coeffs,
        max_angle: float,
        resolution,
        dtype=None,
    ):
        import numpy as np

        if dtype is None:
            dtype = np.float32
        self.focal_length = focal_length.astype(dtype)
        self.principal_point = principal_point.astype(dtype)
        self.radial_coeffs = radial_coeffs.astype(dtype)  # [k1, k2, k3, k4]
        self.max_angle = max_angle
        self.resolution = resolution
        self.dtype = dtype

    def camera_ray_to_image_point_opencv(self, ray):
        """Project ray using OpenCV's fisheye model.

        This uses OpenCV's cv2.fisheye.projectPoints for reference.
        """
        import cv2
        import numpy as np

        ray = np.array(ray, dtype=np.float64)

        if ray[2] <= 0:
            return np.array([0.0, 0.0], dtype=self.dtype), False

        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        K = np.array(
            [
                [self.focal_length[0], 0, self.principal_point[0]],
                [0, self.focal_length[1], self.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        d = self.radial_coeffs.astype(np.float64)

        try:
            p, _ = cv2.fisheye.projectPoints(
                ray.reshape(1, 1, 3), rvec, tvec, K, d, None, 0.0
            )
            image_point = p.reshape(2)

            ray_norm = ray / np.linalg.norm(ray)
            theta = np.arccos(np.clip(ray_norm[2], -1, 1))

            valid = (
                theta <= self.max_angle
                and 0 <= image_point[0] < self.resolution[0]
                and 0 <= image_point[1] < self.resolution[1]
            )

            return image_point.astype(self.dtype), valid
        except cv2.error:
            return np.array([0.0, 0.0], dtype=self.dtype), False
