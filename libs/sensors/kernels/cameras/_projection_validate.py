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

"""Opt-in value-level validation for camera projection components.

The C++ constructors validate cheap shape, dtype, and scalar constraints without
crossing the host/device boundary. The checks here are stricter value-level
invariants that require reading tensor entries, so callers opt in explicitly via
``validate_camera_projection``.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor

from .types import (
    FISHEYE_MAX_FORWARD_POLY_TERMS,
    FTHETA_MAX_POLYNOMIAL_TERMS,
    CameraProjection,
    FThetaProjection,
    OpenCVFisheyeProjection,
    script_class_name,
)


def _to_cpu_floats(t: Tensor, name: str, expected_numel: int) -> list[float]:
    if t.numel() != expected_numel:
        raise ValueError(f"{name} must have {expected_numel} elements, got {t.numel()}")
    return [float(x) for x in t.detach().to("cpu", torch.float32).flatten().tolist()]


def _check_finite(values: list[float], name: str) -> None:
    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise ValueError(f"{name}[{i}] must be finite; got {v!r}")


def _validate_ftheta_projection(projection: FThetaProjection) -> None:
    """Raise ``ValueError`` if ``projection`` violates a load-bearing invariant.

    Cheap shape and scalar-range checks (including ``max_angle`` and
    ``min_2d_norm``) are enforced by the C++ constructor; this function adds
    value-level checks that need a host sync:

    * Component tensors (``principal_point``, ``fw_poly``, ``bw_poly``, ``A``)
      contain only finite values.
    * ``fw_poly[0] == 0`` and ``bw_poly[0] == 0`` -- the radial polynomial must
      pass through the origin so the on-axis ray ``(0, 0, 1)`` projects exactly
      to the principal point.
    * ``A`` is non-singular (``|det(A)| > 1e-12``) -- the inverse is computed on
      demand from ``A`` (closed-form 2x2 inverse), so a zero determinant would
      yield a non-finite ``Ainv`` at kernel-launch time.
    """
    pp = _to_cpu_floats(projection.principal_point, "principal_point", 2)
    _check_finite(pp, "principal_point")

    fw = _to_cpu_floats(projection.fw_poly, "fw_poly", FTHETA_MAX_POLYNOMIAL_TERMS)
    _check_finite(fw, "fw_poly")
    bw = _to_cpu_floats(projection.bw_poly, "bw_poly", FTHETA_MAX_POLYNOMIAL_TERMS)
    _check_finite(bw, "bw_poly")
    if fw[0] != 0.0:
        raise ValueError(
            f"fw_poly[0] must be 0 (radial poly must pass through origin); got {fw[0]!r}"
        )
    if bw[0] != 0.0:
        raise ValueError(
            f"bw_poly[0] must be 0 (inverse poly must pass through origin); got {bw[0]!r}"
        )

    a = _to_cpu_floats(projection.A, "A", 4)
    _check_finite(a, "A")
    det = a[0] * a[3] - a[1] * a[2]
    if abs(det) < 1e-12:
        raise ValueError(
            f"A must be non-singular (det != 0); got det={det!r} for A={a!r}"
        )


def _validate_fisheye_projection(projection: OpenCVFisheyeProjection) -> None:
    """Raise ``ValueError`` if ``projection`` violates a load-bearing invariant.

    Cheap shape and scalar-range checks (including ``max_angle`` and
    ``min_2d_norm``) are enforced by the C++ constructor; this function adds the
    value-level checks that every intrinsic component tensor
    (``principal_point``, ``focal_length``, ``forward_poly``,
    ``approx_backward_factor``) is finite and that both focal lengths are
    strictly positive. The equidistant forward polynomial carries no
    through-origin constraint (its coefficients multiply ``theta`` rather than
    forming an additive term).
    """
    pp = _to_cpu_floats(projection.principal_point, "principal_point", 2)
    _check_finite(pp, "principal_point")
    focal = _to_cpu_floats(projection.focal_length, "focal_length", 2)
    _check_finite(focal, "focal_length")
    for i, v in enumerate(focal):
        if v <= 0.0:
            raise ValueError(f"focal_length[{i}] must be > 0; got {v!r}")
    fw = _to_cpu_floats(
        projection.forward_poly, "forward_poly", FISHEYE_MAX_FORWARD_POLY_TERMS
    )
    _check_finite(fw, "forward_poly")
    ab = _to_cpu_floats(projection.approx_backward_factor, "approx_backward_factor", 1)
    _check_finite(ab, "approx_backward_factor")


_ProjectionValidator = Callable[[CameraProjection], None]

_PROJECTION_VALIDATORS: dict[str, _ProjectionValidator | None] = {
    "OpenCVPinholeProjection": None,
    "FThetaProjection": _validate_ftheta_projection,
    "OpenCVFisheyeProjection": _validate_fisheye_projection,
}


def validate_camera_projection(projection: CameraProjection) -> None:
    """Run opt-in value-level validation for a supported camera projection."""
    class_name = script_class_name(projection)
    try:
        validator = _PROJECTION_VALIDATORS[class_name]
    except KeyError as exc:
        raise TypeError(f"Unknown camera projection class: {class_name}") from exc
    if validator is not None:
        validator(projection)


__all__ = ["validate_camera_projection"]
