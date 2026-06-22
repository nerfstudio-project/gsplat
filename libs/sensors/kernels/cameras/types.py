# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from typing import Any, Union

import torch

from .._backend import _C  # noqa: F401  # loads torch.classes.gsplat_sensors


class ShutterType(IntEnum):
    """OpenCV pinhole shutter modes.

    Mirrors ``gsplat_sensors::ShutterType`` declared in
    ``libs/sensors/kernels/cuda/csrc/shutter_type.h``. The C++ definition is
    the source of truth; this class is verified against it at import time
    by :func:`_verify_shutter_type_matches_cpp`.
    """

    ROLLING_TOP_TO_BOTTOM = 1
    ROLLING_LEFT_TO_RIGHT = 2
    ROLLING_BOTTOM_TO_TOP = 3
    ROLLING_RIGHT_TO_LEFT = 4
    GLOBAL = 5


def _verify_shutter_type_matches_cpp() -> None:
    cpp_enum = getattr(_C, "ShutterType", None)
    if cpp_enum is None:
        raise RuntimeError(
            "Loaded gsplat_sensors_cuda extension does not expose ShutterType. "
            "Rebuild the native extension (see "
            "libs/sensors/kernels/cuda/ext.cpp)."
        )
    for member in ShutterType:
        cpp_member = getattr(cpp_enum, member.name, None)
        if cpp_member is None:
            raise RuntimeError(
                f"C++ ShutterType is missing member {member.name!r}; "
                "Python ShutterType in libs/sensors/kernels/cameras/types.py "
                "and C++ in libs/sensors/kernels/cuda/csrc/shutter_type.h are "
                "out of sync."
            )
        if int(cpp_member) != member.value:
            raise RuntimeError(
                f"ShutterType.{member.name} mismatch: "
                f"Python={member.value}, C++={int(cpp_member)}. "
                "Update one side to match the other (C++ source of truth: "
                "libs/sensors/kernels/cuda/csrc/shutter_type.h)."
            )


_verify_shutter_type_matches_cpp()


class ReferencePolynomial(IntEnum):
    """Reference polynomial direction for bivariate windshield distortion.

    Selects which of the two polynomial directions (forward or backward) is
    treated as the authoritative mapping when evaluating the windshield model.

    Members:
        FORWARD: The horizontal-to-vertical (forward) polynomial is the reference.
        BACKWARD: The vertical-to-horizontal (backward) polynomial is the reference.
    """

    FORWARD = 0
    BACKWARD = 1


# These three names are bound to the TorchScript custom classes registered by
# the gsplat_sensors C++ extension via ``torch::class_<T>``.  They behave like
# Python dataclasses but are actually ``torch.classes.*`` descriptors — created
# by the extension at import time and stored in ``torch.classes.gsplat_sensors``.
OpenCVPinholeProjection = torch.classes.gsplat_sensors.OpenCVPinholeProjection
"""OpenCV pinhole camera projection parameters.

Standard pinhole camera model with radial, tangential, and thin prism
distortion. Parameters are stored as separate per-component tensors on the
C++ ``OpenCVPinholeProjection`` struct (see
``libs/sensors/kernels/cuda/csrc/camera_torch.h``).

Attributes:
    focal_length: (2,) ``[fx, fy]`` focal lengths in pixels.
    principal_point: (2,) ``[cx, cy]`` principal point in pixels.
    radial_coeffs: (6,) ``[k1, k2, k3, k4, k5, k6]`` radial distortion coefficients.
    tangential_coeffs: (2,) ``[p1, p2]`` tangential distortion coefficients.
    thin_prism_coeffs: (4,) ``[s1, s2, s3, s4]`` thin prism distortion coefficients.
    resolution: ``(width, height)`` tuple of ints, image resolution in pixels.

Note:
    Use :func:`script_class_name` to retrieve the registered class name when
    dispatching across projection types.
"""

NoExternalDistortion = torch.classes.gsplat_sensors.NoExternalDistortion
"""No-op external distortion — identity transformation.

Used as the distortion argument when the camera has no windshield or other
external distortion element.
"""

BivariateWindshieldDistortion = (
    torch.classes.gsplat_sensors.BivariateWindshieldDistortion
)
"""Bivariate windshield distortion parameters.

Models optical distortion introduced by a windshield via two pairs of
triangular bivariate polynomials (forward and inverse, horizontal and
vertical).  All 42 coefficients are stored in a single flat tensor for
efficient GPU transfer.

Attributes:
    distortion_coeffs: (42,) packed tensor with layout
        ``[h_poly (6) | v_poly (15) | h_poly_inv (6) | v_poly_inv (15)]``.
    reference_polynomial: :class:`ReferencePolynomial` selecting the
        authoritative mapping direction.
    h_poly_degree: Triangular degree of the horizontal polynomials (active
        coefficients; coefficients beyond this index are zero-padded).
    v_poly_degree: Triangular degree of the vertical polynomials.

Note:
    Polynomial degrees are bounded by fixed-size arrays in the CUDA kernel:
    horizontal polynomials are limited to triangular degree ≤ 2
    (``MAX_H_POLYNOMIAL_TERMS = 6`` coefficients) and vertical polynomials
    to triangular degree ≤ 4 (``MAX_V_POLYNOMIAL_TERMS = 15`` coefficients).
    Use :func:`from_components` to construct this class from unpacked tensors.
"""

# Phase-1 compatibility alias. There is no registered abstract C++ base class.
CameraProjection: Any = OpenCVPinholeProjection
"""Type alias for a camera projection parameter object.

In Phase 1 only :class:`OpenCVPinholeProjection` is supported, so this alias
resolves directly to it.  Code that dispatches across projection types should
use :data:`REGISTERED_CAMERA_PROJECTIONS` and :func:`script_class_name` rather
than an ``isinstance`` check against this alias.
"""

ExternalDistortion = Union[NoExternalDistortion, BivariateWindshieldDistortion]
"""Union type for external (post-projection) distortion parameters.

Either :class:`NoExternalDistortion` (identity) or
:class:`BivariateWindshieldDistortion`.
"""

REGISTERED_CAMERA_PROJECTIONS = (OpenCVPinholeProjection,)
"""Tuple of all supported camera projection classes.

Iterate this tuple to dispatch across projection types without hard-coding
class names in kernel dispatch logic.
"""

REGISTERED_DISTORTIONS = (NoExternalDistortion, BivariateWindshieldDistortion)
"""Tuple of all supported external distortion classes.

Iterate this tuple to dispatch across distortion types without hard-coding
class names in kernel dispatch logic.
"""


def script_class_name(obj: object) -> str:
    """Return the registered TorchScript class name for a camera parameter object.

    Camera projection and distortion classes in sensorlib are registered with
    the PyTorch custom-class registry via ``torch::class_<T>`` in the C++
    extension.  Their ``type(obj).__name__`` is always ``CustomClassHolder``,
    so a plain ``isinstance`` / ``type()`` check cannot distinguish them.
    This helper calls the ``._type()`` method injected by TorchScript to
    recover the original registered name (e.g. ``"OpenCVPinholeProjection"``).

    Falls back to ``type(obj).__name__`` for plain Python objects so callers
    can dispatch uniformly across TorchScript classes and regular classes.

    Args:
        obj: A TorchScript ``CustomClassHolder`` or a plain Python object.

    Returns:
        The registered class name string.
    """
    type_fn = getattr(obj, "_type", None)
    if type_fn is None:
        return type(obj).__name__
    return type_fn().name()


__all__ = [
    "BivariateWindshieldDistortion",
    "CameraProjection",
    "ExternalDistortion",
    "NoExternalDistortion",
    "OpenCVPinholeProjection",
    "ReferencePolynomial",
    "REGISTERED_CAMERA_PROJECTIONS",
    "REGISTERED_DISTORTIONS",
    "script_class_name",
    "ShutterType",
]
