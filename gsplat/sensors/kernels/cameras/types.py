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

from __future__ import annotations

from enum import IntEnum
from typing import Any, Union

import torch

from .. import _backend  # loads torch.classes.gsplat_sensors on backend access


class ShutterType(IntEnum):
    """OpenCV pinhole shutter modes.

    Mirrors ``gsplat_sensors::ShutterType`` declared in
    ``gsplat/sensors/kernels/cuda/csrc/shutter_type.h``. The C++ definition is
    the source of truth; this class is verified against it at import time
    by :func:`_verify_shutter_type_matches_cpp`.
    """

    ROLLING_TOP_TO_BOTTOM = 1
    ROLLING_LEFT_TO_RIGHT = 2
    ROLLING_BOTTOM_TO_TOP = 3
    ROLLING_RIGHT_TO_LEFT = 4
    GLOBAL = 5


def _verify_shutter_type_matches_cpp() -> None:
    cpp_enum = getattr(_backend._SENSORS_CUDA, "ShutterType", None)
    if cpp_enum is None:
        raise RuntimeError(
            "Loaded gsplat_sensors_cuda extension does not expose ShutterType. "
            "Rebuild the native extension (see "
            "gsplat/sensors/kernels/cuda/ext.cpp)."
        )
    for member in ShutterType:
        cpp_member = getattr(cpp_enum, member.name, None)
        if cpp_member is None:
            raise RuntimeError(
                f"C++ ShutterType is missing member {member.name!r}; "
                "Python ShutterType in gsplat/sensors/kernels/cameras/types.py "
                "and C++ in gsplat/sensors/kernels/cuda/csrc/shutter_type.h are "
                "out of sync."
            )
        if int(cpp_member) != member.value:
            raise RuntimeError(
                f"ShutterType.{member.name} mismatch: "
                f"Python={member.value}, C++={int(cpp_member)}. "
                "Update one side to match the other (C++ source of truth: "
                "gsplat/sensors/kernels/cuda/csrc/shutter_type.h)."
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


# These names are bound to the TorchScript custom classes registered by
# the gsplat_sensors C++ extension via ``torch::class_<T>``.  They behave like
# Python dataclasses but are actually ``torch.classes.*`` descriptors — created
# by the extension at import time and stored in ``torch.classes.gsplat_sensors``.
OpenCVPinholeProjection = torch.classes.gsplat_sensors.OpenCVPinholeProjection
"""OpenCV pinhole camera projection parameters.

Standard pinhole camera model with radial, tangential, and thin prism
distortion. Parameters are stored as separate per-component tensors on the
C++ ``OpenCVPinholeProjection`` struct (see
``gsplat/sensors/kernels/cuda/csrc/camera_torch.h``).

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

FThetaProjection = torch.classes.gsplat_sensors.FThetaProjection
"""FTheta fisheye camera projection parameters.

Polynomial fisheye model with a forward (ray-to-angle) and backward
(angle-to-ray) polynomial pair and a 2x2 affine pixel transform. Parameters
are stored as separate per-component tensors on the C++ ``FThetaProjection``
struct (see ``gsplat/sensors/kernels/cuda/csrc/camera_torch.h``).

Attributes:
    principal_point: (2,) ``[cx, cy]`` principal point in pixels.
    fw_poly: (FTHETA_MAX_POLYNOMIAL_TERMS,) forward polynomial coefficients
        (ray angle -> radial pixel distance); coefficients beyond
        ``fw_poly_degree`` are zero-padded.
    bw_poly: (FTHETA_MAX_POLYNOMIAL_TERMS,) backward polynomial coefficients
        (radial pixel distance -> ray angle); coefficients beyond
        ``bw_poly_degree`` are zero-padded.
    A: (4,) row-major 2x2 affine pixel transform applied after the radial polynomial.
        ``Ainv`` is exposed as a read-only property derived from ``A``
        (closed-form 2x2 inverse).
    resolution: ``(width, height)`` tuple of ints, image resolution in pixels.
    reference_polynomial: :class:`ReferencePolynomial` selecting which
        polynomial direction is authoritative when both are evaluated.
    fw_poly_degree: Degree of ``fw_poly`` (active coefficients);
        must satisfy ``fw_poly_degree <= FTHETA_MAX_POLYNOMIAL_TERMS - 1``.
    bw_poly_degree: Degree of ``bw_poly`` (active coefficients);
        must satisfy ``bw_poly_degree <= FTHETA_MAX_POLYNOMIAL_TERMS - 1``.
    newton_iterations: Number of Newton iterations used when inverting the
        forward polynomial in the ray->pixel direction.
    max_angle: Maximum valid ray angle in radians; rays beyond this angle are
        marked invalid by the projection kernels.
    min_2d_norm: Pixel-radius threshold below which the angle-from-pixel
        mapping is treated as degenerate (avoids division by zero near the
        principal point).

Note:
    Use :func:`script_class_name` to retrieve the registered class name when
    dispatching across projection types.
"""

OpenCVFisheyeProjection = torch.classes.gsplat_sensors.OpenCVFisheyeProjection
"""OpenCV equidistant-fisheye camera projection parameters.

Equidistant fisheye model with an odd-power forward polynomial and an
anisotropic focal length. Parameters are stored as separate per-component
tensors on the C++ ``OpenCVFisheyeProjection`` struct (see
``gsplat/sensors/kernels/cuda/csrc/camera_torch.h``).

Attributes:
    principal_point: (2,) ``[cx, cy]`` principal point in pixels.
    focal_length: (2,) ``[fx, fy]`` anisotropic focal lengths in pixels.
    forward_poly: (FISHEYE_MAX_FORWARD_POLY_TERMS,) ``[k1, k2, k3, k4]`` equidistant distortion coefficients
        evaluated as ``theta * (1 + k1*t2 + k2*t4 + k3*t6 + k4*t8)``.
    approx_backward_factor: (1,) Newton initial-guess factor for inverting the
        forward polynomial; never receives a gradient.
    resolution: ``(width, height)`` tuple of ints, image resolution in pixels.
    newton_iterations: Fixed Newton-iteration count used when inverting the
        forward polynomial in the pixel->ray direction.
    max_angle: Maximum valid ray angle in radians; rays beyond this angle are
        marked invalid by the projection kernels.
    min_2d_norm: Pixel-radius threshold below which the angle-from-pixel
        mapping is treated as degenerate.

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

# Maximum number of polynomial coefficients the FTheta CUDA kernel reads from
# fw_poly / bw_poly. The C++ definition is the source of truth; polynomials with
# degree greater than ``FTHETA_MAX_POLYNOMIAL_TERMS - 1`` are rejected at
# construction.
FTHETA_MAX_POLYNOMIAL_TERMS = int(FThetaProjection.get_max_polynomial_terms())

# Number of OpenCV-fisheye forward-polynomial coefficients the CUDA kernel reads
# from forward_poly. The C++ definition is the source of truth; forward_poly must
# carry exactly this many coefficients.
FISHEYE_MAX_FORWARD_POLY_TERMS = int(
    OpenCVFisheyeProjection.get_max_forward_poly_terms()
)

# Compatibility alias kept as ``Any`` (rather than a Union of the registered
# torch script classes) on purpose: type-narrowing a Union of
# ``torch.classes.*`` references inside ``torch.autograd.Function`` is awkward
# because each entry is a ``ScriptClass`` instance, not a regular Python type;
# callers should use :func:`script_class_name` for runtime dispatch instead of
# ``isinstance`` checks. The registered set lives in
# ``REGISTERED_CAMERA_PROJECTIONS``.
#
# The runtime value is bound to ``object`` (not to one of the registered
# projection classes) so the symbol does not falsely imply that any single
# projection is canonical. The annotation ``Any`` is what callers see; the
# right-hand side is required only so ``from .types import CameraProjection``
# resolves at import time.
CameraProjection: Any = object
"""Type alias for a camera projection parameter object.

This is kept as ``Any`` for type-checking ergonomics with TorchScript custom
classes. Code that dispatches across projection types should use
:data:`REGISTERED_CAMERA_PROJECTIONS` and :func:`script_class_name` rather than
an ``isinstance`` check against this alias.
"""

ExternalDistortion = Union[NoExternalDistortion, BivariateWindshieldDistortion]
"""Union type for external (post-projection) distortion parameters.

Either :class:`NoExternalDistortion` (identity) or
:class:`BivariateWindshieldDistortion`.
"""

REGISTERED_CAMERA_PROJECTIONS = (
    OpenCVPinholeProjection,
    FThetaProjection,
    OpenCVFisheyeProjection,
)
"""Tuple of all supported camera projection classes.

Iterate this tuple to dispatch across projection types without hard-coding
class names in kernel dispatch logic.
"""

REGISTERED_DISTORTIONS = (NoExternalDistortion, BivariateWindshieldDistortion)
"""Tuple of all supported external distortion classes.

Iterate this tuple to dispatch across distortion types without hard-coding
class names in kernel dispatch logic.
"""

# Registered TorchScript class names, parallel to REGISTERED_CAMERA_PROJECTIONS
# and REGISTERED_DISTORTIONS. Kept as explicit string tuples because the
# torch.classes.* class objects are torch.ScriptClass instances and do not
# expose a usable __name__ attribute; script_class_name() only works on
# instances, not on the registered class objects themselves.
REGISTERED_CAMERA_PROJECTION_NAMES = (
    "OpenCVPinholeProjection",
    "FThetaProjection",
    "OpenCVFisheyeProjection",
)
REGISTERED_DISTORTION_NAMES = (
    "NoExternalDistortion",
    "BivariateWindshieldDistortion",
)


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
    "FISHEYE_MAX_FORWARD_POLY_TERMS",
    "FTHETA_MAX_POLYNOMIAL_TERMS",
    "FThetaProjection",
    "NoExternalDistortion",
    "OpenCVFisheyeProjection",
    "OpenCVPinholeProjection",
    "ReferencePolynomial",
    "REGISTERED_CAMERA_PROJECTIONS",
    "REGISTERED_CAMERA_PROJECTION_NAMES",
    "REGISTERED_DISTORTIONS",
    "REGISTERED_DISTORTION_NAMES",
    "script_class_name",
    "ShutterType",
]
