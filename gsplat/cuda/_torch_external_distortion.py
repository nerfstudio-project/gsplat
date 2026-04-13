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

"""Python reference implementations for external distortion (bivariate windshield model).

These mirror the CUDA kernels in ExternalDistortion.cuh and ExternalDistortionWrappers.cu,
and are used for cross-validation in tests.
"""

import math

import torch

from ._wrapper import (
    BivariateWindshieldModelParameters,
    ExternalDistortionReferencePolynomial,
)


def ref_compute_order(num_coeffs: int) -> int:
    """Python equivalent of gsplat::extdist::compute_order."""
    sqrt_discriminant = int(math.sqrt(1 + 8 * num_coeffs))
    return (-3 + sqrt_discriminant) // 2


def ref_eval_bivariate_poly(poly_coeffs: list, order: int, x: float, y: float) -> float:
    """Python equivalent of gsplat::extdist::eval_bivariate_poly."""

    def horner_range(poly, val, idx_start, idx_end):
        result = 0.0
        for idx in range(idx_end - 1, idx_start - 1, -1):
            result = result * val + poly[idx]
        return result

    outer_coeffs = [0.0] * (order + 1)
    start_idx = 0
    for inner_order in range(order, -1, -1):
        outer_coeffs[order - inner_order] = horner_range(
            poly_coeffs, x, start_idx, start_idx + inner_order + 1
        )
        start_idx += inner_order + 1
    return horner_range(outer_coeffs, y, 0, order + 1)


def ref_distort_camera_ray(
    ray: tuple,
    horizontal_poly: list,
    vertical_poly: list,
    h_order: int,
    v_order: int,
) -> tuple:
    """Python equivalent of BivariateWindshieldModel::distort_camera_ray."""
    ray_length = math.sqrt(ray[0] ** 2 + ray[1] ** 2 + ray[2] ** 2)
    if ray_length < 1e-6:
        return ray

    phi = math.asin(max(-1.0, min(1.0, ray[0] / ray_length)))
    theta = math.asin(max(-1.0, min(1.0, ray[1] / ray_length)))

    x = math.sin(ref_eval_bivariate_poly(horizontal_poly, h_order, phi, theta))
    y = math.sin(ref_eval_bivariate_poly(vertical_poly, v_order, phi, theta))

    val = max(0.0, min(1.0, x * x + y * y))
    z = math.sqrt(1.0 - val) * (-1.0 if ray[2] < 0.0 else 1.0)
    return (x, y, z)


def num_coeffs_for_order(order: int) -> int:
    """Number of coefficients for a bivariate polynomial of given order."""
    return (order + 1) * (order + 2) // 2


def make_identity_horizontal_poly(order: int = 1) -> list:
    """Order-1 identity polynomial that maps (phi, theta) -> phi.

    For order 1: f(phi, theta) = c0 + c1*phi + c2*theta = phi  =>  [0, 1, 0]
    """
    assert order == 1, "Only order-1 identity is implemented"
    return [0.0, 1.0, 0.0]


def make_identity_vertical_poly(order: int = 1) -> list:
    """Order-1 identity polynomial that maps (phi, theta) -> theta.

    For order 1: f(phi, theta) = c0 + c1*phi + c2*theta = theta  =>  [0, 0, 1]
    """
    assert order == 1, "Only order-1 identity is implemented"
    return [0.0, 0.0, 1.0]


def make_zero_poly(order: int = 1) -> list:
    """All-zero polynomial of given order (always evaluates to 0)."""
    return [0.0] * num_coeffs_for_order(order)


def make_params(
    h_poly: list,
    v_poly: list,
    h_inv: list | None = None,
    v_inv: list | None = None,
    ref_poly: ExternalDistortionReferencePolynomial = ExternalDistortionReferencePolynomial.FORWARD,
    device: torch.device | None = None,
) -> BivariateWindshieldModelParameters:
    """Build BivariateWindshieldModelParameters from Python lists.

    Args:
        h_poly: Horizontal forward polynomial coefficients.
        v_poly: Vertical forward polynomial coefficients.
        h_inv: Horizontal inverse polynomial coefficients (defaults to identity).
        v_inv: Vertical inverse polynomial coefficients (defaults to identity).
        ref_poly: Which polynomial direction is the reference.
        device: Torch device for the tensors (defaults to cuda:0 if available).
    """
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    if h_inv is None:
        h_inv = make_identity_horizontal_poly()
    if v_inv is None:
        v_inv = make_identity_vertical_poly()
    params = BivariateWindshieldModelParameters()
    params.reference_poly = ref_poly
    params.horizontal_poly = torch.tensor(h_poly, dtype=torch.float32, device=device)
    params.vertical_poly = torch.tensor(v_poly, dtype=torch.float32, device=device)
    params.horizontal_poly_inverse = torch.tensor(
        h_inv, dtype=torch.float32, device=device
    )
    params.vertical_poly_inverse = torch.tensor(
        v_inv, dtype=torch.float32, device=device
    )
    return params
