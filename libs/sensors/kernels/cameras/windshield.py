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

"""Windshield distortion factory helpers.

Provides :func:`from_components` for constructing a
:class:`~libs.sensors.kernels.cameras.types.BivariateWindshieldDistortion`
from unpacked polynomial tensors, plus the max-term constants that mirror the
fixed-size arrays in the CUDA kernel.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .types import BivariateWindshieldDistortion, ReferencePolynomial

MAX_H_POLYNOMIAL_TERMS = 6
"""Maximum number of coefficients for horizontal windshield polynomials.

Corresponds to a triangular bivariate polynomial of degree ≤ 2.  Matches the
fixed array size declared in the CUDA kernel.
"""

MAX_V_POLYNOMIAL_TERMS = 15
"""Maximum number of coefficients for vertical windshield polynomials.

Corresponds to a triangular bivariate polynomial of degree ≤ 4.  Matches the
fixed array size declared in the CUDA kernel.
"""


def _compute_poly_order(poly_coeffs: Tensor) -> int:
    """Return the triangular bivariate polynomial degree for a coefficient vector.

    A triangular bivariate polynomial of degree ``d`` has
    ``1 + 2 + ... + (d+1) = (d+1)(d+2)/2`` terms.  This function inverts that
    count to recover ``d``, raising ``ValueError`` if the length does not match
    any valid triangular number.

    Args:
        poly_coeffs: 1D tensor of polynomial coefficients.

    Returns:
        Polynomial degree ``d`` such that ``len(poly_coeffs) == (d+1)(d+2)/2``.

    Raises:
        ValueError: If ``poly_coeffs`` is not 1D or its length is not a
            triangular number.
    """
    if poly_coeffs.dim() != 1:
        raise ValueError("windshield polynomial coefficients must be 1D")

    term_count = poly_coeffs.numel()
    running = 0
    for order in range(term_count):
        running += order + 1
        if running == term_count:
            return order
        if running > term_count:
            break
    raise ValueError(
        "The input length of the windshield distortion coefficients is not "
        "consistent with a triangular bivariate polynomial layout "
        f"(got {term_count} terms; valid sizes: 1, 3, 6, 10, 15, ...)."
    )


def _pad_poly_to_max_terms(poly: Tensor, max_terms: int, name: str) -> Tensor:
    """Right-pad a 1D polynomial coefficient tensor to ``max_terms`` with zeros.

    Args:
        poly: 1D tensor of polynomial coefficients.
        max_terms: Target length after padding.
        name: Name used in error messages.

    Returns:
        Tensor of shape ``(max_terms,)``, zero-padded if necessary.

    Raises:
        ValueError: If ``poly`` is not 1D or has more than ``max_terms`` coefficients.
    """
    if poly.dim() != 1:
        raise ValueError(f"{name} must be 1D")
    if poly.numel() > max_terms:
        raise ValueError(f"{name} must have at most {max_terms} coefficients")
    if poly.numel() == max_terms:
        return poly
    return torch.cat(
        [
            poly,
            torch.zeros(max_terms - poly.numel(), device=poly.device, dtype=poly.dtype),
        ]
    )


def _check_matching_tensor_options(
    reference: Tensor,
    reference_name: str,
    tensors: tuple[tuple[str, Tensor], ...],
) -> None:
    """Assert that all tensors share the device and dtype of ``reference``.

    Args:
        reference: Tensor whose device and dtype serve as the reference.
        reference_name: Name of the reference tensor for error messages.
        tensors: Sequence of ``(name, tensor)`` pairs to check.

    Raises:
        ValueError: If any tensor has a different device or dtype than ``reference``.
    """
    for name, tensor in tensors:
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} has device {tensor.device} but {reference_name} has "
                f"{reference.device}"
            )
        if tensor.dtype != reference.dtype:
            raise ValueError(
                f"{name} has dtype {tensor.dtype} but {reference_name} has "
                f"{reference.dtype}"
            )


def from_components(
    h_poly: Tensor,
    v_poly: Tensor,
    h_poly_inv: Tensor,
    v_poly_inv: Tensor,
    reference_polynomial: ReferencePolynomial,
) -> BivariateWindshieldDistortion:
    """Pack windshield polynomial components into a BivariateWindshieldDistortion.

    The four input polynomials are right-padded with zeros and concatenated into
    a single ``(42,)`` flat coefficient buffer with the layout::

        [ h_poly (6)  |  v_poly (15)  |  h_poly_inv (6)  |  v_poly_inv (15) ]

    where ``h_poly`` / ``h_poly_inv`` hold up to ``MAX_H_POLYNOMIAL_TERMS = 6``
    coefficients (triangular bivariate polynomial of degree <= 2) and ``v_poly``
    / ``v_poly_inv`` hold up to ``MAX_V_POLYNOMIAL_TERMS = 15`` coefficients
    (triangular bivariate polynomial of degree <= 4). All four tensors must
    share the same device and dtype.
    """

    h_poly_degree = _compute_poly_order(h_poly)
    v_poly_degree = _compute_poly_order(v_poly)
    h_poly_inv_degree = _compute_poly_order(h_poly_inv)
    v_poly_inv_degree = _compute_poly_order(v_poly_inv)
    if h_poly_degree != h_poly_inv_degree:
        raise ValueError("h_poly and h_poly_inv must have matching triangular degree")
    if v_poly_degree != v_poly_inv_degree:
        raise ValueError("v_poly and v_poly_inv must have matching triangular degree")
    if h_poly_degree > 2:
        raise ValueError("h_poly degree must be <= 2")
    if v_poly_degree > 4:
        raise ValueError("v_poly degree must be <= 4")
    _check_matching_tensor_options(
        h_poly,
        "h_poly",
        (("v_poly", v_poly), ("h_poly_inv", h_poly_inv), ("v_poly_inv", v_poly_inv)),
    )

    distortion_coeffs = torch.cat(
        [
            _pad_poly_to_max_terms(h_poly, MAX_H_POLYNOMIAL_TERMS, "h_poly"),
            _pad_poly_to_max_terms(v_poly, MAX_V_POLYNOMIAL_TERMS, "v_poly"),
            _pad_poly_to_max_terms(h_poly_inv, MAX_H_POLYNOMIAL_TERMS, "h_poly_inv"),
            _pad_poly_to_max_terms(v_poly_inv, MAX_V_POLYNOMIAL_TERMS, "v_poly_inv"),
        ]
    )
    return BivariateWindshieldDistortion(
        distortion_coeffs,
        int(reference_polynomial),
        h_poly_degree,
        v_poly_degree,
    )


__all__ = [
    "BivariateWindshieldDistortion",
    "MAX_H_POLYNOMIAL_TERMS",
    "MAX_V_POLYNOMIAL_TERMS",
    "from_components",
]
