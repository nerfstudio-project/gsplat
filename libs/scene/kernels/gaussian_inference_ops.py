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

"""Python-level wrapper for the native pack_gaussian_inference_scene op.

Provides input validation before calling into C++, following the same pattern as
:mod:`gsplat_geometry.kernels.quaternion_ops` and ``pose_ops``: callers get
clear ``TypeError`` / ``ValueError`` messages rather than opaque C++ errors.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..sh_compression import SH_COMPRESSION_MODE_VALUES, SHCompressionMode

# Valid sh_degree values; -1 means RGB (no SH)
_VALID_SH_DEGREES = frozenset({-1, 0, 1, 2, 3})


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _expect_tensor(name: str, x: object) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(x).__name__}")
    return x


def _require_cuda(name: str, t: Tensor) -> None:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor; got device {t.device}")


def _require_float32(name: str, t: Tensor) -> None:
    if t.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32; got {t.dtype}")


def _require_shape(name: str, t: Tensor, *expected: int | None) -> None:
    """Check that t.shape matches expected dims, where None means 'any'."""
    if t.dim() != len(expected):
        raise ValueError(
            f"{name} must have {len(expected)}D shape "
            f"{'x'.join(str(d) if d is not None else 'N' for d in expected)}; "
            f"got shape {tuple(t.shape)}"
        )
    for i, (actual, exp) in enumerate(zip(t.shape, expected)):
        if exp is not None and actual != exp:
            raise ValueError(
                f"{name} dim {i} must be {exp}; got shape {tuple(t.shape)}"
            )


def _require_same_device(pairs: list[tuple[str, Tensor]], ref_name: str) -> None:
    ref = pairs[0][1]
    for name, t in pairs[1:]:
        if t.device != ref.device:
            raise ValueError(
                f"{name} must be on the same device as {ref_name} "
                f"({ref.device}); got {t.device}"
            )


# ---------------------------------------------------------------------------
# Public kernel wrapper
# ---------------------------------------------------------------------------


def pack_gaussian_inference_scene(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    sh_degree: int,
    sh_compression_mode: SHCompressionMode,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack activated Gaussian splat tensors into the inference-internal layout.

    This is the kernels-layer entry point.  Downstream code should use
    :func:`gsplat_scene.functional.pack_gaussian_inference_scene` instead.

    All tensor inputs must be contiguous float32 CUDA tensors with N rows on the
    leading dimension.

    Args:
        means:               [N, 3] float32 CUDA — world-space positions
        quats:               [N, 4] float32 CUDA — unit quaternions wxyz
        scales:              [N, 3] float32 CUDA — positive scales (after exp())
        opacities:           [N]    float32 CUDA — opacities in [0, 1] (after sigmoid())
        colors:              [N, 3] float32 CUDA for RGB, or
                             [N, (sh_degree + 1) ** 2, 3] float32 CUDA
                             for SH coefficients
        sh_degree:           -1 for RGB mode; 0–3 for SH
        sh_compression_mode: SHCompressionMode enum. Nonzero modes require SH3.

    Returns:
        means_planar:   [3, N] float32  transposed means
        qso_packed:     [N, 8] float16  packed quaternion cols 0-3,
                        scale cols 4-6, opacity col 7
        colors_packed:  [N, 4] float16 for RGB; [N, K, 3] float32 for SH0-2;
                        [N, 16, 3] float16 for SH3 NONE;
                        [N, 48] float32 for SH3 PACKED_32B;
                        [N, 48] float16 for SH3 PACKED_16B
    """
    # --- scalar validation -----------------------------------------------
    if not isinstance(sh_degree, int):
        raise TypeError(f"sh_degree must be an int; got {type(sh_degree).__name__}")
    if sh_degree not in _VALID_SH_DEGREES:
        raise ValueError(
            f"sh_degree must be one of {sorted(_VALID_SH_DEGREES)}; " f"got {sh_degree}"
        )
    if not isinstance(sh_compression_mode, SHCompressionMode):
        raise TypeError(
            f"sh_compression_mode must be an SHCompressionMode; "
            f"got {type(sh_compression_mode).__name__}"
        )
    if sh_compression_mode not in SH_COMPRESSION_MODE_VALUES:
        raise ValueError(
            f"sh_compression_mode must be one of "
            f"{sorted(SH_COMPRESSION_MODE_VALUES)}; "
            f"got {sh_compression_mode}"
        )
    if sh_compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
        raise ValueError(
            f"sh_compression_mode={sh_compression_mode} requires sh_degree=3; "
            f"got sh_degree={sh_degree}"
        )

    # --- tensor type / device / dtype checks -----------------------------
    means = _expect_tensor("means", means)
    quats = _expect_tensor("quats", quats)
    scales = _expect_tensor("scales", scales)
    opacities = _expect_tensor("opacities", opacities)
    colors = _expect_tensor("colors", colors)

    for name, t in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        _require_cuda(name, t)
        _require_float32(name, t)

    _require_same_device(
        [
            ("means", means),
            ("quats", quats),
            ("scales", scales),
            ("opacities", opacities),
            ("colors", colors),
        ],
        ref_name="means",
    )

    # --- shape checks ----------------------------------------------------
    N = means.shape[0]
    _require_shape("means", means, N, 3)
    _require_shape("quats", quats, N, 4)
    _require_shape("scales", scales, N, 3)
    _require_shape("opacities", opacities, N)

    if sh_degree >= 0:
        expected_K = (sh_degree + 1) ** 2
        if colors.dim() != 3:
            raise ValueError(
                f"sh_degree={sh_degree} requires colors to be 3D "
                f"[N, {expected_K}, 3]; got {colors.dim()}D shape "
                f"{tuple(colors.shape)}"
            )
        if colors.shape != (N, expected_K, 3):
            raise ValueError(
                f"sh_degree={sh_degree} requires colors shape "
                f"({N}, {expected_K}, 3); got {tuple(colors.shape)}"
            )
    else:
        # RGB mode
        _require_shape("colors", colors, N, 3)

    # --- contiguity (C++ CHECK_INPUT requires contiguous) ----------------
    means = means.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    opacities = opacities.contiguous()
    colors = colors.contiguous()

    # --- dispatch --------------------------------------------------------
    from . import _backend

    return _backend._SCENE_CUDA.pack_gaussian_inference_scene(
        means,
        quats,
        scales,
        opacities,
        colors,
        int(sh_degree),
        int(sh_compression_mode),
    )
