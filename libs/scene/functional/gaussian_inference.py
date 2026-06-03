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

"""Public functional API for Gaussian Inference scene operations."""

from __future__ import annotations

from torch import Tensor

from ..sh_compression import SHCompressionMode
from ..kernels.gaussian_inference_ops import (
    pack_gaussian_inference_scene as _pack_gaussian_inference_scene,
)


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

    Converts from the standard training layout (one tensor per attribute, float32)
    into the packed layout used by inference rendering:

    - ``means_planar``: transposed to ``[3, N]`` for cache-friendly access patterns
      in the render kernel.
    - ``qso_packed``: ``[N, 8]`` float16 packing quaternion (cols 0-3), scale (cols 4-6),
      and opacity (col 7).
    - ``colors_packed``: dtype and layout depend on ``sh_degree`` and
      ``sh_compression_mode`` (see below).

    All inputs must be contiguous float32 CUDA tensors.  Activations must already
    be applied: ``scales`` must be positive (``exp()`` applied), ``opacities`` must
    be in ``[0, 1]`` (``sigmoid()`` applied).

    Args:
        means:               ``[N, 3]`` float32 CUDA — world-space positions.
        quats:               ``[N, 4]`` float32 CUDA — unit quaternions in wxyz order.
        scales:              ``[N, 3]`` float32 CUDA — positive scales (after ``exp()``).
        opacities:           ``[N]`` float32 CUDA — opacities in ``[0, 1]``
                             (after ``sigmoid()``).
        colors:              ``[N, 3]`` float32 CUDA for RGB mode
                             (``sh_degree == -1``), or ``[N, K, 3]`` float32 CUDA
                             for SH mode (``sh_degree >= 0``).
        sh_degree:           ``-1`` for RGB (no SH); ``0``–``3`` for SH coefficients.
        sh_compression_mode: SH compression mode enum. ``NONE`` preserves
                             raw SH3 layout in fp16, ``PACKED_32B`` flattens
                             SH3 coefficients into float32 lanes, and
                             ``PACKED_16B`` flattens them into fp16 lanes.

    Returns:
        A 3-tuple ``(means_planar, qso_packed, colors_packed)``:

        - ``means_planar``: ``[3, N]`` float32.
        - ``qso_packed``: ``[N, 8]`` float16.
        - ``colors_packed``: ``[N, 4]`` float16 for RGB; ``[N, K, 3]`` float32
          for SH0–2; ``[N, 16, 3]`` float16 for SH3 ``NONE``;
          ``[N, 48]`` float32 for SH3 ``PACKED_32B``; ``[N, 48]``
          float16 for SH3 ``PACKED_16B``.

    Raises:
        TypeError: if any tensor has the wrong Python type or dtype.
        ValueError: if any tensor is on the wrong device, has the wrong shape, or
            ``sh_degree`` / ``sh_compression_mode`` is out of range.
    """
    return _pack_gaussian_inference_scene(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree,
        sh_compression_mode,
    )
