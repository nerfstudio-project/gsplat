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


"""Accelerated Gaussian Inference render entry points."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from . import _backend


def _require_backend():
    if _backend._C is None:
        raise RuntimeError(
            "Inference CUDA backend is not available. "
            "Ensure CUDA toolkit is installed and the extension compiled."
        )
    return _backend._C


def create_native_gaussian_inference_renderer(
    means_planar: Tensor,
    qso_packed: Tensor,
    colors_packed: Tensor,
    sh_degree: int,
    sh_compression_mode: int,
):
    """Create the pybind stateful renderer wrapper."""
    backend = _require_backend()
    return backend.GaussianInferenceRenderer(
        means_planar,
        qso_packed,
        colors_packed,
        sh_degree,
        sh_compression_mode,
    )


def gaussian_render_inference_only(
    means_planar: Tensor,
    qso_packed: Tensor,
    colors_packed: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_compression_mode: int,
    background: Optional[Tensor],
    *,
    out_renders: Optional[Tensor] = None,
    out_alphas: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Dispatch to the registered CUDA torch op."""
    _require_backend()
    return torch.ops.experimental.gaussian_render_inference_only(
        means_planar,
        qso_packed,
        colors_packed,
        viewmat,
        K,
        width,
        height,
        sh_degree,
        tile_size,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        sh_compression_mode,
        background,
        out_renders=out_renders,
        out_alphas=out_alphas,
    )


__all__ = [
    "create_native_gaussian_inference_renderer",
    "gaussian_render_inference_only",
]
