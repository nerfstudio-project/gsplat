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

"""Differentiable Gaussian NHT (Neural Harmonic Textures) render API."""

from __future__ import annotations

from typing import Any, Optional

from gsplat.rendering import rasterization
from gsplat.scene import GaussianNHTScene

from ..types import RenderReturn

# ---------------------------------------------------------------------------
# rasterize_gaussian_nht_scene  (direct NHT entry point)
# ---------------------------------------------------------------------------


def rasterize_gaussian_nht_scene(
    scene: GaussianNHTScene,
    *,
    out: Optional[RenderReturn] = None,
    **request: Any,
) -> RenderReturn:
    """Render a ``GaussianNHTScene`` through the differentiable NHT raster path.

    Thin wrapper around :func:`gsplat.rendering.rasterization` that pulls
    ``means``/``quats``/``scales``/``opacities``/``features`` from the scene
    (applying the same log-space/logit-space activations the training loop
    expects) and forwards ``scene.nht_params`` plus the rest of ``request``
    unchanged. Camera/render-mode/etc. arguments are exactly
    :func:`rasterization`'s, since NHT training needs the full feature set
    (batched cameras, depth/normal modes, distributed, ...) that the
    Inference scene's fused path intentionally does not support.

    This is the direct, low-level entry point. For a polymorphic API that
    also handles ``GaussianInferenceScene``, use :func:`render_scene`.

    Returns the raw rasterized feature buffer in ``frame`` (not yet passed
    through deferred shading) — callers run their own deferred-shading /
    masking / post-processing pipeline on top, same as before this scene
    wrapper existed.
    """
    if not isinstance(scene, GaussianNHTScene):
        raise TypeError(
            f"rasterize_gaussian_nht_scene requires a GaussianNHTScene; "
            f"got {type(scene).__name__}"
        )
    if out is not None:
        raise TypeError(
            "rasterize_gaussian_nht_scene does not support the out= buffer "
            "contract (training renders a fresh buffer every call)"
        )
    if (
        "means" in request
        or "quats" in request
        or "scales" in request
        or ("opacities" in request or "colors" in request or "sh_degree" in request)
    ):
        raise TypeError(
            "means/quats/scales/opacities/colors/sh_degree are read from "
            "scene.splats; cannot be overridden via request kwargs"
        )
    if "nht_params" in request:
        raise TypeError(
            "nht_params is read from scene.nht_params; cannot be overridden "
            "via request kwargs"
        )

    splats = scene.splats
    render_colors, render_alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=splats["scales"].exp(),
        opacities=splats["opacities"].sigmoid(),
        colors=splats["features"],
        sh_degree=None,
        nht_params=scene.nht_params,
        **request,
    )

    return RenderReturn(
        frame=render_colors,
        metadata={"alphas": render_alphas, **info},
    )


__all__ = ["rasterize_gaussian_nht_scene"]
