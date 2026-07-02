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

"""
NHT Inference Renderer — fully-fused single-kernel rendering.

Pipeline
--------
::

    [pre-pass 1]  fully_fused_projection_with_ut  → radii, depths  (tile intersection only)
    [pre-pass 2]  isect_tiles + isect_offset_encode → tile_offsets, flatten_ids
    [kernel]      rasterize_to_pixels_from_world_nht_3dgs_fused_fwd
                    • same 3D ray-Gaussian intersection as training
                    • same tetrahedral interpolation + harmonic encoding
                    • inline warp-cooperative WMMA MLP + SH3 encoding + sigmoid
                    → fp16 RGB + fp32 alpha in one launch

The MLP weights are converted once (at first render) from tcnn's linear
parameter layout to the warp-fragment "native" layout the kernel consumes;
see ``convert_mlp_params_to_fused_native``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..cuda._wrapper import (
    FThetaCameraDistortionParameters,
    RollingShutterType,
    UnscentedTransformParameters,
    _make_lazy_cuda_obj,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
)
from .deferred_shader import DeferredShaderModule
from ._wrapper import (
    convert_mlp_params_to_fused_native,
    rasterize_to_pixels_from_world_nht_3dgs_fused_fwd,
)


@dataclass
class NHTInferenceConfig:
    tile_size: int = 16
    near_plane: float = 0.01
    far_plane: float = 1e10
    eps2d: float = 0.3
    center_ray_mode: bool = False
    backgrounds: Optional[Tensor] = None


class NHTInferenceRenderer:
    """
    Stateful NHT inference renderer built on the fully-fused kernel.

    On first render the tcnn MLP parameters are re-packed into the kernel's
    native fragment layout (a one-time ~40 KB shuffle). Subsequent calls
    launch projection + intersection + one fused kernel.

    Parameters
    ----------
    shader : DeferredShaderModule
        Trained deferred shader.
    config : NHTInferenceConfig
        Renderer configuration.
    """

    def __init__(
        self,
        shader: DeferredShaderModule,
        config: Optional[NHTInferenceConfig] = None,
    ) -> None:
        self.shader = shader
        self.config = config or NHTInferenceConfig()

        # Projection cache (invalidated whenever geometry or the view changes)
        self._cached_proj: Optional[Tuple[Tensor, Tensor, Tensor]] = None
        self._cached_viewmat: Optional[Tensor] = None
        self._cached_geom_key: Optional[tuple] = None

        # Converted MLP params — re-converted whenever the source parameter
        # tensor changes (optimizer steps, EMA swaps, checkpoint reloads).
        self._mlp_params_native: Optional[Tensor] = None
        self._mlp_params_key: Optional[tuple] = None

    def invalidate_cache(self) -> None:
        self._cached_proj = None
        self._cached_viewmat = None
        self._cached_geom_key = None
        self._mlp_params_native = None
        self._mlp_params_key = None

    @staticmethod
    def _tensor_key(t: Tensor) -> tuple:
        # _version increments on in-place updates (optimizer steps, EMA
        # copy_), id() changes when the tensor is replaced (densification).
        return (id(t), t._version, t.shape[0] if t.dim() > 0 else 0)

    # ── Main render entry ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def render(
        self,
        splats: Dict[str, Tensor],
        viewmat: Tensor,  # [4, 4]
        K: Tensor,  # [3, 3]
        width: int,
        height: int,
        *,
        backgrounds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Render a single view.

        Returns
        -------
        rgb    : [1, H, W, 3]  float32
        alphas : [1, H, W, 1]  float32
        info   : dict
        """
        device = viewmat.device

        means = splats["means"]
        quats = F.normalize(splats["quats"], dim=-1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        features = splats.get("features", splats.get("sh0"))
        assert features is not None, "splats must contain a 'features' key"

        if features.dtype != torch.float16:
            features = features.half()

        N = means.shape[0]
        ts = self.config.tile_size
        tile_width = (width + ts - 1) // ts
        tile_height = (height + ts - 1) // ts

        # ── Pre-pass 1: UT projection for tile intersection bounding boxes ────
        geom_key = (
            self._tensor_key(splats["means"]),
            self._tensor_key(splats["quats"]),
            self._tensor_key(splats["scales"]),
            self._tensor_key(splats["opacities"]),
        )
        view_changed = (
            self._cached_viewmat is None
            or not torch.equal(self._cached_viewmat, viewmat)
            or self._cached_geom_key != geom_key
        )
        if self._cached_proj is None or view_changed:
            radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
                eps2d=self.config.eps2d,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                calc_compensations=False,
                camera_model="pinhole",
                ut_params=UnscentedTransformParameters(),
                rolling_shutter=RollingShutterType.GLOBAL,
                viewmats_rs=None,
                global_z_order=True,
            )
            self._cached_proj = (radii, means2d, depths)
            self._cached_viewmat = viewmat.clone()
            self._cached_geom_key = geom_key
        else:
            radii, means2d, depths = self._cached_proj

        # ── Pre-pass 2: tile intersection ─────────────────────────────────────
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size=ts,
            tile_width=tile_width,
            tile_height=tile_height,
            packed=False,
            n_images=1,
        )
        tile_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

        # ── Fused kernel: rasterization + encoding + MLP + sigmoid ───────────
        (
            render_rgb,
            render_alphas,
            _,
            _,
        ) = rasterize_to_pixels_from_world_nht_3dgs_fused_fwd(
            means=means.unsqueeze(0).unsqueeze(0),
            quats=quats.unsqueeze(0).unsqueeze(0),
            scales=scales.unsqueeze(0).unsqueeze(0),
            colors=features.unsqueeze(0).unsqueeze(0),
            opacities=opacities.unsqueeze(0).unsqueeze(0),
            image_width=width,
            image_height=height,
            tile_size=ts,
            viewmats0=viewmat.unsqueeze(0).unsqueeze(0),
            viewmats1=None,
            Ks=K.unsqueeze(0).unsqueeze(0),
            camera_model=_make_lazy_cuda_obj("CameraModelType.PINHOLE"),
            ut_params=UnscentedTransformParameters(),
            rs_type=int(RollingShutterType.GLOBAL),
            radial_coeffs=None,
            tangential_coeffs=None,
            thin_prism_coeffs=None,
            ftheta_coeffs=FThetaCameraDistortionParameters(),
            lidar_coeffs=None,
            external_distortion_params=None,
            tile_offsets=tile_offsets,
            flatten_ids=flatten_ids,
            center_ray_mode=self.config.center_ray_mode,
            ray_dir_scale=self.shader.ray_dir_scale,
            mlp_params=self._get_mlp_params(device),
            mlp_hidden_dim=self.shader.mlp_hidden_dim,
            mlp_num_layers=self.shader.mlp_num_layers,
        )
        # render_rgb:    [1, 1, H, W, 3] fp16 -> [H, W, 3]
        # render_alphas: [1, 1, H, W]    fp32 -> [H, W]
        rgb = render_rgb.reshape(height, width, 3).float()
        alpha = render_alphas.reshape(height, width)

        bg = backgrounds if backgrounds is not None else self.config.backgrounds
        if bg is not None:
            bg_t = bg.to(device=device, dtype=torch.float32)
            rgb = rgb * alpha.unsqueeze(-1) + bg_t * (1.0 - alpha.unsqueeze(-1))

        return (
            rgb.unsqueeze(0),  # [1, H, W, 3]
            alpha.unsqueeze(0).unsqueeze(-1),  # [1, H, W, 1]
            {"n_gaussians": N},
        )

    def _get_mlp_params(self, device: torch.device) -> Tensor:
        """fp16 network weights in fused-native layout.

        Re-converted whenever ``shader.backbone.params`` changes — detected via
        the tensor's identity + in-place version counter, so optimizer steps,
        EMA swaps, and checkpoint reloads all invalidate the cache.
        """
        src = self.shader.backbone.params
        key = self._tensor_key(src)
        if self._mlp_params_native is None or self._mlp_params_key != key:
            self._mlp_params_native = convert_mlp_params_to_fused_native(
                src.detach().to(device),
                n_feat_in=self.shader.encoded_dim,
                mlp_hidden_dim=self.shader.mlp_hidden_dim,
                mlp_num_layers=self.shader.mlp_num_layers,
            ).contiguous()
            self._mlp_params_key = key
        return self._mlp_params_native
