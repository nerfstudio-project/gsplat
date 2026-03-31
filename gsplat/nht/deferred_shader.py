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

import torch
import tinycudann as tcnn
from typing import Optional, Tuple
from torch import Tensor

# FullyFusedMLP supports a limited output width; above this we use a torch Linear readout.
TCNN_FULLY_FUSED_MAX_OUTPUT_DIM: int = 128


class HarmonicFeatures:
    def __init__(
        self,
        feature_dim: int,
        feature_lr: float,
        features_init_min: float = -5.0,
        features_init_max: float = 5.0,
    ):
        super().__init__()
        assert feature_dim is not None
        assert feature_lr is not None
        self.feature_dim = feature_dim
        self.feature_lr = feature_lr
        self.features_init_min = features_init_min
        self.features_init_max = features_init_max

    def init_features(self, params: torch.nn.ParameterDict, rgbs: torch.Tensor):
        features = self.init_features_random(
            rgbs, self.feature_dim, self.features_init_min, self.features_init_max
        )  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), self.feature_lr))

    @staticmethod
    def init_features_random(
        rgbs: torch.Tensor,
        feature_dim: int,
        features_init_min: float = 0.0,
        features_init_max: float = 1.0,
    ) -> torch.Tensor:
        N = rgbs.shape[0]  # number of points
        features = (
            torch.rand(N, feature_dim) * (features_init_max - features_init_min)
            + features_init_min
        )  # [N, feature_dim]
        return features

class DeferredShaderModule(torch.nn.Module):
    """Deferred shading module."""

    def __init__(
        self,
        feature_dim: int,
        enable_view_encoding: bool,
        view_encoding_type: str = "sh",
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        sh_degree: int = 3,
        sh_scale: float = 1.0,
        fourier_num_freqs: int = 4,
        primitive_type: str = "3dgs",
        center_ray_encoding: bool = False,
        decode_activation: str = "sigmoid",
    ):
        super().__init__()

        # Record parameters
        self.feature_dim = feature_dim
        from gsplat.cuda._wrapper import get_encoding_expansion_factor, get_feature_divisor
        self.encoded_dim = (feature_dim // get_feature_divisor()) * get_encoding_expansion_factor()
        self.view_encoding_type = view_encoding_type
        self.sh_scale = sh_scale
        self.fourier_num_freqs = fourier_num_freqs
        self.center_ray_encoding = center_ray_encoding
        self.primitive_type = primitive_type

        self.enable_view_encoding = enable_view_encoding
        self.view_sh_degree = sh_degree
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers

        # View encoding dimension depends on encoding type
        view_encoding_dim = 0
        if self.enable_view_encoding:
            if view_encoding_type == "sh":
                view_encoding_dim = sh_degree * sh_degree
            elif view_encoding_type == "fourier":
                view_encoding_dim = 6 * fourier_num_freqs
            else:
                raise ValueError(f"Unknown view_encoding_type: {view_encoding_type}")
            print(f"  View encoding: {view_encoding_type}, dim={view_encoding_dim}")

        backbone_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": self.mlp_hidden_dim,
            "n_hidden_layers": max(self.mlp_num_layers, 1),
        }
        if self.enable_view_encoding:
            # Fuse Identity(features) + view encoding with MLP via tcnn Composite
            if view_encoding_type == "sh":
                dir_encoding_config = {
                    "n_dims_to_encode": 3,
                    "otype": "SphericalHarmonics",
                    "degree": sh_degree,
                }
            else:  # fourier
                dir_encoding_config = {
                    "n_dims_to_encode": 3,
                    "otype": "Frequency",
                    "n_frequencies": fourier_num_freqs,
                }
            encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": self.encoded_dim,
                        "otype": "Identity",
                    },
                    dir_encoding_config,
                ],
            }
            mlp_input_dim = self.encoded_dim + 3
            self.backbone = tcnn.NetworkWithInputEncoding(
                n_input_dims=mlp_input_dim,
                n_output_dims=3,
                encoding_config=encoding_config,
                network_config=backbone_config,
            )
        else:
            mlp_input_dim = self.encoded_dim
            self.backbone = tcnn.Network(
                n_input_dims=mlp_input_dim,
                n_output_dims=3,
                network_config=backbone_config,
            )
        # enable jit compilation of the mlp kernel
        # WARNING: it will require dry runs before measuring eval performance, otherwise jit
        # compilation will skew render times
        self.backbone.jit_fusion = tcnn.supports_jit_fusion()

    @property
    def ray_dir_scale(self) -> float:
        """Scaling factor for ray directions in the tcnn [0,1] mapping."""
        if self.enable_view_encoding:
            return self.sh_scale if self.view_encoding_type == "sh" else 1.0
        return 1.0

    def forward(
        self,
        rendered_data: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Decode rasterized features into RGB + optional extras.

        Args:
            rendered_data: (C, H, W, F) where the last 3 channels are
                ray directions from the rasterizer and the preceding
                channels are encoded features (+ optional extras like depth).
            Ks: Unused (kept for backward compat). Ray dirs come from rasterizer.
            camtoworlds: Unused (kept for backward compat).

        Returns:
            colors: (C, H, W, 3)
            extras: Optional tensor of extra features
        """

        C, H, W, D = rendered_data.shape
        if C != 1:
            raise ValueError(
                f"DeferredShader expects single-camera batches (C=1), got C={C}. "
                f"Loop over cameras and call the shader per-camera."
            )

        num_extras = D - self.encoded_dim - 3  # 3 = ray_dir channels
        if self.enable_view_encoding:
            if num_extras > 0:
                mlp_data, extras = rendered_data.split(
                    [self.encoded_dim + 3, num_extras], dim=-1
                )
            else:
                mlp_data = rendered_data
                extras = None
            mlp_inputs = mlp_data.reshape(-1, self.encoded_dim + 3)
            if mlp_inputs.dtype != torch.float32:
                mlp_inputs = mlp_inputs.float()
        else:
            if num_extras > 0:
                features, extras = rendered_data.split(
                    [self.encoded_dim, num_extras + 3], dim=-1
                )
            else:
                features = rendered_data[..., :self.encoded_dim]
                extras = None
            mlp_inputs = features.reshape(-1, self.encoded_dim)
            if mlp_inputs.dtype != torch.float32:
                mlp_inputs = mlp_inputs.float()

        colors = self.backbone(mlp_inputs).float().view(C, H, W, 3)
        return colors, extras


def _nht_encoded_dim(feature_dim: int) -> int:
    from gsplat.cuda._wrapper import get_encoding_expansion_factor, get_feature_divisor

    return (feature_dim // get_feature_divisor()) * get_encoding_expansion_factor()


def _prepare_deferred_mlp_inputs(
    rendered_data: Tensor,
    encoded_dim: int,
    enable_view_encoding: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Split raster tensor into flat MLP inputs and optional extra channels."""
    C, H, W, D = rendered_data.shape
    num_extras = D - encoded_dim - 3  # 3 = ray_dir channels
    if enable_view_encoding:
        if num_extras > 0:
            mlp_data, extras = rendered_data.split(
                [encoded_dim + 3, num_extras], dim=-1
            )
        else:
            mlp_data = rendered_data
            extras = None
        mlp_inputs = mlp_data.reshape(-1, encoded_dim + 3)
    else:
        if num_extras > 0:
            features, extras = rendered_data.split(
                [encoded_dim, num_extras + 3], dim=-1
            )
        else:
            features = rendered_data[..., :encoded_dim]
            extras = None
        mlp_inputs = features.reshape(-1, encoded_dim)
    if mlp_inputs.dtype != torch.float32:
        mlp_inputs = mlp_inputs.float()
    return mlp_inputs, extras


class DeferredShaderModuleAOV(torch.nn.Module):
    """Deferred MLP for RGB plus optional auxiliary channels (multi-head / AOV).

    Architecture is selected with ``split_rgb_head``, ``fused_tcnn_sigmoid``, and
    ``rgb_only_tcnn_sigmoid``:

    * **RGB only** (``auxiliary_output_dim == 0``): tcnn maps to 3 outputs. If
      ``rgb_only_tcnn_sigmoid`` is True, uses ``output_activation="Sigmoid"`` in tcnn
      (same idea as :class:`DeferredShaderModule`); otherwise ``None`` (raw RGB).
    * **Split head** (``split_rgb_head`` and ``auxiliary_output_dim > 0``): tcnn maps
      to ``mlp_hidden_dim`` with ``None``; the first 3 dimensions are RGB (linear
      output from the fused MLP); ``torch.nn.Linear(mlp_hidden_dim - 3, K)`` produces
      auxiliary channels (previous semantic-style AOV path).
    * **Fused direct** (no split, ``3 + K < threshold``): single FullyFusedMLP to
      ``3 + K``. If ``fused_tcnn_sigmoid`` is True, tcnn applies Sigmoid to the **entire**
      ``3 + K`` vector (previous rgb2x-only path); otherwise ``None`` on all outputs.
    * **Full linear readout** (no split, ``3 + K >= threshold``): tcnn to
      ``mlp_hidden_dim`` then ``Linear(mlp_hidden_dim, 3 + K)`` with ``None`` in tcnn.

    ``split_rgb_head`` cannot be combined with ``fused_tcnn_sigmoid``. When
    ``fused_tcnn_sigmoid`` is True on the fused-direct path, returned tensors are
    already squashed by tcnn (document for callers that wrap with extra activations).
    """

    def __init__(
        self,
        feature_dim: int,
        enable_view_encoding: bool,
        auxiliary_output_dim: int = 0,
        view_encoding_type: str = "sh",
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        sh_degree: int = 3,
        sh_scale: float = 1.0,
        fourier_num_freqs: int = 4,
        primitive_type: str = "3dgs",
        center_ray_encoding: bool = False,
        tcnn_fused_output_dim_threshold: int = TCNN_FULLY_FUSED_MAX_OUTPUT_DIM,
        split_rgb_head: bool = False,
        fused_tcnn_sigmoid: bool = False,
        rgb_only_tcnn_sigmoid: bool = True,
    ):
        super().__init__()
        if auxiliary_output_dim < 0:
            raise ValueError("auxiliary_output_dim must be >= 0")
        if tcnn_fused_output_dim_threshold < 1:
            raise ValueError("tcnn_fused_output_dim_threshold must be >= 1")
        if split_rgb_head and auxiliary_output_dim == 0:
            raise ValueError("split_rgb_head requires auxiliary_output_dim > 0")
        if split_rgb_head and fused_tcnn_sigmoid:
            raise ValueError("split_rgb_head cannot be used with fused_tcnn_sigmoid")
        if mlp_hidden_dim <= 3 and split_rgb_head:
            raise ValueError("mlp_hidden_dim must be > 3 when split_rgb_head is True")

        self.feature_dim = feature_dim
        self.encoded_dim = _nht_encoded_dim(feature_dim)
        self.auxiliary_output_dim = auxiliary_output_dim
        self.view_encoding_type = view_encoding_type
        self.sh_scale = sh_scale
        self.fourier_num_freqs = fourier_num_freqs
        self.center_ray_encoding = center_ray_encoding
        self.primitive_type = primitive_type
        self.enable_view_encoding = enable_view_encoding
        self.view_sh_degree = sh_degree
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.tcnn_fused_output_dim_threshold = tcnn_fused_output_dim_threshold
        self.split_rgb_head = split_rgb_head
        self.fused_tcnn_sigmoid = fused_tcnn_sigmoid
        self.rgb_only_tcnn_sigmoid = rgb_only_tcnn_sigmoid

        total_out = 3 + auxiliary_output_dim
        self._total_output_dim = total_out

        if self.enable_view_encoding:
            if view_encoding_type not in ("sh", "fourier"):
                raise ValueError(f"Unknown view_encoding_type: {view_encoding_type}")
            print(f"  View encoding: {view_encoding_type}")

        self.output_proj: Optional[torch.nn.Linear] = None
        self.auxiliary_head: Optional[torch.nn.Linear] = None
        self._architecture: str
        self._tcnn_sigmoid_on_outputs: bool = False

        if auxiliary_output_dim == 0:
            self._architecture = "rgb_only_sigmoid" if rgb_only_tcnn_sigmoid else "rgb_only_raw"
            backbone_out_dim = 3
            out_act = "Sigmoid" if rgb_only_tcnn_sigmoid else "None"
            self._tcnn_sigmoid_on_outputs = rgb_only_tcnn_sigmoid
        elif split_rgb_head:
            self._architecture = "split_rgb_aux_linear"
            backbone_out_dim = mlp_hidden_dim
            out_act = "None"
            self.auxiliary_head = torch.nn.Linear(mlp_hidden_dim - 3, auxiliary_output_dim)
        elif total_out < tcnn_fused_output_dim_threshold:
            self._architecture = "fused_direct"
            backbone_out_dim = total_out
            out_act = "Sigmoid" if fused_tcnn_sigmoid else "None"
            self._tcnn_sigmoid_on_outputs = fused_tcnn_sigmoid
        else:
            self._architecture = "full_linear_readout"
            backbone_out_dim = mlp_hidden_dim
            out_act = "None"
            self.output_proj = torch.nn.Linear(mlp_hidden_dim, total_out)

        backbone_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": out_act,
            "n_neurons": self.mlp_hidden_dim,
            "n_hidden_layers": max(self.mlp_num_layers, 1),
        }

        if self.enable_view_encoding:
            if view_encoding_type == "sh":
                dir_encoding_config = {
                    "n_dims_to_encode": 3,
                    "otype": "SphericalHarmonics",
                    "degree": sh_degree,
                }
            else:
                dir_encoding_config = {
                    "n_dims_to_encode": 3,
                    "otype": "Frequency",
                    "n_frequencies": fourier_num_freqs,
                }
            encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": self.encoded_dim,
                        "otype": "Identity",
                    },
                    dir_encoding_config,
                ],
            }
            mlp_input_dim = self.encoded_dim + 3
            self.backbone = tcnn.NetworkWithInputEncoding(
                n_input_dims=mlp_input_dim,
                n_output_dims=backbone_out_dim,
                encoding_config=encoding_config,
                network_config=backbone_config,
            )
        else:
            mlp_input_dim = self.encoded_dim
            self.backbone = tcnn.Network(
                n_input_dims=mlp_input_dim,
                n_output_dims=backbone_out_dim,
                network_config=backbone_config,
            )

        self.backbone.jit_fusion = tcnn.supports_jit_fusion()

    @property
    def uses_direct_fused_output(self) -> bool:
        return self._architecture == "fused_direct"

    @property
    def uses_split_rgb_aux_linear(self) -> bool:
        return self._architecture == "split_rgb_aux_linear"

    @property
    def uses_full_linear_readout(self) -> bool:
        return self._architecture == "full_linear_readout"

    @property
    def tcnn_emitted_sigmoid_outputs(self) -> bool:
        """True if tcnn already applied Sigmoid to returned RGB (and aux on fused path)."""
        return self._tcnn_sigmoid_on_outputs

    @property
    def ray_dir_scale(self) -> float:
        if self.enable_view_encoding:
            return self.sh_scale if self.view_encoding_type == "sh" else 1.0
        return 1.0

    def forward(
        self,
        rendered_data: Tensor,
        Ks: Optional[Tensor] = None,
        camtoworlds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Decode rasterized features into RGB and optional auxiliary channels.

        When :py:attr:`tcnn_emitted_sigmoid_outputs` is True, the corresponding
        tcnn outputs are already in ``[0, 1]`` (no extra sigmoid needed upstream).

        Args:
            rendered_data: (C, H, W, F) NHT raster layout (features, ray dirs, optional extras).
            Ks, camtoworlds: Unused; kept for API compatibility with older call sites.

        Returns:
            rgb_raw: (C, H, W, 3)
            auxiliary_raw: (C, H, W, K) if K > 0 else None
            extras: optional trailing raster channels
        """
        del Ks, camtoworlds  # unused
        C, H, W, _ = rendered_data.shape
        if C != 1:
            raise ValueError(
                f"DeferredShaderModuleAOV expects single-camera batches (C=1), got C={C}. "
                f"Loop over cameras and call the shader per-camera."
            )

        mlp_inputs, extras = _prepare_deferred_mlp_inputs(
            rendered_data, self.encoded_dim, self.enable_view_encoding
        )
        h = self.backbone(mlp_inputs).float()

        if self._architecture in ("rgb_only_sigmoid", "rgb_only_raw"):
            rgb_raw = h.view(C, H, W, 3)
            auxiliary_raw = None
        elif self._architecture == "split_rgb_aux_linear":
            assert self.auxiliary_head is not None
            rgb_raw = h[:, :3].view(C, H, W, 3)
            auxiliary_raw = self.auxiliary_head(h[:, 3:]).view(
                C, H, W, self.auxiliary_output_dim
            )
        elif self._architecture == "fused_direct":
            rgb_raw = h[:, :3].view(C, H, W, 3)
            auxiliary_raw = (
                h[:, 3:].view(C, H, W, self.auxiliary_output_dim)
                if self.auxiliary_output_dim > 0
                else None
            )
        else:
            assert self.output_proj is not None
            out = self.output_proj(h)
            rgb_raw = out[:, :3].view(C, H, W, 3)
            auxiliary_raw = (
                out[:, 3:].view(C, H, W, self.auxiliary_output_dim)
                if self.auxiliary_output_dim > 0
                else None
            )
        return rgb_raw, auxiliary_raw, extras