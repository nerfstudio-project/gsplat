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

from typing import Any, Optional, Tuple

import torch
from torch import Tensor


# ``tinycudann`` is a heavy optional dependency (it requires a CUDA toolchain
# at install time). Import it lazily so that ``import gsplat`` succeeds even
# when the user has not installed the ``[nht]`` extra. Only callers that
# actually construct a deferred-shader module will hit the import error.
_tcnn: Optional[Any] = None


def _require_tcnn() -> Any:
    global _tcnn
    if _tcnn is not None:
        return _tcnn
    try:
        import tinycudann as tcnn  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "gsplat.nht.deferred_shader requires the optional 'tinycudann' "
            'package. Install with: pip install "gsplat[nht]". Note that '
            "tiny-cuda-nn needs a working CUDA toolchain at install time; "
            "see docs/nht.md for details."
        ) from e
    _tcnn = tcnn
    return _tcnn


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
    """Deferred NHT decoder for RGB plus optional AOV outputs.

    The default RGB-only configuration is backward compatible with the original
    NHT shader. When `auxiliary_output_dim > 0`, the same module selects an
    efficient output architecture (direct fused, split head, or linear readout)
    and returns decoded auxiliary channels in the `extras` output.
    """

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
        auxiliary_output_dim: int = 0,
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
        if split_rgb_head and mlp_hidden_dim <= 3:
            raise ValueError("mlp_hidden_dim must be > 3 when split_rgb_head is True")

        from gsplat.nht._wrapper import (
            get_encoding_expansion_factor,
            get_feature_divisor,
        )

        tcnn = _require_tcnn()
        self.feature_dim = feature_dim
        self.encoded_dim = (
            feature_dim // get_feature_divisor()
        ) * get_encoding_expansion_factor()
        self.view_encoding_type = view_encoding_type
        self.sh_scale = sh_scale
        self.fourier_num_freqs = fourier_num_freqs
        self.center_ray_encoding = center_ray_encoding
        self.primitive_type = primitive_type
        self.enable_view_encoding = enable_view_encoding
        self.view_sh_degree = sh_degree
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.auxiliary_output_dim = auxiliary_output_dim
        self.tcnn_fused_output_dim_threshold = tcnn_fused_output_dim_threshold
        self.split_rgb_head = split_rgb_head
        self.fused_tcnn_sigmoid = fused_tcnn_sigmoid
        self.rgb_only_tcnn_sigmoid = rgb_only_tcnn_sigmoid
        self.decode_activation = decode_activation

        if self.enable_view_encoding:
            if view_encoding_type not in ("sh", "fourier"):
                raise ValueError(f"Unknown view_encoding_type: {view_encoding_type}")
            print(f"  View encoding: {view_encoding_type}")

        total_out = 3 + auxiliary_output_dim
        self._total_output_dim = total_out
        self.output_proj: Optional[torch.nn.Linear] = None
        self.auxiliary_head: Optional[torch.nn.Linear] = None
        self._tcnn_sigmoid_on_outputs = False

        if auxiliary_output_dim == 0:
            self._architecture = (
                "rgb_only_sigmoid" if rgb_only_tcnn_sigmoid else "rgb_only_raw"
            )
            backbone_out_dim = 3
            out_act = "Sigmoid" if rgb_only_tcnn_sigmoid else "None"
            self._tcnn_sigmoid_on_outputs = rgb_only_tcnn_sigmoid
        elif split_rgb_head:
            self._architecture = "split_rgb_aux_linear"
            backbone_out_dim = mlp_hidden_dim
            out_act = "None"
            self.auxiliary_head = torch.nn.Linear(
                mlp_hidden_dim - 3, auxiliary_output_dim
            )
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
            self.backbone = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.encoded_dim + 3,
                n_output_dims=backbone_out_dim,
                encoding_config=encoding_config,
                network_config=backbone_config,
            )
        else:
            self.backbone = tcnn.Network(
                n_input_dims=self.encoded_dim,
                n_output_dims=backbone_out_dim,
                network_config=backbone_config,
            )
        # enable jit compilation of the mlp kernel
        # WARNING: it will require dry runs before measuring eval performance, otherwise jit
        # compilation will skew render times
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
        return self._tcnn_sigmoid_on_outputs

    @property
    def ray_dir_scale(self) -> float:
        """Scaling factor for ray directions in the tcnn [0,1] mapping."""
        if self.enable_view_encoding:
            return self.sh_scale if self.view_encoding_type == "sh" else 1.0
        return 1.0

    def _decode(self, mlp_inputs: Tensor, C: int, H: int, W: int):
        h = self.backbone(mlp_inputs).float()
        if self._architecture in ("rgb_only_sigmoid", "rgb_only_raw"):
            return h.view(C, H, W, 3), None
        if self._architecture == "split_rgb_aux_linear":
            assert self.auxiliary_head is not None
            rgb = h[:, :3].view(C, H, W, 3)
            aux = self.auxiliary_head(h[:, 3:]).view(C, H, W, self.auxiliary_output_dim)
            return rgb, aux
        if self._architecture == "fused_direct":
            rgb = h[:, :3].view(C, H, W, 3)
            aux = (
                h[:, 3:].view(C, H, W, self.auxiliary_output_dim)
                if self.auxiliary_output_dim > 0
                else None
            )
            return rgb, aux
        assert self.output_proj is not None
        out = self.output_proj(h)
        rgb = out[:, :3].view(C, H, W, 3)
        aux = (
            out[:, 3:].view(C, H, W, self.auxiliary_output_dim)
            if self.auxiliary_output_dim > 0
            else None
        )
        return rgb, aux

    def forward(self, rendered_data: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        C, H, W, _ = rendered_data.shape
        if C != 1:
            raise ValueError(
                f"DeferredShader expects single-camera batches (C=1), got C={C}. "
                f"Loop over cameras and call the shader per-camera."
            )

        mlp_inputs, raster_extras = _prepare_deferred_mlp_inputs(
            rendered_data, self.encoded_dim, self.enable_view_encoding
        )
        rgb, aux = self._decode(mlp_inputs, C, H, W)
        if raster_extras is not None and aux is not None:
            extras = torch.cat([raster_extras, aux], dim=-1)
        else:
            extras = raster_extras if raster_extras is not None else aux
        return rgb, extras


def _nht_encoded_dim(feature_dim: int) -> int:
    from gsplat.nht._wrapper import get_encoding_expansion_factor, get_feature_divisor

    return (feature_dim // get_feature_divisor()) * get_encoding_expansion_factor()


def _prepare_deferred_mlp_inputs(
    rendered_data: Tensor,
    encoded_dim: int,
    enable_view_encoding: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Split raster tensor into flat MLP inputs and optional extra channels.

    Layout depends on ``enable_view_encoding`` (see ``DeferredShaderModule.forward``):

    * ``True``: ``[encoded_features (encoded_dim) | ray_dirs (3) | extras (num_extras)]``.
    * ``False``: ``[encoded_features (encoded_dim) | extras (rest)]`` \u2014 ray-dir
      channels are not consumed and don't have to be present.
    """
    C, H, W, D = rendered_data.shape
    if enable_view_encoding:
        num_extras = D - encoded_dim - 3
        if num_extras < 0:
            raise ValueError(
                f"rendered_data has {D} channels but at least encoded_dim + 3 = "
                f"{encoded_dim + 3} are required when view encoding is on."
            )
        features = rendered_data[..., :encoded_dim]
        ray_dirs = rendered_data[..., encoded_dim : encoded_dim + 3]
        extras = rendered_data[..., encoded_dim + 3 :] if num_extras > 0 else None
        mlp_data = torch.cat([features, ray_dirs], dim=-1)
        mlp_inputs = mlp_data.reshape(-1, encoded_dim + 3)
    else:
        num_extras = D - encoded_dim
        if num_extras < 0:
            raise ValueError(
                f"rendered_data has {D} channels but at least encoded_dim = "
                f"{encoded_dim} are required."
            )
        features = rendered_data[..., :encoded_dim]
        extras = rendered_data[..., encoded_dim:] if num_extras > 0 else None
        mlp_inputs = features.reshape(-1, encoded_dim)
    if mlp_inputs.dtype != torch.float32:
        mlp_inputs = mlp_inputs.float()
    return mlp_inputs, extras


class DeferredShaderModuleAOV(DeferredShaderModule):
    """Compatibility wrapper around DeferredShaderModule's auxiliary outputs."""

    def forward(
        self,
        rendered_data: Tensor,
        Ks: Optional[Tensor] = None,
        camtoworlds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        del Ks, camtoworlds
        C, H, W, _ = rendered_data.shape
        mlp_inputs, raster_extras = _prepare_deferred_mlp_inputs(
            rendered_data, self.encoded_dim, self.enable_view_encoding
        )
        rgb, aux = self._decode(mlp_inputs, C, H, W)
        return rgb, aux, raster_extras
