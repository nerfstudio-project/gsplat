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

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..sh_compression import SH_COMPRESSION_MAP, SHCompressionMode
from .base import Scene

_SH_COMPRESSION_MAP = SH_COMPRESSION_MAP


class GaussianInferenceScene(Scene):
    """Activated, packed Gaussian scene for inference rendering.

    Holds packed tensors in the viewer-internal layout. Constructed via
    classmethods; ``__init__`` is not part of the public API.
    """

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.means_planar: Optional[Tensor] = None  # [3, N] float32 CUDA
        self.qso_packed: Optional[Tensor] = None  # [N, 8] float16 CUDA
        self.colors_packed: Optional[Tensor] = None  # varies by sh_degree
        self.sh_degree: Optional[int] = None
        self.sh_compression_mode: Optional[SHCompressionMode] = None
        self.num_gaussians: int = 0
        self.component_names: list[str] = []
        self.component_index: Tensor = torch.zeros(0, dtype=torch.long)

    def is_empty(self) -> bool:
        return (
            self.means_planar is None
            or self.qso_packed is None
            or self.colors_packed is None
            or self.num_gaussians == 0
        )

    def release(self) -> None:
        self.means_planar = None
        self.qso_packed = None
        self.colors_packed = None
        self.sh_degree = None
        self.sh_compression_mode = None
        self.num_gaussians = 0
        self.component_names = []
        self.component_index = torch.zeros(0, dtype=torch.long)

    def put(self, name: str, component: dict) -> None:
        if not name:
            raise ValueError("component name must not be empty")
        scene_empty = self.is_empty()
        if not scene_empty and name in self.component_names:
            raise ValueError(
                f"component '{name}' already present in GaussianInferenceScene"
            )

        means_planar = component["means_planar"]
        qso_packed = component["qso_packed"]
        colors_packed = component["colors_packed"]
        sh_degree = component["sh_degree"]
        sh_compression_mode = component["sh_compression_mode"]

        n_local = self._validate_packed_component(
            means_planar, qso_packed, colors_packed, sh_degree, sh_compression_mode
        )

        if scene_empty:
            # First component
            self.means_planar = means_planar
            self.qso_packed = qso_packed
            self.colors_packed = colors_packed
            self.sh_degree = sh_degree
            self.sh_compression_mode = sh_compression_mode
            self.num_gaussians = n_local
            self.component_names = [name]
            self.component_index = torch.zeros(
                n_local, device=means_planar.device, dtype=torch.long
            )
        else:
            # Validate consistency
            if sh_degree != self.sh_degree:
                raise ValueError(
                    f"sh_degree mismatch: scene has {self.sh_degree}, "
                    f"component has {sh_degree}"
                )
            if sh_compression_mode != self.sh_compression_mode:
                raise ValueError(
                    f"sh_compression_mode mismatch: scene has "
                    f"{self.sh_compression_mode}, "
                    f"component has {sh_compression_mode}"
                )
            self.means_planar = torch.cat([self.means_planar, means_planar], dim=1)
            self.qso_packed = torch.cat([self.qso_packed, qso_packed], dim=0)
            self.colors_packed = torch.cat([self.colors_packed, colors_packed], dim=0)
            self.component_names.append(name)
            self.component_index = torch.cat(
                [
                    self.component_index,
                    torch.full(
                        (n_local,),
                        len(self.component_names) - 1,
                        device=self.component_index.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            self.num_gaussians += n_local

    @staticmethod
    def _validate_packed_component(
        means_planar: Tensor,
        qso_packed: Tensor,
        colors_packed: Tensor,
        sh_degree: int,
        sh_compression_mode: SHCompressionMode,
    ) -> int:
        tensors = {
            "means_planar": means_planar,
            "qso_packed": qso_packed,
            "colors_packed": colors_packed,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if means_planar.dim() != 2 or means_planar.shape[0] != 3:
            raise ValueError(
                f"means_planar must have shape [3, N]; got {list(means_planar.shape)}"
            )
        if qso_packed.dim() != 2 or qso_packed.shape[1] != 8:
            raise ValueError(
                f"qso_packed must have shape [N, 8]; got {list(qso_packed.shape)}"
            )

        n_local = qso_packed.shape[0]
        if means_planar.shape[1] != n_local:
            raise ValueError(
                f"means_planar.shape[1] ({means_planar.shape[1]}) must match "
                f"qso_packed.shape[0] ({n_local})"
            )
        if colors_packed.dim() < 2 or colors_packed.shape[0] != n_local:
            raise ValueError(
                f"colors_packed.shape[0] must match N={n_local}; "
                f"got shape {list(colors_packed.shape)}"
            )

        device = means_planar.device
        for name, tensor in (
            ("qso_packed", qso_packed),
            ("colors_packed", colors_packed),
        ):
            if tensor.device != device:
                raise ValueError(
                    f"{name} device {tensor.device} must match means_planar device {device}"
                )

        if means_planar.dtype != torch.float32:
            raise TypeError(
                f"means_planar must have dtype torch.float32; got {means_planar.dtype}"
            )
        if qso_packed.dtype != torch.float16:
            raise TypeError(
                f"qso_packed must have dtype torch.float16; got {qso_packed.dtype}"
            )

        if not isinstance(sh_compression_mode, SHCompressionMode):
            raise TypeError(
                "sh_compression_mode must be an SHCompressionMode; "
                f"got {type(sh_compression_mode).__name__}"
            )
        if sh_compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
            raise ValueError(
                f"sh_compression_mode={sh_compression_mode} requires sh_degree=3; "
                f"got sh_degree={sh_degree}"
            )

        if sh_degree == -1:
            expected_shape = (n_local, 4)
            expected_dtype = torch.float16
        elif sh_degree in (0, 1, 2):
            expected_shape = (n_local, (sh_degree + 1) ** 2, 3)
            expected_dtype = torch.float32
        elif sh_degree == 3 and sh_compression_mode is SHCompressionMode.NONE:
            expected_shape = (n_local, 16, 3)
            expected_dtype = torch.float16
        elif sh_degree == 3 and sh_compression_mode is SHCompressionMode.PACKED_32B:
            expected_shape = (n_local, 48)
            expected_dtype = torch.float32
        elif sh_degree == 3 and sh_compression_mode is SHCompressionMode.PACKED_16B:
            expected_shape = (n_local, 48)
            expected_dtype = torch.float16
        else:
            raise ValueError(
                f"sh_degree must be one of [-1, 0, 1, 2, 3]; got {sh_degree}"
            )

        if tuple(colors_packed.shape) != expected_shape:
            raise ValueError(
                f"colors_packed must have shape {list(expected_shape)} for "
                f"sh_degree={sh_degree}, sh_compression_mode={sh_compression_mode}; "
                f"got {list(colors_packed.shape)}"
            )
        if colors_packed.dtype != expected_dtype:
            raise TypeError(
                f"colors_packed must have dtype {expected_dtype}; got {colors_packed.dtype}"
            )

        return n_local

    def get(self, component: str | int) -> dict:
        if self.is_empty():
            raise RuntimeError(
                "GaussianInferenceScene has been released and contains no packed tensors"
            )

        if isinstance(component, int):
            if component < 0 or component >= len(self.component_names):
                raise KeyError(f"'{component}'")
            comp_id = component
        else:
            if component not in self.component_names:
                raise KeyError(f"'{component}'")
            comp_id = self.component_names.index(component)

        mask = self.component_index == comp_id
        return {
            "name": self.component_names[comp_id],
            "index": comp_id,
            "mask": mask,
            "means_planar": self.means_planar[:, mask],
            "qso_packed": self.qso_packed[mask],
            "colors_packed": self.colors_packed[mask],
            "sh_degree": self.sh_degree,
            "sh_compression_mode": self.sh_compression_mode,
        }

    @classmethod
    def from_gaussian_scene(
        cls,
        scene,  # GaussianScene - use duck typing to avoid circular import
        *,
        id: str,
        sh_compression: str = "none",
    ) -> "GaussianInferenceScene":
        """Build a GaussianInferenceScene from a training GaussianScene.

        Applies activations automatically: ``F.normalize`` on quaternions,
        ``.exp()`` on scales, ``.sigmoid()`` on opacities.  Colors are read
        from ``splats["colors"]`` or concatenated ``splats["sh0"]``/``splats["shN"]``.

        Appearance-optimized scenes (those containing ``"features"`` in splats) are
        rejected — use :meth:`from_gaussian_tensors` with baked RGB instead.

        Not supported under ``torch.distributed`` with ``world_size > 1``; gather
        tensors first and call :meth:`from_gaussian_tensors`.

        Args:
            scene: A ``GaussianScene`` instance with raw (log-space) splats.
            id: Unique identifier for the inference scene.
            sh_compression: SH packing mode — ``"none"``, ``"32b"``, or ``"16b"``.
                Only meaningful for ``sh_degree == 3``.

        Returns:
            A new ``GaussianInferenceScene`` with packed tensors.  Quaternions
            follow the wxyz convention matching gsplat's rendering path.

        Raises:
            RuntimeError: If called under distributed with ``world_size > 1``.
            ValueError: If splats contain ``"features"`` (appearance-optimized).
            ValueError: If no color data is found in splats.
            ValueError: If activation produces NaN or Inf.
        """
        # Multi-component rejection
        if hasattr(scene, "component_names") and len(scene.component_names) > 1:
            raise ValueError(
                "from_gaussian_scene does not support multi-component scenes; "
                "convert each component individually via from_gaussian_tensors"
            )

        # Distributed rejection
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "GaussianInferenceScene.from_gaussian_scene is not supported under "
                "distributed (world_size > 1); gather first and use "
                "from_gaussian_tensors"
            )

        splats = scene.splats

        if "features" in splats:
            raise ValueError(
                "from_gaussian_scene does not support appearance-optimized scenes "
                "(splats contain 'features'). Bake activated per-view RGB and use "
                "from_gaussian_tensors() instead."
            )

        means = splats["means"]
        quats = F.normalize(splats["quats"], dim=-1)
        scales = splats["scales"].exp()
        opacities = splats["opacities"].sigmoid()

        for act_name, act_tensor in [
            ("quats (after normalize)", quats),
            ("scales (after exp)", scales),
            ("opacities (after sigmoid)", opacities),
        ]:
            if not torch.isfinite(act_tensor).all():
                raise ValueError(
                    f"from_gaussian_scene: {act_name} contains NaN or Inf after "
                    f"activation; check the source GaussianScene for invalid values"
                )

        colors = splats.get("colors", None)
        if colors is None:
            sh0 = splats.get("sh0", None)
            if sh0 is not None and "shN" in splats:
                colors = torch.cat([sh0, splats["shN"]], dim=1)
            elif sh0 is not None:
                warnings.warn(
                    "GaussianScene has sh0 but no shN; degrading to SH degree 0",
                    UserWarning,
                    stacklevel=2,
                )
                colors = sh0
            else:
                colors = sh0
        if colors is None:
            raise ValueError("GaussianScene must contain 'colors' or 'sh0' in splats")

        # Determine sh_degree from colors shape
        if colors.dim() == 3:
            K_sh = colors.shape[1]
            basis_width = math.isqrt(K_sh)
            if basis_width * basis_width != K_sh:
                raise ValueError(
                    f"colors SH basis dimension must be a perfect square; got {K_sh}"
                )
            sh_degree_val = basis_width - 1
        else:
            sh_degree_val = None

        return cls._build(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree_val,
            sh_compression=sh_compression,
            id=id,
            skip_activation_checks=True,
        )

    @classmethod
    def from_gaussian_tensors(
        cls,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        sh_degree: Optional[int],
        sh_compression: str,
        *,
        id: str,
    ) -> "GaussianInferenceScene":
        """Build a GaussianInferenceScene from pre-activated tensors.

        All inputs must already have activations applied:

        - ``quats``: unit-norm quaternions in wxyz order
          (apply ``F.normalize(quats, dim=-1)`` if needed).
        - ``scales``: positive values (apply ``.exp()`` if in log-space).
        - ``opacities``: values in ``[0, 1]`` (apply ``.sigmoid()`` if in
          logit-space).
        - ``colors``: ``[N, 3]`` for RGB or ``[N, K, 3]`` for SH coefficients.

        Args:
            means: ``[N, 3]`` float32 CUDA — world-space positions.
            quats: ``[N, 4]`` float32 CUDA — unit quaternions (wxyz).
            scales: ``[N, 3]`` float32 CUDA — positive scales.
            opacities: ``[N]`` float32 CUDA — opacities in ``[0, 1]``.
            colors: ``[N, 3]`` or ``[N, K, 3]`` float32 CUDA.
            sh_degree: ``None`` or ``-1`` for RGB; ``0``–``3`` for SH.
            sh_compression: ``"none"``, ``"32b"``, or ``"16b"``.
            id: Unique identifier for the inference scene.

        Returns:
            A new ``GaussianInferenceScene`` with packed tensors.

        Raises:
            ValueError: If activation contracts are violated (non-positive scales,
                opacities outside ``[0, 1]``, non-unit quaternions, NaN/Inf).
        """
        return cls._build(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            sh_compression=sh_compression,
            id=id,
            skip_activation_checks=False,
        )

    @classmethod
    def _build(
        cls,
        *,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        sh_degree: Optional[int],
        sh_compression: str,
        id: str,
        skip_activation_checks: bool,
    ) -> "GaussianInferenceScene":
        # Validate sh_compression
        if sh_compression not in _SH_COMPRESSION_MAP:
            raise ValueError(
                f"sh_compression must be one of {{'none', '32b', '16b'}}; "
                f"got '{sh_compression}'"
            )
        sh_compression_mode = _SH_COMPRESSION_MAP[sh_compression]

        # Validate means shape: must be [N, 3]
        if means.dim() != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape [N, 3]; got {list(means.shape)}")

        # Validate sh_degree vs colors shape consistency
        if sh_degree is not None and sh_degree >= 0:
            if colors.dim() != 3:
                raise ValueError(
                    f"sh_degree={sh_degree} requires colors to be 3-D [N, K, 3]; "
                    f"got {colors.dim()}-D shape {list(colors.shape)}"
                )
            expected_K = (sh_degree + 1) ** 2
            if colors.shape[1] != expected_K:
                raise ValueError(
                    f"sh_degree={sh_degree} requires colors.shape[1]={expected_K} "
                    f"(got {colors.shape[1]})"
                )
        elif sh_degree is None or sh_degree < 0:
            if colors.dim() != 2:
                raise ValueError(
                    f"sh_degree=None (pre-activated RGB) requires colors to be "
                    f"2-D [N, 3]; got {colors.dim()}-D shape {list(colors.shape)}"
                )

        # Validate sh_compression is only used with sh_degree == 3
        if sh_compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
            raise ValueError(
                f"sh_compression='{sh_compression}' requires sh_degree=3; "
                f"got sh_degree={sh_degree}"
            )

        # Activation contract checks (only for from_gaussian_tensors)
        if not skip_activation_checks:
            _check_activation_contract(means, quats, scales, opacities, colors)

        # Detach-and-warn
        tensors = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }
        detached_names = []
        for name, t in tensors.items():
            if t.requires_grad:
                detached_names.append(name)
                tensors[name] = t.detach()

        if detached_names:
            warnings.warn(
                f"GaussianInferenceScene: detached grad-tracked inputs: "
                f"{detached_names}. "
                f"The resulting packed tensors will not participate in "
                f"autograd.",
                RuntimeWarning,
                stacklevel=3,
            )

        means = tensors["means"]
        quats = tensors["quats"]
        scales = tensors["scales"]
        opacities = tensors["opacities"]
        colors = tensors["colors"]

        # fp16 clamp warning check (before calling C++ op which also clamps)
        fp16_max = torch.finfo(torch.float16).max
        clamp_warnings: list = []
        _check_fp16_range(scales, "scales", fp16_max, clamp_warnings)
        _check_fp16_range(colors, "colors", fp16_max, clamp_warnings)

        if clamp_warnings:
            msg_lines = ["GaussianInferenceScene: fp16 clamping applied:"]
            for entry in clamp_warnings:
                if len(entry) == 5:
                    name, count, orig_min, orig_max, nonfinite_count = entry
                    msg = f"  {name}: clamped {count} elements (original range [{orig_min:.4g}, {orig_max:.4g}])"
                    if nonfinite_count > 0:
                        msg += f" ({nonfinite_count} non-finite)"
                else:
                    name, count, orig_min, orig_max = entry
                    msg = f"  {name}: clamped {count} elements (original range [{orig_min:.4g}, {orig_max:.4g}])"
                msg_lines.append(msg)
            warnings.warn("\n".join(msg_lines), RuntimeWarning, stacklevel=3)

        # Encode sh_degree for C++ op
        sh_deg = -1 if sh_degree is None else sh_degree

        # Call C++ packing op through the scene kernels layer
        from ..kernels.gaussian_inference_ops import (
            pack_gaussian_inference_scene as _pack,
        )

        means_planar, qso_packed, colors_packed = _pack(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_deg,
            sh_compression_mode,
        )

        # Build scene
        scene = cls(id)
        scene.put(
            id,
            {
                "means_planar": means_planar,
                "qso_packed": qso_packed,
                "colors_packed": colors_packed,
                "sh_degree": sh_deg,
                "sh_compression_mode": sh_compression_mode,
            },
        )
        return scene


def _check_activation_contract(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
) -> None:
    """Validate activation contract for from_gaussian_tensors."""
    # Check for NaN/Inf in all tensors
    for name, t in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        bad = ~torch.isfinite(t)
        if bad.any():
            indices = (
                torch.nonzero(bad.reshape(-1), as_tuple=False).squeeze(-1).tolist()
            )
            # Limit displayed indices
            display = indices[:10]
            suffix = f"... ({len(indices)} total)" if len(indices) > 10 else ""
            raise ValueError(
                f"tensor '{name}' contains NaN or Inf at indices " f"{display}{suffix}"
            )

    # scales must be positive (exp(x) > 0 always)
    bad_scales = scales <= 0
    if bad_scales.any():
        indices = (
            torch.nonzero(bad_scales.any(dim=-1), as_tuple=False).squeeze(-1).tolist()
        )
        display = indices[:10]
        suffix = f"... ({len(indices)} total)" if len(indices) > 10 else ""
        raise ValueError(
            f"scales contain non-positive values at indices "
            f"{display}{suffix}; did you forget to call .exp()?"
        )

    # opacities must be in [0, 1] (sigmoid bounded)
    bad_opacities = (opacities < 0) | (opacities > 1)
    if bad_opacities.any():
        indices = torch.nonzero(bad_opacities, as_tuple=False).squeeze(-1).tolist()
        display = indices[:10]
        suffix = f"... ({len(indices)} total)" if len(indices) > 10 else ""
        raise ValueError(
            f"opacities outside [0, 1] at indices {display}{suffix}; "
            f"did you forget to call .sigmoid()?"
        )

    # quats must be unit-norm (F.normalize applied)
    norms = quats.norm(dim=-1)
    bad_norms = (norms - 1).abs() > 1e-3
    if bad_norms.any():
        indices = torch.nonzero(bad_norms, as_tuple=False).squeeze(-1).tolist()
        display = indices[:10]
        suffix = f"... ({len(indices)} total)" if len(indices) > 10 else ""
        raise ValueError(
            f"quats are not unit-norm at indices {display}{suffix}; "
            f"did you forget F.normalize(quats, dim=-1)?"
        )


def _check_fp16_range(
    tensor: Tensor, name: str, fp16_max: float, warnings_list: list
) -> None:
    """Check if tensor has values outside fp16 range or non-finite."""
    with torch.no_grad():
        nonfinite = ~torch.isfinite(tensor)
        abs_vals = tensor.abs()
        exceed_mask = (abs_vals > fp16_max) | nonfinite
        count = int(exceed_mask.sum().item())
        if count > 0:
            finite_vals = tensor[torch.isfinite(tensor)]
            if finite_vals.numel() > 0:
                orig_min = float(finite_vals.min().item())
                orig_max = float(finite_vals.max().item())
            else:
                orig_min = float("nan")
                orig_max = float("nan")
            nonfinite_count = int(nonfinite.sum().item())
            warnings_list.append((name, count, orig_min, orig_max, nonfinite_count))
