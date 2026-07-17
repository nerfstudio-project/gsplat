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

"""Python-level wrappers for the native env-map texture-sampling ops.

:class:`gsplat.scene.components.EnvMapBackground` samples its texture through
the fused CUDA fwd/bwd kernels (``gsplat_scene_cuda``) ONLY — there is no
runtime CPU/PyTorch fallback. Inputs must be float32 CUDA tensors; the public
dispatch wrappers raise a clear error otherwise.

The CUDA kernels differentiate w.r.t. ``textures`` only; ``rays_d`` is a
constant input (its gradient is ``None``). Accessing the lazy ``_SCENE_CUDA``
backend is deferred to inside forward/backward so importing this module never
triggers a JIT build.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Internal validation helpers (mirrors gaussian_inference_ops.py)
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


def _require_same_device(name: str, t: Tensor, ref_name: str, ref: Tensor) -> None:
    if t.device != ref.device:
        raise ValueError(
            f"{name} must be on the same device as {ref_name} "
            f"({ref.device}); got {t.device}"
        )


def _validate_equirect_inputs(rays_d: Tensor, textures: Tensor) -> None:
    rays_d = _expect_tensor("rays_d", rays_d)
    textures = _expect_tensor("textures", textures)
    _require_cuda("rays_d", rays_d)
    _require_cuda("textures", textures)
    _require_float32("rays_d", rays_d)
    _require_float32("textures", textures)
    _require_same_device("textures", textures, "rays_d", rays_d)
    if rays_d.dim() != 2 or rays_d.shape[1] != 3:
        raise ValueError(f"rays_d must have shape [N, 3]; got {tuple(rays_d.shape)}")
    if textures.dim() != 4 or textures.shape[0] != 1 or textures.shape[3] != 3:
        raise ValueError(
            f"equirectangular textures must have shape [1, H, W, 3]; "
            f"got {tuple(textures.shape)}"
        )


def _validate_cubemap_inputs(rays_d: Tensor, textures: Tensor) -> None:
    rays_d = _expect_tensor("rays_d", rays_d)
    textures = _expect_tensor("textures", textures)
    _require_cuda("rays_d", rays_d)
    _require_cuda("textures", textures)
    _require_float32("rays_d", rays_d)
    _require_float32("textures", textures)
    _require_same_device("textures", textures, "rays_d", rays_d)
    if rays_d.dim() != 2 or rays_d.shape[1] != 3:
        raise ValueError(f"rays_d must have shape [N, 3]; got {tuple(rays_d.shape)}")
    if (
        textures.dim() != 5
        or textures.shape[0] != 1
        or textures.shape[1] != 6
        or textures.shape[4] != 3
    ):
        raise ValueError(
            f"cubemap textures must have shape [1, 6, H, W, 3]; "
            f"got {tuple(textures.shape)}"
        )
    if textures.shape[2] != textures.shape[3]:
        raise ValueError(
            f"cubemap textures must have H == W; got "
            f"H={textures.shape[2]}, W={textures.shape[3]}"
        )


# ---------------------------------------------------------------------------
# Autograd Functions (CUDA path only)
# ---------------------------------------------------------------------------


class EquirectEnvMapSampleFunction(torch.autograd.Function):
    """Differentiable equirectangular env-map texture sampling (CUDA).

    ``rays_d`` is assumed already unit-normalized. Returns raw pre-activation
    radiance ``[N, 3]``. Backward produces a gradient for ``textures`` only;
    ``rays_d`` receives ``None`` (it is a constant input to the kernel).
    """

    @staticmethod
    def forward(ctx, rays_d: Tensor, textures: Tensor) -> Tensor:
        from . import _backend

        rays_d = rays_d.contiguous()
        textures = textures.contiguous()
        out = _backend._SCENE_CUDA.sample_env_map_equirect_fwd(rays_d, textures)
        ctx.save_for_backward(rays_d, textures)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        from . import _backend

        grad_out = grad_outputs[0].contiguous()
        rays_d, textures = ctx.saved_tensors
        grad_textures = _backend._SCENE_CUDA.sample_env_map_equirect_bwd(
            rays_d, textures, grad_out
        )
        return None, grad_textures


class CubemapEnvMapSampleFunction(torch.autograd.Function):
    """Differentiable cubemap env-map texture sampling (CUDA).

    ``rays_d`` is assumed already unit-normalized. Returns raw pre-activation
    radiance ``[N, 3]``. Backward produces a gradient for ``textures`` only;
    ``rays_d`` receives ``None`` (it is a constant input to the kernel).
    """

    @staticmethod
    def forward(ctx, rays_d: Tensor, textures: Tensor) -> Tensor:
        from . import _backend

        rays_d = rays_d.contiguous()
        textures = textures.contiguous()
        out = _backend._SCENE_CUDA.sample_env_map_cubemap_fwd(rays_d, textures)
        ctx.save_for_backward(rays_d, textures)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        from . import _backend

        grad_out = grad_outputs[0].contiguous()
        rays_d, textures = ctx.saved_tensors
        grad_textures = _backend._SCENE_CUDA.sample_env_map_cubemap_bwd(
            rays_d, textures, grad_out
        )
        return None, grad_textures


# ---------------------------------------------------------------------------
# Public dispatch wrappers (CUDA-only — NO fallback)
# ---------------------------------------------------------------------------


def sample_env_map_equirect(rays_d: Tensor, textures: Tensor) -> Tensor:
    """Sample an equirectangular env map on CUDA → raw pre-activation radiance.

    Validates that both inputs are float32 CUDA tensors on the same device with
    the expected shapes, then dispatches to the fused CUDA kernel. There is NO
    CPU/PyTorch fallback; non-CUDA or non-float32 inputs raise a clear error.

    Args:
        rays_d: ``[N, 3]`` float32 CUDA directions, assumed unit-normalized.
        textures: ``[1, H, W, 3]`` float32 CUDA texture.

    Returns:
        ``[N, 3]`` raw pre-activation radiance.
    """
    _validate_equirect_inputs(rays_d, textures)
    return EquirectEnvMapSampleFunction.apply(rays_d, textures)


def sample_env_map_cubemap(rays_d: Tensor, textures: Tensor) -> Tensor:
    """Sample a cubemap env map on CUDA → raw pre-activation radiance.

    Validates that both inputs are float32 CUDA tensors on the same device with
    the expected shapes (``H == W``), then dispatches to the fused CUDA kernel.
    There is NO CPU/PyTorch fallback; non-CUDA or non-float32 inputs raise a
    clear error.

    Args:
        rays_d: ``[N, 3]`` float32 CUDA directions, assumed unit-normalized.
        textures: ``[1, 6, H, W, 3]`` float32 CUDA texture with ``H == W``.

    Returns:
        ``[N, 3]`` raw pre-activation radiance.
    """
    _validate_cubemap_inputs(rays_d, textures)
    return CubemapEnvMapSampleFunction.apply(rays_d, textures)


__all__ = [
    "EquirectEnvMapSampleFunction",
    "CubemapEnvMapSampleFunction",
    "sample_env_map_equirect",
    "sample_env_map_cubemap",
]
