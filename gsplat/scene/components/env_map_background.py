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

"""Learnable environment-map background (equirectangular or cubemap)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..kernels.env_map_sample_ops import (
    sample_env_map_cubemap,
    sample_env_map_equirect,
)
from .background_scene import BackgroundScene


class EnvMapType(str, Enum):
    """Projection used by an :class:`EnvMapBackground` texture."""

    EQUIRECTANGULAR = "equirectangular"
    CUBEMAP = "cubemap"


@dataclass
class EnvMapBackgroundConfig:
    """Configuration for :class:`EnvMapBackground`.

    Defaults mirror NRE's deployed ``configs/model/background/sky_env_map.yaml``.
    """

    envmap_type: Literal["cubemap", "equirectangular"] = "cubemap"
    width: int = 512
    height: int = 512  # must equal width for cubemap
    saturate_radiance: bool = False  # False = relu (HDR); True = clamp [0, 1]
    min_grad_updates: int = 1000  # gradient tracking starts after this many updates
    # Run the (expensive) inpaint gradient-tracking pass only every Nth tracked
    # ``composite`` call. 1 = every step (original behavior); larger values trade
    # a slightly staler inpaint map for a proportionally cheaper extra backward.
    # Gated on the rank-uniform ``n_grad_updates`` counter so every DDP rank
    # runs the tracking ``all_reduce`` on the same steps (see ``_track_gradients``).
    grad_track_interval: int = 1
    # Consumed only by the external AV trainer (nre!4012), not by this
    # component; serialized here so checkpoints round-trip the trainer's setting.
    should_inpaint: bool = True  # enable unobserved-texel inpainting (trainer-driven)
    inpaint_threshold: float = 5e-2  # texture_grads below this → inpaint mask
    inpaint_kernel_size: int = 10  # dilation kernel size for the inpaint mask


class EnvMapBackground(BackgroundScene):
    """Learnable environment-map background.

    The background is stored as a single :class:`torch.nn.Parameter` texture
    initialized to neutral mid-grey (``0.5``). Two projections are supported:
    equirectangular (``[1, H, W, 3]``) and cubemap (``[1, 6, H, W, 3]``).

    ``EnvMapBackground`` is a plain :class:`Scene` (not an ``nn.Module``), so it
    exposes trainable tensors through :meth:`parameters` and serializes state
    through hand-written :meth:`state_dict` / :meth:`from_state_dict`.

    Args:
        config: Configuration; defaults to :class:`EnvMapBackgroundConfig`.
        id: Non-empty scene id.
        device: Device for the ``textures``/``texture_grads`` tensors; ``None``
            keeps them on CPU. Use :meth:`to` to move an existing instance.
    """

    def __init__(
        self,
        config: EnvMapBackgroundConfig | None = None,
        id: str = "background",
        device: torch.device | str | None = None,
    ) -> None:
        config = config or EnvMapBackgroundConfig()
        super().__init__(id)
        self.envmap_type = EnvMapType(config.envmap_type)
        self.width = int(config.width)
        self.height = int(config.height)
        self.saturate_radiance = bool(config.saturate_radiance)
        self.min_grad_updates = int(config.min_grad_updates)
        self.grad_track_interval = max(1, int(config.grad_track_interval))
        self.should_inpaint = bool(config.should_inpaint)
        self.inpaint_threshold = float(config.inpaint_threshold)
        self.inpaint_kernel_size = int(config.inpaint_kernel_size)
        # Plain int (not a tensor/buffer) — a tensor counter would force a
        # GPU→CPU sync on every gate check.
        self.n_grad_updates = 0

        if self.envmap_type == EnvMapType.CUBEMAP:
            if self.width != self.height:
                raise ValueError(
                    "cubemap env map requires width == height, "
                    f"got width={self.width}, height={self.height}"
                )
            tex_shape = (1, 6, self.height, self.width, 3)
            grads_shape = (6, self.height, self.width)
        else:
            tex_shape = (1, self.height, self.width, 3)
            grads_shape = (self.height, self.width)

        self.textures = nn.Parameter(torch.full(tex_shape, 0.5, device=device))
        self.texture_grads = torch.zeros(grads_shape, device=device)

    def to(self, device: torch.device | str) -> "EnvMapBackground":
        """Move ``textures``/``texture_grads`` to ``device`` (in place).

        Mutates in place and preserves the ``textures`` ``nn.Parameter`` object
        identity (standard ``nn.Module.to()`` semantics), so an optimizer built
        earlier from :meth:`parameters` keeps updating the live texture across
        device moves. ``requires_grad`` is carried over automatically since the
        same Parameter is reused. Any pending ``textures.grad`` is moved too so a
        stale gradient is never stranded on the previous device.
        """
        self.textures.data = self.textures.data.to(device)
        if self.textures.grad is not None:
            self.textures.grad = self.textures.grad.to(device)
        self.texture_grads = self.texture_grads.to(device)
        return self

    # -- Sampling ----------------------------------------------------------

    def sample(self, rays_d: Tensor) -> Tensor:
        """Return background RGB ``[N, 3]`` for ray directions ``[N, 3]``.

        ``rays_d`` need not be unit length; normalization is handled internally
        (the equirectangular polar angle assumes a unit direction).
        """
        return self._activate(self._sample_raw(rays_d))

    def _sample_raw(self, rays_d: Tensor) -> Tensor:
        """Sample the texture without the radiance activation.

        Gradient tracking operates on this pre-activation radiance so that
        bright/HDR texels still receive gradient under ``saturate_radiance``.

        Sampling is CUDA-only: it dispatches to the fused ``gsplat_scene_cuda``
        kernel via :func:`sample_env_map_equirect` / :func:`sample_env_map_cubemap`.
        There is no CPU/PyTorch runtime fallback — non-CUDA float32 inputs raise
        a clear error from the ops layer. The CUDA kernel differentiates w.r.t.
        ``textures`` only; ``rays_d`` is a constant input.
        """
        if self.envmap_type == EnvMapType.EQUIRECTANGULAR:
            # Only the equirectangular polar angle needs a unit direction, so
            # normalize on that path exclusively. Cubemap dominant-axis routing
            # is scale-invariant, so normalization there is wasted work (a full
            # N×3 norm/division/allocation per forward).
            rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return self._sample_equirect(rays_d)
        return self._sample_cubemap(rays_d)

    def _sample_equirect(self, rays_d: Tensor) -> Tensor:
        """Sample the equirectangular texture → raw pre-activation radiance.

        ``rays_d`` is assumed unit-normalized (the polar angle needs a unit
        direction); :meth:`_sample_raw` normalizes before calling. Kept as an
        overridable method (not inlined into :meth:`_sample_raw`) so subclasses
        can swap the per-projection sampler — e.g. NRE's ``SkyEnvMapBackground``
        overrides this to preserve its historical texture layout while inheriting
        the cubemap path below.
        """
        return sample_env_map_equirect(rays_d, self.textures)

    def _sample_cubemap(self, rays_d: Tensor) -> Tensor:
        """Sample the cubemap texture → raw pre-activation radiance.

        Cubemap routing is scale-invariant, so ``rays_d`` need not be unit
        length. Kept as an overridable method for the same reason as
        :meth:`_sample_equirect`; NRE inherits this one directly.
        """
        return sample_env_map_cubemap(rays_d, self.textures)

    def _activate(self, rgb: Tensor) -> Tensor:
        """Radiance activation keeping output non-negative."""
        if self.saturate_radiance:
            return rgb.clamp(0.0, 1.0)
        return F.relu(rgb)

    # -- Compositing with gradient tracking --------------------------------

    def composite(
        self,
        gaussian_rgb: Tensor,  # [N, 3]
        opacity: Tensor,  # [N] or [N, 1]
        rays_d: Tensor,  # [N, 3]
        is_training: bool = False,
    ) -> Tensor:
        """Sample, track per-texel gradients, then blend behind the render.

        Extends :meth:`BackgroundScene.composite` with the gradient-tracking
        pass NRE relies on to identify unobserved texels. ``opacity`` is the
        accumulated Gaussian alpha and may be ``[N]`` or ``[N, 1]``.

        .. warning::
            **DDP precondition.** When ``is_training=True`` the gradient-tracking
            pass runs an ``all_reduce`` gated on the rank-local ``n_grad_updates``
            counter (advanced once per tracked call). Every rank must therefore call
            ``composite(is_training=True)`` the **same number of times per step** —
            uneven counts (e.g. a rank whose microbatch has no sky rays skipping the
            call, or per-rank-variable gradient accumulation) desynchronize the
            collective and **deadlock**. Do not make the number of training
            ``composite`` calls per step depend on rank-local or data-dependent state.
            See :meth:`sync_texture_grad`, which must also be called uniformly.
        """
        raw_rgb = self._sample_raw(rays_d)
        bg_rgb = self._activate(raw_rgb)
        # Track gradients against the pre-activation radiance so bright/HDR
        # texels are not mislabeled unobserved when ``saturate_radiance`` clamps.
        self._track_gradients(raw_rgb, opacity, is_training)
        return self._blend_background(gaussian_rgb, bg_rgb, opacity)

    def _track_gradients(
        self, raw_rgb: Tensor, opacity: Tensor, is_training: bool
    ) -> None:
        """Accumulate a running max of per-texel mean gradient magnitude.

        ``n_grad_updates`` counts tracked training ``composite`` calls (not
        optimizer steps); if ``composite`` runs multiple times per step the
        ``min_grad_updates`` warm-up and ``grad_track_interval`` gate advance
        that many times faster.

        DDP lockstep invariant: the gate below reads only ``n_grad_updates`` /
        ``min_grad_updates`` / ``grad_track_interval``, which advance identically
        on every rank, so all ranks enter the ``all_reduce`` on the same steps.
        Do NOT gate this on rank-local or data-dependent state — a rank that
        skips the collective while others enter it deadlocks (the same hazard
        :meth:`sync_texture_grad` guards against with zeros-participation).
        """
        # ``raw_rgb.requires_grad`` is False under ``torch.no_grad()`` and when
        # ``textures`` is frozen — both cases must skip the autograd.grad call.
        # This is uniform across ranks as long as every rank calls ``composite``
        # in training each step (the lockstep invariant documented above).
        if not (is_training and raw_rgb.requires_grad):
            return
        warmed_up = self.n_grad_updates >= self.min_grad_updates
        due = self.n_grad_updates % self.grad_track_interval == 0
        if warmed_up and due:
            grad = torch.autograd.grad(
                raw_rgb,
                self.textures,
                grad_outputs=(1 - opacity.reshape(-1, 1)).expand_as(raw_rgb),
                retain_graph=True,
            )[
                0
            ].detach()  # [1, (6,) H, W, 3]

            # DDP: autograd.grad only sees this rank's shard, so average across
            # ranks to keep texture_grads consistent (matches NRE).
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                grad = grad / torch.distributed.get_world_size()
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)

            mag = grad.squeeze(0).mean(dim=-1)  # [H, W] or [6, H, W]
            # ``mag`` and ``texture_grads`` share a device by construction
            # (textures/grads are placed together at setup and via ``to``), so
            # the in-place max keeps the tensor identity stable and callers that
            # cached a reference to ``texture_grads`` still observe updates.
            self.texture_grads.copy_(torch.maximum(mag, self.texture_grads))
        self.n_grad_updates += 1

    # -- Inpainting helper -------------------------------------------------

    def inpaint_mask(self) -> Tensor:
        """Boolean mask of unobserved texels, dilated by ``inpaint_kernel_size``.

        Computed unconditionally: this is independent of the ``should_inpaint``
        config flag, which this component stores/serializes for round-trip but
        never acts on (it is trainer-driven, nre!4012). The caller decides
        whether to apply the mask.

        Returns:
            Bool tensor with the same spatial shape as ``texture_grads``
            (``[H, W]`` or ``[6, H, W]``); ``True`` where inpainting is needed.
        """
        mask = (self.texture_grads < self.inpaint_threshold).float()
        # Force an odd kernel (even sizes round up to the next odd) so that
        # padding=k//2 is symmetric and max_pool2d returns the input spatial
        # size exactly (an even kernel would dilate off-center).
        k = int(self.inpaint_kernel_size) | 1
        if k <= 1:
            return mask > 0.5
        H, W = mask.shape[-2], mask.shape[-1]
        reshaped = mask.reshape(-1, 1, H, W)  # [D or 1, 1, H, W]
        dilated = F.max_pool2d(reshaped, kernel_size=k, stride=1, padding=k // 2)
        return dilated.reshape(mask.shape) > 0.5

    # -- Optimizer / persistence ------------------------------------------

    def parameters(self) -> list[nn.Parameter]:
        """Return the trainable tensors (mirrors ``splats.values()``)."""
        return [self.textures]

    def sync_texture_grad(self) -> None:
        """All-reduce (mean) the photometric gradient on ``textures`` across DDP ranks.

        Call after ``loss.backward()`` and before ``optimizer.step()``. Because this
        scene is a plain ``Scene`` (not an ``nn.Module``), DDP does not register a
        reduction hook on ``textures``; the trainer must call this each step under DDP.
        No-op when distributed is unavailable/uninitialized or single-rank. Ranks with
        no background contribution (``textures.grad is None``) contribute zeros so every
        rank participates in the collective. Like :meth:`composite`, this must be called
        uniformly across ranks — exactly once per optimizer step on every rank.
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        grad = self.textures.grad
        if grad is None:
            grad = torch.zeros_like(self.textures)
            self.textures.grad = grad
        grad.div_(world_size)
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)

    def state_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "envmap_type": self.envmap_type.value,  # "cubemap" | "equirectangular"
            "width": self.width,
            "height": self.height,
            "saturate_radiance": self.saturate_radiance,
            "min_grad_updates": self.min_grad_updates,
            "grad_track_interval": self.grad_track_interval,
            "should_inpaint": self.should_inpaint,
            "inpaint_threshold": self.inpaint_threshold,
            "inpaint_kernel_size": self.inpaint_kernel_size,
            "textures": self.textures.detach().clone(),  # [1, (6,) H, W, 3]
            "textures_requires_grad": bool(self.textures.requires_grad),
            "texture_grads": self.texture_grads.detach().clone(),  # [H,W] or [6,H,W]
            "n_grad_updates": self.n_grad_updates,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        device: torch.device | str | None = None,
    ) -> "EnvMapBackground":
        config = EnvMapBackgroundConfig(
            envmap_type=str(state["envmap_type"]),
            width=int(state["width"]),
            height=int(state["height"]),
            saturate_radiance=bool(state["saturate_radiance"]),
            min_grad_updates=int(state.get("min_grad_updates", 1000)),
            grad_track_interval=int(state.get("grad_track_interval", 1)),
            should_inpaint=bool(state.get("should_inpaint", True)),
            inpaint_threshold=float(state.get("inpaint_threshold", 5e-2)),
            inpaint_kernel_size=int(state.get("inpaint_kernel_size", 10)),
        )
        bg = cls(config, id=str(state.get("id", "background")), device=device)
        textures = state["textures"]
        # Guard against a checkpoint whose texture shape disagrees with the
        # scalar config fields (wrong envmap_type / width / height), which would
        # otherwise restore silently and fail later in an opaque place.
        if tuple(textures.shape) != tuple(bg.textures.shape):
            raise ValueError(
                "state['textures'] shape "
                f"{tuple(textures.shape)} does not match the shape implied by "
                f"config (envmap_type={config.envmap_type}, width={config.width}, "
                f"height={config.height}): expected {tuple(bg.textures.shape)}"
            )
        expected_grads = tuple(bg.texture_grads.shape)
        if tuple(state["texture_grads"].shape) != expected_grads:
            raise ValueError(
                "state['texture_grads'] shape "
                f"{tuple(state['texture_grads'].shape)} does not match expected "
                f"{expected_grads}"
            )
        restored_grads = state["texture_grads"].detach().clone()
        restored_textures = textures.detach().clone()
        if device is not None:
            restored_textures = restored_textures.to(device)
            restored_grads = restored_grads.to(device)
        bg.textures = nn.Parameter(
            restored_textures,
            requires_grad=bool(state.get("textures_requires_grad", True)),
        )
        bg.texture_grads = restored_grads
        bg.n_grad_updates = int(state.get("n_grad_updates", 0))
        return bg
