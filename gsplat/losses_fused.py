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

"""Fused CUDA loss modules with pure-PyTorch fallbacks."""

from enum import IntFlag
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import has_losses
from gsplat.losses import (
    background_in_track_loss,
    bilateral_grid_drift_loss,
    create_ssim_window,
    deform_smoothness_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    node_semantic_loss,
    out_of_bound_loss,
    semantic_cross_entropy_masked,
    torch_ssim_loss,
)

try:
    import gsplat.csrc as _C
except ImportError:
    # Keep the pure-PyTorch fallbacks importable from an unbuilt source tree.
    _C = None


def _cuda_losses_available() -> bool:
    """Check if the fused CUDA loss ops are compiled."""
    return _C is not None and hasattr(torch.ops.gsplat, "gaussian_losses_fwd")


def _cuda_camera_losses_available() -> bool:
    """Check if the fused CUDA camera losses op is compiled."""
    return _C is not None and hasattr(torch.ops.gsplat, "camera_losses_fwd")


def _cuda_lidar_losses_available() -> bool:
    """Check if the fused CUDA lidar losses op is compiled."""
    return _C is not None and hasattr(torch.ops.gsplat, "lidar_losses_fwd")


def _cuda_ssim_losses_available() -> bool:
    """Check if the fused CUDA SSIM loss op is compiled."""
    return _C is not None and hasattr(torch.ops.gsplat, "ssim_losses_fwd")


def _require_uniform_cuda(name, int_tensors, float_tensors) -> bool:
    """Return ``True`` iff every input tensor is on CUDA.

    When all inputs are on CUDA, also assert they share one device and that the
    float inputs share one dtype before the native op runs — the kernels'
    ``CHECK_INPUT`` guards contiguity/device per tensor but does not enforce
    cross-tensor agreement, and the launchers infer the device/dtype from only
    one input. Returns ``False`` (→ pure-PyTorch path) when any input is on CPU,
    so mixed CPU/CUDA inputs never reach the native op.
    """
    tensors = (*int_tensors, *float_tensors)
    if not all(t.is_cuda for t in tensors):
        return False
    devices = {str(t.device) for t in tensors}
    if len(devices) != 1:
        raise ValueError(
            f"{name}: all inputs must be on one CUDA device, got {sorted(devices)}"
        )
    dtypes = {t.dtype for t in float_tensors}
    if len(dtypes) != 1:
        raise ValueError(
            f"{name}: all float inputs must share one dtype, "
            f"got {sorted(map(str, dtypes))}"
        )
    return True


def _exact_zero_like_ssim(pred: Tensor) -> Tensor:
    shape = (*pred.shape[:3], 1)
    return pred.new_zeros(shape) + pred[..., :0].sum()


def _normalize_ssim_flags(flags: Tensor, pred: Tensor) -> Tensor:
    if pred.dim() != 4:
        raise ValueError(f"pred must be [B, H, W, C], got {tuple(pred.shape)}")
    expected = pred.shape[:3]
    if flags.shape == expected:
        return flags
    if flags.shape == (*expected, 1):
        return flags.reshape(expected)
    raise ValueError(
        f"flags must be [B, H, W] or [B, H, W, 1], got {tuple(flags.shape)} "
        f"for pred shape {tuple(pred.shape)}"
    )


class FusedGaussianLosses(torch.nn.Module):
    """Fused CUDA implementation of four gaussian regularization losses.

    Computes :func:`~gsplat.losses.gaussian_scale_reg`,
    :func:`~gsplat.losses.gaussian_density_reg`,
    :func:`~gsplat.losses.gaussian_z_scale_reg`, and
    :func:`~gsplat.losses.out_of_bound_loss` in a single fused CUDA kernel.

    Each member owns its row count (heterogeneous regularization domains):
    ``scales [N_scales, 3]``, ``densities [N_densities]``,
    ``z_scales [N_z_scales]``, ``positions``/``cuboid_dims [N_oob, 3]``. Any
    count may be zero (that member simply produces an empty output);
    callers whose members share one gaussian cloud pass equal counts.
    ``visibility`` spans the scale and density members, so it must have
    ``max(N_scales, N_densities)`` elements when given.

    Pre-activation mode (``preactivation=True``) switches the member math to
    log-space inputs:

    - ``scales``/``z_scales`` arrive as log-space pre-activations and the
      ``exp()`` is fused into the kernel. Per-segment weights fold in
      caller-side as ``+log(w)`` on the pre-activations (see
      :func:`~gsplat.losses.fold_log_space_weight`); a non-positive weight
      arrives as a ``-inf`` fill, which ``exp()`` maps to an exact zero for
      both the value and the gradient — weighting after the ``exp()`` would
      instead risk ``0 * inf = NaN`` on pre-activation overflow.

      .. warning::
          The ``+log(w)`` fold is exact only for the **scale** member, whose
          loss is linear in the activated value. It must **not** be
          pre-applied to ``z_scales``: the z member subtracts its threshold
          after the ``exp()``, so a folded weight lands inside the relu —
          ``relu(exp(z + log(w)) - T) == relu(w * exp(z) - T)``, which is not
          ``w * relu(exp(z) - T)`` for any ``w`` outside ``{w <= 0, 1}``. To
          weight the z member per segment, scale its output post-relu (or
          equivalently fold the weight into the threshold too:
          ``relu(w * exp(z) - w * T) == w * relu(exp(z) - T)`` for
          ``w > 0``). See :func:`~gsplat.losses.fold_log_space_weight`.
    - ``densities`` stay post-activation but carry per-segment weights folded
      by plain multiplication, so they are treated as signed magnitudes
      (``|.|`` applied) rather than relying on the ``>= 0`` contract.

    With the default ``preactivation=False`` every member computes the
    post-activation math bit for bit as before.

    Grouped dispatch: when the optional ``deformation`` input
    is given, the same single op call additionally runs the deform-smoothness
    loss (see :func:`~gsplat.losses.deform_smoothness_loss` for the exact
    contract; its CUDA kernels are launched from the gaussian host entry) and
    a fifth output — the scalar deform loss — is appended to the returned
    tuple. Without it, the module behaves exactly as before and returns the
    four-tuple. The deform member is independent of the gaussian members: it
    composes with either mode and is never gated on the gaussian counts.

    Falls back to the pure-PyTorch implementations when CUDA is
    unavailable or the inputs are on CPU.

    All gaussian outputs are **unreduced** (per-element), matching the
    pure-PyTorch contract; the deform loss is a scalar (masked mean), matching
    :func:`~gsplat.losses.deform_smoothness_loss`.

    .. note::
        The CUDA kernel dispatches on ``fp32`` and ``fp64`` only. AMP training
        with ``fp16``/``bf16`` inputs will raise at kernel dispatch; cast to
        ``float`` before calling, or disable AMP for this module. Extending
        to half-precision is a possible future enhancement.

    Args:
        z_scale_threshold: Threshold above which z-scale penalty is applied
            (always in post-activation units, in both modes). Must be
            non-negative: the exact-zero contract for disabled (``w <= 0``
            folded) or empty z-scale members relies on ``relu(0 - T) == 0``.
        preactivation: When ``True``, ``scales``/``z_scales`` are log-space
            pre-activations (``exp()`` fused in-kernel) and ``densities`` are
            signed magnitudes. Defaults to ``False`` (post-activation inputs).
    """

    def __init__(self, z_scale_threshold: float = 0.0, preactivation: bool = False):
        super().__init__()
        # `not (T >= 0)` mirrors the native launcher predicate: it also
        # rejects NaN, which `T < 0` would let through (nan < 0 is False).
        if not z_scale_threshold >= 0:
            raise ValueError(
                "z_scale_threshold must be non-negative (relu(0 - T) must be an "
                f"exact zero for disabled or empty z-scale members), got "
                f"{z_scale_threshold}"
            )
        self.z_scale_threshold = z_scale_threshold
        self.preactivation = preactivation
        self._cuda_available = has_losses()

    def forward(
        self,
        scales: Tensor,  # [N_scales, 3]
        densities: Tensor,  # [N_densities]
        z_scales: Tensor,  # [N_z_scales]
        positions: Tensor,  # [N_oob, 3]
        cuboid_dims: Tensor,  # [N_oob, 3]
        visibility: Optional[Tensor] = None,  # [max(N_scales, N_densities)] float
        deformation: Optional[Tensor] = None,  # [M, D]
        deform_mask: Optional[Tensor] = None,  # broadcastable to [M, D]
    ) -> Tuple[Tensor, ...]:
        """Compute the four gaussian regularization losses (plus, optionally,
        the grouped deform-smoothness loss).

        Returns:
            Tuple of ``(loss_scale, loss_density, loss_z_scale, loss_oob)``,
            each unreduced per-element. When ``deformation`` is given, the
            scalar ``deform_loss`` is appended as a fifth element.
        """
        if deformation is None and deform_mask is not None:
            raise ValueError("FusedGaussianLosses: deform_mask requires deformation.")
        if deformation is not None and deformation.dim() != 2:
            # The CUDA kernels require a 2-D [N, D] deformation while the
            # pure-PyTorch reference is shape-agnostic; validate at the module
            # boundary, before dispatch, so both paths accept exactly the same
            # shapes instead of training on one path and raising on the other.
            raise ValueError(
                "FusedGaussianLosses: deformation must be 2-D [N, D], got "
                f"shape {tuple(deformation.shape)}."
            )
        if visibility is not None:
            # One shared visibility tensor spans the scale and density
            # members; validate at the public boundary so the native and
            # fallback paths reject a wrong length with one contract (the
            # fallback slices per member and would otherwise silently accept
            # an overlong tensor).
            n_visibility = max(scales.shape[0], densities.shape[0])
            if visibility.reshape(-1).shape[0] != n_visibility:
                raise ValueError(
                    "FusedGaussianLosses: visibility must have "
                    f"max(N_scales, N_densities) = {n_visibility} elements, "
                    f"got {visibility.reshape(-1).shape[0]}."
                )
        # Dispatch to CUDA only when every input is on the same GPU — otherwise
        # fall through to pure-PyTorch instead of tripping CHECK_INPUT deep in C++.
        all_cuda = scales.is_cuda and all(
            t.is_cuda for t in (densities, z_scales, positions, cuboid_dims)
        )
        for optional_input in (visibility, deformation, deform_mask):
            if optional_input is not None:
                all_cuda = all_cuda and optional_input.is_cuda
        if self._cuda_available and all_cuda:
            from gsplat.cuda._losses_wrapper import _FusedGaussianLosses

            return _FusedGaussianLosses.apply(
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                self.z_scale_threshold,
                visibility,
                deformation,
                deform_mask,
                self.preactivation,
            )

        # Pure-PyTorch fallback (grouped mode composes the existing torch
        # references, exactly like the ungrouped modules would). Each member
        # owns its row count, so visibility is sliced per member — a no-op
        # for shared-cloud callers, where every count matches.
        if visibility is not None:
            visibility = visibility.reshape(-1)
            vis_scales = visibility[: scales.shape[0]]
            vis_densities = visibility[: densities.shape[0]]
        else:
            vis_scales = vis_densities = None
        if self.preactivation:
            # Log-space members: the explicit exp() mirrors the fused kernel
            # (a -inf pre-activation — a folded non-positive segment weight —
            # becomes an exact zero for value and gradient). Densities carry
            # folded signed weights, so take |.| directly instead of relying
            # on gaussian_density_reg's >= 0 contract.
            if vis_scales is not None:
                # A zero-visibility lane takes the same exact-zero branch as a
                # folded non-positive segment weight: substitute -inf BEFORE
                # the exp() (exp(-inf) == 0 for value and gradient, mirroring
                # the kernel's no-read branch). Multiplying after the exp()
                # would yield 0 * inf = NaN when a pre-activation overflows —
                # and masking after the exp() would still leak the NaN through
                # the masked branch's backward.
                scales = torch.where(
                    (vis_scales == 0).view(-1, 1),
                    torch.full_like(scales, float("-inf")),
                    scales,
                )
            loss_density = densities.abs()
            if vis_densities is not None:
                loss_density = loss_density * vis_densities.view(
                    -1, *[1] * (loss_density.ndim - 1)
                )
            gaussian_losses = (
                gaussian_scale_reg(scales.exp(), visibility=vis_scales),
                loss_density,
                gaussian_z_scale_reg(z_scales.exp(), self.z_scale_threshold),
                out_of_bound_loss(positions, cuboid_dims),
            )
        else:
            gaussian_losses = (
                gaussian_scale_reg(scales, visibility=vis_scales),
                gaussian_density_reg(densities, visibility=vis_densities),
                gaussian_z_scale_reg(z_scales, self.z_scale_threshold),
                out_of_bound_loss(positions, cuboid_dims),
            )
        if deformation is None:
            return gaussian_losses
        return (*gaussian_losses, deform_smoothness_loss(deformation, deform_mask))


# Per-ray flag bitmask values are single-sourced in csrc/LossFlags.h and surfaced
# by the compiled extension via `m.attr` (honoring any -D overrides). They are
# read from the extension rather than duplicated as Python literals — they are
# only meaningful when the loss kernels are built. With no compiled loss kernels
# (`_C` is None, or built without GSPLAT_BUILD_LOSSES) the fused camera/LiDAR
# losses are unavailable, so the flags resolve to None.
if _C is not None and hasattr(_C, "LOSS_FLAG_RGB_LABEL"):

    class LossFlag(IntFlag):
        """Per-ray loss flag bitmask for :class:`FusedCameraLosses` /
        :class:`FusedLidarLosses`.

        Public vocabulary for building the ``flags`` tensor those modules
        consume. Values are sourced from the compiled extension (defined in
        ``csrc/LossFlags.h``) and honor any ``-D`` build overrides, so the
        flags are only available when the loss kernels are built.
        """

        RGB_LABEL = int(_C.LOSS_FLAG_RGB_LABEL)
        SKY_SEMANTIC = int(_C.LOSS_FLAG_SKY_SEMANTIC)
        DROPPED = int(_C.LOSS_FLAG_DROPPED)
        INVALID = int(_C.LOSS_FLAG_INVALID)
        DIFIXED = int(_C.LOSS_FLAG_DIFIXED)
        SYNTHETIC = int(_C.LOSS_FLAG_SYNTHETIC)

    # Plain-int aliases for internal bitwise math on flag tensors.
    _FLAG_RGB_LABEL = int(LossFlag.RGB_LABEL)
    _FLAG_SKY_SEMANTIC = int(LossFlag.SKY_SEMANTIC)
    _FLAG_DROPPED = int(LossFlag.DROPPED)
    _FLAG_INVALID = int(LossFlag.INVALID)
    _FLAG_DIFIXED = int(LossFlag.DIFIXED)
    _FLAG_SYNTHETIC = int(LossFlag.SYNTHETIC)
else:
    LossFlag = None
    _FLAG_RGB_LABEL = _FLAG_SKY_SEMANTIC = _FLAG_DROPPED = None
    _FLAG_INVALID = _FLAG_DIFIXED = _FLAG_SYNTHETIC = None


class FusedSSIMLosses(torch.nn.Module):
    """Fused SSIM loss with invalid-pixel blending.

    Inputs are channel-last image tensors ``[B, H, W, C]`` plus a per-pixel
    int32 flag tensor. Pixels with :class:`LossFlag.INVALID` set are blended out
    before SSIM is computed; valid pixels receive ``(1 - SSIM) * factor`` after
    averaging channels. The output is a per-pixel map ``[B, H, W, 1]``.

    Unlike :class:`FusedCameraLosses` / :class:`FusedLidarLosses`, which give
    ``factor == 0`` and ``factor < 0`` distinct meanings, this module collapses
    any ``factor <= 0`` to an exact (NaN-safe, differentiable) zero map and skips
    both the CUDA and PyTorch SSIM paths.

    Requires the compiled gsplat loss extension for the flag vocabulary. CUDA
    inputs with ``window_size == 11`` use the fused kernel; CPU inputs and other
    window sizes use an equivalent pure-PyTorch path.
    """

    def __init__(
        self,
        window_size: int = 11,
        mask_mode: str = "target",
        constant_mask_value: float = 0.0,
    ):
        super().__init__()
        if _FLAG_INVALID is None:
            raise RuntimeError(
                "FusedSSIMLosses requires the compiled gsplat loss extension "
                "(the per-pixel flag vocabulary is defined by the loss build). "
                "Rebuild gsplat with loss support enabled."
            )
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        mode = mask_mode.lower()
        if mode not in {"target", "gt", "constant"}:
            raise ValueError("mask_mode must be 'target', 'gt', or 'constant'")
        self.window_size = window_size
        self.mask_mode_target = mode != "constant"
        self.constant_mask_value = constant_mask_value
        self._cuda_available = _cuda_ssim_losses_available()

    def forward(
        self,
        flags: Tensor,  # [B, H, W] or [B, H, W, 1]
        pred: Tensor,  # [B, H, W, C]
        target: Tensor,  # [B, H, W, C]
        factor: float = 1.0,
    ) -> Tensor:
        """Returns a per-pixel SSIM loss map ``[B, H, W, 1]``."""
        flags = _normalize_ssim_flags(flags, pred)
        if pred.shape != target.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match pred shape {tuple(pred.shape)}"
            )
        if flags.dtype != torch.int32:
            raise ValueError(f"flags must be int32, got {flags.dtype}")
        if pred.dtype != torch.float32 or target.dtype != torch.float32:
            raise ValueError(
                f"FusedSSIMLosses expects float32 inputs, got {pred.dtype} and {target.dtype}"
            )
        if target.requires_grad:
            raise ValueError(
                "FusedSSIMLosses does not support gradients w.r.t. target; "
                "pass target with requires_grad=False."
            )
        if (
            factor <= 0.0
            or pred.shape[0] == 0
            or pred.shape[1] == 0
            or pred.shape[2] == 0
        ):
            return _exact_zero_like_ssim(pred)

        if (
            self.window_size == 11
            and self._cuda_available
            and _require_uniform_cuda("FusedSSIMLosses", (flags,), (pred, target))
        ):
            from gsplat.cuda._losses_wrapper import _FusedSSIMLosses

            return _FusedSSIMLosses.apply(
                flags,
                pred,
                target,
                factor,
                self.mask_mode_target,
                self.constant_mask_value,
            )

        channel = pred.shape[3]
        pred_bchw = pred.permute(0, 3, 1, 2).contiguous()
        target_bchw = target.permute(0, 3, 1, 2).contiguous()
        valid_bchw = ((flags & _FLAG_INVALID) == 0).unsqueeze(1)
        mask_value = (
            target_bchw
            if self.mask_mode_target
            else torch.full_like(target_bchw, self.constant_mask_value)
        )
        pred_blended = torch.where(valid_bchw, pred_bchw, mask_value)
        target_blended = torch.where(valid_bchw, target_bchw, mask_value)
        window = create_ssim_window(
            self.window_size,
            channel,
            device=pred.device,
        ).type_as(pred)
        ssim_map = torch_ssim_loss(
            pred_blended,
            target_blended,
            window,
            window_size=self.window_size,
            channel=channel,
        )
        loss = (
            (1.0 - ssim_map).mean(dim=1, keepdim=True)
            * valid_bchw.to(pred.dtype)
            * factor
        )
        return loss.permute(0, 2, 3, 1).contiguous()


class FusedCameraLosses(torch.nn.Module):
    """Fused CUDA camera losses: RGB L1 + background MSE with flag masking.

    Each pixel is masked by per-ray flags (int32 bitmask; see :class:`LossFlag`).
    Sub-losses are individually enabled/disabled via their factor:
    ``factor > 0`` = active, ``factor == 0`` = no valid pixels,
    ``factor < 0`` = disabled.

    The masked semantic cross-entropy loss (see
    :func:`~gsplat.losses.semantic_cross_entropy_masked` for the exact
    contract) can be folded into the same dispatch — one host op call
    launching the whole group of kernels — by passing the optional
    ``semantic_logits``/``semantic_targets``/``semantic_valid`` inputs
    together. The forward then
    returns a third output, the scalar CE loss, and the CE gradient flows to
    ``semantic_logits`` only. The semantic row count is independent of the
    camera ray count (the CE carries its own ``valid`` row mask). Omitting the
    semantic inputs keeps the pre-fold two-output behavior bit-for-bit.

    Requires the compiled gsplat loss extension, which defines the per-ray
    flag vocabulary (:class:`LossFlag`); constructing the module raises
    ``RuntimeError`` if the extension is unavailable. With the extension built,
    CUDA inputs use the fused kernel and CPU inputs use an equivalent
    pure-PyTorch path.

    The camera outputs are per-pixel (unreduced); the optional semantic CE
    output is a scalar (masked mean).
    """

    def __init__(self):
        super().__init__()
        if _FLAG_RGB_LABEL is None:
            raise RuntimeError(
                "FusedCameraLosses requires the compiled gsplat loss extension "
                "(the per-ray flag vocabulary is defined by the loss build). "
                "Rebuild gsplat with loss support enabled."
            )
        self._cuda_available = _cuda_camera_losses_available()

    def forward(
        self,
        flags: Tensor,  # [N] int32
        rgb_pred: Tensor,  # [N, 3]
        rgb_gt: Tensor,  # [N, 3]
        bg_pred: Tensor,  # [N]
        rgb_factor: float = 1.0,
        bg_factor: float = 1.0,
        semantic_logits: Optional[Tensor] = None,  # [M, C] fp32/fp64
        semantic_targets: Optional[Tensor] = None,  # [M] uint8 or int64
        semantic_valid: Optional[Tensor] = None,  # [M] bool
        semantic_ignore_index: int = -100,
    ) -> Tuple[Tensor, ...]:
        """Returns ``(rgb_loss [N], bg_loss [N])``, plus a scalar
        ``semantic_loss`` third element when the semantic inputs are given."""
        if rgb_gt.requires_grad:
            raise ValueError(
                "FusedCameraLosses does not support gradients w.r.t. the "
                "target rgb_gt; pass it with requires_grad=False. The fused "
                "CUDA backward returns no target gradient, so the pure-PyTorch "
                "fallback is held to the same contract to keep both paths "
                "consistent."
            )
        semantic_inputs = (semantic_logits, semantic_targets, semantic_valid)
        has_semantic = any(t is not None for t in semantic_inputs)
        if has_semantic and any(t is None for t in semantic_inputs):
            raise ValueError(
                "FusedCameraLosses: semantic_logits, semantic_targets, and "
                "semantic_valid enable the grouped semantic CE loss together; "
                "pass all three or none."
            )
        use_cuda = self._cuda_available and _require_uniform_cuda(
            "FusedCameraLosses", (flags,), (rgb_pred, rgb_gt, bg_pred)
        )
        if has_semantic and use_cuda:
            # The CE kernels dispatch on the logits dtype independently of the
            # camera trio, so fp64 CE may ride beside fp32 camera tensors:
            # check the semantic group's uniformity separately (flags in both
            # groups ties them to one device).
            use_cuda = _require_uniform_cuda(
                "FusedCameraLosses(semantic)",
                (flags, semantic_targets, semantic_valid),
                (semantic_logits,),
            )
        if use_cuda:
            from gsplat.cuda._losses_wrapper import _FusedCameraLosses

            return _FusedCameraLosses.apply(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                rgb_factor,
                bg_factor,
                semantic_logits,
                semantic_targets,
                semantic_valid,
                semantic_ignore_index,
            )

        # Pure-PyTorch fallback
        N = flags.size(0)
        f = flags
        rgb_loss = torch.zeros(N, device=rgb_pred.device, dtype=rgb_pred.dtype)
        bg_loss = torch.zeros(N, device=bg_pred.device, dtype=bg_pred.dtype)

        if rgb_factor > 0:
            mask = ((f & _FLAG_RGB_LABEL) != 0) & ((f & _FLAG_INVALID) == 0)
            diff = (rgb_pred - rgb_gt).abs().sum(dim=-1)
            rgb_loss = torch.where(mask, diff * rgb_factor, rgb_loss)

        if bg_factor > 0:
            mask = (
                ((f & _FLAG_INVALID) == 0)
                & ((f & _FLAG_DIFIXED) == 0)
                & ((f & _FLAG_SYNTHETIC) == 0)
            )
            # Non-sky rays target opacity 1.0, sky rays target 0.0. A scalar
            # cast avoids allocating two full-size zeros/ones tensors.
            target = ((f & _FLAG_SKY_SEMANTIC) == 0).to(bg_pred.dtype)
            pred = bg_pred.clamp(0, 1)
            bg_loss = torch.where(mask, (pred - target).pow(2) * bg_factor, bg_loss)

        if has_semantic:
            semantic_loss = semantic_cross_entropy_masked(
                semantic_logits,
                semantic_targets,
                semantic_valid,
                ignore_index=semantic_ignore_index,
            )
            return rgb_loss, bg_loss, semantic_loss
        return rgb_loss, bg_loss


class FusedLidarLosses(torch.nn.Module):
    """Fused CUDA LiDAR losses: distance L1 + intensity/raydrop/bg MSE.

    Each pixel is masked by per-ray flags (see :class:`LossFlag`).
    Sub-losses are individually enabled/disabled via their factor:
    ``factor > 0`` = active, ``factor == 0`` = no valid data,
    ``factor < 0`` = disabled.

    Requires the compiled gsplat loss extension, which defines the per-ray
    flag vocabulary (:class:`LossFlag`); constructing the module raises
    ``RuntimeError`` if the extension is unavailable. With the extension built,
    CUDA inputs use the fused kernel and CPU inputs use an equivalent
    pure-PyTorch path.

    All outputs are per-pixel (unreduced).
    """

    def __init__(self):
        super().__init__()
        if _FLAG_RGB_LABEL is None:
            raise RuntimeError(
                "FusedLidarLosses requires the compiled gsplat loss extension "
                "(the per-ray flag vocabulary is defined by the loss build). "
                "Rebuild gsplat with loss support enabled."
            )
        self._cuda_available = _cuda_lidar_losses_available()

    def forward(
        self,
        flags: Tensor,  # [N] int32
        distance_pred: Tensor,  # [N]
        distance_gt: Tensor,  # [N]
        intensity_pred: Tensor,  # [N]
        intensity_gt: Tensor,  # [N]
        raydrop_pred: Tensor,  # [N]
        raydrop_gt: Tensor,  # [N]
        bg_pred: Tensor,  # [N]
        distance_factor: float = 1.0,
        intensity_factor: float = 1.0,
        raydrop_factor: float = 1.0,
        bg_factor: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns ``(distance_loss, intensity_loss, raydrop_loss, bg_loss)``."""
        if (
            distance_gt.requires_grad
            or intensity_gt.requires_grad
            or raydrop_gt.requires_grad
        ):
            raise ValueError(
                "FusedLidarLosses does not support gradients w.r.t. the "
                "targets distance_gt/intensity_gt/raydrop_gt; pass them with "
                "requires_grad=False. The fused CUDA backward returns no "
                "target gradients, so the pure-PyTorch fallback is held to the "
                "same contract to keep both paths consistent."
            )
        if self._cuda_available and _require_uniform_cuda(
            "FusedLidarLosses",
            (flags,),
            (
                distance_pred,
                distance_gt,
                intensity_pred,
                intensity_gt,
                raydrop_pred,
                raydrop_gt,
                bg_pred,
            ),
        ):
            from gsplat.cuda._losses_wrapper import _FusedLidarLosses

            return _FusedLidarLosses.apply(
                flags,
                distance_pred,
                distance_gt,
                intensity_pred,
                intensity_gt,
                raydrop_pred,
                raydrop_gt,
                bg_pred,
                distance_factor,
                intensity_factor,
                raydrop_factor,
                bg_factor,
            )

        # Pure-PyTorch fallback
        f = flags
        valid = (f & _FLAG_INVALID) == 0
        valid_nd = valid & ((f & _FLAG_DROPPED) == 0)
        zeros = torch.zeros_like(distance_pred)

        distance_loss = (
            torch.where(
                valid_nd, (distance_pred - distance_gt).abs() * distance_factor, zeros
            )
            if distance_factor > 0
            else zeros
        )
        intensity_loss = (
            torch.where(
                valid_nd,
                (intensity_pred - intensity_gt).pow(2) * intensity_factor,
                zeros,
            )
            if intensity_factor > 0
            else zeros
        )
        raydrop_loss = (
            torch.where(
                valid, (raydrop_pred - raydrop_gt).pow(2) * raydrop_factor, zeros
            )
            if raydrop_factor > 0
            else zeros
        )
        bg_loss = zeros
        if bg_factor > 0:
            pred = bg_pred.clamp(0, 1)
            target = torch.where(
                (f & _FLAG_SKY_SEMANTIC) != 0,
                torch.zeros_like(pred),
                torch.ones_like(pred),
            )
            bg_loss = torch.where(valid_nd, (pred - target).pow(2) * bg_factor, zeros)

        return distance_loss, intensity_loss, raydrop_loss, bg_loss


# ---------------------------------------------------------------------------
# Ground-gaussian distortion loss
# ---------------------------------------------------------------------------


def _quat_to_so3_matrix(quat: Tensor) -> Tensor:
    """Normalized quaternion ``(..., 4)`` in ``(x, y, z, w)`` order to a
    rotation matrix ``(..., 3, 3)``."""
    q = quat / quat.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def _so3_matrix_to_quat(R: Tensor) -> Tensor:
    """Rotation matrix ``(3, 3)`` to a normalized quaternion ``(4,)`` in
    ``(x, y, z, w)`` order, using the largest-element (decision-matrix) form."""
    t0, t1, t2 = R[0, 0], R[1, 1], R[2, 2]
    trace = t0 + t1 + t2
    decision = torch.stack([t0, t1, t2, trace])
    choice = int(torch.argmax(decision).item())

    q = R.new_zeros(4)
    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3
        q[i] = 1 + R[i, i] - R[j, j] - R[k, k]
        q[j] = R[j, i] + R[i, j]
        q[k] = R[k, i] + R[i, k]
        q[3] = R[k, j] - R[j, k]
    else:
        q[0] = R[2, 1] - R[1, 2]
        q[1] = R[0, 2] - R[2, 0]
        q[2] = R[1, 0] - R[0, 1]
        q[3] = 1 + trace
    return q / q.norm()


def _tquat_to_se3_matrix(tquat: Tensor) -> Tensor:
    """``[tx, ty, tz, qx, qy, qz, qw]`` to a 4x4 SE3 matrix."""
    T = torch.eye(4, dtype=tquat.dtype, device=tquat.device)
    T[:3, :3] = _quat_to_so3_matrix(tquat[3:])
    T[:3, 3] = tquat[:3]
    return T


def _se3_matrix_inverse(T: Tensor) -> Tensor:
    """Inverse of an SE3 matrix ``[R t; 0 1]`` as ``[R^T  -R^T t; 0 1]``."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.t()
    T_inv[:3, 3] = -(R.t() @ t)
    return T_inv


def _quat_to_roll_pitch(quat: Tensor) -> Tuple[Tensor, Tensor]:
    """Roll (about z) and pitch (about x) Euler angles from a batch of
    quaternions ``(..., 4)`` in ``(x, y, z, w)`` order."""
    x, y, z, w = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return roll, pitch


def ground_gaussians_loss(
    positions: Tensor,  # [N, 3]
    rotations: Tensor,  # [N, 4] quaternion (x, y, z, w)
    cam_tquat: Tensor,  # [7] camera-from-world (tx,ty,tz,qx,qy,qz,qw)
    random_values: Tensor,  # [B] bin offsets in [0, 1)
    min_bias: float,
    range_bias: float,
    grid_len: float,
    rotation_lambda: float,
) -> Tensor:
    """Pure-PyTorch ground-gaussian distortion loss (the fallback reference).

    Transforms each gaussian into the camera frame, then for each of ``B``
    randomly placed depth bins along the camera z-axis accumulates the
    standard deviation of the camera-space height (y), roll, and pitch of the
    gaussians inside the bin. The per-bin contribution is
    ``std(y) + rotation_lambda * (std(roll) + std(pitch))`` and the scalar loss
    is the mean over the ``B`` bins.

    Bins with one or fewer members are skipped (std is undefined). Standard
    deviation uses the unbiased (``N - 1``) estimator.
    """
    T_cam_world = _tquat_to_se3_matrix(cam_tquat)
    T_world_cam = _se3_matrix_inverse(T_cam_world)

    positions_cam = positions @ T_world_cam[:3, :3].t() + T_world_cam[:3, 3]

    R_world = _quat_to_so3_matrix(rotations)  # [N, 3, 3]
    R_cam = T_cam_world[:3, :3].t() @ R_world @ T_cam_world[:3, :3]
    rotations_cam = torch.stack(
        [_so3_matrix_to_quat(R_cam[i]) for i in range(R_cam.shape[0])]
    )

    roll, pitch = _quat_to_roll_pitch(rotations_cam)
    z_cam = positions_cam[:, 2]
    y_cam = positions_cam[:, 1]

    biases = min_bias + range_bias * random_values

    loss = positions.new_zeros(())
    if random_values.shape[0] == 0:
        # No bins: return a differentiable zero connected to the inputs, matching
        # the CUDA path (which returns a zero scalar with zero gradients for
        # n_bins == 0) instead of dividing by zero. The empty-view sums are
        # exactly 0 and keep the autograd edge without reading any values.
        return loss + positions[:0].sum() + rotations[:0].sum()
    for bias in biases:
        mask = (bias <= z_cam) & (z_cam < (bias + grid_len))
        if int(torch.sum(mask)) <= 1:  # std on <=1 element is undefined
            continue
        loss = loss + torch.std(y_cam[mask])
        loss = loss + rotation_lambda * (torch.std(pitch[mask]) + torch.std(roll[mask]))

    return loss / random_values.shape[0]


class FusedGroundGaussiansLosses(torch.nn.Module):
    """Fused CUDA ground-gaussian distortion loss with a pure-PyTorch fallback.

    Constrains the camera-space height (y) and roll/pitch rotation variance of
    gaussians that fall inside randomly placed depth bins along the camera
    z-axis (HUGSIM-style ground regularization). Returns a single scalar loss;
    gradients flow to ``positions`` and ``rotations``.

    A typical use is regularizing a near-planar surface layer of a scene toward
    local flatness -- for example, the road surface in an autonomous-driving
    reconstruction.

    Falls back to :func:`ground_gaussians_loss` when CUDA is unavailable or the
    inputs are on CPU.

    .. note::
        The CUDA kernel dispatches on ``fp32`` and ``fp64`` only. AMP training
        with ``fp16``/``bf16`` inputs will raise at kernel dispatch; cast to
        ``float`` before calling, or disable AMP for this module.

    Args:
        min_bias: Lower edge of the bin-sampling range along camera z.
        range_bias: Span of the bin-sampling range (``bin_min = min_bias +
            range_bias * random_value``).
        grid_len: Bin width along camera z (``bin_max = bin_min + grid_len``).
        rotation_lambda: Weight on the roll + pitch standard-deviation terms
            relative to the height term.
    """

    def __init__(
        self,
        min_bias: float,
        range_bias: float,
        grid_len: float,
        rotation_lambda: float,
    ):
        super().__init__()
        self.min_bias = min_bias
        self.range_bias = range_bias
        self.grid_len = grid_len
        self.rotation_lambda = rotation_lambda
        self._cuda_available = has_losses()

    def forward(
        self,
        positions: Tensor,  # [N, 3]
        rotations: Tensor,  # [N, 4] quaternion (x, y, z, w)
        cam_tquat: Tensor,  # [7]
        random_values: Tensor,  # [B]
    ) -> Tensor:
        """Compute the scalar ground-gaussian distortion loss."""
        all_cuda = positions.is_cuda and all(
            t.is_cuda for t in (rotations, cam_tquat, random_values)
        )
        if self._cuda_available and all_cuda:
            from gsplat.cuda._losses_wrapper import _FusedGroundGaussiansLosses

            return _FusedGroundGaussiansLosses.apply(
                positions,
                rotations,
                cam_tquat,
                random_values,
                self.min_bias,
                self.range_bias,
                self.grid_len,
                self.rotation_lambda,
            )

        return ground_gaussians_loss(
            positions,
            rotations,
            cam_tquat,
            random_values,
            self.min_bias,
            self.range_bias,
            self.grid_len,
            self.rotation_lambda,
        )


class FusedBgGridLosses(torch.nn.Module):
    """Fused CUDA implementation of sky-envmap TV + bilateral-grid drift +
    bilateral-grid spatial TV.

    Computes up to five per-element/per-cell losses in a single kernel pair:

    - Sky env-map TV on ``bg_tex`` (planar or cubemap)
    - Grid drift (Frobenius from identity) on ``grids_camera``
    - Grid spatial TV on ``grids_camera``
    - Grid drift on ``grids_frame``
    - Grid spatial TV on ``grids_frame``

    Any of ``bg_tex`` / ``grids_camera`` / ``grids_frame`` can be ``None``
    to skip its sub-losses entirely. Individual sub-losses can also be
    disabled by passing a negative ``*_factor`` at forward time.

    Falls back to an equivalent pure-PyTorch reference when CUDA isn't
    available or inputs are on CPU.

    .. note::
        **float32 only.** The fused bg-grid CUDA op is single-precision (the
        kernels read/write ``float``); passing ``fp64``/``fp16`` CUDA tensors
        raises at the native op. Cast to ``float`` before calling.

    Tensor shapes:

    - ``bg_tex``: ``[B * bg_tex_depth, H, W, C]`` (flat). Set
      ``bg_tex_depth = 1`` for a planar envmap or ``6`` for a cubemap; for a
      cubemap the caller is expected to pre-flatten
      ``[B, 6, H, W, C] → [B*6, H, W, C]`` before calling. The sky-envmap TV
      uses **ordered face-neighbour** forward differences along the flattened
      D axis (no wraparound): each face ``d`` is paired with face ``d + 1`` for
      ``d < bg_tex_depth - 1``, in addition to the within-face H/W differences.
    - ``grids_camera`` / ``grids_frame``: ``[B * 12, D, H, W]`` with the 12
      channels representing the flattened ``3 × 4`` affine matrix. The per-
      cell drift output has length ``B * D * H * W`` (NOT ``B * 12 * D * H * W``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._cuda_available = has_losses()

    def forward(
        self,
        bg_tex: Optional[Tensor] = None,
        grids_camera: Optional[Tensor] = None,
        grids_frame: Optional[Tensor] = None,
        bg_tex_factor: float = 1.0,
        grid_drift_camera_factor: float = 1.0,
        grid_drift_frame_factor: float = 1.0,
        grid_camera_tv_factor: float = 1.0,
        grid_frame_tv_factor: float = 1.0,
        bg_tex_depth: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the five fused losses.

        Returns:
            Tuple of ``(bg_tex_loss, grids_drift_loss, grid_camera_tv_loss,
            grid_frame_tv_loss)``:

            - ``bg_tex_loss``: ``[B * D * H * W * C]`` flat, per-element
              squared-difference contributions (each element contributes its
              own forward-difference cost to the sum; take ``.sum()`` to
              reduce).
            - ``grids_drift_loss``: ``[numel_gc_cells + numel_gf_cells]``
              (camera cells first, frame cells second), one Frobenius-distance
              scalar per ``(b, d, h, w)`` cell.
            - ``grid_camera_tv_loss``: ``[numel_gc_cells]`` per-cell TV.
            - ``grid_frame_tv_loss``: ``[numel_gf_cells]`` per-cell TV.

            When a sub-loss is disabled (via ``*_factor < 0`` or missing
            input tensor), the corresponding output is returned as a zero
            tensor of the expected shape.
        """
        # Validate bg_tex_depth at the public boundary so the native and
        # fallback paths reject invalid depths with one contract (the fallback
        # otherwise reaches a reshape / divide on bad input).
        if bg_tex is not None:
            if bg_tex_depth < 1:
                raise ValueError(
                    "bg_tex_depth must be >= 1 (1 for planar, 6 for cubemap)."
                )
            if bg_tex.shape[0] % bg_tex_depth != 0:
                raise ValueError(
                    "bg_tex.shape[0] must be divisible by bg_tex_depth "
                    f"(got {bg_tex.shape[0]} and {bg_tex_depth})."
                )
        # Device-uniformity check: if any provided input is on CPU, fall
        # through to the pure-PyTorch path (same policy as FusedGaussianLosses).
        provided = [t for t in (bg_tex, grids_camera, grids_frame) if t is not None]
        all_cuda = bool(provided) and all(t.is_cuda for t in provided)

        if self._cuda_available and all_cuda:
            # All provided inputs must share one CUDA device before dispatch —
            # the native op picks a single reference device and the kernel
            # launches on the current stream, so mixed devices would corrupt.
            devices = {t.device for t in provided}
            if len(devices) != 1:
                raise ValueError(
                    "FusedBgGridLosses: all provided inputs must be on the same "
                    f"CUDA device, got {sorted(map(str, devices))}"
                )
            from gsplat.cuda._losses_wrapper import _FusedBgGridLosses

            return _FusedBgGridLosses.apply(
                bg_tex,
                bg_tex_depth,
                grids_camera,
                grids_frame,
                bg_tex_factor,
                grid_drift_camera_factor,
                grid_drift_frame_factor,
                grid_camera_tv_factor,
                grid_frame_tv_factor,
            )

        # Pure-PyTorch fallback. Matches the kernel output shapes so
        # callers can use either path interchangeably.
        return _bg_grid_losses_pytorch(
            bg_tex,
            bg_tex_depth,
            grids_camera,
            grids_frame,
            bg_tex_factor,
            grid_drift_camera_factor,
            grid_drift_frame_factor,
            grid_camera_tv_factor,
            grid_frame_tv_factor,
        )


def _bg_grid_losses_pytorch(
    bg_tex: Optional[Tensor],
    bg_tex_depth: int,
    grids_camera: Optional[Tensor],
    grids_frame: Optional[Tensor],
    bg_tex_factor: float,
    grid_drift_camera_factor: float,
    grid_drift_frame_factor: float,
    grid_camera_tv_factor: float,
    grid_frame_tv_factor: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pure-PyTorch reference for :class:`FusedBgGridLosses`.

    Returns tensors in the same flat per-element / per-cell layout that the
    CUDA path produces so tests can compare shape-for-shape.
    """
    # Reference device/dtype from the first provided tensor.
    ref = (
        bg_tex
        if bg_tex is not None
        else (grids_camera if grids_camera is not None else grids_frame)
    )
    if ref is None:
        raise ValueError(
            "At least one of bg_tex / grids_camera / grids_frame must be provided."
        )
    device, dtype = ref.device, ref.dtype

    def _disabled_zero(numel: int, ref_in: Optional[Tensor]) -> Tensor:
        # Zero output for a disabled (negative factor) or absent sub-loss. When
        # the input tensor is present, keep the zero connected to its autograd
        # graph so the fallback backward yields an exact-zero gradient to the
        # input, matching the CUDA custom Function (which writes zeros without
        # reading the disabled input). ``ref_in[:0].sum()`` is an empty-view sum:
        # exactly 0 and differentiable, but it never reads the input's values, so
        # a present-but-disabled input containing NaN or +/-Inf still yields a
        # zero output (``0.0 * ref_in.sum()`` would be NaN); it also avoids the
        # O(N) reduction. An absent input needs no such connection.
        z = torch.zeros(numel, device=device, dtype=dtype)
        return z if ref_in is None else z + ref_in[:0].sum()

    def _bg_tex_loss() -> Tensor:
        if bg_tex is None or bg_tex_factor < 0:
            numel = bg_tex.numel() if bg_tex is not None else 0
            return _disabled_zero(numel, bg_tex)
        # bg_tex is [B*D, H, W, C]. Per-element squared forward differences
        # along D (one-sided across faces), H, W. Each (ti) contributes the
        # sum over its forward-neighbour diffs.
        tex = bg_tex
        BD, H, W, C = tex.shape
        D = bg_tex_depth
        B = BD // D
        tex5d = tex.reshape(B, D, H, W, C)  # [B, D, H, W, C]
        parts = torch.zeros_like(tex)
        # Use the per-element layout: we fold contributions into a
        # [BD, H, W, C] tensor where each element holds the sum of squared
        # forward-diffs originating from itself.
        parts5d = parts.view(B, D, H, W, C)
        if D > 1:
            diff = tex5d[:, :-1] - tex5d[:, 1:]  # [B, D-1, H, W, C]
            parts5d[:, :-1] += diff * diff
        if H > 1:
            diff = tex5d[:, :, :-1] - tex5d[:, :, 1:]
            parts5d[:, :, :-1] += diff * diff
        if W > 1:
            diff = tex5d[:, :, :, :-1] - tex5d[:, :, :, 1:]
            parts5d[:, :, :, :-1] += diff * diff
        return (parts5d.reshape(-1)) * bg_tex_factor

    def _grid_drift_loss(grid: Optional[Tensor], factor: float) -> Tensor:
        if grid is None or factor < 0:
            numel = (
                (grid.shape[0] // 12) * grid.shape[1] * grid.shape[2] * grid.shape[3]
                if grid is not None
                else 0
            )
            return _disabled_zero(numel, grid)
        B12, D, H, W = grid.shape
        B = B12 // 12
        # Per-cell Frobenius distance from the 3x4 affine identity. Inlined
        # (rather than calling identity_distance) to reproduce the CUDA kernel's
        # backward dead-zone: the kernel guards ``sum_sq > 1e-12`` before the
        # 1/sqrt(sum_sq) drift gradient, so a near-identity cell gets zero drift
        # gradient. The forward value stays the exact ``sqrt(sum_sq)`` (the
        # kernel does an unconditional sqrtf), so only the gradient is masked —
        # via a straight-through that detaches the dead-zone branch
        # (a plain ``torch.where(sqrt(...))`` would propagate a 0/0 NaN at
        # exactly identity).
        grid_mat = grid.reshape(B, 3, 4, D, H, W)
        identity = torch.eye(3, 4, device=device, dtype=dtype).view(1, 3, 4, 1, 1, 1)
        sum_sq = (grid_mat - identity).pow(2).sum(dim=(1, 2))  # [B,D,H,W] = ||·||_F^2
        active = sum_sq > 1e-12
        norm = torch.sqrt(sum_sq.clamp(min=1e-12))  # exact for active cells
        per_cell = torch.where(active, norm, torch.sqrt(sum_sq).detach())
        return per_cell.reshape(-1) * factor

    def _grid_tv_loss(grid: Optional[Tensor], factor: float) -> Tensor:
        if grid is None or factor < 0:
            numel = (
                (grid.shape[0] // 12) * grid.shape[1] * grid.shape[2] * grid.shape[3]
                if grid is not None
                else 0
            )
            return _disabled_zero(numel, grid)
        B12, D, H, W = grid.shape
        B = B12 // 12
        # Per-cell TV: sum of forward-diff-squared across D, H, W over all 12
        # channels, divided by 12 (matches the CUDA kernel).
        grid5d = grid.reshape(B, 12, D, H, W)
        acc = torch.zeros(B, D, H, W, device=device, dtype=dtype)
        if D > 1:
            sq_diff = (
                (grid5d[:, :, :-1] - grid5d[:, :, 1:]).pow(2).sum(dim=1)
            )  # [B, D-1, H, W]
            acc[:, :-1] += sq_diff
        if H > 1:
            sq_diff = (grid5d[:, :, :, :-1] - grid5d[:, :, :, 1:]).pow(2).sum(dim=1)
            acc[:, :, :-1] += sq_diff
        if W > 1:
            sq_diff = (
                (grid5d[:, :, :, :, :-1] - grid5d[:, :, :, :, 1:]).pow(2).sum(dim=1)
            )
            acc[:, :, :, :-1] += sq_diff
        return (acc.reshape(-1) / 12.0) * factor

    bg_loss = _bg_tex_loss()
    camera_drift = _grid_drift_loss(grids_camera, grid_drift_camera_factor)
    frame_drift = _grid_drift_loss(grids_frame, grid_drift_frame_factor)
    grids_drift_loss = torch.cat([camera_drift, frame_drift])
    grid_camera_tv_loss = _grid_tv_loss(grids_camera, grid_camera_tv_factor)
    grid_frame_tv_loss = _grid_tv_loss(grids_frame, grid_frame_tv_factor)
    return bg_loss, grids_drift_loss, grid_camera_tv_loss, grid_frame_tv_loss


# ---------------------------------------------------------------------------
# Background-in-track + node-semantic Gaussian losses
# ---------------------------------------------------------------------------


def _cuda_bg_track_node_semantic_available() -> bool:
    """Check if the fused CUDA bg-track + node-semantic op is compiled."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401

        return hasattr(torch.ops.gsplat, "bg_track_node_semantic_losses_fwd")
    except Exception:
        return False


class FusedBgTrackNodeSemanticLosses(torch.nn.Module):
    """Fused background-in-track + node-semantic grouped BCE losses.

    One forward computes up to two scalar loss pairs that share a single
    selection pass:

    - **Background-in-track** (enabled iff ``background_lambda >= 0``):
      penalizes ``BCEWithLogits(density_logit, 0)`` for background Gaussians
      whose density logit exceeds *density_logits_min*, whose semantic argmax
      is in the allowed class set, and whose position lies inside any dynamic
      cuboid track at the mean camera timestamp (SE(3)-interpolated, inclusive
      AABB test, time-window gated).
    - **Node-semantic** (enabled iff ``node_lambda >= 0``): penalizes the
      density of Gaussians whose semantic argmax fails a per-node
      ``(class_ids, select_matches)`` predicate — ``select_matches=True``
      penalizes argmax **in** the id set (an exclusion list),
      ``select_matches=False`` penalizes argmax **not in** the id set (an
      allow list). Empty ids are legal on either polarity (penalize
      nothing / penalize everything). Nodes without semantic logits are the
      caller's responsibility to omit.

    Each raw loss is a selected-count mean (exactly 0 when nothing is
    selected) and each weighted loss is ``lambda * raw``. Only the two density
    tensors receive gradients.

    Falls back to the pure-PyTorch implementations
    (:func:`~gsplat.losses.background_in_track_loss`,
    :func:`~gsplat.losses.node_semantic_loss`) when CUDA or the compiled op
    is unavailable or any input is on CPU.

    Domain modes, mirroring the CUDA host entry:

    - **joint** (``n_semantic_points=None``, node enabled): the primary node
      shares the background point domain. Pass the shared ``[Nbg, C]``
      ``semantic_logits``, the background filter as
      ``background_allowed_class_ids``, and an explicit
      ``node_primary_predicate`` (required — an empty predicate is legal,
      absence is not). Remaining nodes are packed into
      ``other_density_logits`` / ``other_segments``.
    - **packed** (``n_semantic_points`` given, node enabled): the background
      member runs its own segmented semantic filter
      (``background_semantic_logits`` / ``background_segments`` cover the
      first ``n_semantic_points`` points; the rest are unfiltered, covering
      layers that carry no semantic logits) and all node-semantic points are
      packed separately. Requires the background member enabled.
    - **background-only** (``node_lambda < 0``): requires
      ``n_semantic_points`` (0 with no segments = fully unfiltered).

    .. note::
        The CUDA kernels dispatch on ``fp32``/``fp64`` only; cast half
        precision inputs before calling.

    Args:
        density_logits_min: Background selection threshold on the raw density
            logit (strictly-greater comparison). Default: ``-20.0``.
    """

    def __init__(self, density_logits_min: float = -20.0):
        super().__init__()
        self.density_logits_min = density_logits_min
        self._cuda_available = _cuda_bg_track_node_semantic_available()

    def forward(
        self,
        positions: Tensor,  # [Nbg, 3]
        density_logits: Tensor,  # [Nbg] (differentiable)
        camera_timestamps_startend_us: Tensor,  # [B, 2] int64
        tracks_packinfo: Tensor,  # [T, 2] int32
        tracks_poses: Tensor,  # [P, 7] (tx, ty, tz, qx, qy, qz, qw)
        tracks_timestamps_us: Tensor,  # [P] int64
        cuboids_dims: Tensor,  # [T, 3]
        semantic_logits: Optional[Tensor] = None,  # [Nbg, C] joint mode
        background_allowed_class_ids: Sequence[int] = (),  # joint mode
        node_primary_predicate: Optional[
            Tuple[Sequence[int], bool]
        ] = None,  # joint mode
        n_semantic_points: Optional[int] = None,  # generic/packed modes
        background_semantic_logits: Sequence[Tensor] = (),  # generic/packed
        background_segments: Sequence[Tuple[int, Sequence[int]]] = (),  # generic/packed
        other_density_logits: Optional[Tensor] = None,  # [Nother]
        other_semantic_logits: Sequence[Tensor] = (),
        other_segments: Sequence[Tuple[int, Sequence[int], bool]] = (),
        background_lambda: float = 1.0,
        node_lambda: float = -1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns ``(background_raw, background_weighted, node_raw,
        node_weighted)`` scalars; disabled members return zeros."""
        background_enabled = background_lambda >= 0.0
        node_enabled = node_lambda >= 0.0
        if not background_enabled and not node_enabled:
            raise ValueError(
                "FusedBgTrackNodeSemanticLosses: at least one of "
                "background_lambda and node_lambda must be >= 0."
            )
        if other_density_logits is None:
            other_density_logits = density_logits.new_zeros(0)
        background_semantic_logits = list(background_semantic_logits)
        background_segments = [
            (int(end), list(ids)) for end, ids in background_segments
        ]
        other_semantic_logits = list(other_semantic_logits)
        other_segments = [
            (int(end), list(ids), bool(sel)) for end, ids, sel in other_segments
        ]

        if node_enabled and n_semantic_points is None:
            # Joint mode: shared primary domain.
            if semantic_logits is None:
                raise ValueError(
                    "FusedBgTrackNodeSemanticLosses: the joint mode "
                    "(n_semantic_points=None) requires the shared "
                    "semantic_logits tensor for the primary domain."
                )
            if node_primary_predicate is None:
                raise ValueError(
                    "FusedBgTrackNodeSemanticLosses: the joint node-semantic "
                    "path requires an explicit node_primary_predicate; pass "
                    "(class_ids, select_matches). Empty ids with "
                    "select_matches=True penalize no primary point, with "
                    "select_matches=False they penalize all of them."
                )
        elif node_enabled:
            # Packed mode mirrors the kernel's requirement of independent
            # background metadata.
            if not background_enabled:
                raise ValueError(
                    "FusedBgTrackNodeSemanticLosses: the packed node mode "
                    "(n_semantic_points given) requires the background member "
                    "enabled (background_lambda >= 0)."
                )
        else:
            if n_semantic_points is None:
                raise ValueError(
                    "FusedBgTrackNodeSemanticLosses: the background-only mode "
                    "requires n_semantic_points (pass 0 with no segments for "
                    "a fully unfiltered domain)."
                )
        if n_semantic_points is not None:
            last_end = background_segments[-1][0] if background_segments else 0
            if last_end != int(n_semantic_points):
                raise ValueError(
                    "FusedBgTrackNodeSemanticLosses: the final background "
                    f"segment end ({last_end}) must equal n_semantic_points "
                    f"({int(n_semantic_points)})."
                )

        cuda_inputs: List[Tensor] = [
            positions,
            density_logits,
            other_density_logits,
            camera_timestamps_startend_us,
            tracks_packinfo,
            tracks_poses,
            tracks_timestamps_us,
            cuboids_dims,
            *background_semantic_logits,
            *other_semantic_logits,
        ]
        if semantic_logits is not None:
            cuda_inputs.append(semantic_logits)
        if self._cuda_available and all(t.is_cuda for t in cuda_inputs):
            from gsplat.cuda._losses_wrapper import _FusedBgTrackNodeSemantic

            return _FusedBgTrackNodeSemantic.apply(
                positions,
                density_logits,
                semantic_logits,
                other_density_logits,
                camera_timestamps_startend_us,
                tracks_packinfo,
                tracks_poses,
                tracks_timestamps_us,
                cuboids_dims,
                n_semantic_points,
                background_semantic_logits,
                background_segments,
                other_semantic_logits,
                other_segments,
                list(background_allowed_class_ids),
                node_primary_predicate,
                background_lambda,
                node_lambda,
                self.density_logits_min,
            )

        # Pure-PyTorch fallback: compose the standalone reference losses. The
        # node member's selected-count denominator is shared across all node
        # segments (kernel accumulates one global sum/count), so the joint
        # mode packs the primary domain and the other nodes into one call.
        n_background = int(density_logits.shape[0])
        if background_enabled:
            if n_semantic_points is None:
                bg_filter_logits: List[Tensor] = [semantic_logits]
                bg_filter_segments: List[Tuple[int, Sequence[int]]] = [
                    (n_background, list(background_allowed_class_ids))
                ]
            else:
                bg_filter_logits = background_semantic_logits
                bg_filter_segments = background_segments
            background_raw, background_weighted = background_in_track_loss(
                positions,
                density_logits,
                camera_timestamps_startend_us,
                tracks_packinfo,
                tracks_poses,
                tracks_timestamps_us,
                cuboids_dims,
                semantic_logits=bg_filter_logits,
                semantic_segments=bg_filter_segments,
                background_lambda=background_lambda,
                density_logits_min=self.density_logits_min,
            )
        else:
            # Disabled member: differentiable exact zero via an empty-view sum
            # (never reads values, so a present-but-non-finite input cannot
            # poison it; matches the CUDA path's zero-filled outputs).
            background_raw = density_logits[:0].sum()
            background_weighted = density_logits[:0].sum()

        if node_enabled:
            if n_semantic_points is None:
                primary_ids, primary_select = node_primary_predicate
                packed_density = torch.cat([density_logits, other_density_logits])
                packed_logits = [semantic_logits] + other_semantic_logits
                packed_segments = [
                    (n_background, list(primary_ids), bool(primary_select))
                ] + [(n_background + end, ids, sel) for end, ids, sel in other_segments]
                node_raw, node_weighted = node_semantic_loss(
                    packed_density,
                    packed_logits,
                    packed_segments,
                    node_lambda=node_lambda,
                )
            else:
                node_raw, node_weighted = node_semantic_loss(
                    other_density_logits,
                    other_semantic_logits,
                    other_segments,
                    node_lambda=node_lambda,
                )
        else:
            # Disabled member: differentiable exact zero (empty-view sum).
            node_raw = other_density_logits[:0].sum()
            node_weighted = other_density_logits[:0].sum()

        return background_raw, background_weighted, node_raw, node_weighted
