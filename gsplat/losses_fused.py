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
from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._backend import _C
from gsplat.cuda._wrapper import has_losses
from gsplat.losses import (
    bilateral_grid_drift_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)


def _cuda_losses_available() -> bool:
    """Check if the fused CUDA loss ops are compiled."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401

        return hasattr(torch.ops.gsplat, "gaussian_losses_fwd")
    except Exception:
        return False


def _cuda_camera_losses_available() -> bool:
    """Check if the fused CUDA camera losses op is compiled."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401

        return hasattr(torch.ops.gsplat, "camera_losses_fwd")
    except Exception:
        return False


def _cuda_lidar_losses_available() -> bool:
    """Check if the fused CUDA lidar losses op is compiled."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401

        return hasattr(torch.ops.gsplat, "lidar_losses_fwd")
    except Exception:
        return False


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


class FusedGaussianLosses(torch.nn.Module):
    """Fused CUDA implementation of four gaussian regularization losses.

    Computes :func:`~gsplat.losses.gaussian_scale_reg`,
    :func:`~gsplat.losses.gaussian_density_reg`,
    :func:`~gsplat.losses.gaussian_z_scale_reg`, and
    :func:`~gsplat.losses.out_of_bound_loss` in a single fused CUDA kernel.

    Falls back to the pure-PyTorch implementations when CUDA is
    unavailable or the inputs are on CPU.

    All outputs are **unreduced** (per-element), matching the pure-PyTorch
    contract.

    .. note::
        The CUDA kernel dispatches on ``fp32`` and ``fp64`` only. AMP training
        with ``fp16``/``bf16`` inputs will raise at kernel dispatch; cast to
        ``float`` before calling, or disable AMP for this module. Extending
        to half-precision is a possible future enhancement.

    Args:
        z_scale_threshold: Threshold above which z-scale penalty is applied.
    """

    def __init__(self, z_scale_threshold: float = 0.0):
        super().__init__()
        self.z_scale_threshold = z_scale_threshold
        self._cuda_available = has_losses()

    def forward(
        self,
        scales: Tensor,  # [N, 3]
        densities: Tensor,  # [N]
        z_scales: Tensor,  # [N]
        positions: Tensor,  # [N, 3]
        cuboid_dims: Tensor,  # [N, 3]
        visibility: Optional[Tensor] = None,  # [N] float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute all four gaussian regularization losses.

        Returns:
            Tuple of ``(loss_scale, loss_density, loss_z_scale, loss_oob)``,
            each unreduced per-element.
        """
        # Dispatch to CUDA only when every input is on the same GPU — otherwise
        # fall through to pure-PyTorch instead of tripping CHECK_INPUT deep in C++.
        all_cuda = scales.is_cuda and all(
            t.is_cuda for t in (densities, z_scales, positions, cuboid_dims)
        )
        if visibility is not None:
            all_cuda = all_cuda and visibility.is_cuda
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
            )

        # Pure-PyTorch fallback
        return (
            gaussian_scale_reg(scales, visibility=visibility),
            gaussian_density_reg(densities, visibility=visibility),
            gaussian_z_scale_reg(z_scales, self.z_scale_threshold),
            out_of_bound_loss(positions, cuboid_dims),
        )


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


class FusedCameraLosses(torch.nn.Module):
    """Fused CUDA camera losses: RGB L1 + background MSE with flag masking.

    Each pixel is masked by per-ray flags (int32 bitmask; see :class:`LossFlag`).
    Sub-losses are individually enabled/disabled via their factor:
    ``factor > 0`` = active, ``factor == 0`` = no valid pixels,
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
    ) -> Tuple[Tensor, Tensor]:
        """Returns ``(rgb_loss [N], bg_loss [N])``."""
        if rgb_gt.requires_grad:
            raise ValueError(
                "FusedCameraLosses does not support gradients w.r.t. the "
                "target rgb_gt; pass it with requires_grad=False. The fused "
                "CUDA backward returns no target gradient, so the pure-PyTorch "
                "fallback is held to the same contract to keep both paths "
                "consistent."
            )
        if self._cuda_available and _require_uniform_cuda(
            "FusedCameraLosses", (flags,), (rgb_pred, rgb_gt, bg_pred)
        ):
            from gsplat.cuda._losses_wrapper import _FusedCameraLosses

            return _FusedCameraLosses.apply(
                flags, rgb_pred, rgb_gt, bg_pred, rgb_factor, bg_factor
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
