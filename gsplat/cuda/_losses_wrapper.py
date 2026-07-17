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

"""Autograd.Function wrappers for fused CUDA loss kernels.

Dedicated companion to :mod:`gsplat.losses_fused` — this module groups the
autograd bridges for the fused-loss CUDA kernels, mirroring the
``losses_*`` / ``cuda/_*_wrapper`` split used elsewhere in the package.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import _make_lazy_cuda_func


class _FusedGaussianLosses(torch.autograd.Function):
    """Fused forward+backward for gaussian_scale_reg, gaussian_density_reg,
    gaussian_z_scale_reg, and out_of_bound_loss via a single CUDA kernel.

    Each member owns its row count (``N_scales``/``N_densities``/
    ``N_z_scales``/``N_oob``, any of which may be zero); ``visibility`` spans
    the scale and density members. ``preactivation`` selects the log-space
    member math (see :class:`gsplat.losses_fused.FusedGaussianLosses`).

    Grouped dispatch: when ``deformation`` is given, the same
    single op call additionally launches the deform-smoothness kernels and a
    fifth output — the scalar deform loss — is returned. Its gradients flow to
    ``deformation`` and, when present, ``deform_mask``.
    """

    @staticmethod
    def forward(
        ctx,
        scales: Tensor,  # [N_scales, 3]
        densities: Tensor,  # [N_densities]
        z_scales: Tensor,  # [N_z_scales]
        positions: Tensor,  # [N_oob, 3]
        cuboid_dims: Tensor,  # [N_oob, 3]
        z_scale_threshold: float,
        visibility: Optional[Tensor] = None,  # [max(N_scales, N_densities)] float
        deformation: Optional[Tensor] = None,  # [M, D]
        deform_mask: Optional[Tensor] = None,  # broadcastable to [M, D]
        preactivation: bool = False,
    ) -> Tuple[Tensor, ...]:
        # Unused outputs must reach backward as None rather than materialized
        # zeros: backward() normalizes each None base cotangent to a defined
        # exact zero and skips the deform member outright when its cotangent
        # is None, so an unused output can never poison a sibling's gradient.
        ctx.set_materialize_grads(False)
        # The CUDA backward does not compute d(oob)/d(cuboid_dims); failing loudly
        # here is safer than silently returning a zero gradient that the
        # pure-PyTorch path would have propagated.
        if cuboid_dims.requires_grad:
            raise ValueError(
                "FusedGaussianLosses does not support grad w.r.t. cuboid_dims; "
                "pass cuboid_dims with requires_grad=False."
            )
        if deformation is None and deform_mask is not None:
            raise ValueError("_FusedGaussianLosses: deform_mask requires deformation.")
        scales = scales.contiguous()
        densities = densities.contiguous()
        z_scales = z_scales.contiguous()
        positions = positions.contiguous()
        cuboid_dims = cuboid_dims.contiguous()
        if visibility is not None:
            # The pure-PyTorch path accepts visibility as [N] or [N, 1]; the
            # CUDA kernel expects [N]. Reshape here so the public API matches
            # the pure-PyTorch path.
            visibility = visibility.reshape(-1).contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps.
        loss_scale = torch.empty_like(scales)
        loss_density = torch.empty_like(densities)
        loss_z_scale = torch.empty_like(z_scales)
        loss_oob = torch.empty_like(positions)

        # Grouped deform-smoothness member: the running-sums buffer and scalar
        # loss must start zeroed (the kernel accumulates into sums via
        # atomics).
        deform_sums = None
        deform_loss = None
        if deformation is not None:
            deformation = deformation.contiguous()
            if deform_mask is not None:
                deform_mask = deform_mask.contiguous()
            deform_sums = torch.zeros(
                2, dtype=deformation.dtype, device=deformation.device
            )
            deform_loss = torch.zeros(
                (), dtype=deformation.dtype, device=deformation.device
            )

        _make_lazy_cuda_func("gaussian_losses_fwd")(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility,
            z_scale_threshold,
            loss_scale,
            loss_density,
            loss_z_scale,
            loss_oob,
            deformation,
            deform_mask,
            deform_sums,
            deform_loss,
            preactivation,
        )

        saved = [scales, densities, z_scales, positions, cuboid_dims]
        if visibility is not None:
            saved.append(visibility)
        if deformation is not None:
            saved.extend([deformation, deform_sums])
            if deform_mask is not None:
                saved.append(deform_mask)
        ctx.save_for_backward(*saved)
        ctx.has_visibility = visibility is not None
        ctx.has_deform = deformation is not None
        ctx.has_deform_mask = deform_mask is not None
        ctx.z_scale_threshold = z_scale_threshold
        ctx.preactivation = preactivation

        if deformation is not None:
            return loss_scale, loss_density, loss_z_scale, loss_oob, deform_loss
        return loss_scale, loss_density, loss_z_scale, loss_oob

    @staticmethod
    def backward(
        ctx,
        v_loss_scale: Optional[Tensor],
        v_loss_density: Optional[Tensor],
        v_loss_z_scale: Optional[Tensor],
        v_loss_oob: Optional[Tensor],
        *v_extra: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], ...]:
        saved = list(ctx.saved_tensors)
        scales, densities, z_scales, positions, cuboid_dims = saved[:5]
        idx = 5
        visibility = None
        if ctx.has_visibility:
            visibility = saved[idx]
            idx += 1
        deformation = None
        deform_sums = None
        deform_mask = None
        if ctx.has_deform:
            deformation, deform_sums = saved[idx : idx + 2]
            idx += 2
            if ctx.has_deform_mask:
                deform_mask = saved[idx]
                idx += 1

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_scales = torch.empty_like(scales)
        v_densities = torch.empty_like(densities)
        v_z_scales = torch.empty_like(z_scales)
        v_positions = torch.empty_like(positions)

        # set_materialize_grads(False) delivers unused outputs' cotangents as
        # None. The base op requires defined per-element upstream grads, so
        # normalize those to exact zeros; the per-element base backward turns
        # a zero upstream into a zero gradient.
        def _upstream(grad: Optional[Tensor], ref: Tensor) -> Tensor:
            return torch.zeros_like(ref) if grad is None else grad.contiguous()

        # Grouped deform-smoothness member: v_deformation is fully written by
        # the kernel; v_deform_mask accumulates via atomics, so it must start
        # zeroed. A None cotangent means the deform loss is not part of the
        # objective: the member's backward launch is skipped outright (all
        # deform arguments passed as None) so an unused-but-present
        # non-finite deformation can never poison any gradient.
        v_deform_loss = v_extra[0] if v_extra else None
        deform_active = ctx.has_deform and v_deform_loss is not None
        v_deformation = None
        v_deform_mask = None
        if deform_active:
            v_deform_loss = v_deform_loss.reshape(()).contiguous()
            v_deformation = torch.empty_like(deformation)
            if deform_mask is not None:
                v_deform_mask = torch.zeros_like(deform_mask)

        _make_lazy_cuda_func("gaussian_losses_bwd")(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility,
            ctx.z_scale_threshold,
            _upstream(v_loss_scale, scales),
            _upstream(v_loss_density, densities),
            _upstream(v_loss_z_scale, z_scales),
            _upstream(v_loss_oob, positions),
            v_scales,
            v_densities,
            v_z_scales,
            v_positions,
            deformation if deform_active else None,
            deform_mask if deform_active else None,
            deform_sums if deform_active else None,
            v_deform_loss if deform_active else None,
            v_deformation,
            v_deform_mask,
            ctx.preactivation,
        )

        # Gradients for: scales, densities, z_scales, positions, cuboid_dims,
        #   z_scale_threshold, visibility, deformation, deform_mask,
        #   preactivation.
        # (The trailing entries are None whenever the caller omitted the
        # optional inputs — autograd drops extra trailing None grads.)
        return (
            v_scales,
            v_densities,
            v_z_scales,
            v_positions,
            None,
            None,
            None,
            v_deformation,
            v_deform_mask,
            None,
        )


class _FusedCameraLosses(torch.autograd.Function):
    """Fused RGB L1 + background MSE with per-ray flag masking, plus an
    optional grouped masked semantic cross-entropy member of the fused
    camera dispatch.

    When the ``semantic_*`` inputs are provided, the same single
    ``camera_losses_fwd``/``camera_losses_bwd`` op call additionally launches
    the semantic-CE kernels and a third output — the scalar CE loss — is
    returned. The semantic row count is independent of the camera ray count
    (the CE carries its own ``valid`` row mask). Gradient routing: the CE
    member contributes ``v_semantic_logits`` only; the camera members
    contribute ``v_rgb_pred``/``v_bg_pred`` exactly as in ungrouped mode.
    """

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [N] int32
        rgb_pred: Tensor,  # [N, 3]
        rgb_gt: Tensor,  # [N, 3]
        bg_pred: Tensor,  # [N]
        rgb_factor: float,
        bg_factor: float,
        semantic_logits: Optional[Tensor] = None,  # [M, C] fp32/fp64
        semantic_targets: Optional[Tensor] = None,  # [M] uint8 or int64
        semantic_valid: Optional[Tensor] = None,  # [M] bool
        semantic_ignore_index: int = -100,
    ) -> Tuple[Tensor, ...]:
        # Unused outputs must reach backward as None rather than materialized
        # zeros: backward() normalizes each None camera cotangent to a defined
        # exact zero and skips the CE member outright when its cotangent is
        # None, so an unused output can never poison a sibling's gradient.
        ctx.set_materialize_grads(False)
        flags = flags.contiguous()
        rgb_pred = rgb_pred.contiguous()
        rgb_gt = rgb_gt.contiguous()
        bg_pred = bg_pred.contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps. rgb_loss
        # is per-ray [N] (the RGB channels are reduced inside the kernel), so it
        # is allocated from bg_pred's [N] shape rather than rgb_pred's [N, 3].
        rgb_loss = torch.empty_like(bg_pred)
        bg_loss = torch.empty_like(bg_pred)

        has_semantic = semantic_logits is not None
        if has_semantic:
            semantic_logits = semantic_logits.contiguous()
            semantic_targets = semantic_targets.reshape(-1).contiguous()
            semantic_valid = semantic_valid.reshape(-1).contiguous()
            # CE workspace + scalar loss. The kernel accumulates into
            # loss_sum/valid_count via atomics, so both must start zeroed.
            semantic_loss_sum = torch.zeros(
                (), dtype=semantic_logits.dtype, device=semantic_logits.device
            )
            semantic_valid_count = torch.zeros(
                (), dtype=torch.int32, device=semantic_logits.device
            )
            semantic_loss = torch.zeros(
                (), dtype=semantic_logits.dtype, device=semantic_logits.device
            )
        else:
            semantic_loss_sum = None
            semantic_valid_count = None
            semantic_loss = None

        _make_lazy_cuda_func("camera_losses_fwd")(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            rgb_factor,
            bg_factor,
            rgb_loss,
            bg_loss,
            semantic_logits,
            semantic_targets,
            semantic_valid,
            int(semantic_ignore_index),
            semantic_loss_sum,
            semantic_valid_count,
            semantic_loss,
        )

        if has_semantic:
            # The CE backward only needs the valid-row count from the
            # workspace (the per-row softmax is re-derived from logits).
            ctx.save_for_backward(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits,
                semantic_targets,
                semantic_valid,
                semantic_valid_count,
            )
        else:
            ctx.save_for_backward(flags, rgb_pred, rgb_gt, bg_pred)
        ctx.has_semantic = has_semantic
        ctx.rgb_factor = rgb_factor
        ctx.bg_factor = bg_factor
        ctx.semantic_ignore_index = int(semantic_ignore_index)

        if has_semantic:
            return rgb_loss, bg_loss, semantic_loss
        return rgb_loss, bg_loss

    @staticmethod
    def backward(
        ctx,
        v_rgb_loss: Optional[Tensor],
        v_bg_loss: Optional[Tensor],
        v_semantic_loss: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], ...]:
        if ctx.has_semantic:
            (
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits,
                semantic_targets,
                semantic_valid,
                semantic_valid_count,
            ) = ctx.saved_tensors
        else:
            flags, rgb_pred, rgb_gt, bg_pred = ctx.saved_tensors
            semantic_logits = None
            semantic_targets = None
            semantic_valid = None
            semantic_valid_count = None

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_rgb_pred = torch.empty_like(rgb_pred)
        v_bg_pred = torch.empty_like(bg_pred)

        # set_materialize_grads(False) delivers unused outputs' cotangents as
        # None. The camera op requires defined per-ray upstream grads, so
        # normalize those to exact zeros; the per-ray backward turns a zero
        # upstream into a zero gradient.
        def _upstream(grad: Optional[Tensor]) -> Tensor:
            return torch.zeros_like(bg_pred) if grad is None else grad.contiguous()

        # Grouped CE member: a None cotangent means the CE loss is not part of
        # the objective — its backward launch is skipped outright (all
        # semantic arguments passed as None), so present-but-unused non-finite
        # logits can never poison any gradient.
        semantic_active = ctx.has_semantic and v_semantic_loss is not None
        v_semantic_logits = None
        if semantic_active:
            # The CE backward kernel writes every (row, class) element of
            # v_semantic_logits exactly once (gradients for contributing rows,
            # exact zeros otherwise), so the buffer needs no pre-zeroing.
            v_semantic_logits = torch.empty_like(semantic_logits)
            v_semantic_loss = v_semantic_loss.reshape(()).contiguous()

        _make_lazy_cuda_func("camera_losses_bwd")(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            ctx.rgb_factor,
            ctx.bg_factor,
            _upstream(v_rgb_loss),
            _upstream(v_bg_loss),
            v_rgb_pred,
            v_bg_pred,
            semantic_logits if semantic_active else None,
            semantic_targets if semantic_active else None,
            semantic_valid if semantic_active else None,
            ctx.semantic_ignore_index,
            semantic_valid_count if semantic_active else None,
            v_semantic_loss if semantic_active else None,
            v_semantic_logits,
        )

        # Gradients for: flags, rgb_pred, rgb_gt, bg_pred, rgb_factor,
        #   bg_factor, semantic_logits, semantic_targets, semantic_valid,
        #   semantic_ignore_index
        return (
            None,
            v_rgb_pred,
            None,
            v_bg_pred,
            None,
            None,
            v_semantic_logits,
            None,
            None,
            None,
        )


class _FusedLidarLosses(torch.autograd.Function):
    """Fused distance L1 + intensity/raydrop/bg MSE with per-ray flag masking."""

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [N] int32
        distance_pred: Tensor,  # [N]
        distance_gt: Tensor,  # [N]
        intensity_pred: Tensor,  # [N]
        intensity_gt: Tensor,  # [N]
        raydrop_pred: Tensor,  # [N]
        raydrop_gt: Tensor,  # [N]
        bg_pred: Tensor,  # [N]
        distance_factor: float,
        intensity_factor: float,
        raydrop_factor: float,
        bg_factor: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        flags = flags.contiguous()
        distance_pred = distance_pred.contiguous()
        distance_gt = distance_gt.contiguous()
        intensity_pred = intensity_pred.contiguous()
        intensity_gt = intensity_gt.contiguous()
        raydrop_pred = raydrop_pred.contiguous()
        raydrop_gt = raydrop_gt.contiguous()
        bg_pred = bg_pred.contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps. Each loss
        # is per-ray [N], matching its corresponding *_pred input.
        distance_loss = torch.empty_like(distance_pred)
        intensity_loss = torch.empty_like(intensity_pred)
        raydrop_loss = torch.empty_like(raydrop_pred)
        bg_loss = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("lidar_losses_fwd")(
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
            distance_loss,
            intensity_loss,
            raydrop_loss,
            bg_loss,
        )

        ctx.save_for_backward(
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
        )
        ctx.factors = (distance_factor, intensity_factor, raydrop_factor, bg_factor)

        return distance_loss, intensity_loss, raydrop_loss, bg_loss

    @staticmethod
    def backward(
        ctx,
        v_distance_loss: Tensor,
        v_intensity_loss: Tensor,
        v_raydrop_loss: Tensor,
        v_bg_loss: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        (
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
        ) = ctx.saved_tensors
        distance_factor, intensity_factor, raydrop_factor, bg_factor = ctx.factors

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_distance_pred = torch.empty_like(distance_pred)
        v_intensity_pred = torch.empty_like(intensity_pred)
        v_raydrop_pred = torch.empty_like(raydrop_pred)
        v_bg_pred = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("lidar_losses_bwd")(
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
            v_distance_loss.contiguous(),
            v_intensity_loss.contiguous(),
            v_raydrop_loss.contiguous(),
            v_bg_loss.contiguous(),
            v_distance_pred,
            v_intensity_pred,
            v_raydrop_pred,
            v_bg_pred,
        )

        # Gradients for: flags, distance_pred, distance_gt, intensity_pred,
        #   intensity_gt, raydrop_pred, raydrop_gt, bg_pred, 4x factors
        return (
            None,
            v_distance_pred,
            None,
            v_intensity_pred,
            None,
            v_raydrop_pred,
            None,
            v_bg_pred,
            None,
            None,
            None,
            None,
        )


class _FusedGroundGaussiansLosses(torch.autograd.Function):
    """Fused forward+backward for the ground-gaussian distortion loss.

    Computes a scalar loss constraining the height (y) and roll/pitch
    rotation variance of gaussians inside randomly placed camera-space depth
    bins. Gradients flow to ``positions`` and ``rotations``; the camera pose
    and the bin parameters are treated as constants.
    """

    @staticmethod
    def forward(
        ctx,
        positions: Tensor,  # [N, 3]
        rotations: Tensor,  # [N, 4] quaternion (x, y, z, w)
        cam_tquat: Tensor,  # [7] camera-from-world (tx,ty,tz,qx,qy,qz,qw)
        random_values: Tensor,  # [B] bin offsets in [0, 1)
        min_bias: float,
        range_bias: float,
        grid_len: float,
        rotation_lambda: float,
    ) -> Tensor:
        positions = positions.contiguous()
        rotations = rotations.contiguous()
        cam_tquat = cam_tquat.reshape(-1).contiguous()
        random_values = random_values.reshape(-1).contiguous()

        n_bins = random_values.shape[0]
        # Pre-allocate the statistics buffer and scalar loss in Python so
        # lifetimes are explicit and the caching allocator can reuse them.
        stats = torch.zeros(n_bins, 7, dtype=positions.dtype, device=positions.device)
        loss = torch.zeros((), dtype=positions.dtype, device=positions.device)

        _make_lazy_cuda_func("ground_gaussians_losses_fwd")(
            positions,
            rotations,
            cam_tquat,
            random_values,
            float(min_bias),
            float(range_bias),
            float(grid_len),
            float(rotation_lambda),
            stats,
            loss,
        )

        ctx.save_for_backward(positions, rotations, cam_tquat, random_values, stats)
        ctx.min_bias = float(min_bias)
        ctx.range_bias = float(range_bias)
        ctx.grid_len = float(grid_len)
        ctx.rotation_lambda = float(rotation_lambda)

        return loss

    @staticmethod
    def backward(ctx, v_loss: Tensor) -> Tuple[Optional[Tensor], ...]:
        positions, rotations, cam_tquat, random_values, stats = ctx.saved_tensors

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_positions = torch.zeros_like(positions)
        v_rotations = torch.zeros_like(rotations)

        _make_lazy_cuda_func("ground_gaussians_losses_bwd")(
            positions,
            rotations,
            cam_tquat,
            random_values,
            stats,
            v_loss.reshape(()).contiguous(),
            ctx.min_bias,
            ctx.range_bias,
            ctx.grid_len,
            ctx.rotation_lambda,
            v_positions,
            v_rotations,
        )

        # Gradients for: positions, rotations, cam_tquat, random_values,
        #                 min_bias, range_bias, grid_len, rotation_lambda
        return v_positions, v_rotations, None, None, None, None, None, None


class _FusedBgTrackNodeSemantic(torch.autograd.Function):
    """Fused forward+backward for the background-in-track + node-semantic
    grouped BCE losses.

    One call computes up to two scalar loss pairs sharing a single selection
    pass: the background-in-track member (suppress background Gaussians whose
    position falls inside any dynamic cuboid track) and the node-semantic
    member (suppress Gaussians whose semantic argmax fails a per-node
    ``(class_ids, select_matches)`` predicate). A member is enabled iff its
    lambda is ``>= 0``; at least one must be enabled. Only the two density
    tensors are differentiable — positions and semantic logits feed
    non-differentiable selections.

    Modes:

    - **joint** (``n_semantic_points is None``): node member enabled and the
      primary node shares the background point domain; ``semantic_logits``
      is that domain's single ``[Nbg, C]`` semantic tensor and an explicit
      ``node_primary_predicate`` is REQUIRED (an *empty* predicate is legal:
      empty ids + ``select_matches=True`` penalize no primary point, empty
      ids + ``select_matches=False`` penalize all of them; the C++ op
      rejects a missing predicate with a TORCH_CHECK).
    - **packed** (``n_semantic_points`` given, node enabled): background runs
      the generic segmented path over its own semantic segments (points past
      ``n_semantic_points`` are unfiltered); nodes run packed segments over
      ``other_density_logits``. Requires the background member enabled.
    - **background-only** (node lambda ``< 0``): generic segmented background
      path alone; requires ``n_semantic_points``.

    Op contract for the C++/CUDA side (TORCH_LIBRARY schema, ext.cpp)::

        bg_track_node_semantic_losses_fwd(
            Tensor positions, Tensor density_logits, Tensor semantic_logits,
            int? n_semantic_points,
            Tensor[] background_semantic_logits,
            int[] background_segment_ends,
            int[] background_segment_class_ids,
            int[] background_segment_class_id_ends,
            Tensor other_density_logits,
            Tensor[] other_semantic_logits,
            int[] other_segment_ends,
            int[] other_segment_class_ids,
            int[] other_segment_class_id_ends,
            int[] other_segment_select_matches,
            Tensor camera_timestamps_startend_us, Tensor tracks_packinfo,
            Tensor tracks_poses, Tensor tracks_timestamps_us,
            Tensor cuboids_dims,
            int[] background_allowed_class_ids,
            int[]? node_primary_class_ids, bool node_primary_select_matches,
            float density_logits_min,
            float background_lambda, float node_lambda,
            Tensor(a!) track_boxes, Tensor(b!) selection,
            Tensor(c!) workspace,
            Tensor(d!) background_unweighted_loss,
            Tensor(e!) background_weighted_loss,
            Tensor(f!) node_unweighted_loss,
            Tensor(g!) node_weighted_loss) -> ()

        bg_track_node_semantic_losses_bwd(
            Tensor background_density_logits, Tensor other_density_logits,
            Tensor selection, Tensor workspace,
            Tensor? grad_background_unweighted,
            Tensor? grad_background_weighted,
            Tensor? grad_node_unweighted, Tensor? grad_node_weighted,
            float background_lambda, float node_lambda,
            Tensor(a!)? grad_background_density_logits,
            Tensor(b!)? grad_other_density_logits) -> ()

    Boundary conventions (all caller-allocated, written by the op):

    - Ragged per-segment class-id lists cross as a flat ``int[]`` plus a
      ``len(segments)`` exclusive-end ``int[]`` with no leading zero
      (segment *i* owns ``class_ids[ends[i-1]:ends[i]]``, the first segment
      starting at 0); segment ends are exclusive and nondecreasing, and the
      select flags cross as 0/1 ``int[]`` (checked host-side). ``node_primary_class_ids is None`` means "no
      predicate" (only legal when the joint mode is not selected).
    - ``semantic_logits`` is only read in joint mode; other modes pass an
      empty ``[0, 1]`` placeholder.
    - ``camera_timestamps_startend_us`` is ``[B, 2]`` int64 with ``B >= 1``
      (the op rejects ``B == 0``); only row 0 — the reference camera — is
      read, matching the pure-PyTorch fallback. Extra rows are ignored.
    - ``track_boxes`` is a ``[T, 16]`` float workspace (three world-to-local
      rotation rows with the world center in the 4th column, then half dims
      + valid flag); ``selection_bits`` is ``uint8 [Nbg + Nother]``
      (bit 0 = background-in-track, bit 1 = node-semantic); ``workspace``
      is a zeroed, 16-byte-aligned ``int32[8]`` reduction buffer for BOTH
      dtypes (the fp32 layout occupies the first 16 bytes, fp64 all 32)
      holding (loss_sum, selected_count) per member. The selected counts are
      written only by the forward and divided by in the backward: the same
      ``selection_bits``/``workspace`` instances the forward filled must
      reach the backward unmodified (they ride ``ctx.save_for_backward``);
      re-zeroing, pooling, or reusing them between the passes silently
      corrupts every gradient of that step.
    - The four loss outputs are one-element float tensors; raw values are
      selected-count means (0 when nothing is selected) and weighted values
      are ``lambda * raw``. Disabled members' outputs are left untouched
      (they are zero-initialized here).
    - The backward upstream loss grads are always defined (absent grads are
      normalized to zeros before dispatch); the two density-grad outputs are
      optional and skipped for inputs that do not need gradients.
    """

    @staticmethod
    def forward(
        ctx,
        positions: Tensor,  # [Nbg, 3]
        density_logits: Tensor,  # [Nbg] (differentiable)
        semantic_logits: Optional[Tensor],  # [Nbg, C] joint mode only
        other_density_logits: Tensor,  # [Nother] (differentiable)
        camera_timestamps_startend_us: Tensor,  # [B, 2] int64, B >= 1; row 0 only
        tracks_packinfo: Tensor,  # [T, 2] int32
        tracks_poses: Tensor,  # [P, 7]
        tracks_timestamps_us: Tensor,  # [P] int64
        cuboids_dims: Tensor,  # [T, 3]
        n_semantic_points,  # Optional[int]; None selects the joint mode
        background_semantic_logits,  # list[Tensor [Ni, Ci]] (generic mode)
        background_segments,  # list[(end, class_ids)] (generic mode)
        other_semantic_logits,  # list[Tensor [Ni, Ci]]
        other_segments,  # list[(end, class_ids, select_matches)]
        background_allowed_class_ids,  # list[int] (joint mode)
        node_primary_predicate,  # Optional[(class_ids, select_matches)]
        background_lambda: float,
        node_lambda: float,
        density_logits_min: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Unused outputs must reach backward as None rather than materialized
        # zeros; backward() then normalizes each None to a defined exact-zero
        # upstream, so an unused output can never feed the native op anything
        # but zero (see _upstream below).
        ctx.set_materialize_grads(False)
        background_enabled = background_lambda >= 0.0
        node_enabled = node_lambda >= 0.0
        if not background_enabled and not node_enabled:
            raise ValueError(
                "_FusedBgTrackNodeSemantic: at least one of background_lambda "
                "and node_lambda must be >= 0."
            )
        if (
            node_enabled
            and n_semantic_points is None
            and node_primary_predicate is None
        ):
            # Mirrors the C++ TORCH_CHECK: the joint path requires an explicit
            # primary-domain predicate (an empty one is legal, absence is not).
            raise ValueError(
                "_FusedBgTrackNodeSemantic: the joint node-semantic path "
                "requires an explicit node_primary_predicate; pass "
                "(class_ids, select_matches). Empty ids with "
                "select_matches=True penalize no primary point, with "
                "select_matches=False they penalize all of them."
            )

        positions = positions.contiguous()
        density_logits = density_logits.contiguous()
        other_density_logits = other_density_logits.contiguous()
        if semantic_logits is None:
            semantic_logits = density_logits.new_zeros((0, 1))
        semantic_logits = semantic_logits.contiguous()
        camera_timestamps_startend_us = camera_timestamps_startend_us.contiguous()
        tracks_packinfo = tracks_packinfo.contiguous()
        tracks_poses = tracks_poses.contiguous()
        tracks_timestamps_us = tracks_timestamps_us.contiguous()
        cuboids_dims = cuboids_dims.contiguous()
        background_semantic_logits = [
            t.contiguous() for t in background_semantic_logits
        ]
        other_semantic_logits = [t.contiguous() for t in other_semantic_logits]

        def _flatten_id_groups(groups):
            flat = []
            ends = []
            for ids in groups:
                flat.extend(int(i) for i in ids)
                ends.append(len(flat))
            return flat, ends

        background_segment_ends = [int(end) for end, _ in background_segments]
        background_ids_flat, background_ids_ends = _flatten_id_groups(
            [ids for _, ids in background_segments]
        )
        other_segment_ends = [int(end) for end, _, _ in other_segments]
        other_ids_flat, other_ids_ends = _flatten_id_groups(
            [ids for _, ids, _ in other_segments]
        )
        other_select_matches = [bool(sel) for _, _, sel in other_segments]

        if node_primary_predicate is not None:
            node_primary_class_ids = [int(i) for i in node_primary_predicate[0]]
            node_primary_select_matches = bool(node_primary_predicate[1])
        else:
            node_primary_class_ids = None
            node_primary_select_matches = False

        n_background = density_logits.shape[0]
        n_other = other_density_logits.shape[0]
        n_tracks = tracks_packinfo.shape[0]
        device = density_logits.device
        dtype = density_logits.dtype

        # Caller-allocated workspaces and outputs (see the op contract above).
        track_boxes = torch.zeros((n_tracks, 16), dtype=dtype, device=device)
        selection_bits = torch.zeros(
            n_background + n_other, dtype=torch.uint8, device=device
        )
        # int32[8] for both dtypes: the reduction struct needs 32 B under fp64
        # and the kernel checks a dtype-independent size.
        workspace = torch.zeros(8, dtype=torch.int32, device=device)
        background_unweighted_loss = torch.zeros(1, dtype=dtype, device=device)
        background_weighted_loss = torch.zeros(1, dtype=dtype, device=device)
        node_unweighted_loss = torch.zeros(1, dtype=dtype, device=device)
        node_weighted_loss = torch.zeros(1, dtype=dtype, device=device)

        _make_lazy_cuda_func("bg_track_node_semantic_losses_fwd")(
            positions,
            density_logits,
            semantic_logits,
            None if n_semantic_points is None else int(n_semantic_points),
            background_semantic_logits,
            background_segment_ends,
            background_ids_flat,
            background_ids_ends,
            other_density_logits,
            other_semantic_logits,
            other_segment_ends,
            other_ids_flat,
            other_ids_ends,
            other_select_matches,
            camera_timestamps_startend_us,
            tracks_packinfo,
            tracks_poses,
            tracks_timestamps_us,
            cuboids_dims,
            [int(i) for i in background_allowed_class_ids],
            node_primary_class_ids,
            node_primary_select_matches,
            float(density_logits_min),
            float(background_lambda),
            float(node_lambda),
            track_boxes,
            selection_bits,
            workspace,
            background_unweighted_loss,
            background_weighted_loss,
            node_unweighted_loss,
            node_weighted_loss,
        )

        # Forward-to-backward contract: selection_bits and workspace (whose
        # selected counts the backward divides by) are written only by the
        # forward op above. save_for_backward carries these exact instances
        # to backward(); they must arrive unmodified — re-zeroing or pooling
        # them between the passes is a silent-gradient-corruption bug.
        ctx.save_for_backward(
            density_logits, other_density_logits, selection_bits, workspace
        )
        ctx.background_lambda = float(background_lambda)
        ctx.node_lambda = float(node_lambda)

        return (
            background_unweighted_loss.reshape(()),
            background_weighted_loss.reshape(()),
            node_unweighted_loss.reshape(()),
            node_weighted_loss.reshape(()),
        )

    @staticmethod
    def backward(
        ctx,
        v_background_raw: Optional[Tensor],
        v_background_weighted: Optional[Tensor],
        v_node_raw: Optional[Tensor],
        v_node_weighted: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], ...]:
        (
            density_logits,
            other_density_logits,
            selection_bits,
            workspace,
        ) = ctx.saved_tensors

        def _upstream(grad: Optional[Tensor]) -> Tensor:
            # Normalize absent upstream grads to defined zero scalars so the
            # kernel can dereference all four unconditionally.
            if grad is None:
                return density_logits.new_zeros(1)
            return grad.reshape(1).contiguous()

        needs_background = ctx.needs_input_grad[1]
        # The node member owns the only gradient path into
        # other_density_logits; with it disabled the generic backward never
        # writes that buffer, so skip it entirely.
        needs_other = ctx.needs_input_grad[3] and ctx.node_lambda >= 0.0

        v_density_logits = (
            torch.empty_like(density_logits) if needs_background else None
        )
        v_other_density_logits = (
            torch.empty_like(other_density_logits) if needs_other else None
        )

        if needs_background or needs_other:
            _make_lazy_cuda_func("bg_track_node_semantic_losses_bwd")(
                density_logits,
                other_density_logits,
                selection_bits,
                workspace,
                _upstream(v_background_raw),
                _upstream(v_background_weighted),
                _upstream(v_node_raw),
                _upstream(v_node_weighted),
                ctx.background_lambda,
                ctx.node_lambda,
                v_density_logits,
                v_other_density_logits,
            )

        # Gradients for: positions, density_logits, semantic_logits,
        # other_density_logits, camera_timestamps_startend_us,
        # tracks_packinfo, tracks_poses, tracks_timestamps_us, cuboids_dims,
        # n_semantic_points, background_semantic_logits, background_segments,
        # other_semantic_logits, other_segments,
        # background_allowed_class_ids, node_primary_predicate,
        # background_lambda, node_lambda, density_logits_min
        return (
            None,
            v_density_logits,
            None,
            v_other_density_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FusedBgGridLosses(torch.autograd.Function):
    """Fused forward+backward for sky-envmap TV + bilateral-grid drift + grid
    spatial TV via a single pair of CUDA kernels.

    Any of ``bg_tex`` / ``grids_camera`` / ``grids_frame`` can be ``None``
    to skip its sub-losses entirely. Per-sub-loss enable/disable is also
    available via a negative ``factor`` (loss output is written as zeros).
    """

    @staticmethod
    def forward(
        ctx,
        bg_tex: Optional[Tensor],  # [B*D, H, W, C] flat (D=1 planar, D=6 cubemap)
        bg_tex_depth: int,
        grids_camera: Optional[Tensor],  # [B*12, D, H, W]
        grids_frame: Optional[Tensor],  # [B*12, D, H, W]
        bg_tex_factor: float,
        grid_drift_camera_factor: float,
        grid_drift_frame_factor: float,
        grid_camera_tv_factor: float,
        grid_frame_tv_factor: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Unused outputs must reach backward as None rather than materialized zeros, so a
        # NaN-weighted-but-unused sub-loss can't be fed to the native op and poison a finite
        # sibling's shared-input gradient via NaN*0. backward() disables each None output.
        ctx.set_materialize_grads(False)
        bg_tex = bg_tex.contiguous() if bg_tex is not None else None
        grids_camera = grids_camera.contiguous() if grids_camera is not None else None
        grids_frame = grids_frame.contiguous() if grids_frame is not None else None

        # Reference device for allocation (first non-None input).
        ref = (
            bg_tex
            if bg_tex is not None
            else (grids_camera if grids_camera is not None else grids_frame)
        )
        if ref is None:
            raise ValueError(
                "FusedBgGridLosses.forward needs at least one of "
                "bg_tex / grids_camera / grids_frame to be non-None."
            )
        device, dtype = ref.device, ref.dtype

        # Per-cell element counts (1 scalar per (b, d, h, w) cell for grids;
        # 1 scalar per element for bg_tex).
        def _cell_numel(g: Optional[Tensor]) -> int:
            if g is None:
                return 0
            # g is [B*12, D, H, W]; cells = B * D * H * W. The 12 is the fixed
            # 3x4 affine channel count (LOSSES_GRID_NUM_CHANNELS in Config.h,
            # which is non-overrideable so this constant can't drift from it).
            return (g.shape[0] // 12) * g.shape[1] * g.shape[2] * g.shape[3]

        numel_bg = bg_tex.numel() if bg_tex is not None else 0
        numel_gc = _cell_numel(grids_camera)
        numel_gf = _cell_numel(grids_frame)

        # Pre-allocate outputs on the Python side (keeps buffer lifetime
        # explicit and lets torch's caching allocator reuse across steps).
        bg_tex_loss = torch.empty(numel_bg, device=device, dtype=dtype)
        grids_drift_loss = torch.empty(numel_gc + numel_gf, device=device, dtype=dtype)
        grid_camera_tv_loss = torch.empty(numel_gc, device=device, dtype=dtype)
        grid_frame_tv_loss = torch.empty(numel_gf, device=device, dtype=dtype)

        _make_lazy_cuda_func("bg_grid_losses_fwd")(
            bg_tex,
            int(bg_tex_depth),
            grids_camera,
            grids_frame,
            float(bg_tex_factor),
            float(grid_drift_camera_factor),
            float(grid_drift_frame_factor),
            float(grid_camera_tv_factor),
            float(grid_frame_tv_factor),
            bg_tex_loss,
            grids_drift_loss,
            grid_camera_tv_loss,
            grid_frame_tv_loss,
        )

        ctx.save_for_backward(
            (
                bg_tex
                if bg_tex is not None
                else torch.empty(0, device=device, dtype=dtype)
            ),
            (
                grids_camera
                if grids_camera is not None
                else torch.empty(0, device=device, dtype=dtype)
            ),
            (
                grids_frame
                if grids_frame is not None
                else torch.empty(0, device=device, dtype=dtype)
            ),
        )
        ctx.has_bg_tex = bg_tex is not None
        ctx.has_grids_camera = grids_camera is not None
        ctx.has_grids_frame = grids_frame is not None
        ctx.bg_tex_depth = int(bg_tex_depth)
        # Output element counts so backward can build shape-correct zero placeholders
        # for any output whose cotangent arrives as None (unused).
        ctx.numel_bg = numel_bg
        ctx.numel_gc = numel_gc
        ctx.numel_gf = numel_gf
        ctx.factors = (
            float(bg_tex_factor),
            float(grid_drift_camera_factor),
            float(grid_drift_frame_factor),
            float(grid_camera_tv_factor),
            float(grid_frame_tv_factor),
        )

        return bg_tex_loss, grids_drift_loss, grid_camera_tv_loss, grid_frame_tv_loss

    @staticmethod
    def backward(
        ctx,
        v_bg_tex_loss: Optional[Tensor],
        v_grids_drift_loss: Optional[Tensor],
        v_grid_camera_tv_loss: Optional[Tensor],
        v_grid_frame_tv_loss: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], ...]:
        bg_tex, grids_camera, grids_frame = ctx.saved_tensors

        bg_tex_arg = bg_tex if ctx.has_bg_tex else None
        grids_camera_arg = grids_camera if ctx.has_grids_camera else None
        grids_frame_arg = grids_frame if ctx.has_grids_frame else None

        # set_materialize_grads(False) means an unused output arrives as None. Replace it
        # with a shape-correct zero placeholder for the native op and disable that output's
        # factor(s) (negative), so the native backward skips it — otherwise a NaN factor on
        # an unused output would contaminate a finite sibling's shared-input gradient.
        bg_f, drift_c_f, drift_f_f, tv_c_f, tv_f_f = ctx.factors
        _ref = next(
            t for t in (bg_tex_arg, grids_camera_arg, grids_frame_arg) if t is not None
        )
        _dev, _dt = _ref.device, _ref.dtype
        if v_bg_tex_loss is None:
            v_bg_tex_loss = torch.zeros(ctx.numel_bg, device=_dev, dtype=_dt)
            bg_f = -1.0
        if v_grids_drift_loss is None:
            v_grids_drift_loss = torch.zeros(
                ctx.numel_gc + ctx.numel_gf, device=_dev, dtype=_dt
            )
            # Combined camera+frame drift output -> disable both drift factors.
            drift_c_f = -1.0
            drift_f_f = -1.0
        if v_grid_camera_tv_loss is None:
            v_grid_camera_tv_loss = torch.zeros(ctx.numel_gc, device=_dev, dtype=_dt)
            tv_c_f = -1.0
        if v_grid_frame_tv_loss is None:
            v_grid_frame_tv_loss = torch.zeros(ctx.numel_gf, device=_dev, dtype=_dt)
            tv_f_f = -1.0

        # Gradient buffers — the gather backward writes every entry exactly once
        # (zero when a sub-loss is disabled), so no pre-zeroing is needed.
        v_bg_tex = (
            torch.empty_like(bg_tex)
            if ctx.has_bg_tex
            else torch.empty(0, device=v_bg_tex_loss.device, dtype=v_bg_tex_loss.dtype)
        )
        v_grids_camera = (
            torch.empty_like(grids_camera)
            if ctx.has_grids_camera
            else torch.empty(
                0, device=v_grids_drift_loss.device, dtype=v_grids_drift_loss.dtype
            )
        )
        v_grids_frame = (
            torch.empty_like(grids_frame)
            if ctx.has_grids_frame
            else torch.empty(
                0, device=v_grids_drift_loss.device, dtype=v_grids_drift_loss.dtype
            )
        )

        _make_lazy_cuda_func("bg_grid_losses_bwd")(
            bg_tex_arg,
            ctx.bg_tex_depth,
            grids_camera_arg,
            grids_frame_arg,
            bg_f,
            drift_c_f,
            drift_f_f,
            tv_c_f,
            tv_f_f,
            v_bg_tex_loss.contiguous(),
            v_grids_drift_loss.contiguous(),
            v_grid_camera_tv_loss.contiguous(),
            v_grid_frame_tv_loss.contiguous(),
            v_bg_tex,
            v_grids_camera,
            v_grids_frame,
        )

        return (
            v_bg_tex if ctx.has_bg_tex else None,
            None,  # bg_tex_depth
            v_grids_camera if ctx.has_grids_camera else None,
            v_grids_frame if ctx.has_grids_frame else None,
            None,
            None,
            None,
            None,
            None,  # 5 factors
        )


class _FusedSSIMLosses(torch.autograd.Function):
    """Fused invalid-pixel blend + Gaussian SSIM + masked reduction."""

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [B, H, W] int32
        pred: Tensor,  # [B, H, W, C]
        target: Tensor,  # [B, H, W, C]
        factor: float,
        mask_mode_target: bool,
        constant_mask_value: float,
    ) -> Tensor:
        ctx.set_materialize_grads(False)

        flags = flags.contiguous()
        pred = pred.contiguous()
        target = target.contiguous()
        B, H, W, C = pred.shape

        loss = torch.empty((B, H, W, 1), device=pred.device, dtype=pred.dtype)
        dm_dmu1 = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)
        dm_dsigma1_sq = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)
        dm_dsigma12 = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)

        _make_lazy_cuda_func("ssim_losses_fwd")(
            flags,
            pred,
            target,
            factor,
            mask_mode_target,
            constant_mask_value,
            loss,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        )

        ctx.save_for_backward(flags, pred, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.factor = factor
        ctx.mask_mode_target = mask_mode_target
        ctx.constant_mask_value = constant_mask_value
        return loss

    @staticmethod
    def backward(ctx, v_loss: Optional[Tensor]) -> Tuple[Optional[Tensor], ...]:
        flags, pred, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        if v_loss is None:
            v_pred = torch.zeros_like(pred)
        else:
            v_pred = torch.empty_like(pred)
            _make_lazy_cuda_func("ssim_losses_bwd")(
                flags,
                pred,
                target,
                ctx.factor,
                ctx.mask_mode_target,
                ctx.constant_mask_value,
                v_loss.contiguous(),
                dm_dmu1,
                dm_dsigma1_sq,
                dm_dsigma12,
                v_pred,
            )

        # Gradients for: flags, pred, target, factor, mask_mode_target,
        # constant_mask_value
        return None, v_pred, None, None, None, None
