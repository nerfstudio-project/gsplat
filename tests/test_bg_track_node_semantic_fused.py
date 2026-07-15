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

"""Tests for the fused CUDA background-in-track + node-semantic BCE losses.

The
:class:`~gsplat.losses_fused.FusedBgTrackNodeSemanticLosses` module wraps the
``gsplat::bg_track_node_semantic_losses_fwd`` / ``gsplat::bg_track_node_semantic_losses_bwd``
ops with a pure-PyTorch fallback (:func:`gsplat.losses.background_in_track_loss`
+ :func:`gsplat.losses.node_semantic_loss`) for CPU inputs; the module's
house-style ``_cuda_available`` flag lets tests force the fallback on CUDA
tensors, which is the parity reference here. An independent in-test
re-derivation of the kernel math additionally pins both paths so a shared
bug cannot slip through.

Semantics being pinned:

* Track boxes interpolate at the truncated midpoint of camera row 0,
  ``t = start + trunc((end - start) / 2)``. Tracks with <= 1 poses, a midpoint
  outside their timestamp span, or a degenerate bracketing span are invalid
  (they contain nothing). Containment is inclusive:
  ``|R_w2l @ (p - center)| <= dims / 2``.
* Background member (enabled iff ``background_lambda >= 0``): selected iff
  ``density_logit > density_logits_min`` AND the semantic argmax is allowed
  (per-segment ids in generic/packed mode with an unfiltered suffix past
  ``n_semantic_points``; ``background_allowed_class_ids`` over the shared
  domain in joint mode) AND inside any valid box. Raw loss is
  ``mean(softplus(logit))`` over selected points (exactly 0 when none);
  ``weighted = lambda * raw``.
* Node member (enabled iff ``node_lambda >= 0``): per predicate
  ``(class_ids, select_matches)``, selected iff
  ``(argmax in ids) == select_matches``; one pooled mean over primary + other
  selections. Empty ids + ``True`` selects NOTHING; empty ids + ``False``
  selects EVERYTHING (both legal). The joint path REQUIRES an explicit
  primary predicate.
* Gradients flow only to the two density-logit tensors:
  ``d raw / d logit_i = 1[selected_i] * sigmoid(logit_i) / count`` per member,
  members summing on shared (joint primary) points, and the weighted output
  contributing ``lambda`` times that.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from gsplat.losses_fused import FusedBgTrackNodeSemanticLosses


def _cuda_ops_available() -> bool:
    """Check that the fused bg-track + node-semantic native ops are compiled."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401

        return hasattr(
            torch.ops.gsplat, "bg_track_node_semantic_losses_fwd"
        ) and hasattr(torch.ops.gsplat, "bg_track_node_semantic_losses_bwd")
    except Exception:
        return False


CUDA_AVAILABLE = torch.cuda.is_available() and _cuda_ops_available()

DENSITY_LOGITS_MIN = -20.0
BG_LAMBDA = 0.125
NODE_LAMBDA = 1.7
DISABLED = -1.0

N_CLASSES = 40  # > 32 => the semantic class ids span two 32-bit mask words
CLS_PERSON = 1
CLS_TREE = 7
CLS_CAR = 32  # first bit of the second mask word
CLS_ROAD = 33  # >= 32 exercises multi-word class masks
_CLASS_CYCLE = (CLS_PERSON, CLS_ROAD, CLS_CAR, CLS_TREE, 2)

N_BG = 1027  # non-multiple of 256 exercises the grid boundary guard
N_OTHER = 771

# Production-scale point counts for the smoke tests (hundreds of thousands of
# background Gaussians per dispatch, many blocks per launch).
PROD_N_BG = 200_000
PROD_N_OTHER = 120_000

# 45 degrees about z for the rotated track box.
_ROT_QUAT = (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))

FWD_TOL = dict(atol=1e-5, rtol=1e-5)
GRAD_TOL = dict(atol=1e-4, rtol=1e-4)
FP64_TOL = dict(atol=1e-10, rtol=1e-10)
HAND_TOL = dict(atol=1e-6, rtol=1e-6)

_OUTPUT_LABELS = ("background_raw", "background_weighted", "node_raw", "node_weighted")


def _softplus_f(x: float) -> float:
    """Closed-form stable softplus for hand-computed expectations."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _sigmoid_f(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# In-test reference (independent re-derivation of the kernel math)
# ---------------------------------------------------------------------------


def _quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Normalized quaternion ``[4]`` in (x, y, z, w) order to ``[3, 3]``."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
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
        ]
    ).reshape(3, 3)


def _reference_track_boxes(inp):
    """Interpolated (R_world_to_local, center, half_dims) per VALID track.

    Mirrors ``interpolate_track_boxes_kernel``: truncated midpoint of camera
    row 0, invalid for single-pose tracks, midpoints outside the track's
    timestamp span, and degenerate spans. Fixtures only use two-pose tracks
    with a constant rotation, so pose interpolation reduces to a center lerp.
    """
    camera_ts = inp["camera_timestamps_startend_us"]
    packinfo = inp["tracks_packinfo"]
    poses = inp["tracks_poses"]
    ts = inp["tracks_timestamps_us"]
    dims = inp["cuboids_dims"]

    t_start = int(camera_ts[0, 0].item())
    t_end = int(camera_ts[0, 1].item())
    t_mid = t_start + (t_end - t_start) // 2  # fixtures keep end >= start

    boxes = []
    for track_idx in range(int(packinfo.shape[0])):
        start = int(packinfo[track_idx, 0].item())
        n_poses = int(packinfo[track_idx, 1].item())
        if n_poses <= 1:
            continue
        assert n_poses == 2, "fixtures use two-pose tracks"
        t0 = int(ts[start].item())
        t1 = int(ts[start + 1].item())
        if t_mid < t0 or t_mid > t1 or t1 == t0:
            continue
        alpha = float(t_mid - t0) / float(t1 - t0)
        pose0 = poses[start]
        pose1 = poses[start + 1]
        assert torch.equal(pose0[3:], pose1[3:]), "fixtures use constant rotations"
        center = (1.0 - alpha) * pose0[:3] + alpha * pose1[:3]
        r_world_to_local = _quat_xyzw_to_rotmat(pose0[3:]).transpose(0, 1)
        boxes.append((r_world_to_local, center, 0.5 * dims[track_idx]))
    return boxes


def _inside_any_box(positions: torch.Tensor, boxes) -> torch.Tensor:
    inside = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    for r_world_to_local, center, half_dims in boxes:
        local = (positions - center) @ r_world_to_local.transpose(0, 1)
        inside |= ((local >= -half_dims) & (local <= half_dims)).all(dim=1)
    return inside


def _class_isin(classes: torch.Tensor, class_ids) -> torch.Tensor:
    if len(class_ids) == 0:
        return torch.zeros_like(classes, dtype=torch.bool)
    ids = torch.tensor(tuple(class_ids), dtype=classes.dtype, device=classes.device)
    return torch.isin(classes, ids)


def _predicate_selected(semantic_logits, class_ids, select_matches):
    """selected = (argmax class in ids) == select_matches."""
    matches = _class_isin(semantic_logits.argmax(dim=1), class_ids)
    return matches if select_matches else ~matches


def _reference(inp, bg_lambda, node_lambda):
    """Reference forward: dict of losses plus the selection masks."""
    density_logits = inp["density_logits"]
    other_logits = inp["other_density_logits"]
    dtype = density_logits.dtype
    n_bg = density_logits.numel()
    is_joint = inp["n_semantic_points"] is None

    bg_selected = torch.zeros(n_bg, dtype=torch.bool, device=density_logits.device)
    if bg_lambda >= 0:
        boxes = _reference_track_boxes(inp)
        if is_joint:
            allowed = _class_isin(
                inp["semantic_logits"].argmax(dim=1),
                inp["background_allowed_class_ids"],
            )
        else:
            allowed = torch.zeros(n_bg, dtype=torch.bool, device=density_logits.device)
            begin = 0
            for (end, class_ids), segment_logits in zip(
                inp["background_segments"], inp["background_semantic_logits"]
            ):
                allowed[begin:end] = _class_isin(
                    segment_logits.argmax(dim=1), class_ids
                )
                begin = end
            allowed[inp["n_semantic_points"] :] = True  # unfiltered suffix
        bg_selected = (
            (density_logits > DENSITY_LOGITS_MIN)
            & allowed
            & _inside_any_box(inp["positions"], boxes)
        )
    bg_sel = bg_selected.to(dtype)
    bg_raw = (F.softplus(density_logits) * bg_sel).sum() / bg_sel.sum().clamp(min=1.0)

    node_primary_selected = torch.zeros(
        n_bg, dtype=torch.bool, device=density_logits.device
    )
    node_other_selected = torch.zeros(
        other_logits.numel(), dtype=torch.bool, device=other_logits.device
    )
    if node_lambda >= 0:
        if is_joint:
            predicate = inp["node_primary_predicate"]
            assert (
                predicate is not None
            ), "the joint node-semantic path requires an explicit predicate"
            node_primary_selected = _predicate_selected(
                inp["semantic_logits"], predicate[0], predicate[1]
            )
        begin = 0
        for (end, class_ids, select_matches), segment_logits in zip(
            inp["other_segments"], inp["other_semantic_logits"]
        ):
            node_other_selected[begin:end] = _predicate_selected(
                segment_logits, class_ids, select_matches
            )
            begin = end
    primary_sel = node_primary_selected.to(dtype)
    other_sel = node_other_selected.to(dtype)
    node_numerator = (F.softplus(density_logits) * primary_sel).sum() + (
        F.softplus(other_logits) * other_sel
    ).sum()
    node_count = primary_sel.sum() + other_sel.sum()
    node_raw = node_numerator / node_count.clamp(min=1.0)

    return {
        "outputs": (bg_raw, bg_lambda * bg_raw, node_raw, node_lambda * node_raw),
        "bg_selected": bg_selected,
        "node_primary_selected": node_primary_selected,
        "node_other_selected": node_other_selected,
    }


# ---------------------------------------------------------------------------
# Fixtures: synthetic tracks and point clouds (built on CPU, then moved)
# ---------------------------------------------------------------------------


def _make_tracks():
    """Five tracks: static axis-aligned, moving, rotated, plus two invalid.

    * track 0: static box at the origin, dims (2,2,2) => |xyz| <= 1 inclusive.
    * track 1: center moving x: -1 -> +1 over ts [0, 10]; at the camera
      midpoint t=5 it sits at (0, 4, 0) with half dims (0.5, 1, 1).
    * track 2: static box at (4, -4, 0) rotated 45 deg about z, ts [2, 8].
    * track 3: time-gated out (timestamps [100, 110] exclude midpoint 5).
    * track 4: single pose => always invalid.
    """
    sq, cq = _ROT_QUAT[2], _ROT_QUAT[3]
    poses = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, -4.0, 0.0, 0.0, 0.0, sq, cq],
            [4.0, -4.0, 0.0, 0.0, 0.0, sq, cq],
            [-4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    timestamps = torch.tensor([0, 10, 0, 10, 2, 8, 100, 110, 5], dtype=torch.int64)
    packinfo = torch.tensor([[0, 2], [2, 2], [4, 2], [6, 2], [8, 1]], dtype=torch.int32)
    dims = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 2.0],
            [2.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    return packinfo, poses, timestamps, dims


def _make_empty_tracks():
    return (
        torch.zeros((0, 2), dtype=torch.int32),
        torch.zeros((0, 7), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.int64),
        torch.zeros((0, 3), dtype=torch.float32),
    )


def _camera_timestamps():
    # Only row 0 is read (midpoint 5); row 1 would invalidate every track if
    # an implementation wrongly consumed it, so parity would catch that.
    return torch.tensor([[4, 6], [100, 200]], dtype=torch.int64)


def _cycle_classes(n, offset=0):
    return torch.tensor(
        [_CLASS_CYCLE[(i + offset) % len(_CLASS_CYCLE)] for i in range(n)],
        dtype=torch.long,
    )


def _one_hot_logits(classes, n_classes):
    logits = torch.full((classes.numel(), n_classes), -7.0, dtype=torch.float32)
    logits[torch.arange(classes.numel()), classes] = 7.0
    return logits


def _background_cloud(n=N_BG):
    """Positions + density logits + argmax classes with pinned containment.

    Pinned rows (track geometry from :func:`_make_tracks`):

    * 0: exactly on track 0's +x face (inclusive containment), person class.
    * 1: just outside track 0.
    * 2: center of the moving track at t=5 (inside).
    * 3: center of the time-gated track (must stay OUTSIDE: invalid track).
    * 4: center of the single-pose track (must stay OUTSIDE: invalid track).
    * 5..24: interior cluster of track 0 (rows 6..9 forced below
      ``density_logits_min`` to pin the density gate on allowed classes).
    * 25..34: interior of the rotated track 2 box.
    * 35..39: interior of the moving track 1 box.
    * 950..959: interior cluster of track 0 (lands in the second-segment /
      unfiltered region of the generic-mode scenarios).
    """
    generator = torch.Generator().manual_seed(20260708)
    positions = (torch.rand(n, 3, generator=generator) - 0.5) * 12.0
    positions[0] = torch.tensor([1.0, 0.0, 0.0])
    positions[1] = torch.tensor([1.001, 0.0, 0.0])
    positions[2] = torch.tensor([0.0, 4.0, 0.0])
    positions[3] = torch.tensor([-4.0, 4.0, 0.0])
    positions[4] = torch.tensor([-4.0, -4.0, 0.0])
    positions[5:25] = (torch.rand(20, 3, generator=generator) - 0.5) * 1.6
    rotated_local = (torch.rand(10, 3, generator=generator) - 0.5) * torch.tensor(
        [1.6, 0.8, 1.6]
    )
    rotation = _quat_xyzw_to_rotmat(torch.tensor(_ROT_QUAT))
    positions[25:35] = torch.tensor([4.0, -4.0, 0.0]) + rotated_local @ rotation.t()
    positions[35:40, 0] = (torch.rand(5, generator=generator) - 0.5) * 0.6
    positions[35:40, 1] = 4.0 + (torch.rand(5, generator=generator) - 0.5) * 1.2
    positions[35:40, 2] = (torch.rand(5, generator=generator) - 0.5) * 1.2
    positions[950:960] = (torch.rand(10, 3, generator=generator) - 0.5) * 1.6

    density_logits = torch.randn(n, generator=generator) * 2.0
    density_logits[0] = 2.0
    density_logits[6:10] = -25.0  # below DENSITY_LOGITS_MIN despite containment

    classes = _cycle_classes(n)
    classes[0] = CLS_PERSON
    return positions, density_logits, classes


def _other_cloud(n=N_OTHER):
    generator = torch.Generator().manual_seed(31415926)
    density_logits = torch.randn(n, generator=generator) * 2.0
    classes = _cycle_classes(n, offset=3)
    return density_logits, classes


def _base_inputs():
    packinfo, poses, timestamps, dims = _make_tracks()
    return {
        "camera_timestamps_startend_us": _camera_timestamps(),
        "tracks_packinfo": packinfo,
        "tracks_poses": poses,
        "tracks_timestamps_us": timestamps,
        "cuboids_dims": dims,
        "semantic_logits": None,
        "background_allowed_class_ids": (),
        "node_primary_predicate": None,
        "n_semantic_points": None,
        "background_semantic_logits": (),
        "background_segments": (),
        "other_density_logits": torch.zeros((0,), dtype=torch.float32),
        "other_semantic_logits": (),
        "other_segments": (),
    }


def _with_empty_tracks(inp):
    packinfo, poses, timestamps, dims = _make_empty_tracks()
    inp.update(
        tracks_packinfo=packinfo,
        tracks_poses=poses,
        tracks_timestamps_us=timestamps,
        cuboids_dims=dims,
    )
    return inp


# --- scenarios -------------------------------------------------------------
# Each returns (inputs_cpu_fp32, background_lambda, node_lambda,
# expect_bg_selected, expect_node_selected).


def _scenario_bg_only_filtered():
    """(1) Background-in-track alone via the filtered (generic) path.

    Two semantic segments with independent class counts; segment 1 configures
    a class id >= 32 so the generic path also walks the second mask word.
    """
    positions, density_logits, classes = _background_cloud()
    inp = _base_inputs()
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["background_semantic_logits"] = (
        _one_hot_logits(classes[:600], N_CLASSES),
        _one_hot_logits(classes[600:] % 8, 8),
    )
    inp["background_segments"] = ((600, (CLS_PERSON, CLS_ROAD)), (N_BG, (2, 7)))
    inp["n_semantic_points"] = N_BG
    return inp, BG_LAMBDA, DISABLED, True, False


def _scenario_node_only_use_exclude():
    """(2) Node-semantic alone: a labels_to_use primary + a labels_to_exclude
    other segment (joint path with the background member disabled)."""
    positions, density_logits, classes = _background_cloud()
    other_density, other_classes = _other_cloud()
    inp = _with_empty_tracks(_base_inputs())
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["semantic_logits"] = _one_hot_logits(classes, N_CLASSES)
    # labels_to_use => penalize argmax NOT in ids => select_matches=False.
    inp["node_primary_predicate"] = ((CLS_PERSON,), False)
    inp["other_density_logits"] = other_density
    inp["other_semantic_logits"] = (_one_hot_logits(other_classes, N_CLASSES),)
    # labels_to_exclude => penalize argmax in ids => select_matches=True.
    inp["other_segments"] = ((N_OTHER, (CLS_ROAD,), True),)
    return inp, DISABLED, NODE_LAMBDA, False, True


def _scenario_joint_shared_primary():
    """(3) Joint mode: bg-in-track on the shared primary domain plus
    node-semantic on primary (exclude road) and the road node (use road).

    A focused mixed case: bg-track uses ``{background: [person]}``,
    node-semantic uses ``{road: [road]}`` and excludes ``{background:
    [road]}``. Class id 33 makes the joint launch multi-word.
    """
    positions, density_logits, classes = _background_cloud()
    other_density, other_classes = _other_cloud()
    inp = _base_inputs()
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["semantic_logits"] = _one_hot_logits(classes, N_CLASSES)
    inp["background_allowed_class_ids"] = (CLS_PERSON,)
    inp["node_primary_predicate"] = ((CLS_ROAD,), True)
    inp["other_density_logits"] = other_density
    inp["other_semantic_logits"] = (_one_hot_logits(other_classes, N_CLASSES),)
    inp["other_segments"] = ((N_OTHER, (CLS_ROAD,), False),)
    return inp, BG_LAMBDA, NODE_LAMBDA, True, True


def _scenario_select_none_pin():
    """(4) SELECT-NONE pin: the bg-track primary layer is not
    node-semantic-configured, so the explicit ``((), True)`` predicate must
    keep every primary point out of the node loss."""
    inp, _, _, _, _ = _scenario_joint_shared_primary()
    inp["node_primary_predicate"] = ((), True)
    return inp, BG_LAMBDA, NODE_LAMBDA, True, True


def _scenario_select_all_pin():
    """(5a) SELECT-ALL pin: an explicitly empty labels_to_use list means
    "use nothing" => predicate ``((), False)`` penalizes EVERY primary point.

    No other segments: the launch stays single-word, and primary points with
    argmax >= 32 pin the first-word class-range guard."""
    positions, density_logits, classes = _background_cloud()
    inp = _with_empty_tracks(_base_inputs())
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["semantic_logits"] = _one_hot_logits(classes, N_CLASSES)
    inp["node_primary_predicate"] = ((), False)
    return inp, DISABLED, NODE_LAMBDA, False, True


def _scenario_select_all_pin_multiword():
    """(5b) SELECT-ALL under multi-word launches: an other segment configured
    with class id 33 (>= 32) forces the multi-word kernel loop while the
    empty ``((), False)`` predicate must still select every primary point
    exactly once across the word launches."""
    inp, _, node_lambda, _, _ = _scenario_select_all_pin()
    # Rows 6:10 pin the density gate at -25 for the track scenarios, but the
    # kernel's parity-pinned fp32 tanh-form sigmoid flushes to an exact zero
    # there — and whether a GPU's approx-tanh saturates at that magnitude is
    # arch-dependent. Tracks are empty here, so lift those rows into a range
    # where "selected => strictly nonzero gradient" holds on every arch.
    inp["density_logits"][6:10] = -2.5
    other_density, other_classes = _other_cloud()
    inp["other_density_logits"] = other_density
    inp["other_semantic_logits"] = (_one_hot_logits(other_classes, N_CLASSES),)
    inp["other_segments"] = ((N_OTHER, (CLS_ROAD,), True),)
    return inp, DISABLED, node_lambda, False, True


def _scenario_packed_unfiltered():
    """(6) Packed mode: generic bg-track with an unfiltered suffix
    (``n_semantic_points < Nbg``, covering layers without semantic logits)
    plus packed node-semantic segments."""
    positions, density_logits, classes = _background_cloud()
    other_density, other_classes = _other_cloud()
    inp = _base_inputs()
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["background_semantic_logits"] = (_one_hot_logits(classes[:600], N_CLASSES),)
    inp["background_segments"] = ((600, (CLS_PERSON, CLS_ROAD)),)
    inp["n_semantic_points"] = 600  # rows 600.. are unfiltered
    inp["other_density_logits"] = other_density
    inp["other_semantic_logits"] = (
        _one_hot_logits(other_classes[:400], N_CLASSES),
        _one_hot_logits(other_classes[400:], N_CLASSES),
    )
    inp["other_segments"] = (
        (400, (CLS_ROAD,), False),
        (N_OTHER, (CLS_TREE, CLS_ROAD), True),
    )
    return inp, BG_LAMBDA, NODE_LAMBDA, True, True


def _scenario_empty_tracks():
    """(7a) T == 0: the background member must produce a hard zero."""
    inp, bg_lambda, node_lambda, _, _ = _scenario_bg_only_filtered()
    return _with_empty_tracks(inp), bg_lambda, node_lambda, False, False


def _scenario_zero_points():
    """(7b) Nbg == 0 with real tracks: a hard zero on the generic path."""
    inp = _base_inputs()
    inp["positions"] = torch.zeros((0, 3), dtype=torch.float32)
    inp["density_logits"] = torch.zeros((0,), dtype=torch.float32)
    inp["n_semantic_points"] = 0
    return inp, BG_LAMBDA, DISABLED, False, False


def _scenario_joint_empty_other():
    """(7c) Joint mode with an empty packed-other domain: the node loss comes
    from the primary predicate alone."""
    inp, bg_lambda, node_lambda, _, _ = _scenario_joint_shared_primary()
    inp["other_density_logits"] = torch.zeros((0,), dtype=torch.float32)
    inp["other_semantic_logits"] = ()
    inp["other_segments"] = ()
    return inp, bg_lambda, node_lambda, True, True


def _scenario_production_scale():
    """(8) Production-scale joint dispatch: both members enabled over hundreds
    of thousands of points per domain (kept out of ``_SCENARIOS`` so only the
    dedicated smoke tests pay for it)."""
    positions, density_logits, classes = _background_cloud(PROD_N_BG)
    other_density, other_classes = _other_cloud(PROD_N_OTHER)
    inp = _base_inputs()
    inp["positions"] = positions
    inp["density_logits"] = density_logits
    inp["semantic_logits"] = _one_hot_logits(classes, N_CLASSES)
    inp["background_allowed_class_ids"] = (CLS_PERSON,)
    inp["node_primary_predicate"] = ((CLS_ROAD,), True)
    inp["other_density_logits"] = other_density
    inp["other_semantic_logits"] = (_one_hot_logits(other_classes, N_CLASSES),)
    inp["other_segments"] = ((PROD_N_OTHER, (CLS_ROAD,), False),)
    return inp, BG_LAMBDA, NODE_LAMBDA, True, True


_SCENARIOS = {
    "bg_only_filtered": _scenario_bg_only_filtered,
    "node_only_use_exclude": _scenario_node_only_use_exclude,
    "joint_shared_primary": _scenario_joint_shared_primary,
    "select_none_pin": _scenario_select_none_pin,
    "select_all_pin": _scenario_select_all_pin,
    "select_all_pin_multiword": _scenario_select_all_pin_multiword,
    "packed_unfiltered": _scenario_packed_unfiltered,
    "empty_tracks": _scenario_empty_tracks,
    "zero_points": _scenario_zero_points,
    "joint_empty_other": _scenario_joint_empty_other,
}
_SCENARIO_IDS = tuple(_SCENARIOS)


# ---------------------------------------------------------------------------
# Module-facing helpers
# ---------------------------------------------------------------------------


def _move_inputs(inp, device, dtype=torch.float32):
    """Move fixture tensors to (device, dtype); ints keep their dtype."""

    def convert(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if (
            isinstance(value, tuple)
            and len(value) > 0
            and isinstance(value[0], torch.Tensor)
        ):
            return tuple(convert(item) for item in value)
        return value

    return {key: convert(value) for key, value in inp.items()}


def _make_module():
    return FusedBgTrackNodeSemanticLosses(density_logits_min=DENSITY_LOGITS_MIN)


def _fallback_module():
    """A module forced onto the pure-PyTorch path (the parity reference)."""
    module = _make_module()
    assert hasattr(module, "_cuda_available"), (
        "FusedBgTrackNodeSemanticLosses must expose the house-style "
        "_cuda_available flag so tests can force the pure-PyTorch fallback"
    )
    module._cuda_available = False
    return module


def _call(module, inp, bg_lambda, node_lambda):
    """Single adaptation point for the module forward signature."""
    return module(**inp, background_lambda=bg_lambda, node_lambda=node_lambda)


def _assert_outputs_match(actual, expected, tol):
    actual = tuple(actual)
    expected = tuple(expected)
    assert len(actual) == 4, f"expected 4 outputs, got {len(actual)}"
    for label, actual_value, expected_value in zip(_OUTPUT_LABELS, actual, expected):
        torch.testing.assert_close(
            actual_value.reshape(()),
            expected_value.reshape(()),
            **tol,
            msg=lambda m, label=label: f"{label} mismatch: {m}",
        )


def _objective(outputs, bg_lambda, node_lambda):
    """Backward driver: distinct raw/weighted upstream coefficients
    exercise both gradient slots of each enabled member."""
    bg_raw, bg_weighted, node_raw, node_weighted = outputs
    terms = []
    if bg_lambda >= 0:
        terms.append(2.0 * bg_raw.reshape(()))
        terms.append(3.0 * bg_weighted.reshape(()))
    if node_lambda >= 0:
        terms.append(5.0 * node_raw.reshape(()))
        terms.append(7.0 * node_weighted.reshape(()))
    return torch.stack(terms).sum()


def _with_grad_leaves(inp):
    """Fresh grad leaves for the two differentiable inputs."""
    out = dict(inp)
    leaves = []
    for key in ("density_logits", "other_density_logits"):
        leaf = inp[key].detach().clone().requires_grad_(True)
        out[key] = leaf
        leaves.append(leaf)
    return out, leaves


def _grads_of(objective, leaves):
    if not objective.requires_grad:
        return tuple(torch.zeros_like(leaf) for leaf in leaves)
    grads = torch.autograd.grad(objective, leaves, allow_unused=True)
    return tuple(
        torch.zeros_like(leaf) if grad is None else grad
        for grad, leaf in zip(grads, leaves)
    )


def _noncontiguous_view(t):
    """The same values as ``t`` seen through a stride-doubled view."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return t
    view = torch.repeat_interleave(t.unsqueeze(-1), 2, dim=-1)[..., 0]
    assert not view.is_contiguous()
    return view


def _with_noncontiguous_inputs(inp):
    """Non-contiguous views of every non-empty per-point tensor, plus fresh
    grad leaves for the two differentiable inputs (each view stays in its
    leaf's graph, so gradients land on ordinary contiguous leaf storage)."""
    out = dict(inp)
    for key in ("positions", "semantic_logits"):
        if isinstance(out.get(key), torch.Tensor):
            out[key] = _noncontiguous_view(out[key])
    out["background_semantic_logits"] = tuple(
        _noncontiguous_view(t) for t in out["background_semantic_logits"]
    )
    out["other_semantic_logits"] = tuple(
        _noncontiguous_view(t) for t in out["other_semantic_logits"]
    )
    leaves = []
    for key in ("density_logits", "other_density_logits"):
        leaf = inp[key].detach().clone().requires_grad_(True)
        out[key] = _noncontiguous_view(leaf)
        leaves.append(leaf)
    return out, leaves


def _assert_expected_selection(reference, expect_bg, expect_node):
    """Guard against vacuous parity: enabled members must select something."""
    if expect_bg:
        assert int(reference["bg_selected"].sum()) > 0
    if expect_node:
        node_count = int(reference["node_primary_selected"].sum()) + int(
            reference["node_other_selected"].sum()
        )
        assert node_count > 0


# ---------------------------------------------------------------------------
# Miniature fixtures with hand-computed expectations
# ---------------------------------------------------------------------------


def _miniature_bg_inputs(device):
    """One static unit-half box; every selection gate pinned by hand.

    Points (class ids in a 4-class space, allowed ids = (1,)):

    * (0.5, 0, 0)  logit  2.0 class 1 -> SELECTED
    * (0.5, .2, 0) logit -25  class 1 -> density-gated out
    * (0.5,-.2, 0) logit  1.0 class 2 -> semantic-gated out
    * (5, 0, 0)    logit  3.0 class 1 -> outside the box
    * (-0.5, 0, 0) logit -1.0 class 1 -> SELECTED

    bg raw loss = (softplus(2) + softplus(-1)) / 2.
    """
    inp = _base_inputs()
    inp["tracks_packinfo"] = torch.tensor([[0, 2]], dtype=torch.int32)
    inp["tracks_poses"] = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32
    )
    inp["tracks_timestamps_us"] = torch.tensor([0, 10], dtype=torch.int64)
    inp["cuboids_dims"] = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    inp["positions"] = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, -0.2, 0.0],
            [5.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    inp["density_logits"] = torch.tensor(
        [2.0, -25.0, 1.0, 3.0, -1.0], dtype=torch.float32
    )
    classes = torch.tensor([1, 1, 2, 1, 1], dtype=torch.long)
    inp["background_semantic_logits"] = (_one_hot_logits(classes, 4),)
    inp["background_segments"] = ((5, (1,)),)
    inp["n_semantic_points"] = 5
    return _move_inputs(inp, device)


def _miniature_node_inputs(device, predicate=((CLS_PERSON,), False), with_other=True):
    """Three primary points (classes 1, 33, 1) + two other points (7, 33)."""
    inp = _with_empty_tracks(_base_inputs())
    inp["positions"] = torch.zeros((3, 3), dtype=torch.float32)
    inp["density_logits"] = torch.tensor([0.5, -0.25, 1.5], dtype=torch.float32)
    inp["semantic_logits"] = _one_hot_logits(
        torch.tensor([CLS_PERSON, CLS_ROAD, CLS_PERSON]), N_CLASSES
    )
    inp["node_primary_predicate"] = predicate
    if with_other:
        inp["other_density_logits"] = torch.tensor([0.75, -2.0], dtype=torch.float32)
        inp["other_semantic_logits"] = (
            _one_hot_logits(torch.tensor([CLS_TREE, CLS_ROAD]), N_CLASSES),
        )
        inp["other_segments"] = ((2, (CLS_ROAD,), True),)
    return _move_inputs(inp, device)


# ---------------------------------------------------------------------------
# Track-box slerp interpolation: nontrivially different pose quaternions,
# pinned by a torch/scipy-free pure-Python fp64 reference
# ---------------------------------------------------------------------------


def _axis_angle_quat(axis, angle_deg):
    """Unit quaternion ``(x, y, z, w)`` from axis + angle, pure Python."""
    norm = math.sqrt(sum(a * a for a in axis))
    half = 0.5 * math.radians(angle_deg)
    s = math.sin(half) / norm
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def _quat_neg(q):
    return tuple(-c for c in q)


def _quat_dot(q0, q1):
    return sum(a * b for a, b in zip(q0, q1))


def _slerp_reference(q0, q1, t):
    """Textbook shortest-arc slerp in pure-Python fp64.

    Independent of the kernel's formulation: exact ``sin`` ratio weights
    with a normalized-lerp fallback only for numerically degenerate
    ``sin(omega)``. The kernel's shared geometry helper (sin-ratio weights
    with a wider normalized-lerp branch above ``cos(omega) >
    _NLERP_DOT_THRESHOLD``) agrees with this reference far below the probe
    margins used by the tests.
    """
    dot = _quat_dot(q0, q1)
    if dot < 0.0:
        q1 = _quat_neg(q1)
        dot = -dot
    dot = min(dot, 1.0)
    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    if sin_omega < 1e-9:
        w0, w1 = 1.0 - t, t
    else:
        w0 = math.sin((1.0 - t) * omega) / sin_omega
        w1 = math.sin(t * omega) / sin_omega
    out = tuple(w0 * a + w1 * b for a, b in zip(q0, q1))
    norm = math.sqrt(sum(c * c for c in out))
    return tuple(c / norm for c in out)


def _quat_rotmat_rows(q):
    """Row-major local-to-world rotation ``R(q)``, pure-Python fp64."""
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


# Nearby-quaternion cutoff of the shared geometry helper the kernel and the
# fallback both delegate to (gsplat_geometry::kSlerpSmallAngleDotThreshold).
_NLERP_DOT_THRESHOLD = 0.9995

# Probe margin around the interpolated box faces. Far above every acceptable
# numerical divergence between implementations (the helper's nlerp branch vs
# the reference's exact slerp above the cutoff: <= ~1e-8 here) and far below
# any real math error (wrong branch, missing flip, wrong bracket: >= ~1e-2).
_SLERP_PROBE_MARGIN = 1e-4

# Non-cubic half dims so every rotation error is observable on some face.
_SLERP_BOX_DIMS = (2.0, 1.2, 0.6)


def _slerp_cases():
    """Track fixtures exercising every slerp branch nontrivially.

    * ``sign_flip``: raw pose quaternions with a negative dot product pin
      the shortest-arc flip.
    * ``sign_flip_off_mid``: same coaxial negative-dot pair sampled at
      ``alpha = 0.25``. This one actually FAILS if the flip is removed:
      at ``alpha = 0.5`` the long-way result is off by exactly 180 deg
      about the box's own z axis, and a cuboid is symmetric under that,
      so containment cannot tell — off-midpoint the error becomes a
      90 deg local-z rotation, which swaps the unequal x/y half extents.
    * ``sign_flip_skew_axes``: negative dot with distinct rotation axes
      at ``alpha = 0.3``, so the long-way result is not related to the
      correct one by any box symmetry.
    * ``near_parallel_lerp``: ``cos(omega) > _NLERP_DOT_THRESHOLD`` pins
      the helper's normalized-lerp branch, at ``alpha = 0.25`` where lerp
      and slerp are not trivially identical (they are at 0.5).
    * ``nlerp_threshold_band``: ``cos(omega)`` inside ``(1 - 1e-2,
      _NLERP_DOT_THRESHOLD)`` with a tightened probe margin, so widening
      the normalized-lerp acceptance to ``1 - 1e-2`` (or more) routes
      this case through nlerp and moves the faces past the probes.
      Together with ``near_parallel_lerp`` this pins the branch threshold
      from both sides.
    * ``generic_mid_interval``: distinct rotation axes at ``alpha = 0.3``
      pin the generic ``sin``-path.
    * ``three_pose_second_interval``: a three-pose track whose camera
      midpoint falls in the second interval pins the bracketing binary
      search together with a nontrivial slerp.
    """
    return {
        "sign_flip": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((0.0, 0.0, 1.0), 40.0)),
                (
                    (0.4, -0.2, 0.6),
                    _quat_neg(_axis_angle_quat((0.0, 0.0, 1.0), 100.0)),
                ),
            ),
            timestamps_us=(0, 10),
            camera_startend_us=(4, 6),  # midpoint 5 -> alpha 0.5
            bracket=(0, 1),
            branch="negative_dot",
        ),
        "sign_flip_off_mid": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((0.0, 0.0, 1.0), 40.0)),
                (
                    (0.4, -0.2, 0.6),
                    _quat_neg(_axis_angle_quat((0.0, 0.0, 1.0), 100.0)),
                ),
            ),
            timestamps_us=(0, 12),
            camera_startend_us=(2, 4),  # midpoint 3 -> alpha 0.25
            bracket=(0, 1),
            branch="negative_dot",
        ),
        "sign_flip_skew_axes": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((1.0, 2.0, 3.0), 50.0)),
                (
                    (0.5, -0.4, 0.3),
                    _quat_neg(_axis_angle_quat((-2.0, 1.0, 1.0), 120.0)),
                ),
            ),
            timestamps_us=(0, 10),
            camera_startend_us=(2, 4),  # midpoint 3 -> alpha 0.3
            bracket=(0, 1),
            branch="negative_dot",
        ),
        "near_parallel_lerp": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((0.0, 0.0, 1.0), 10.0)),
                ((0.6, 0.3, -0.2), _axis_angle_quat((0.0, 0.0, 1.0), 10.6)),
            ),
            timestamps_us=(0, 12),
            camera_startend_us=(2, 4),  # midpoint 3 -> alpha 0.25
            bracket=(0, 1),
            branch="near_parallel",
        ),
        "nlerp_threshold_band": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((0.0, 0.0, 1.0), 15.0)),
                ((0.6, 0.3, -0.2), _axis_angle_quat((0.0, 0.0, 1.0), 27.0)),
            ),
            timestamps_us=(0, 12),
            camera_startend_us=(2, 4),  # midpoint 3 -> alpha 0.25
            bracket=(0, 1),
            branch="threshold_band",
            # nlerp-vs-slerp face displacement here is ~4e-5 (quat dot
            # cos(6 deg) ~ 0.9945, alpha 0.25, ~1 m lever), so the default
            # 1e-4 margin would not notice a widened nlerp acceptance;
            # 1e-5 does, while staying orders of magnitude above the
            # correct slerp path's fp64 divergence from the reference.
            probe_margin=1e-5,
        ),
        "generic_mid_interval": dict(
            poses=(
                ((0.0, 0.0, 0.0), _axis_angle_quat((1.0, 2.0, 3.0), 50.0)),
                ((0.5, -0.4, 0.3), _axis_angle_quat((-2.0, 1.0, 1.0), 120.0)),
            ),
            timestamps_us=(0, 10),
            camera_startend_us=(2, 4),  # midpoint 3 -> alpha 0.3
            bracket=(0, 1),
            branch="generic",
        ),
        "three_pose_second_interval": dict(
            poses=(
                ((5.0, 5.0, 5.0), _axis_angle_quat((1.0, 0.0, 0.0), 0.0)),
                ((0.0, 0.0, 0.0), _axis_angle_quat((0.0, 0.0, 1.0), 90.0)),
                ((1.0, 0.5, -0.25), _axis_angle_quat((1.0, 1.0, 0.0), 60.0)),
            ),
            timestamps_us=(0, 10, 20),
            camera_startend_us=(14, 16),  # midpoint 15 -> poses 1..2, alpha 0.5
            bracket=(1, 2),
            branch="generic",
        ),
    }


_SLERP_CASE_IDS = tuple(_slerp_cases())


def _slerp_case_inputs(case):
    """Build a single-track fixture whose probe points straddle the faces of
    the reference-interpolated box; returns ``(inputs_fp64, expected_inside,
    expected_raw)``."""
    poses = case["poses"]
    timestamps = case["timestamps_us"]
    cam_start, cam_end = case["camera_startend_us"]
    i0, i1 = case["bracket"]

    # Reference bracketing per the documented semantics (truncated midpoint,
    # right-bound search) — asserted, not searched, so a kernel bracket bug
    # cannot silently steer the reference.
    t_mid = cam_start + (cam_end - cam_start) // 2
    t0, t1 = timestamps[i0], timestamps[i1]
    assert i1 == i0 + 1 and t0 < t_mid < t1, "fixture must pin one interval"
    assert timestamps[0] <= t_mid <= timestamps[-1]
    alpha = (t_mid - t0) / (t1 - t0)

    q0, q1 = poses[i0][1], poses[i1][1]
    dot = _quat_dot(q0, q1)
    if case["branch"] == "negative_dot":
        assert dot < 0.0 and -dot < _NLERP_DOT_THRESHOLD, "must pin the sign flip"
    elif case["branch"] == "near_parallel":
        assert dot > _NLERP_DOT_THRESHOLD, "must pin the normalized-lerp branch"
    elif case["branch"] == "threshold_band":
        assert 1.0 - 1e-2 < dot < _NLERP_DOT_THRESHOLD, "must sit below the cutoff"
    else:
        assert 0.0 < dot < _NLERP_DOT_THRESHOLD, "must pin the generic sin branch"

    q_ref = _slerp_reference(q0, q1, alpha)
    c0, c1 = poses[i0][0], poses[i1][0]
    center = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(c0, c1))
    rot = _quat_rotmat_rows(q_ref)
    half = tuple(0.5 * d for d in _SLERP_BOX_DIMS)

    margin = case.get("probe_margin", _SLERP_PROBE_MARGIN)
    positions = []
    expected_inside = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            for factor, inside in (
                (1.0 - margin, True),
                (1.0 + margin, False),
            ):
                local = [0.0, 0.0, 0.0]
                local[axis] = sign * factor * half[axis]
                positions.append(
                    tuple(
                        center[row]
                        + sum(rot[row][col] * local[col] for col in range(3))
                        for row in range(3)
                    )
                )
                expected_inside.append(inside)
    # Corner probes: inside on every axis / outside on every axis.
    for factor, inside in ((0.99, True), (1.0 + margin, False)):
        local = [factor * h for h in half]
        positions.append(
            tuple(
                center[row] + sum(rot[row][col] * local[col] for col in range(3))
                for row in range(3)
            )
        )
        expected_inside.append(inside)
    # All-sign corner probes at (1 -/+ margin). Face-center probes move
    # tangentially under a small rotation error (first-order invisible to
    # containment), while a corner picks up a first-order perpendicular
    # component on the neighboring axes — these are what actually pin
    # small-angle errors such as a widened normalized-lerp acceptance.
    for factor, inside in ((1.0 - margin, True), (1.0 + margin, False)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                for sz in (1.0, -1.0):
                    local = [
                        sx * factor * half[0],
                        sy * factor * half[1],
                        sz * factor * half[2],
                    ]
                    positions.append(
                        tuple(
                            center[row]
                            + sum(rot[row][col] * local[col] for col in range(3))
                            for row in range(3)
                        )
                    )
                    expected_inside.append(inside)

    n_probes = len(positions)
    density_logits = [0.4 + 0.15 * i for i in range(n_probes)]
    selected_softplus = [
        _softplus_f(logit)
        for logit, inside in zip(density_logits, expected_inside)
        if inside
    ]
    expected_raw = sum(selected_softplus) / len(selected_softplus)

    inp = _base_inputs()
    inp["tracks_packinfo"] = torch.tensor([[0, len(poses)]], dtype=torch.int32)
    inp["tracks_poses"] = torch.tensor(
        [list(c) + list(q) for c, q in poses], dtype=torch.float64
    )
    inp["tracks_timestamps_us"] = torch.tensor(list(timestamps), dtype=torch.int64)
    inp["cuboids_dims"] = torch.tensor([list(_SLERP_BOX_DIMS)], dtype=torch.float64)
    inp["camera_timestamps_startend_us"] = torch.tensor(
        [[cam_start, cam_end]], dtype=torch.int64
    )
    inp["positions"] = torch.tensor(positions, dtype=torch.float64)
    inp["density_logits"] = torch.tensor(density_logits, dtype=torch.float64)
    inp["n_semantic_points"] = 0  # fully unfiltered background domain
    return inp, expected_inside, expected_raw


def _run_slerp_case(case_key, device):
    """Forward + backward on (device); selection and mean pinned to the
    pure-Python fp64 reference."""
    inp, expected_inside, expected_raw = _slerp_case_inputs(_slerp_cases()[case_key])
    inp = _move_inputs(inp, device, dtype=torch.float64)
    inp, (bg_leaf, _) = _with_grad_leaves(inp)
    outputs = _call(_make_module(), inp, 1.0, DISABLED)

    raw = outputs[0].reshape(())
    torch.testing.assert_close(
        raw.cpu(),
        torch.tensor(expected_raw, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    # The gradient support is exactly the reference-selected probe set.
    (grad,) = _grads_of(raw, (bg_leaf,))
    assert (grad != 0).cpu().tolist() == expected_inside


class TestTrackBoxSlerpInterpolation:
    """Pose-pair quaternions genuinely differ, so the interpolated box only
    lands where a correct slerp puts it: sign-flip, normalized-lerp, and
    generic branches plus multi-pose bracketing, against a torch/scipy-free
    fp64 reference."""

    @pytest.mark.parametrize("case", _SLERP_CASE_IDS)
    def test_fallback_matches_slerp_reference(self, case):
        _run_slerp_case(case, "cpu")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
    @pytest.mark.parametrize("case", _SLERP_CASE_IDS)
    def test_cuda_matches_slerp_reference(self, case):
        _run_slerp_case(case, "cuda")


# ---------------------------------------------------------------------------
# CPU fallback tests — always runnable
# ---------------------------------------------------------------------------


class TestBgTrackNodeSemanticFallback:
    """Pure-PyTorch (CPU) fallback vs hand-computed cases and the in-test
    reference re-derivation of the kernel semantics."""

    def test_miniature_bg_track_hand_computed(self):
        inp = _miniature_bg_inputs("cpu")
        inp, (bg_leaf, _) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, BG_LAMBDA, DISABLED)

        expected = (_softplus_f(2.0) + _softplus_f(-1.0)) / 2.0
        torch.testing.assert_close(
            outputs[0].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        torch.testing.assert_close(
            outputs[1].reshape(()),
            torch.tensor(BG_LAMBDA * expected, dtype=torch.float32),
            **HAND_TOL,
        )
        # Disabled node member returns exact zeros.
        assert outputs[2].reshape(()).item() == 0.0
        assert outputs[3].reshape(()).item() == 0.0

        # d raw / d logit = 1[selected] * sigmoid(logit) / count.
        (grad,) = _grads_of(outputs[0].reshape(()), (bg_leaf,))
        expected_grad = torch.tensor(
            [_sigmoid_f(2.0) / 2.0, 0.0, 0.0, 0.0, _sigmoid_f(-1.0) / 2.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(grad, expected_grad, **HAND_TOL)

    def test_miniature_node_semantic_use_and_exclude_hand_computed(self):
        """labels_to_use on the primary (penalize NOT person) + a
        labels_to_exclude other segment (penalize road)."""
        inp = _miniature_node_inputs("cpu")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, DISABLED, NODE_LAMBDA)

        # Primary: class 33 is not "person" -> selected (softplus(-0.25)).
        # Other: class 33 is excluded -> selected (softplus(-2.0)).
        expected = (_softplus_f(-0.25) + _softplus_f(-2.0)) / 2.0
        torch.testing.assert_close(
            outputs[2].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        torch.testing.assert_close(
            outputs[3].reshape(()),
            torch.tensor(NODE_LAMBDA * expected, dtype=torch.float32),
            **HAND_TOL,
        )

        bg_grad, other_grad = _grads_of(outputs[2].reshape(()), (bg_leaf, other_leaf))
        torch.testing.assert_close(
            bg_grad,
            torch.tensor([0.0, _sigmoid_f(-0.25) / 2.0, 0.0], dtype=torch.float32),
            **HAND_TOL,
        )
        torch.testing.assert_close(
            other_grad,
            torch.tensor([0.0, _sigmoid_f(-2.0) / 2.0], dtype=torch.float32),
            **HAND_TOL,
        )

    def test_miniature_select_all_hand_computed(self):
        """Explicitly empty labels_to_use => ((), False) penalizes EVERY
        primary point (including the class-33 one)."""
        inp = _miniature_node_inputs("cpu", predicate=((), False), with_other=False)
        outputs = _call(_make_module(), inp, DISABLED, NODE_LAMBDA)
        expected = (_softplus_f(0.5) + _softplus_f(-0.25) + _softplus_f(1.5)) / 3.0
        torch.testing.assert_close(
            outputs[2].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )

    def test_miniature_select_none_hand_computed(self):
        """((), True) keeps every primary point out; only the excluded road
        point of the other segment contributes."""
        inp = _miniature_node_inputs("cpu", predicate=((), True))
        outputs = _call(_make_module(), inp, DISABLED, NODE_LAMBDA)
        expected = _softplus_f(-2.0)
        torch.testing.assert_close(
            outputs[2].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )

    @pytest.mark.parametrize("scenario", _SCENARIO_IDS)
    def test_fallback_matches_reference_forward(self, scenario):
        inp, bg_lambda, node_lambda, expect_bg, expect_node = _SCENARIOS[scenario]()
        inp = _move_inputs(inp, "cpu")
        reference = _reference(inp, bg_lambda, node_lambda)
        _assert_expected_selection(reference, expect_bg, expect_node)

        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        _assert_outputs_match(outputs, reference["outputs"], FWD_TOL)

    @pytest.mark.parametrize("scenario", _SCENARIO_IDS)
    def test_fallback_backward_matches_reference(self, scenario):
        inp, bg_lambda, node_lambda, _, _ = _SCENARIOS[scenario]()
        inp = _move_inputs(inp, "cpu")

        module_inp, module_leaves = _with_grad_leaves(inp)
        module_grads = _grads_of(
            _objective(
                _call(_make_module(), module_inp, bg_lambda, node_lambda),
                bg_lambda,
                node_lambda,
            ),
            module_leaves,
        )

        ref_inp, ref_leaves = _with_grad_leaves(inp)
        ref_grads = _grads_of(
            _objective(
                _reference(ref_inp, bg_lambda, node_lambda)["outputs"],
                bg_lambda,
                node_lambda,
            ),
            ref_leaves,
        )

        for label, module_grad, ref_grad in zip(
            ("density_logits", "other_density_logits"), module_grads, ref_grads
        ):
            torch.testing.assert_close(
                module_grad,
                ref_grad,
                **FWD_TOL,
                msg=lambda m, label=label: f"{label} grad mismatch: {m}",
            )

    def test_empty_tracks_zero_loss_and_grads(self):
        inp, bg_lambda, node_lambda, _, _ = _scenario_empty_tracks()
        inp = _move_inputs(inp, "cpu")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        assert outputs[0].reshape(()).item() == 0.0
        assert outputs[1].reshape(()).item() == 0.0
        grads = _grads_of(
            _objective(outputs, bg_lambda, node_lambda), (bg_leaf, other_leaf)
        )
        assert torch.count_nonzero(grads[0]) == 0
        assert torch.count_nonzero(grads[1]) == 0

    def test_zero_points_zero_loss(self):
        inp, bg_lambda, node_lambda, _, _ = _scenario_zero_points()
        inp = _move_inputs(inp, "cpu")
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        assert outputs[0].reshape(()).item() == 0.0
        assert torch.isfinite(outputs[0].reshape(()))

    def test_joint_missing_predicate_raises(self):
        """The joint node-semantic path requires an explicit predicate (a
        TORCH_CHECK in the C++ op); a silent default would flip
        select-none/select-all semantics."""
        inp = _miniature_node_inputs("cpu")
        inp["node_primary_predicate"] = None
        with pytest.raises((ValueError, RuntimeError)):
            _call(_make_module(), inp, DISABLED, NODE_LAMBDA)

    def test_contract_validation_errors(self):
        module = _make_module()
        # Both members disabled.
        inp = _miniature_bg_inputs("cpu")
        with pytest.raises((ValueError, RuntimeError)):
            _call(module, inp, DISABLED, DISABLED)
        # Background-only mode requires n_semantic_points.
        inp = _miniature_bg_inputs("cpu")
        inp["n_semantic_points"] = None
        inp["background_semantic_logits"] = ()
        inp["background_segments"] = ()
        with pytest.raises((ValueError, RuntimeError)):
            _call(module, inp, BG_LAMBDA, DISABLED)
        # Joint mode requires the shared semantic_logits tensor.
        inp = _miniature_node_inputs("cpu")
        inp["semantic_logits"] = None
        with pytest.raises((ValueError, RuntimeError)):
            _call(module, inp, DISABLED, NODE_LAMBDA)
        # The final background segment end must equal n_semantic_points.
        inp = _miniature_bg_inputs("cpu")
        inp["n_semantic_points"] = 4
        with pytest.raises((ValueError, RuntimeError)):
            _call(module, inp, BG_LAMBDA, DISABLED)

    def test_noncontiguous_inputs_match_contiguous(self):
        """Non-contiguous views take the same pure-PyTorch path and reproduce
        the contiguous call bitwise (CPU vs CPU)."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_joint_shared_primary()
        inp = _move_inputs(inp, "cpu")

        base_inp, base_leaves = _with_grad_leaves(inp)
        base_outputs = _call(_make_module(), base_inp, bg_lambda, node_lambda)
        base_grads = _grads_of(
            _objective(base_outputs, bg_lambda, node_lambda), base_leaves
        )

        nc_inp, nc_leaves = _with_noncontiguous_inputs(inp)
        nc_outputs = _call(_make_module(), nc_inp, bg_lambda, node_lambda)
        nc_grads = _grads_of(_objective(nc_outputs, bg_lambda, node_lambda), nc_leaves)

        for label, base_value, nc_value in zip(
            _OUTPUT_LABELS, base_outputs, nc_outputs
        ):
            assert torch.equal(
                base_value.reshape(()), nc_value.reshape(())
            ), f"{label} differs on non-contiguous inputs"
        for base_grad, nc_grad in zip(base_grads, nc_grads):
            assert torch.equal(base_grad, nc_grad)

    def test_nonfinite_unselected_values_do_not_poison(self):
        """Non-finite density logits in gated-out (never-selected) points must
        not leak into any loss output or gradient."""
        # Background member: the density-gated, semantic-gated, and
        # outside-every-box miniature points each carry a non-finite logit.
        # NaN fails the strictly-greater density gate, +inf passes it but is
        # semantic-gated, -inf sits outside every box.
        inp = _miniature_bg_inputs("cpu")
        inp["density_logits"] = inp["density_logits"].clone()
        inp["density_logits"][1] = float("nan")
        inp["density_logits"][2] = float("inf")
        inp["density_logits"][3] = float("-inf")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, BG_LAMBDA, DISABLED)
        expected = (_softplus_f(2.0) + _softplus_f(-1.0)) / 2.0
        torch.testing.assert_close(
            outputs[0].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        # The disabled node member's exact zeros never read the values.
        assert outputs[2].reshape(()).item() == 0.0
        assert outputs[3].reshape(()).item() == 0.0
        grads = _grads_of(
            _objective(outputs, BG_LAMBDA, DISABLED), (bg_leaf, other_leaf)
        )
        assert torch.isfinite(grads[0]).all()
        assert torch.count_nonzero(grads[0][1:4]) == 0

        # Node member: selection is purely semantic, so unselected primary
        # (person) and other (tree) points never have their logits read.
        inp = _miniature_node_inputs("cpu")
        inp["density_logits"] = inp["density_logits"].clone()
        inp["density_logits"][0] = float("nan")
        inp["density_logits"][2] = float("inf")
        inp["other_density_logits"] = inp["other_density_logits"].clone()
        inp["other_density_logits"][0] = float("-inf")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, DISABLED, NODE_LAMBDA)
        expected = (_softplus_f(-0.25) + _softplus_f(-2.0)) / 2.0
        torch.testing.assert_close(
            outputs[2].reshape(()),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        bg_grad, other_grad = _grads_of(
            _objective(outputs, DISABLED, NODE_LAMBDA), (bg_leaf, other_leaf)
        )
        assert torch.isfinite(bg_grad).all()
        assert torch.isfinite(other_grad).all()
        assert bg_grad[0] == 0.0 and bg_grad[2] == 0.0
        assert other_grad[0] == 0.0

    def test_nonfinite_poisons_leave_participating_gradients_bit_identical(self):
        """Beyond staying finite: gated-out non-finite logits must leave every
        output and every participating gradient bit-identical to a clean run
        (the poisoned reads are skipped entirely, not merely damped).

        The poison placement is gate-aware. In the background miniature the
        density gate is value-dependent (it excludes nan and -inf but admits
        +inf), so index 1 only carries values the gate rejects; the semantic
        gate (index 2) and the box gate (index 3) exclude regardless of value,
        so the rotation runs every non-finite value through at least one
        value-independent gate. Node-member selection is purely semantic, so
        all three values rotate freely there.
        """
        for member, make_inputs, lambdas, poison_runs in (
            (
                "background",
                _miniature_bg_inputs,
                (BG_LAMBDA, DISABLED),
                (
                    {
                        ("density_logits", 1): float("nan"),
                        ("density_logits", 2): float("inf"),
                        ("density_logits", 3): float("-inf"),
                    },
                    {
                        ("density_logits", 1): float("-inf"),
                        ("density_logits", 2): float("nan"),
                        ("density_logits", 3): float("inf"),
                    },
                ),
            ),
            (
                "node",
                _miniature_node_inputs,
                (DISABLED, NODE_LAMBDA),
                (
                    {
                        ("density_logits", 0): float("nan"),
                        ("density_logits", 2): float("inf"),
                        ("other_density_logits", 0): float("-inf"),
                    },
                    {
                        ("density_logits", 0): float("inf"),
                        ("density_logits", 2): float("-inf"),
                        ("other_density_logits", 0): float("nan"),
                    },
                ),
            ),
        ):
            clean_inp, clean_leaves = _with_grad_leaves(make_inputs("cpu"))
            clean_outputs = _call(_make_module(), clean_inp, *lambdas)
            clean_grads = _grads_of(_objective(clean_outputs, *lambdas), clean_leaves)

            for run_idx, poisons in enumerate(poison_runs):
                tag = f"rotation {run_idx}"
                inp = make_inputs("cpu")
                for (key, idx), value in poisons.items():
                    inp[key] = inp[key].clone()
                    inp[key][idx] = value
                inp, leaves = _with_grad_leaves(inp)
                outputs = _call(_make_module(), inp, *lambdas)
                for label, out, clean_out in zip(
                    _OUTPUT_LABELS, outputs, clean_outputs
                ):
                    assert torch.equal(
                        out, clean_out
                    ), f"{member}/{tag}: output {label} differs from clean run"

                grads = _grads_of(_objective(outputs, *lambdas), leaves)
                for grad, clean_grad in zip(grads, clean_grads):
                    assert torch.equal(
                        grad, clean_grad
                    ), f"{member}/{tag}: gradient differs from clean run"

    def test_production_scale_smoke(self):
        """The fallback stays finite and differentiable at production point
        counts."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_production_scale()
        inp = _move_inputs(inp, "cpu")
        inp, leaves = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        for label, value in zip(_OUTPUT_LABELS, outputs):
            assert torch.isfinite(value.reshape(())), f"{label} not finite"
        assert outputs[0].reshape(()).item() > 0.0
        assert outputs[2].reshape(()).item() > 0.0
        grads = _grads_of(_objective(outputs, bg_lambda, node_lambda), leaves)
        for grad in grads:
            assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grads[0]) > 0
        assert torch.count_nonzero(grads[1]) > 0


# ---------------------------------------------------------------------------
# CUDA tests — only run when GPU + compiled extension available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestBgTrackNodeSemanticCUDA:
    """Compare the CUDA fused kernels against the pure-PyTorch fallback
    (forced via ``_cuda_available = False``) and the in-test reference."""

    @pytest.mark.parametrize("scenario", _SCENARIO_IDS)
    def test_forward_matches_fallback(self, scenario):
        inp, bg_lambda, node_lambda, expect_bg, expect_node = _SCENARIOS[scenario]()
        inp = _move_inputs(inp, "cuda")
        reference = _reference(inp, bg_lambda, node_lambda)
        _assert_expected_selection(reference, expect_bg, expect_node)

        cuda_outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        fallback_outputs = _call(_fallback_module(), inp, bg_lambda, node_lambda)
        _assert_outputs_match(cuda_outputs, fallback_outputs, FWD_TOL)
        # Also pin against the independent reference so a bug shared by both
        # module paths cannot slip through.
        _assert_outputs_match(cuda_outputs, reference["outputs"], FWD_TOL)

    @pytest.mark.parametrize("scenario", _SCENARIO_IDS)
    def test_backward_matches_fallback(self, scenario):
        inp, bg_lambda, node_lambda, _, _ = _SCENARIOS[scenario]()
        inp = _move_inputs(inp, "cuda")

        cuda_inp, cuda_leaves = _with_grad_leaves(inp)
        cuda_grads = _grads_of(
            _objective(
                _call(_make_module(), cuda_inp, bg_lambda, node_lambda),
                bg_lambda,
                node_lambda,
            ),
            cuda_leaves,
        )

        fallback_inp, fallback_leaves = _with_grad_leaves(inp)
        fallback_grads = _grads_of(
            _objective(
                _call(_fallback_module(), fallback_inp, bg_lambda, node_lambda),
                bg_lambda,
                node_lambda,
            ),
            fallback_leaves,
        )

        for label, cuda_grad, fallback_grad in zip(
            ("density_logits", "other_density_logits"), cuda_grads, fallback_grads
        ):
            torch.testing.assert_close(
                cuda_grad,
                fallback_grad,
                **GRAD_TOL,
                msg=lambda m, label=label: f"{label} grad mismatch: {m}",
            )

    def test_select_none_pin_zero_primary_node_grads(self):
        """Structural pin: under ((), True) no primary point joins the node
        loss, so a node-only objective must send zero gradient to the primary
        density logits while the other segment still gets gradient."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_select_none_pin()
        inp = _move_inputs(inp, "cuda")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        node_objective = 5.0 * outputs[2].reshape(()) + 7.0 * outputs[3].reshape(())
        bg_grad, other_grad = _grads_of(node_objective, (bg_leaf, other_leaf))
        assert torch.count_nonzero(bg_grad) == 0
        assert torch.count_nonzero(other_grad) > 0

    def test_select_all_pin_every_primary_point_penalized(self):
        """Structural pin: under ((), False) with multi-word launches every
        primary point (argmax classes on both sides of 32) is selected exactly
        once, so each primary logit receives a strictly nonzero gradient."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_select_all_pin_multiword()
        inp = _move_inputs(inp, "cuda")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        node_objective = 5.0 * outputs[2].reshape(()) + 7.0 * outputs[3].reshape(())
        bg_grad, _ = _grads_of(node_objective, (bg_leaf, other_leaf))
        assert (bg_grad != 0).all(), "select-all must penalize every primary point"

    def test_empty_tracks_zero_loss_and_grads(self):
        inp, bg_lambda, node_lambda, _, _ = _scenario_empty_tracks()
        inp = _move_inputs(inp, "cuda")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        assert outputs[0].reshape(()).item() == 0.0
        assert outputs[1].reshape(()).item() == 0.0
        grads = _grads_of(
            _objective(outputs, bg_lambda, node_lambda), (bg_leaf, other_leaf)
        )
        assert torch.count_nonzero(grads[0]) == 0
        assert torch.count_nonzero(grads[1]) == 0

    def test_background_only_backward_accepts_shared_selection_buffer(self):
        """Regression: in background-only mode the autograd wrapper still
        allocates the shared uint8 selection buffer sized [Nbg + Nother];
        the backward must apply the forward's windowed acceptance (>= Nbg,
        background bits leading) instead of demanding exactly [Nbg]."""
        inp, bg_lambda, _, _, _ = _scenario_bg_only_filtered()
        other_density, _ = _other_cloud()
        inp["other_density_logits"] = other_density  # nonempty, node disabled
        inp = _move_inputs(inp, "cuda")

        cuda_inp, cuda_leaves = _with_grad_leaves(inp)
        cuda_outputs = _call(_make_module(), cuda_inp, bg_lambda, DISABLED)
        cuda_grads = _grads_of(
            _objective(cuda_outputs, bg_lambda, DISABLED), cuda_leaves
        )

        fallback_inp, fallback_leaves = _with_grad_leaves(inp)
        fallback_outputs = _call(_fallback_module(), fallback_inp, bg_lambda, DISABLED)
        fallback_grads = _grads_of(
            _objective(fallback_outputs, bg_lambda, DISABLED), fallback_leaves
        )

        _assert_outputs_match(cuda_outputs, fallback_outputs, FWD_TOL)
        for label, cuda_grad, fallback_grad in zip(
            ("density_logits", "other_density_logits"), cuda_grads, fallback_grads
        ):
            torch.testing.assert_close(
                cuda_grad,
                fallback_grad,
                **GRAD_TOL,
                msg=lambda m, label=label: f"{label} grad mismatch: {m}",
            )
        assert torch.count_nonzero(cuda_grads[0]) > 0
        # The disabled node member owns the only path into the other domain.
        assert torch.count_nonzero(cuda_grads[1]) == 0

    def test_zero_points_zero_loss(self):
        inp, bg_lambda, node_lambda, _, _ = _scenario_zero_points()
        inp = _move_inputs(inp, "cuda")
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        assert outputs[0].reshape(()).item() == 0.0
        assert torch.isfinite(outputs[0].reshape(()))

    @pytest.mark.parametrize(
        "scenario", ("bg_only_filtered", "joint_shared_primary", "packed_unfiltered")
    )
    def test_fp64_parity(self, scenario):
        """The kernels are templated over fp64 as well as fp32."""
        inp, bg_lambda, node_lambda, _, _ = _SCENARIOS[scenario]()
        inp = _move_inputs(inp, "cuda", dtype=torch.float64)

        cuda_inp, cuda_leaves = _with_grad_leaves(inp)
        cuda_outputs = _call(_make_module(), cuda_inp, bg_lambda, node_lambda)
        fallback_inp, fallback_leaves = _with_grad_leaves(inp)
        fallback_outputs = _call(
            _fallback_module(), fallback_inp, bg_lambda, node_lambda
        )
        _assert_outputs_match(cuda_outputs, fallback_outputs, FP64_TOL)

        cuda_grads = _grads_of(
            _objective(cuda_outputs, bg_lambda, node_lambda), cuda_leaves
        )
        fallback_grads = _grads_of(
            _objective(fallback_outputs, bg_lambda, node_lambda), fallback_leaves
        )
        for cuda_grad, fallback_grad in zip(cuda_grads, fallback_grads):
            torch.testing.assert_close(cuda_grad, fallback_grad, **FP64_TOL)

    def test_positions_and_semantic_logits_receive_no_grad(self):
        """Only density logits are differentiable: positions and semantic
        logits feed non-differentiable selection only."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_joint_shared_primary()
        inp = _move_inputs(inp, "cuda")
        inp, _ = _with_grad_leaves(inp)
        positions = inp["positions"].detach().clone().requires_grad_(True)
        semantics = inp["semantic_logits"].detach().clone().requires_grad_(True)
        inp["positions"] = positions
        inp["semantic_logits"] = semantics
        outputs = _call(_make_module(), inp, bg_lambda, node_lambda)
        objective = _objective(outputs, bg_lambda, node_lambda)
        grads = torch.autograd.grad(
            objective, (positions, semantics), allow_unused=True
        )
        for grad in grads:
            assert grad is None or torch.count_nonzero(grad) == 0

    def test_joint_missing_predicate_raises(self):
        """The explicit-predicate requirement (the op's TORCH_CHECK) must hold
        on the CUDA route as well."""
        inp = _miniature_node_inputs("cuda")
        inp["node_primary_predicate"] = None
        with pytest.raises((ValueError, RuntimeError)):
            _call(_make_module(), inp, DISABLED, NODE_LAMBDA)

    def test_miniature_bg_track_hand_computed_cuda(self):
        """The CUDA path reproduces the hand-computed miniature exactly."""
        inp = _miniature_bg_inputs("cuda")
        outputs = _call(_make_module(), inp, BG_LAMBDA, DISABLED)
        expected = (_softplus_f(2.0) + _softplus_f(-1.0)) / 2.0
        torch.testing.assert_close(
            outputs[0].reshape(()).cpu(),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        # The disabled node member's outputs stay exact zeros.
        assert outputs[2].reshape(()).item() == 0.0
        assert outputs[3].reshape(()).item() == 0.0

    def test_noncontiguous_inputs_match_contiguous(self):
        """The CUDA route accepts non-contiguous inputs (contiguified at the
        wrapper boundary) and matches the contiguous call."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_joint_shared_primary()
        inp = _move_inputs(inp, "cuda")

        base_inp, base_leaves = _with_grad_leaves(inp)
        base_outputs = _call(_make_module(), base_inp, bg_lambda, node_lambda)
        base_grads = _grads_of(
            _objective(base_outputs, bg_lambda, node_lambda), base_leaves
        )

        nc_inp, nc_leaves = _with_noncontiguous_inputs(inp)
        nc_outputs = _call(_make_module(), nc_inp, bg_lambda, node_lambda)
        nc_grads = _grads_of(_objective(nc_outputs, bg_lambda, node_lambda), nc_leaves)

        _assert_outputs_match(nc_outputs, base_outputs, FWD_TOL)
        for nc_grad, base_grad in zip(nc_grads, base_grads):
            torch.testing.assert_close(nc_grad, base_grad, **GRAD_TOL)

    def test_nonfinite_unselected_values_do_not_poison(self):
        """Non-finite logits in gated-out points must not poison the CUDA
        outputs or gradients (the kernels predicate every read on the
        selection bit)."""
        inp = _miniature_bg_inputs("cuda")
        inp["density_logits"] = inp["density_logits"].clone()
        inp["density_logits"][1] = float("nan")
        inp["density_logits"][2] = float("inf")
        inp["density_logits"][3] = float("-inf")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, BG_LAMBDA, DISABLED)
        expected = (_softplus_f(2.0) + _softplus_f(-1.0)) / 2.0
        torch.testing.assert_close(
            outputs[0].reshape(()).cpu(),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        assert outputs[2].reshape(()).item() == 0.0
        assert outputs[3].reshape(()).item() == 0.0
        grads = _grads_of(
            _objective(outputs, BG_LAMBDA, DISABLED), (bg_leaf, other_leaf)
        )
        assert torch.isfinite(grads[0]).all()
        assert torch.count_nonzero(grads[0][1:4]) == 0

        inp = _miniature_node_inputs("cuda")
        inp["density_logits"] = inp["density_logits"].clone()
        inp["density_logits"][0] = float("nan")
        inp["density_logits"][2] = float("inf")
        inp["other_density_logits"] = inp["other_density_logits"].clone()
        inp["other_density_logits"][0] = float("-inf")
        inp, (bg_leaf, other_leaf) = _with_grad_leaves(inp)
        outputs = _call(_make_module(), inp, DISABLED, NODE_LAMBDA)
        expected = (_softplus_f(-0.25) + _softplus_f(-2.0)) / 2.0
        torch.testing.assert_close(
            outputs[2].reshape(()).cpu(),
            torch.tensor(expected, dtype=torch.float32),
            **HAND_TOL,
        )
        bg_grad, other_grad = _grads_of(
            _objective(outputs, DISABLED, NODE_LAMBDA), (bg_leaf, other_leaf)
        )
        assert torch.isfinite(bg_grad).all()
        assert torch.isfinite(other_grad).all()
        assert bg_grad[0].item() == 0.0 and bg_grad[2].item() == 0.0
        assert other_grad[0].item() == 0.0

    def test_nonfinite_poisons_leave_participating_gradients_bit_identical(self):
        """Beyond staying finite: gated-out non-finite logits must leave every
        output and every participating gradient bit-identical to a clean run
        (the poisoned reads are skipped entirely, not merely damped).

        The poison placement is gate-aware. In the background miniature the
        density gate is value-dependent (it excludes nan and -inf but admits
        +inf), so index 1 only carries values the gate rejects; the semantic
        gate (index 2) and the box gate (index 3) exclude regardless of value,
        so the rotation runs every non-finite value through at least one
        value-independent gate. Node-member selection is purely semantic, so
        all three values rotate freely there.
        """
        for member, make_inputs, lambdas, poison_runs in (
            (
                "background",
                _miniature_bg_inputs,
                (BG_LAMBDA, DISABLED),
                (
                    {
                        ("density_logits", 1): float("nan"),
                        ("density_logits", 2): float("inf"),
                        ("density_logits", 3): float("-inf"),
                    },
                    {
                        ("density_logits", 1): float("-inf"),
                        ("density_logits", 2): float("nan"),
                        ("density_logits", 3): float("inf"),
                    },
                ),
            ),
            (
                "node",
                _miniature_node_inputs,
                (DISABLED, NODE_LAMBDA),
                (
                    {
                        ("density_logits", 0): float("nan"),
                        ("density_logits", 2): float("inf"),
                        ("other_density_logits", 0): float("-inf"),
                    },
                    {
                        ("density_logits", 0): float("inf"),
                        ("density_logits", 2): float("-inf"),
                        ("other_density_logits", 0): float("nan"),
                    },
                ),
            ),
        ):
            clean_inp, clean_leaves = _with_grad_leaves(make_inputs("cuda"))
            clean_outputs = _call(_make_module(), clean_inp, *lambdas)
            clean_grads = _grads_of(_objective(clean_outputs, *lambdas), clean_leaves)

            for run_idx, poisons in enumerate(poison_runs):
                tag = f"rotation {run_idx}"
                inp = make_inputs("cuda")
                for (key, idx), value in poisons.items():
                    inp[key] = inp[key].clone()
                    inp[key][idx] = value
                inp, leaves = _with_grad_leaves(inp)
                outputs = _call(_make_module(), inp, *lambdas)
                # Loss sums are atomicAdd-reduced, so clean-vs-poisoned
                # launches may differ in summation order; the per-point
                # gradients are direct writes and must be bit-identical.
                for label, out, clean_out in zip(
                    _OUTPUT_LABELS, outputs, clean_outputs
                ):
                    torch.testing.assert_close(
                        out,
                        clean_out,
                        msg=f"{member}/{tag}: output {label} differs from clean run",
                        **FWD_TOL,
                    )
                grads = _grads_of(_objective(outputs, *lambdas), leaves)
                for grad, clean_grad in zip(grads, clean_grads):
                    assert torch.equal(
                        grad, clean_grad
                    ), f"{member}/{tag}: gradient differs from clean run"

    def test_joint_forward_tolerates_prefilled_selection_buffer(self):
        """Regression: the joint launcher must zero the selection buffer itself
        (like the generic launcher) rather than relying on the wrapper's
        zero-allocation plus full mask-word coverage — backward reads every
        selection byte, so stale bytes would silently corrupt gradients.

        Dirty and clean runs are separate launches, so block-completion
        order can move the reduction's fp32 atomic adds by a few ulp; the
        comparison uses an ulp-scale tolerance rather than bitwise equality.
        A surviving stale selection byte pulls whole extra points into the
        member means, which shows up orders of magnitude above that."""
        import unittest.mock as mock

        inp, bg_lambda, node_lambda, _, _ = _scenario_joint_shared_primary()
        inp = _move_inputs(inp, "cuda")

        clean_inp, clean_leaves = _with_grad_leaves(inp)
        clean_outputs = _call(_make_module(), clean_inp, bg_lambda, node_lambda)
        clean_grads = _grads_of(
            _objective(clean_outputs, bg_lambda, node_lambda), clean_leaves
        )

        real_zeros = torch.zeros

        def poisoned_zeros(*args, **kwargs):
            out = real_zeros(*args, **kwargs)
            if kwargs.get("dtype") is torch.uint8:
                out.fill_(255)  # garbage selection bytes: bits 0/1 both set
            return out

        dirty_inp, dirty_leaves = _with_grad_leaves(inp)
        with mock.patch("gsplat.cuda._losses_wrapper.torch.zeros", poisoned_zeros):
            dirty_outputs = _call(_make_module(), dirty_inp, bg_lambda, node_lambda)
            dirty_grads = _grads_of(
                _objective(dirty_outputs, bg_lambda, node_lambda), dirty_leaves
            )

        tol = dict(rtol=1e-6, atol=1e-7)
        _assert_outputs_match(dirty_outputs, clean_outputs, tol)
        for clean_grad, dirty_grad in zip(clean_grads, dirty_grads):
            torch.testing.assert_close(
                dirty_grad,
                clean_grad,
                **tol,
                msg=lambda m: (
                    f"gradients depend on the selection buffer's initial contents: {m}"
                ),
            )

    def test_production_scale_matches_fallback(self):
        """Production-scale smoke: the CUDA path stays finite and agrees with
        the fallback at real point counts (many blocks per launch)."""
        inp, bg_lambda, node_lambda, _, _ = _scenario_production_scale()
        inp = _move_inputs(inp, "cuda")

        cuda_inp, cuda_leaves = _with_grad_leaves(inp)
        cuda_outputs = _call(_make_module(), cuda_inp, bg_lambda, node_lambda)
        fallback_inp, fallback_leaves = _with_grad_leaves(inp)
        fallback_outputs = _call(
            _fallback_module(), fallback_inp, bg_lambda, node_lambda
        )

        for label, value in zip(_OUTPUT_LABELS, cuda_outputs):
            assert torch.isfinite(value.reshape(())), f"{label} not finite"
        _assert_outputs_match(cuda_outputs, fallback_outputs, FWD_TOL)

        cuda_grads = _grads_of(
            _objective(cuda_outputs, bg_lambda, node_lambda), cuda_leaves
        )
        fallback_grads = _grads_of(
            _objective(fallback_outputs, bg_lambda, node_lambda), fallback_leaves
        )
        for cuda_grad, fallback_grad in zip(cuda_grads, fallback_grads):
            torch.testing.assert_close(cuda_grad, fallback_grad, **GRAD_TOL)
