# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def expand_named_params(named_params):
    """
    Expand a list of (name, value) tuples into pytest.param objects with IDs.

    Args:
        named_params: List of (name, value) tuples where name is the test ID
                     and value is the parameter value.

    Returns:
        List of pytest.param objects with IDs set.

    Example:
        >>> ROLLING_SHUTTER_TYPES = [
        ...     ("L2R", RollingShutterType.ROLLING_LEFT_TO_RIGHT),
        ...     ("R2L", RollingShutterType.ROLLING_RIGHT_TO_LEFT),
        ... ]
        >>> @pytest.mark.parametrize("rs_type", expand_named_params(ROLLING_SHUTTER_TYPES))
    """
    import pytest

    return [pytest.param(value, id=name) for name, value in named_params]


def load_test_data(
    data_path: Optional[str] = None,
    device="cuda",
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),
    scene_grid: int = 1,
):
    """Load the test data."""
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz")
    data = np.load(data_path)
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    Ks = torch.from_numpy(data["Ks"]).float().to(device)
    means = torch.from_numpy(data["means3d"]).float().to(device)
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)
    C = len(viewmats)

    # crop
    aabb = torch.tensor(scene_crop, device=device)
    edges = aabb[3:] - aabb[:3]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, colors = means[sel], colors[sel]

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)
    colors = colors.repeat(repeats**2, 1)

    # create gaussian attributes
    N = len(means)
    # Generate scales in range [1e-4, 0.02] to avoid numerical instability
    # Gradient of 1/scale is -1/scale², which explodes for extremely small scales
    min_scale = 1e-4
    max_scale = 0.02
    scales = torch.rand((N, 3), device=device) * (max_scale - min_scale) + min_scale

    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)
    opacities = torch.rand((N,), device=device)

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height


def get_inlier_abserror_mask(actual, expected, *, quantile=None, atol=None, rtol=None):
    """
    Create mask for inliers based on error thresholds.

    Combines quantile-based filtering with absolute/relative tolerance checks.
    Uses the same condition as torch.testing.assert_close.

    Args:
        actual: Actual tensor (e.g., CUDA implementation)
        expected: Expected tensor (e.g., reference implementation)
        quantile: Quantile threshold in [0, 1] (e.g., 0.99 = mask out worst 1%). Optional.
        atol: Absolute tolerance. Optional.
        rtol: Relative tolerance (relative to expected). Optional. Requires atol if specified.

    Returns:
        Boolean mask same shape as inputs, True for inliers (values within all specified thresholds)
    """
    # Validate arguments
    assert (
        rtol is None or atol is not None
    ), "If rtol is specified, atol must also be specified"

    abs_diff = (actual - expected).abs()

    # Build mask by combining conditions
    mask = torch.ones_like(abs_diff, dtype=torch.bool)

    # Apply quantile threshold if specified
    if quantile is not None:
        quantile_threshold = torch.quantile(abs_diff, quantile).item()
        mask = mask & (abs_diff <= quantile_threshold)

    # Apply atol/rtol threshold if specified
    if atol is not None:
        rtol_val = rtol if rtol is not None else 0
        # Relative tolerance (element-wise, depends on expected values)
        threshold = atol + rtol_val * expected.abs()
        mask = mask & (abs_diff <= threshold)

    return mask


def assert_shape(name: str, t: torch.Tensor, shape: tuple):
    """
    Check if the shape of a tensor matches a given shape.

    Args:
        name: Name of the tensor
        t: Tensor to check
        shape: Shape to check against
    """

    try:
        torch.broadcast_shapes(t.shape, shape)
        return True
    except Exception:
        raise ValueError(f"{name} must have shape {shape}, got {t.shape}")


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
):
    # rtol, atol = 0,0

    torch.testing.assert_close(
        actual,
        expected,
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
    )


def assert_mismatch_ratio(actual, expected, *, max=1e-5):
    """
    Assert that the mismatch ratio is less than a given tolerance.
    """
    if max is None:
        max = 1e-5

    # max=0

    assert actual.shape == expected.shape, f"{actual.shape=} {expected.shape=}"

    mismatch = (actual != expected).sum().item()
    total = expected.numel()
    mismatch_ratio = mismatch / total if total > 0 else 1
    assert (
        mismatch_ratio <= max
    ), f"Too many validity mismatches: {mismatch}/{total} ({mismatch_ratio*100:.2f}%) "


def assert_grad_sparsity(
    actual,
    expected,
    *,
    min_ratio,
    reduce_dim=-1,
    msg="",
):
    """Per-row (per-Gaussian / per-pixel) magnitude-ratio sparsity check.

    Reduces ``actual`` and ``expected`` to a magnitude per row by summing
    ``|x|`` over ``reduce_dim`` and asserts ``min(a, e) / max(a, e) >=
    min_ratio`` for every row -- the two magnitudes never differ by more
    than ``1 / min_ratio``.

    Catches the bug class that per-element ``assert_close`` admits: a
    CUDA backward that zeros a real gradient (one side ~0, other side
    non-zero -> ratio ~0), an atomic-add that's forgotten (entire input's
    gradient missing), a backward branch that returns 0 in some cases.
    These survive tolerance-based checks because ``|0 - tiny|`` is always
    within atol; the ratio check sees that the magnitudes are
    fundamentally inconsistent regardless of absolute scale.

    Args:
        actual, expected: tensors of equal shape and dtype. Must be finite
                          (NaN / Inf are rejected up front; a NaN-tolerant
                          ratio admits the very bug class this helper exists
                          to catch).
        min_ratio:        REQUIRED, must be > 0. Minimum allowed
                          ``min(a, e) / max(a, e)`` per row. Tighter values
                          catch more sparsity bugs but admit fewer magnitude
                          variations. Set per call site -- the appropriate
                          value depends on the gradient's magnitude variance.
        reduce_dim:       dim (or tuple) to reduce for the per-row magnitude
                          (default last dim).
        msg:              prefix for any AssertionError.

    Magnitude is computed as L1 (``sum(|x|)``) rather than L2
    (``sqrt(sum(x*x))``). The choice does not affect missing-grad
    detection (which only fires when one side's magnitude is identically
    zero), and L1 avoids a per-row ``sqrt`` and is overflow-stable on
    the ~1e3-magnitude fisheye ``z<=0`` grads the helper must accept.

    Edge cases:
        * Fully-zero rows on both sides -> skipped (ratio undefined; both
          sides agree at zero so there is no missing-grad signal to catch).
        * All rows zero on both sides (e.g. degenerate test fixture
          producing no gradient flow) -> all rows skipped, helper
          returns silently. This is row-level by design (the bug class
          this helper catches is sparsity, not absence of signal); for
          a tensor-level "expected has gradient flow" precondition,
          add an upstream ``expected.abs().sum() > 0`` assert at the
          call site.
        * NaN / Inf in actual or expected -> rejected before the ratio
          check; rows containing them would silently mark as "skipped"
          (NaN > 0 is False) and the missing-grad bug class would survive.
    """
    assert actual.shape == expected.shape, f"{actual.shape=} {expected.shape=}"
    assert actual.dtype == expected.dtype, f"{msg}: {actual.dtype=} {expected.dtype=}"
    assert min_ratio > 0, f"{msg}: min_ratio must be > 0 (got {min_ratio})"
    assert torch.isfinite(actual).all(), f"{msg}: actual contains NaN / Inf"
    assert torch.isfinite(expected).all(), f"{msg}: expected contains NaN / Inf"
    mag_actual = actual.abs().sum(dim=reduce_dim)
    mag_expected = expected.abs().sum(dim=reduce_dim)
    larger = torch.maximum(mag_actual, mag_expected)
    smaller = torch.minimum(mag_actual, mag_expected)
    # Rows where both are essentially zero -- skip (ratio undefined).
    valid = larger > 0
    ratio = torch.ones_like(larger)
    ratio[valid] = smaller[valid] / larger[valid]
    mismatch = ratio < min_ratio
    n_mismatch = int(mismatch.sum().item())
    if n_mismatch > 0:
        # Direction: which side is the smaller (i.e. which side is missing
        # gradient signal)?
        actual_smaller = (mag_actual < mag_expected) & mismatch
        expected_smaller = (mag_actual > mag_expected) & mismatch
        raise AssertionError(
            f"{msg}: {n_mismatch} per-row sparsity mismatches "
            f"(min_ratio={min_ratio}; "
            f"actual_smaller={int(actual_smaller.sum().item())}, "
            f"expected_smaller={int(expected_smaller.sum().item())})"
        )


def assert_close_with_boundary_band(
    actual,
    expected,
    *,
    boundary_mask,
    interior_atol,
    interior_rtol,
    boundary_max_flip_ratio,
    boundary_symmetry_tol,
    flip_predicate=None,
    boundary_cross_predicate=None,
    msg="",
):
    """
    Two-tier comparison for quantities sensitive to algorithmic discontinuities
    (e.g. floor() of a projected coordinate, or strict < at an image edge).

    The caller supplies ``boundary_mask`` (True = element sits in the band where
    a discontinuity lives, False = element is in the smooth interior). The
    helper applies:

      INTERIOR (boundary_mask=False): tight per-element ``assert_close`` with
      (interior_atol, interior_rtol). This catches real numerical regressions:
      a bug that nudges most elements past interior_atol fires here.

      BOUNDARY (boundary_mask=True): allow disagreement only if
        (i)  the disagreeing fraction is <= boundary_max_flip_ratio,
        (ii) disagreements are roughly symmetric (no directional bias) -
             |a_only - r_only| / (a_only + r_only) <= boundary_symmetry_tol,
       (iii) if ``boundary_cross_predicate`` is provided, every disagreement in
             the band must satisfy it. The predicate confirms each flip is a
             true boundary cross (actual and expected agree within an upstream
             ULP AND fall on opposite sides of the same discontinuity).
             Disagreements in the band that are NOT crosses indicate real bugs.

    For boolean inputs, symmetry compares (a & ~e) vs (~a & e) counts.
    For non-boolean inputs, symmetry uses the mean of sign(a-e) over flips.
    Both formulas normalize to [0, 1] and produce comparable values: a
    ``boundary_symmetry_tol=0.5`` means the same thing on a float test as on
    a bool test ("at most 50% net imbalance among flips").

    Predicate-input asymmetry: ``flip_predicate`` receives the band-only
    slices ``actual[boundary_mask]`` / ``expected[boundary_mask]``;
    ``boundary_cross_predicate`` receives the FULL-shape mask so its closure
    can index unrelated upstream tensors (e.g. the projected image_point).
    Each predicate's parameter doc states this contract; check the relevant
    block before writing a new closure.

    Args:
        actual, expected:
            Tensors of equal shape and dtype.
        boundary_mask:
            Bool tensor of same shape; True = element is in the discontinuity
            band, False = interior.
        interior_atol, interior_rtol:
            Forwarded to ``torch.testing.assert_close`` for the interior.
        boundary_max_flip_ratio:
            Maximum allowed disagreeing fraction within the band.
        boundary_symmetry_tol:
            Bias guardrail (see above). Set to 1.0 to disable when there are
            too few flips to be statistically meaningful (e.g. n=1).
        flip_predicate:
            Callable ``(a_band, e_band) -> bool tensor`` of length
            ``boundary_mask.sum()``. Inputs are the band-only slices
            ``actual[boundary_mask]`` and ``expected[boundary_mask]``.
            Returns True for elements that COUNT toward
            ``boundary_max_flip_ratio``; False elements are admitted into the
            band without counting.

            Two regimes; pick by what the boundary looks like:

            * **Residual-based** (default for non-bool:
              ``(a-e).abs() > interior_atol``). Right when the boundary is a
              quantization step (``floor()``) -- ULP-scale disagreements at
              the step are noise; anything larger is a real flip. Pass a
              closure that mirrors ``interior_atol`` so a regression nudging
              every band element by ``interior_atol`` still trips the cap.

            * **Magnitude-based** (caller-supplied, e.g.
              ``lambda a, e, _t=nz_thresh: a.abs() > 10 * _t``). Right when
              the boundary is a near-zero region (Newton fall-off, fisheye
              ``z->0``): both sides agree on FP noise around zero; only
              large-magnitude disagreement is a real bug. Most call sites in
              ``test_basic`` / ``test_2dgs`` use this shape because their
              gradients fall to zero on near-zero-depth Gaussians.

            Default for bool dtype is ``a != e``.
        boundary_cross_predicate:
            Optional callable ``(boundary_mask) -> bool tensor`` of length
            ``boundary_mask.sum()``. The input is the FULL-shape mask (not
            the band-only slices, unlike ``flip_predicate``); the closure
            may capture upstream tensors (e.g. the projected image_point)
            needed to verify the cross. Returns one bool per band element
            (i.e. one per ``True`` entry of ``boundary_mask``); every flip
            in the band must satisfy it.

            Idiom (capture upstream tensors, compute full-shape, index by
            the mask)::

                def cross_pred(m, _a=image_point, _b=image_point_ref):
                    full = _a.floor() != _b.floor()  # full shape
                    return full[m]                   # band length
        msg:
            String prefix included in any AssertionError.

    Calibration tip (used during initial setup):
        Set ``interior_atol=0``, ``interior_rtol=0``,
        ``boundary_max_flip_ratio=0``, ``boundary_symmetry_tol=1.0`` (off),
        ``boundary_cross_predicate=None``. The first failure tells you whether
        the disagreement is interior or in-band, and gives you the magnitude
        and count to set tight values.

    Calibration policy:
        Existing call-site tolerances are envelope-of-worst-observed x 1.05
        across the calibration GPUs. When a new GPU produces a value above
        the envelope, prefer (1) bisecting the per-test calibration trace
        comments to identify the new worst-case, (2) confirming the new
        value is FP-noise-level rather than an algorithmic regression, and
        (3) only then bumping the envelope to ``new_worst x 1.05``. Avoid
        widening tolerances solely to chase a CI failure -- the in-tree
        ``test_basic.py`` per-GPU traces are the audit trail.

    Edge cases:
        * Empty boundary band -> only the interior assert runs.
        * Empty interior     -> a UserWarning is emitted; boundary checks
                                still run. Callers should ensure
                                ``boundary_mask`` isn't trivializing the test.
        * NaN / Inf in actual or expected -> rejected up front; the default
          flip-predicate (``(a-e).abs() > interior_atol``) and any caller
          predicates of the same shape silently treat NaN as "not a flip"
          (NaN > anything is False), which would let band NaN escape the cap.
        * Per-flip magnitude is unbounded by the band check. Admitted in-band
          flips can be arbitrarily large; the cross-predicate is the intended
          escape hatch when an upstream invariant is verifiable. When no
          cross-predicate is feasible, callers should add a per-element
          ``(actual - expected).abs() <= atol_outlier + rtol_outlier *
          expected.abs()`` assertion alongside this helper to catch a NaN /
          catastrophic single-element bug from hiding inside the budget
          (see ``test_basic.py`` proj covars2d for an example).
    """
    assert actual.shape == expected.shape, f"{actual.shape=} {expected.shape=}"
    assert actual.dtype == expected.dtype, f"{msg}: {actual.dtype=} {expected.dtype=}"
    assert (
        boundary_mask.shape == actual.shape
    ), f"{boundary_mask.shape=} {actual.shape=}"
    assert boundary_mask.dtype == torch.bool, f"{boundary_mask.dtype=}"
    assert 0.0 <= boundary_max_flip_ratio <= 1.0, (
        f"{msg}: boundary_max_flip_ratio must be in [0, 1] "
        f"(got {boundary_max_flip_ratio})"
    )
    assert 0.0 <= boundary_symmetry_tol <= 1.0, (
        f"{msg}: boundary_symmetry_tol must be in [0, 1] "
        f"(got {boundary_symmetry_tol})"
    )
    if actual.dtype != torch.bool:
        assert torch.isfinite(actual).all(), f"{msg}: actual contains NaN / Inf"
        assert torch.isfinite(expected).all(), f"{msg}: expected contains NaN / Inf"

    interior = ~boundary_mask

    # --- Interior assert: regression catcher ----------------------------------
    if interior.any():
        torch.testing.assert_close(
            actual[interior],
            expected[interior],
            atol=interior_atol,
            rtol=interior_rtol,
            msg=lambda default_msg: f"{msg}: interior failure: {default_msg}",
        )

    if not boundary_mask.any():
        return

    # --- Empty-interior warning (hoisted before band checks so a degenerate
    # mask is flagged even if a subsequent band check raises) ----------------
    if not interior.any():
        warnings.warn(
            f"[boundary-band] {msg}: all elements in band; no interior "
            f"coverage. Verify the boundary_mask is not trivially True.",
            UserWarning,
            stacklevel=2,
        )

    a_b = actual[boundary_mask]
    e_b = expected[boundary_mask]

    # --- flip predicate -------------------------------------------------------
    if flip_predicate is None:
        if actual.dtype == torch.bool:
            flips = a_b != e_b
        else:
            flips = (a_b - e_b).abs() > interior_atol
    else:
        flips = flip_predicate(a_b, e_b)
        assert flips.shape == a_b.shape, (
            f"{msg}: flip predicate shape mismatch "
            f"(got {flips.shape}, expected {a_b.shape})"
        )

    n_flips = int(flips.sum().item())
    n_band = int(flips.numel())
    flip_ratio = n_flips / n_band if n_band > 0 else 0.0

    # --- Cross verification: every band flip must be a true boundary cross ---
    if boundary_cross_predicate is not None and n_flips > 0:
        crosses = boundary_cross_predicate(boundary_mask)
        assert crosses.shape == flips.shape, (
            f"{msg}: cross predicate shape mismatch "
            f"(got {crosses.shape}, expected {flips.shape})"
        )
        non_cross_flips = flips & ~crosses
        n_non_cross = int(non_cross_flips.sum().item())
        assert n_non_cross == 0, (
            f"{msg}: {n_non_cross}/{n_flips} band flips are NOT true boundary "
            f"crosses (real disagreements, not FP noise at the discontinuity)"
        )

    # --- Flip-rate cap --------------------------------------------------------
    assert flip_ratio <= boundary_max_flip_ratio, (
        f"{msg}: band flip ratio {flip_ratio*100:.4f}% "
        f"({n_flips}/{n_band}) > {boundary_max_flip_ratio*100:.4f}%"
    )

    # --- Symmetry guardrail ---------------------------------------------------
    if actual.dtype == torch.bool:
        a_only = int((a_b & ~e_b).sum().item())
        r_only = int((~a_b & e_b).sum().item())
        if a_only + r_only > 0:
            asym = abs(a_only - r_only) / (a_only + r_only)
            assert asym <= boundary_symmetry_tol, (
                f"{msg}: directional asymmetry in band flips: "
                f"a_only={a_only} r_only={r_only} (|delta|/sum={asym:.3f} > "
                f"{boundary_symmetry_tol:.3f})"
            )
    else:
        if n_flips > 0:
            signs = (a_b - e_b).sign()[flips]
            sign_mean = signs.float().mean().abs().item()
            assert sign_mean <= boundary_symmetry_tol, (
                f"{msg}: directional asymmetry in band flips: "
                f"|mean(sign(a-e))|={sign_mean:.3f} > "
                f"{boundary_symmetry_tol:.3f}"
            )
