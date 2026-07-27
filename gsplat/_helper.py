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
from contextvars import ContextVar
from functools import lru_cache
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


def ensure_shape(name: str, t: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Expand a tensor to an exact shape, or raise if that is not possible.

    Returns a tensor whose shape is exactly ``shape``. It accepts size-1 and
    missing leading broadcast dimensions, but rejects inputs that would
    broadcast to a larger shape than requested.

    Args:
        name: Name of the tensor, used in error messages.
        t: Tensor to validate and expand.
        shape: Exact shape to return.

    Returns:
        ``t`` itself when it already has the exact shape, otherwise an expanded
        view with shape ``shape``.
    """

    actual_shape = tuple(t.shape)
    expected_shape = tuple(shape)
    try:
        broadcast_shape = tuple(torch.broadcast_shapes(actual_shape, expected_shape))
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {actual_shape}."
        ) from exc
    # Extra leading dimensions can be broadcast-compatible while producing a
    # larger result than the kernel shape. Those are not safe to flatten.
    if broadcast_shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {actual_shape}."
        )

    if actual_shape == expected_shape:
        return t
    return t.expand(shape)


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


_ACTIVE_EXPECT_GROUP = ContextVar("gsplat_active_expect_group", default=None)


def _record_expect_result(passed: bool) -> bool:
    """Track a pytest-check result for an active :func:`expect_group`."""

    group = _ACTIVE_EXPECT_GROUP.get()
    if group is not None and not passed:
        group._record_failure()
    return passed


@lru_cache(maxsize=128)
def _checked(assert_func):
    """Return the cached pytest-check wrapper for an assert helper."""

    # pytest-check is a test-only dependency; keep runtime imports of this
    # shared helper module independent of pytest plugins.
    from pytest_check import check

    return check.check_func(assert_func)


def _assert_true(condition, msg=""):
    """Assert a boolean condition for :func:`expect_true`."""

    assert bool(condition), msg or "expected condition to be true"


class _ExpectGroup:
    """Scoped soft-check collector with a hard barrier at context exit."""

    def __init__(self, name: str = "expect group"):
        self.name = name
        self._token = None
        self._n_failures = 0

    def __enter__(self):
        self._token = _ACTIVE_EXPECT_GROUP.set(self)
        return None

    def __exit__(self, exc_type, exc, tb):
        assert self._token is not None
        _ACTIVE_EXPECT_GROUP.reset(self._token)
        if exc_type is not None:
            return False
        assert self._n_failures == 0, (
            f"{self.name}: {self._n_failures} soft check(s) failed; "
            "see pytest-check diagnostics above"
        )
        return False

    def _record_failure(self) -> None:
        self._n_failures += 1


def expect_group(name: str = "expect group") -> _ExpectGroup:
    """Create a scoped group of soft checks with a hard exit barrier.

    All ``expect`` calls inside the context run to completion. If any of them
    records a pytest-check failure, exiting the context raises once so later
    dependent test-body code does not run.
    """

    return _ExpectGroup(name)


def expect_call(assert_func, *args, **kwargs):
    """Soft-check an arbitrary assert-style callable."""

    return _record_expect_result(_checked(assert_func)(*args, **kwargs))


def expect_true(condition, msg=""):
    """Soft-check that ``condition`` is truthy."""

    return _record_expect_result(_checked(_assert_true)(condition, msg=msg))


def expect_close(*args, **kwargs):
    """Soft-check counterpart to :func:`assert_close`.

    Failures are recorded through pytest-check and the caller continues
    executing. At the end of the test, pytest-check reports any collected
    failures and marks the test failed.
    """

    return _record_expect_result(_checked(assert_close)(*args, **kwargs))


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


def expect_mismatch_ratio(*args, **kwargs):
    """Soft-check counterpart to :func:`assert_mismatch_ratio`."""

    return _record_expect_result(_checked(assert_mismatch_ratio)(*args, **kwargs))


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


def expect_grad_sparsity(*args, **kwargs):
    """Soft-check counterpart to :func:`assert_grad_sparsity`."""

    return _record_expect_result(_checked(assert_grad_sparsity)(*args, **kwargs))


def assert_grad_reference_close(
    actual,
    expected,
    *,
    atol,
    rtol,
    mask=None,
    max_element_fail_ratio=0.0,
    max_rel_l2=None,
    max_rel_l1=None,
    min_cosine=None,
    max_signed_bias=None,
    eps=1e-30,
    require_nonempty=True,
    msg="",
):
    """Compare a gradient tensor against a reference as both values and a vector.

    ``torch.testing.assert_close`` is a good scalar predicate, but large sparse
    gradient tensors also need aggregate guards: a few local outliers should not
    hide a directional bias, and a small absolute tolerance near zero should not
    hide a missing-gradient bug. This helper keeps the usual scale-aware
    element bound and optionally adds direction / magnitude / bias checks over
    the selected tensor as a whole.

    Boolean tensors are accepted for equality-style boundary checks. For bool
    inputs, ``atol`` and ``rtol`` are ignored, ``max_element_fail_ratio`` caps
    the ``actual != expected`` fraction, and aggregate vector metrics are
    skipped.

    Args:
        actual, expected: Gradient tensors with identical shape and dtype.
        atol, rtol: Elementwise bound ``abs(actual - expected) <=
            atol + rtol * abs(expected)``.
        mask: Optional boolean mask selecting elements to compare. Broadcasts to
            ``actual.shape``.
        max_element_fail_ratio: Maximum fraction of selected elements allowed to
            exceed the elementwise bound.
        max_rel_l2: Optional cap for ``||actual - expected||_2 / ||expected||_2``.
        max_rel_l1: Optional cap for ``||actual - expected||_1 / ||expected||_1``.
        min_cosine: Optional lower bound on cosine similarity.
        max_signed_bias: Optional cap for ``abs(sum(actual - expected)) /
            sum(abs(expected))``.
        eps: Denominator floor for aggregate metrics.
        require_nonempty: Whether an empty mask is an error.
        msg: Prefix included in assertion messages.
    """

    assert actual.shape == expected.shape, f"{actual.shape=} {expected.shape=}"
    assert actual.dtype == expected.dtype, f"{msg}: {actual.dtype=} {expected.dtype=}"
    assert 0.0 <= max_element_fail_ratio <= 1.0, (
        f"{msg}: max_element_fail_ratio must be in [0, 1] "
        f"(got {max_element_fail_ratio})"
    )
    assert eps > 0.0, f"{msg}: eps must be > 0 (got {eps})"

    if mask is None:
        selected = torch.ones_like(actual, dtype=torch.bool)
    else:
        assert mask.dtype == torch.bool, f"{msg}: mask must be bool, got {mask.dtype}"
        try:
            selected = mask.expand_as(actual)
        except RuntimeError as exc:
            raise AssertionError(
                f"{msg}: mask shape {tuple(mask.shape)} cannot broadcast to "
                f"{tuple(actual.shape)}"
            ) from exc

    n_total = int(selected.sum().item())
    if n_total == 0:
        if require_nonempty:
            raise AssertionError(f"{msg}: mask selected no elements")
        return

    a = actual[selected]
    e = expected[selected]

    if actual.dtype == torch.bool:
        # Boolean boundary checks only have equality semantics. The vector
        # metrics below are intentionally numeric-only.
        fail = a != e
        diff_for_diag = fail.to(torch.float32)
    else:
        assert torch.isfinite(a).all(), f"{msg}: actual contains NaN / Inf"
        assert torch.isfinite(e).all(), f"{msg}: expected contains NaN / Inf"
        diff = (a - e).abs()
        bound = atol + rtol * e.abs()
        fail = diff > bound
        diff_for_diag = diff

    n_fail = int(fail.sum().item())
    fail_ratio = n_fail / n_total

    if actual.dtype != torch.bool:
        a64 = a.to(torch.float64)
        e64 = e.to(torch.float64)
        d64 = a64 - e64
        abs_e_sum = e64.abs().sum().item()
        l1_expected = max(abs_e_sum, eps)
        l2_expected = max(torch.linalg.vector_norm(e64).item(), eps)
        rel_l1 = d64.abs().sum().item() / l1_expected
        rel_l2 = torch.linalg.vector_norm(d64).item() / l2_expected
        signed_bias = abs(d64.sum().item()) / l1_expected
        actual_l2 = torch.linalg.vector_norm(a64).item()
        expected_l2_raw = torch.linalg.vector_norm(e64).item()
        if actual_l2 <= eps and expected_l2_raw <= eps:
            cosine = 1.0
        elif actual_l2 <= eps or expected_l2_raw <= eps:
            cosine = 0.0
        else:
            cosine = (a64 * e64).sum().item() / (actual_l2 * expected_l2_raw)

        vector_metrics_msg = (
            f"rel_l2={rel_l2:.6e}, rel_l1={rel_l1:.6e}, "
            f"cosine={cosine:.12f}, signed_bias={signed_bias:.6e}"
        )
        metrics_msg = f"{msg}: {vector_metrics_msg}"

    def _format_common_diagnostics():
        flat_diff = torch.zeros_like(actual, dtype=diff_for_diag.dtype)
        flat_diff[selected] = diff_for_diag
        worst_flat = int(flat_diff.reshape(-1).argmax().item())
        worst_index = tuple(int(i) for i in np.unravel_index(worst_flat, actual.shape))
        worst_actual = actual[worst_index].item()
        worst_expected = expected[worst_index].item()
        max_abs = flat_diff[worst_index].item()
        if actual.dtype == torch.bool:
            max_rel = float("nan")
        else:
            denom = max(abs(float(worst_expected)), eps)
            max_rel = abs(float(worst_actual) - float(worst_expected)) / denom
        vector_metrics_suffix = ""
        if actual.dtype != torch.bool:
            vector_metrics_suffix = f"; {vector_metrics_msg}"
        return (
            f"{msg}: element failures {n_fail}/{n_total} "
            f"({fail_ratio:.6%}) > {max_element_fail_ratio:.6%}; "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
            f"worst_index={worst_index}, actual={worst_actual}, "
            f"expected={worst_expected}, atol={atol}, rtol={rtol}"
            f"{vector_metrics_suffix}"
        )

    assert fail_ratio <= max_element_fail_ratio, _format_common_diagnostics()

    if actual.dtype == torch.bool:
        return
    if max_rel_l2 is not None:
        assert rel_l2 <= max_rel_l2, f"{metrics_msg}; rel_l2 exceeds {max_rel_l2:.6e}"
    if max_rel_l1 is not None:
        assert rel_l1 <= max_rel_l1, f"{metrics_msg}; rel_l1 exceeds {max_rel_l1:.6e}"
    if min_cosine is not None:
        assert cosine >= min_cosine, f"{metrics_msg}; cosine below {min_cosine:.12f}"
    if max_signed_bias is not None:
        assert (
            signed_bias <= max_signed_bias
        ), f"{metrics_msg}; signed_bias exceeds {max_signed_bias:.6e}"


def expect_grad_reference_close(*args, **kwargs):
    """Soft-check counterpart to :func:`assert_grad_reference_close`."""

    return _record_expect_result(_checked(assert_grad_reference_close)(*args, **kwargs))


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
            Elementwise tolerances for the interior comparison. The interior is
            checked through ``assert_grad_reference_close`` with
            ``max_element_fail_ratio=0.0``, so failures include the shared
            worst-index diagnostics.
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
        assert_grad_reference_close(
            actual,
            expected,
            mask=interior,
            atol=interior_atol,
            rtol=interior_rtol,
            max_element_fail_ratio=0.0,
            require_nonempty=False,
            msg=f"{msg}: interior failure",
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


def expect_close_with_boundary_band(*args, **kwargs):
    """Soft-check counterpart to :func:`assert_close_with_boundary_band`."""

    return _record_expect_result(
        _checked(assert_close_with_boundary_band)(*args, **kwargs)
    )
