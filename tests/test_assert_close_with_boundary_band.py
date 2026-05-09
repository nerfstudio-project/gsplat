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

"""Tests for ``gsplat._helper.assert_close_with_boundary_band``.

The helper has three independent layers, each guarded by a separate
caller-tunable knob:

  1. INTERIOR    -- per-element ``torch.testing.assert_close`` outside band
  2. BAND CAP    -- flip ratio inside band <= boundary_max_flip_ratio
  3. SYMMETRY    -- bias of band flips bounded by boundary_symmetry_tol

plus optional ``flip_predicate`` (overrides the default flip rule) and
``boundary_cross_predicate`` (every flip must be a verified cross), plus
up-front shape / dtype / NaN-Inf guards. Each parametrize entry exercises
exactly one layer / branch so a regression localizes immediately.

Each parametrize entry is a ``(case_factory, expected)`` pair:

  case_factory : zero-arg callable returning the kwargs dict to pass to
                 the helper. Lazy so each test gets fresh tensors and so
                 module import is cheap.
  expected     : one of
                   "pass"            - call returns without error / warning
                   ("warn",  regex)  - call returns; UserWarning matches regex
                   ("error", regex)  - AssertionError matches regex

All inputs live on CPU. The helper is dtype-checked but otherwise
device-agnostic, so CPU coverage exercises every branch.
"""

import warnings

import pytest
import torch

from gsplat._helper import assert_close_with_boundary_band


def _run(case_factory, expected):
    kwargs = case_factory()
    if expected == "pass":
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            assert_close_with_boundary_band(**kwargs)
        return

    kind, pattern = expected
    if kind == "warn":
        with pytest.warns(UserWarning, match=pattern):
            assert_close_with_boundary_band(**kwargs)
    elif kind == "error":
        with pytest.raises(AssertionError, match=pattern):
            assert_close_with_boundary_band(**kwargs)
    else:
        raise ValueError(f"unknown expected outcome kind: {kind!r}")


# ---- 1. INTERIOR layer ---------------------------------------------------- #


def _case_interior_exact_match():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return dict(
        actual=a.clone(),
        expected=a.clone(),
        boundary_mask=torch.zeros(4, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_interior_within_atol():
    return dict(
        actual=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        expected=torch.tensor([1.0 + 5e-7, 2.0 - 5e-7, 3.0, 4.0]),
        boundary_mask=torch.zeros(4, dtype=torch.bool),
        interior_atol=1e-6,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_interior_within_rtol():
    expected = torch.tensor([10.0, 100.0, 1000.0])
    actual = expected * (1.0 + 5e-4)
    return dict(
        actual=actual,
        expected=expected,
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=1e-3,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_interior_exceeds_atol():
    return dict(
        actual=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        expected=torch.tensor([1.0, 2.0, 3.0, 4.0 + 1e-3]),
        boundary_mask=torch.zeros(4, dtype=torch.bool),
        interior_atol=1e-6,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
        msg="proj covars2d (forward)",
    )


def _case_interior_exceeds_rtol():
    expected = torch.tensor([10.0, 100.0, 1000.0])
    actual = expected * (1.0 + 5e-2)
    return dict(
        actual=actual,
        expected=expected,
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=1e-3,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


# ---- 2. BAND-CAP layer ---------------------------------------------------- #


def _band_inputs(n_band, n_flip, *, signs=None):
    """Build two band-only float tensors of length ``n_band`` such that
    exactly ``n_flip`` elements differ. ``signs`` is a list of +/-1 of
    length ``n_flip`` controlling flip direction (default all +1)."""
    actual = torch.zeros(n_band)
    expected = torch.zeros(n_band)
    if signs is None:
        signs = [+1] * n_flip
    assert len(signs) == n_flip
    for i, s in enumerate(signs):
        actual[i] = s * 1.0  # well above interior_atol=0
    return actual, expected


def _case_band_no_flip():
    a, e = _band_inputs(n_band=10, n_flip=0)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_band_flip_within_budget_symmetric():
    # 50 in band, 50 in interior. 4 band flips = 8% < 10% budget; 2 positive
    # + 2 negative = symmetric. Interior is 50 zero/zero -> tight pass. No
    # empty-interior warning fires because the interior layer has coverage.
    actual = torch.zeros(100)
    expected = torch.zeros(100)
    actual[0] = +1.0
    actual[1] = +1.0
    actual[2] = -1.0
    actual[3] = -1.0
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:50] = True
    return dict(
        actual=actual,
        expected=expected,
        boundary_mask=mask,
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.1,
        boundary_symmetry_tol=0.5,
    )


def _case_band_flip_exceeds_budget():
    # 4 flips out of 10 = 40% > 10% budget.
    a, e = _band_inputs(n_band=10, n_flip=4, signs=[+1, -1, +1, -1])
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.1,
        boundary_symmetry_tol=1.0,
    )


# ---- 3. SYMMETRY layer ---------------------------------------------------- #


def _case_band_symmetry_violated_float():
    # All 4 flips have the same sign -> sign_mean = 1.0 > 0.5.
    a, e = _band_inputs(n_band=10, n_flip=4, signs=[+1, +1, +1, +1])
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,  # cap disabled
        boundary_symmetry_tol=0.5,
    )


def _case_band_symmetry_disabled_via_tol_one():
    # Same one-sided flips; symmetry_tol=1.0 disables the guardrail.
    a, e = _band_inputs(n_band=10, n_flip=4, signs=[+1, +1, +1, +1])
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
    )


def _case_band_symmetry_bool_balanced():
    # bool dtype, 5 in band + 3 in interior. Band: 2 a_only + 2 r_only +
    # 1 match -> asym = 0. Interior: 3 matching False/False -> tight pass.
    actual = torch.tensor([True, True, False, False, False, False, False, False])
    expected = torch.tensor([False, False, True, True, False, False, False, False])
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:5] = True
    return dict(
        actual=actual,
        expected=expected,
        boundary_mask=mask,
        interior_atol=0.0,  # unused for bool, but required by API
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=0.0,
    )


def _case_band_symmetry_bool_violated():
    # bool dtype: 4 a_only + 0 r_only -> asym = 1.0 > 0.0.
    actual = torch.tensor([True, True, True, True, False])
    expected = torch.tensor([False, False, False, False, False])
    return dict(
        actual=actual,
        expected=expected,
        boundary_mask=torch.ones(5, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=0.0,
    )


# ---- 4. flip_predicate override ------------------------------------------- #


def _case_flip_predicate_suppresses():
    # 4 nominal flips; predicate accepts none (returns all-False).
    a, e = _band_inputs(n_band=10, n_flip=4)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,  # 0% budget -> would fail without override
        boundary_symmetry_tol=1.0,
        flip_predicate=lambda a, *_: torch.zeros_like(a, dtype=torch.bool),
    )


def _case_flip_predicate_amplifies():
    # 0 nominal flips; predicate marks all 10 as flips -> 100% > 10% budget.
    a, e = _band_inputs(n_band=10, n_flip=0)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(10, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.1,
        boundary_symmetry_tol=1.0,
        flip_predicate=lambda a, *_: torch.ones_like(a, dtype=torch.bool),
    )


# ---- 5. boundary_cross_predicate ------------------------------------------ #


def _case_cross_predicate_all_crosses():
    # 4 flips, predicate confirms all 4 are true crosses -> pass.
    a, e = _band_inputs(n_band=10, n_flip=4, signs=[+1, -1, +1, -1])
    mask = torch.ones(10, dtype=torch.bool)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=mask,
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
        boundary_cross_predicate=lambda m: torch.ones(
            int(m.sum().item()), dtype=torch.bool
        ),
    )


def _case_cross_predicate_one_non_cross():
    # 4 flips; predicate marks exactly one as a non-cross.
    a, e = _band_inputs(n_band=10, n_flip=4, signs=[+1, -1, +1, -1])
    mask = torch.ones(10, dtype=torch.bool)

    def cross_pred(m):
        out = torch.ones(int(m.sum().item()), dtype=torch.bool)
        out[0] = False  # this flip is NOT a true cross
        return out

    return dict(
        actual=a,
        expected=e,
        boundary_mask=mask,
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
        boundary_cross_predicate=cross_pred,
    )


def _case_cross_predicate_shape_mismatch():
    a, e = _band_inputs(n_band=10, n_flip=4)
    mask = torch.ones(10, dtype=torch.bool)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=mask,
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
        # Wrong length (5 instead of 10).
        boundary_cross_predicate=lambda *_: torch.ones(5, dtype=torch.bool),
    )


# ---- 6. Empty band / empty interior --------------------------------------- #


def _case_empty_band_only_interior_runs():
    # All-False mask: only interior assert runs, must succeed without warn.
    a = torch.tensor([1.0, 2.0, 3.0])
    return dict(
        actual=a.clone(),
        expected=a.clone(),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_empty_interior_warns():
    # All-True mask, no flips, budget+symmetry OK -> reaches the trailing
    # ``warnings.warn(...)`` for empty interior.
    a, e = _band_inputs(n_band=4, n_flip=0)
    return dict(
        actual=a,
        expected=e,
        boundary_mask=torch.ones(4, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
        msg="empty-interior probe",
    )


# ---- 7. Up-front guards: shape / dtype / NaN-Inf -------------------------- #


def _case_shape_mismatch_actual_expected():
    return dict(
        actual=torch.zeros(4),
        expected=torch.zeros(5),
        boundary_mask=torch.zeros(4, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_shape_mismatch_mask():
    return dict(
        actual=torch.zeros(4),
        expected=torch.zeros(4),
        boundary_mask=torch.zeros(5, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_mask_wrong_dtype():
    return dict(
        actual=torch.zeros(4),
        expected=torch.zeros(4),
        boundary_mask=torch.zeros(4, dtype=torch.int32),  # not bool
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
    )


def _case_actual_has_nan():
    actual = torch.tensor([1.0, float("nan"), 3.0])
    return dict(
        actual=actual,
        expected=torch.tensor([1.0, 2.0, 3.0]),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
    )


def _case_expected_has_inf():
    return dict(
        actual=torch.tensor([1.0, 2.0, 3.0]),
        expected=torch.tensor([1.0, float("inf"), 3.0]),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
    )


def _case_dtype_mismatch_actual_expected():
    return dict(
        actual=torch.zeros(3, dtype=torch.float64),
        expected=torch.zeros(3, dtype=torch.float32),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
    )


def _case_negative_max_flip_ratio():
    return dict(
        actual=torch.zeros(3),
        expected=torch.zeros(3),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=-0.1,
        boundary_symmetry_tol=1.0,
    )


def _case_max_flip_ratio_above_one():
    return dict(
        actual=torch.zeros(3),
        expected=torch.zeros(3),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=1.5,
        boundary_symmetry_tol=1.0,
    )


def _case_negative_symmetry_tol():
    return dict(
        actual=torch.zeros(3),
        expected=torch.zeros(3),
        boundary_mask=torch.zeros(3, dtype=torch.bool),
        interior_atol=1.0,
        interior_rtol=1.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=-0.1,
    )


def _case_flip_predicate_wrong_shape():
    # Predicate returns a tensor of length n_band+1 instead of n_band.
    return dict(
        actual=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        boundary_mask=torch.tensor([True, True, False, False]),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=1.0,
        boundary_symmetry_tol=1.0,
        flip_predicate=lambda a, _e: torch.zeros(a.numel() + 1, dtype=torch.bool),
    )


# ---- 8. msg propagation --------------------------------------------------- #


def _case_msg_appears_in_interior_error():
    # interior_atol=0 + a single 1.0-vs-0.0 element -> interior failure.
    return dict(
        actual=torch.tensor([1.0]),
        expected=torch.tensor([0.0]),
        boundary_mask=torch.zeros(1, dtype=torch.bool),
        interior_atol=0.0,
        interior_rtol=0.0,
        boundary_max_flip_ratio=0.0,
        boundary_symmetry_tol=1.0,
        msg="MY_TAG",
    )


# Parametrize table: (id, factory, expected).
_CASES = [
    # interior layer
    ("interior/exact", _case_interior_exact_match, "pass"),
    ("interior/within-atol", _case_interior_within_atol, "pass"),
    ("interior/within-rtol", _case_interior_within_rtol, "pass"),
    (
        "interior/exceeds-atol",
        _case_interior_exceeds_atol,
        ("error", r"interior failure"),
    ),
    (
        "interior/exceeds-rtol",
        _case_interior_exceeds_rtol,
        ("error", r"interior failure"),
    ),
    # band-cap layer
    ("band/no-flips", _case_band_no_flip, ("warn", r"all elements in band")),
    ("band/within-budget-symmetric", _case_band_flip_within_budget_symmetric, "pass"),
    (
        "band/exceeds-budget",
        _case_band_flip_exceeds_budget,
        ("error", r"band flip ratio"),
    ),
    # symmetry layer
    (
        "sym/float-violated",
        _case_band_symmetry_violated_float,
        ("error", r"directional asymmetry"),
    ),
    (
        "sym/float-tol-disables",
        _case_band_symmetry_disabled_via_tol_one,
        ("warn", r"all elements in band"),
    ),
    ("sym/bool-balanced", _case_band_symmetry_bool_balanced, "pass"),
    (
        "sym/bool-violated",
        _case_band_symmetry_bool_violated,
        ("error", r"directional asymmetry"),
    ),
    # flip predicate override
    (
        "flip-pred/suppresses",
        _case_flip_predicate_suppresses,
        ("warn", r"all elements in band"),
    ),
    (
        "flip-pred/amplifies",
        _case_flip_predicate_amplifies,
        ("error", r"band flip ratio"),
    ),
    # cross predicate
    (
        "cross-pred/all-crosses",
        _case_cross_predicate_all_crosses,
        ("warn", r"all elements in band"),
    ),
    (
        "cross-pred/one-non-cross",
        _case_cross_predicate_one_non_cross,
        ("error", r"NOT true boundary crosses"),
    ),
    (
        "cross-pred/shape-mismatch",
        _case_cross_predicate_shape_mismatch,
        ("error", r"cross predicate shape mismatch"),
    ),
    # empty band / empty interior
    ("edge/empty-band", _case_empty_band_only_interior_runs, "pass"),
    (
        "edge/empty-interior-warn",
        _case_empty_interior_warns,
        ("warn", r"all elements in band"),
    ),
    # up-front guards
    (
        "guard/shape-mismatch-ae",
        _case_shape_mismatch_actual_expected,
        ("error", r"actual\.shape="),
    ),
    (
        "guard/shape-mismatch-mask",
        _case_shape_mismatch_mask,
        ("error", r"boundary_mask\.shape="),
    ),
    (
        "guard/mask-wrong-dtype",
        _case_mask_wrong_dtype,
        ("error", r"boundary_mask\.dtype="),
    ),
    ("guard/nan-actual", _case_actual_has_nan, ("error", r"actual contains NaN / Inf")),
    (
        "guard/inf-expected",
        _case_expected_has_inf,
        ("error", r"expected contains NaN / Inf"),
    ),
    (
        "guard/dtype-mismatch-ae",
        _case_dtype_mismatch_actual_expected,
        ("error", r"actual\.dtype="),
    ),
    (
        "guard/negative-max-flip-ratio",
        _case_negative_max_flip_ratio,
        ("error", r"boundary_max_flip_ratio must be in \[0, 1\]"),
    ),
    (
        "guard/max-flip-ratio-above-one",
        _case_max_flip_ratio_above_one,
        ("error", r"boundary_max_flip_ratio must be in \[0, 1\]"),
    ),
    (
        "guard/negative-symmetry-tol",
        _case_negative_symmetry_tol,
        ("error", r"boundary_symmetry_tol must be in \[0, 1\]"),
    ),
    (
        "guard/flip-predicate-wrong-shape",
        _case_flip_predicate_wrong_shape,
        ("error", r"flip predicate shape mismatch"),
    ),
    # msg propagation
    ("msg/prefix-in-error", _case_msg_appears_in_interior_error, ("error", r"MY_TAG")),
]


@pytest.mark.parametrize(
    "case_factory,expected",
    [pytest.param(f, x, id=i) for (i, f, x) in _CASES],
)
def test_assert_close_with_boundary_band(case_factory, expected):
    _run(case_factory, expected)


# --------------------------------------------------------------------------- #
# Cross-cutting: directional-asymmetry side reporting in band errors.
#
# When the band fails on symmetry, the error message names which side
# dominates (a_only / r_only for bool, sign mean for float). A regression
# that reports the wrong side is silent under the regex-match cases above,
# so verify the count attribution explicitly here.
# --------------------------------------------------------------------------- #


def test_band_bool_error_message_attributes_a_only():
    actual = torch.tensor([True, True, True, True, False])
    expected = torch.tensor([False, False, False, False, False])
    with pytest.raises(AssertionError) as exc_info:
        assert_close_with_boundary_band(
            actual=actual,
            expected=expected,
            boundary_mask=torch.ones(5, dtype=torch.bool),
            interior_atol=0.0,
            interior_rtol=0.0,
            boundary_max_flip_ratio=1.0,
            boundary_symmetry_tol=0.0,
        )
    msg = str(exc_info.value)
    assert "a_only=4" in msg, msg
    assert "r_only=0" in msg, msg


def test_band_float_error_reports_negative_sign_mean_when_actual_smaller():
    # All 4 flips have actual < expected -> mean(sign(a-e)) = -1, |.|=1.0.
    actual = torch.zeros(10)
    expected = torch.zeros(10)
    expected[:4] = 1.0  # a=0, e=1 -> sign(a-e)=-1
    with pytest.raises(AssertionError, match=r"directional asymmetry") as exc_info:
        assert_close_with_boundary_band(
            actual=actual,
            expected=expected,
            boundary_mask=torch.ones(10, dtype=torch.bool),
            interior_atol=0.0,
            interior_rtol=0.0,
            boundary_max_flip_ratio=1.0,
            boundary_symmetry_tol=0.5,
        )
    assert "1.000" in str(exc_info.value), str(exc_info.value)
