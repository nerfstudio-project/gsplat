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

"""Tests for ``gsplat._helper.assert_grad_sparsity``.

Per-row magnitude-ratio check: reduces ``actual`` and ``expected`` along
``reduce_dim`` to one magnitude per row, then asserts ``min(a, e) /
max(a, e) >= min_ratio`` for every row whose magnitude is non-zero on at
least one side. Bug class is "missing gradient" -- one side is ~0 while
the other is non-zero, which slips past per-element ``assert_close``.

Each parametrize entry is a ``(case_factory, expected)`` pair:

  case_factory : zero-arg callable returning the kwargs dict to pass to
                 the helper.
  expected     : one of
                   "pass"            - call returns without error
                   ("error", regex)  - AssertionError matches regex

All inputs live on CPU.
"""

import pytest
import torch

from gsplat._helper import assert_grad_sparsity


def _run(case_factory, expected):
    kwargs = case_factory()
    if expected == "pass":
        assert_grad_sparsity(**kwargs)
        return

    kind, pattern = expected
    if kind == "error":
        with pytest.raises(AssertionError, match=pattern):
            assert_grad_sparsity(**kwargs)
    else:
        raise ValueError(f"unknown expected outcome kind: {kind!r}")


# ---- Ratio behavior ------------------------------------------------------- #


def _case_equal_magnitudes():
    g = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    return dict(actual=g.clone(), expected=g.clone(), min_ratio=0.99)


def _case_within_ratio():
    # 5% magnitude bias per row: ratio = 0.95... >= 0.9 -> pass.
    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    actual = expected * 1.05
    return dict(actual=actual, expected=expected, min_ratio=0.9)


def _case_below_ratio_actual_smaller():
    # actual row magnitude is 0.5x expected -> ratio 0.5 < 0.9.
    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    actual = expected * 0.5
    return dict(actual=actual, expected=expected, min_ratio=0.9, msg="MY_GRAD_TAG")


def _case_below_ratio_expected_smaller():
    # expected row magnitude is 0.5x actual -> ratio 0.5 < 0.9.
    actual = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    expected = actual * 0.5
    return dict(actual=actual, expected=expected, min_ratio=0.9)


def _case_one_zero_row_each_side():
    # Row 0: actual zero, expected non-zero -> ratio 0 < 0.5, FLAGGED.
    # Row 1: both zero -> skipped (ratio undefined).
    actual = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    expected = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
    return dict(actual=actual, expected=expected, min_ratio=0.5)


def _case_both_zero_rows_skipped():
    # Both sides fully zero: every row's `larger == 0`, so all skipped -> pass.
    z = torch.zeros(3, 4)
    return dict(actual=z.clone(), expected=z.clone(), min_ratio=0.99)


def _case_reduce_dim_explicit():
    # reduce_dim=0 -> per-column magnitudes [3, 7] vs [3, 7] -> pass.
    actual = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    expected = torch.tensor([[2.0, 4.0], [1.0, 3.0]])  # same column sums
    return dict(actual=actual, expected=expected, min_ratio=0.99, reduce_dim=0)


# ---- Up-front guards ------------------------------------------------------ #


def _case_min_ratio_zero():
    # Helper requires min_ratio > 0.
    g = torch.tensor([[1.0, 2.0]])
    return dict(actual=g, expected=g, min_ratio=0.0)


def _case_min_ratio_negative():
    g = torch.tensor([[1.0, 2.0]])
    return dict(actual=g, expected=g, min_ratio=-0.1)


def _case_shape_mismatch():
    return dict(
        actual=torch.zeros(2, 3),
        expected=torch.zeros(2, 4),
        min_ratio=0.5,
    )


def _case_actual_has_nan():
    actual = torch.tensor([[1.0, float("nan")]])
    expected = torch.tensor([[1.0, 1.0]])
    return dict(actual=actual, expected=expected, min_ratio=0.5)


def _case_expected_has_inf():
    actual = torch.tensor([[1.0, 1.0]])
    expected = torch.tensor([[1.0, float("inf")]])
    return dict(actual=actual, expected=expected, min_ratio=0.5)


def _case_dtype_mismatch():
    return dict(
        actual=torch.zeros(2, 3, dtype=torch.float64),
        expected=torch.zeros(2, 3, dtype=torch.float32),
        min_ratio=0.5,
    )


# ---- Shape coverage ------------------------------------------------------- #


def _case_reduce_dim_tuple():
    # 3D tensor with reduce_dim as a tuple: per-(dim 0) row magnitude is
    # sum of |x| over (dim 1, dim 2). Equal magnitudes per row -> pass.
    actual = torch.ones(2, 3, 4)
    expected = torch.ones(2, 3, 4)
    return dict(actual=actual, expected=expected, min_ratio=0.99, reduce_dim=(1, 2))


def _case_3d_default_reduce():
    # (C, N, F) shape matching production call sites: per-(C, N) row magnitude
    # is sum of |x| over F. Inject one (C, N) row where actual is half of
    # expected so the per-row ratio falls below min_ratio=0.6.
    actual = torch.ones(3, 4, 5)
    expected = torch.ones(3, 4, 5)
    actual[0, 0, :] *= 0.5  # row (0, 0) ratio = 0.5 < 0.6
    return dict(actual=actual, expected=expected, min_ratio=0.6)


def _case_1d_default_reduce():
    # (N,) shape: reduce_dim=-1 collapses to scalar magnitude. Helper
    # treats this as a single "row" comparison.
    actual = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([1.0, 2.0, 3.0])
    return dict(actual=actual, expected=expected, min_ratio=0.99)


# Parametrize table: (id, factory, expected).
_CASES = [
    ("ratio/equal-magnitudes", _case_equal_magnitudes, "pass"),
    ("ratio/within-bound", _case_within_ratio, "pass"),
    (
        "ratio/below-actual-smaller",
        _case_below_ratio_actual_smaller,
        ("error", r"sparsity mismatches"),
    ),
    (
        "ratio/below-expected-smaller",
        _case_below_ratio_expected_smaller,
        ("error", r"sparsity mismatches"),
    ),
    (
        "ratio/one-zero-row-flagged",
        _case_one_zero_row_each_side,
        ("error", r"sparsity mismatches"),
    ),
    ("ratio/both-zero-rows-skipped", _case_both_zero_rows_skipped, "pass"),
    ("ratio/reduce-dim-0", _case_reduce_dim_explicit, "pass"),
    # guards
    ("guard/min-ratio-zero", _case_min_ratio_zero, ("error", r"min_ratio must be > 0")),
    (
        "guard/min-ratio-negative",
        _case_min_ratio_negative,
        ("error", r"min_ratio must be > 0"),
    ),
    ("guard/shape-mismatch", _case_shape_mismatch, ("error", r"actual\.shape=")),
    ("guard/dtype-mismatch", _case_dtype_mismatch, ("error", r"actual\.dtype=")),
    ("guard/nan-actual", _case_actual_has_nan, ("error", r"actual contains NaN / Inf")),
    (
        "guard/inf-expected",
        _case_expected_has_inf,
        ("error", r"expected contains NaN / Inf"),
    ),
    # shape coverage
    ("shape/reduce-dim-tuple", _case_reduce_dim_tuple, "pass"),
    (
        "shape/3d-default-reduce",
        _case_3d_default_reduce,
        ("error", r"sparsity mismatches"),
    ),
    ("shape/1d-default-reduce", _case_1d_default_reduce, "pass"),
    # msg propagation
    (
        "msg/prefix-in-error",
        _case_below_ratio_actual_smaller,
        ("error", r"MY_GRAD_TAG"),
    ),
]


@pytest.mark.parametrize(
    "case_factory,expected",
    [pytest.param(f, x, id=i) for (i, f, x) in _CASES],
)
def test_assert_grad_sparsity(case_factory, expected):
    _run(case_factory, expected)


# --------------------------------------------------------------------------- #
# Cross-cutting: error message attributes which side has the smaller row.
#
# A regression that swaps actual/expected in the side count is silent under
# the regex-match parametrize cases above.
# --------------------------------------------------------------------------- #


def test_grad_sparsity_error_counts_actual_smaller_rows():
    # 2 rows where actual is smaller, 1 row where expected is smaller.
    actual = torch.tensor(
        [
            [0.1, 0.1],  # mag 0.2 vs 2.0 -> actual_smaller
            [0.1, 0.1],  # mag 0.2 vs 2.0 -> actual_smaller
            [2.0, 2.0],  # mag 4.0 vs 0.2 -> expected_smaller
        ]
    )
    expected = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.1, 0.1],
        ]
    )
    with pytest.raises(AssertionError) as exc_info:
        assert_grad_sparsity(actual, expected, min_ratio=0.5)
    msg = str(exc_info.value)
    assert "actual_smaller=2" in msg, msg
    assert "expected_smaller=1" in msg, msg
