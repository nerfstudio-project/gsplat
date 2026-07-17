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

"""Tests for ``gsplat._helper.assert_grad_reference_close``.

The helper intentionally combines local and aggregate gradient predicates:
per-element scale-aware bounds, optional sparse-outlier budget, relative
vector norms, cosine similarity, and signed-bias detection.
"""

import importlib.util

import pytest
import torch

from .._package_paths import gsplat_package_file

_HELPER_PATH = gsplat_package_file("_helper.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "gsplat_helper_under_test", _HELPER_PATH
)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
assert _HELPER_SPEC.loader is not None
_HELPER_SPEC.loader.exec_module(_HELPER)

assert_grad_reference_close = _HELPER.assert_grad_reference_close


def _case_exact_match():
    expected = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    actual = expected.clone()
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=0.0,
        max_rel_l2=0.0,
        max_rel_l1=0.0,
        min_cosine=1.0 - 1e-15,
        max_signed_bias=0.0,
        msg="exact",
    )


def _case_scale_aware():
    expected = torch.tensor([1.0, 10.0, 100.0])
    actual = expected * 1.001
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=2e-3,
        max_rel_l2=2e-3,
        max_rel_l1=2e-3,
        min_cosine=0.999999,
        max_signed_bias=2e-3,
    )


def _case_budgeted_sparse_outlier():
    expected = torch.ones(100)
    actual = expected.clone()
    actual[0] = 2.0
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=1e-3,
        max_element_fail_ratio=0.02,
    )


def _case_masked_outlier():
    expected = torch.ones(4)
    actual = expected.clone()
    actual[3] = 100.0
    mask = torch.tensor([True, True, True, False])
    return dict(
        actual=actual,
        expected=expected,
        mask=mask,
        atol=0.0,
        rtol=0.0,
        max_rel_l2=0.0,
        max_rel_l1=0.0,
        min_cosine=1.0,
        max_signed_bias=0.0,
    )


def _case_sparse_outlier_failure():
    expected = torch.ones(100)
    actual = expected.clone()
    actual[0] = 2.0
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=1e-3,
        max_element_fail_ratio=0.0,
    )


def _case_relative_norm_failure():
    expected = torch.ones(100)
    actual = expected.clone()
    actual[:10] *= 2.0
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=1e-3,
        max_element_fail_ratio=0.20,
        max_rel_l2=0.1,
    )


def _case_cosine_failure():
    expected = torch.tensor([1.0, -2.0, 3.0])
    actual = -expected
    return dict(
        actual=actual,
        expected=expected,
        atol=0.0,
        rtol=0.0,
        max_element_fail_ratio=1.0,
        min_cosine=0.99,
    )


def _case_signed_bias_failure():
    expected = torch.ones(100)
    actual = expected + 0.01
    return dict(
        actual=actual,
        expected=expected,
        atol=0.02,
        rtol=0.0,
        max_signed_bias=0.005,
    )


def _case_empty_mask_failure():
    return dict(
        actual=torch.ones(2),
        expected=torch.ones(2),
        mask=torch.zeros(2, dtype=torch.bool),
        atol=0.0,
        rtol=0.0,
    )


def _case_bool_failure():
    return dict(
        actual=torch.tensor([True, False, True]),
        expected=torch.tensor([True, False, False]),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "case_factory",
    [
        pytest.param(_case_exact_match, id="exact"),
        pytest.param(_case_scale_aware, id="scale-aware"),
        pytest.param(_case_budgeted_sparse_outlier, id="budgeted-sparse-outlier"),
        pytest.param(_case_masked_outlier, id="masked-outlier"),
    ],
)
def test_assert_grad_reference_close_passes(case_factory):
    assert_grad_reference_close(**case_factory())


@pytest.mark.parametrize(
    "case_factory,pattern",
    [
        pytest.param(_case_sparse_outlier_failure, "element failures", id="element"),
        pytest.param(_case_relative_norm_failure, "rel_l2 exceeds", id="rel-l2"),
        pytest.param(_case_cosine_failure, "cosine below", id="cosine"),
        pytest.param(_case_signed_bias_failure, "signed_bias exceeds", id="bias"),
        pytest.param(_case_empty_mask_failure, "mask selected no elements", id="empty"),
        pytest.param(_case_bool_failure, "element failures", id="bool"),
    ],
)
def test_assert_grad_reference_close_fails(case_factory, pattern):
    with pytest.raises(AssertionError, match=pattern):
        assert_grad_reference_close(**case_factory())


def test_assert_grad_reference_close_element_failure_includes_vector_diagnostics():
    case = _case_sparse_outlier_failure()
    case.update(
        max_rel_l2=10.0,
        max_rel_l1=10.0,
        min_cosine=-1.0,
        max_signed_bias=10.0,
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_grad_reference_close(**case)

    message = str(exc_info.value)
    for label in ("rel_l2=", "rel_l1=", "cosine=", "signed_bias="):
        assert label in message
