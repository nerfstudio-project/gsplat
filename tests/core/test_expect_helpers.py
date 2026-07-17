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

"""Tests for pytest-check backed ``gsplat._helper.expect_*`` helpers."""

import importlib.util
import textwrap

import pytest
import torch

from .._package_paths import gsplat_package_file

pytest_plugins = ["pytester"]

_HELPER_PATH = gsplat_package_file("_helper.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "gsplat_helper_under_test", _HELPER_PATH
)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
assert _HELPER_SPEC.loader is not None
_HELPER_SPEC.loader.exec_module(_HELPER)

expect_close = _HELPER.expect_close
expect_close_with_boundary_band = _HELPER.expect_close_with_boundary_band
expect_call = _HELPER.expect_call
expect_grad_reference_close = _HELPER.expect_grad_reference_close
expect_grad_sparsity = _HELPER.expect_grad_sparsity
expect_group = _HELPER.expect_group
expect_mismatch_ratio = _HELPER.expect_mismatch_ratio
expect_true = _HELPER.expect_true


@pytest.mark.parametrize(
    "helper,kwargs",
    [
        pytest.param(
            expect_close,
            dict(actual=torch.ones(2), expected=torch.ones(2), rtol=0, atol=0),
            id="close",
        ),
        pytest.param(
            expect_mismatch_ratio,
            dict(actual=torch.tensor([1, 2]), expected=torch.tensor([1, 2]), max=0),
            id="mismatch-ratio",
        ),
        pytest.param(
            expect_grad_sparsity,
            dict(
                actual=torch.ones(2, 3),
                expected=torch.ones(2, 3),
                min_ratio=1.0,
                msg="sparsity",
            ),
            id="grad-sparsity",
        ),
        pytest.param(
            expect_grad_reference_close,
            dict(
                actual=torch.tensor([1.0, -2.0]),
                expected=torch.tensor([1.0, -2.0]),
                rtol=0,
                atol=0,
                max_rel_l2=0,
                max_rel_l1=0,
                min_cosine=1.0 - 1e-15,
                max_signed_bias=0,
                msg="grad",
            ),
            id="grad-reference-close",
        ),
        pytest.param(
            expect_close_with_boundary_band,
            dict(
                actual=torch.tensor([1.0, 2.0]),
                expected=torch.tensor([1.0, 2.0]),
                boundary_mask=torch.tensor([False, True]),
                interior_atol=0,
                interior_rtol=0,
                boundary_max_flip_ratio=0,
                boundary_symmetry_tol=1,
                msg="band",
            ),
            id="boundary-band",
        ),
        pytest.param(
            expect_call,
            dict(
                assert_func=_HELPER.assert_close,
                actual=torch.ones(2),
                expected=torch.ones(2),
                rtol=0,
                atol=0,
            ),
            id="call",
        ),
        pytest.param(
            expect_true,
            dict(condition=True, msg="truthy"),
            id="true",
        ),
    ],
)
def test_expect_helpers_pass_through_success(helper, kwargs):
    helper(**kwargs)


def test_expect_call_wrapper_cache_is_bounded_for_arbitrary_callables():
    """Arbitrary ``expect_call`` callables should not grow the wrapper cache forever."""

    _HELPER._checked.cache_clear()
    try:
        for expected in range(256):

            def assert_matches(value, _expected=expected):
                assert value == _expected

            expect_call(assert_matches, expected)

        cache_info = _HELPER._checked.cache_info()
        assert cache_info.maxsize == 128
        assert cache_info.currsize <= cache_info.maxsize
    finally:
        _HELPER._checked.cache_clear()


@pytest.mark.parametrize(
    "case_name,call_source",
    [
        pytest.param(
            "close",
            """
helper.expect_close(
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="soft close",
)
""",
            id="close",
        ),
        pytest.param(
            "mismatch_ratio",
            """
helper.expect_mismatch_ratio(
    torch.tensor([1]),
    torch.tensor([2]),
    max=0,
)
""",
            id="mismatch-ratio",
        ),
        pytest.param(
            "grad_sparsity",
            """
helper.expect_grad_sparsity(
    torch.zeros(1, 3),
    torch.ones(1, 3),
    min_ratio=0.5,
    msg="soft sparsity",
)
""",
            id="grad-sparsity",
        ),
        pytest.param(
            "grad_reference_close",
            """
helper.expect_grad_reference_close(
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="soft grad",
)
""",
            id="grad-reference-close",
        ),
        pytest.param(
            "boundary_band",
            """
helper.expect_close_with_boundary_band(
    torch.tensor([1.0, 2.0]),
    torch.tensor([1.0, 2.1]),
    boundary_mask=torch.tensor([False, False]),
    interior_atol=0,
    interior_rtol=0,
    boundary_max_flip_ratio=0,
    boundary_symmetry_tol=1,
    msg="soft band",
)
""",
            id="boundary-band",
        ),
        pytest.param(
            "call",
            """
helper.expect_call(
    helper.assert_close,
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="soft call",
)
""",
            id="call",
        ),
        pytest.param(
            "true",
            """
helper.expect_true(False, msg="soft true")
""",
            id="true",
        ),
    ],
)
def test_expect_helper_records_failure_without_stopping(
    pytester, case_name, call_source
):
    """A failing expect helper should behave like GoogleTest EXPECT_*.

    The child test must continue past the failed check and still be reported as
    failed at the end of the test by pytest-check.
    """

    indented_call = textwrap.indent(textwrap.dedent(call_source).strip(), "    ")
    pytester.makepyfile(
        test_soft_expect=f"""
import importlib.util
import pathlib

import torch

helper_path = pathlib.Path({str(_HELPER_PATH)!r})
spec = importlib.util.spec_from_file_location("gsplat_helper_under_test", helper_path)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)


def test_soft_expect_continues():
{indented_call}
    pathlib.Path("continued_{case_name}.txt").write_text("continued")
"""
    )

    result = pytester.runpytest("-q", "-p", "pytest_check")

    result.assert_outcomes(failed=1)
    assert (pytester.path / f"continued_{case_name}.txt").read_text() == "continued"


@pytest.mark.parametrize(
    "case_name,call_source",
    [
        pytest.param(
            "close",
            """
helper.expect_close(
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="group close",
)
""",
            id="close",
        ),
        pytest.param(
            "mismatch_ratio",
            """
helper.expect_mismatch_ratio(
    torch.tensor([1]),
    torch.tensor([2]),
    max=0,
)
""",
            id="mismatch-ratio",
        ),
        pytest.param(
            "grad_sparsity",
            """
helper.expect_grad_sparsity(
    torch.zeros(1, 3),
    torch.ones(1, 3),
    min_ratio=0.5,
    msg="group sparsity",
)
""",
            id="grad-sparsity",
        ),
        pytest.param(
            "grad_reference_close",
            """
helper.expect_grad_reference_close(
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="group grad",
)
""",
            id="grad-reference-close",
        ),
        pytest.param(
            "boundary_band",
            """
helper.expect_close_with_boundary_band(
    torch.tensor([1.0, 2.0]),
    torch.tensor([1.0, 2.1]),
    boundary_mask=torch.tensor([False, False]),
    interior_atol=0,
    interior_rtol=0,
    boundary_max_flip_ratio=0,
    boundary_symmetry_tol=1,
    msg="group band",
)
""",
            id="boundary-band",
        ),
        pytest.param(
            "call",
            """
helper.expect_call(
    helper.assert_close,
    torch.tensor([1.0]),
    torch.tensor([2.0]),
    rtol=0,
    atol=0,
    msg="group call",
)
""",
            id="call",
        ),
        pytest.param(
            "true",
            """
helper.expect_true(False, msg="group true")
""",
            id="true",
        ),
    ],
)
def test_expect_group_allows_direct_expect_calls_then_stops_after_group(
    pytester, case_name, call_source
):
    """Direct expect_* calls should run inside a group, then trip its barrier."""

    indented_call = textwrap.indent(textwrap.dedent(call_source).strip(), "        ")
    pytester.makepyfile(
        test_group=f"""
import importlib.util
import pathlib

import torch

helper_path = pathlib.Path({str(_HELPER_PATH)!r})
spec = importlib.util.spec_from_file_location("gsplat_helper_under_test", helper_path)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)


def test_group_barrier():
    with helper.expect_group("group {case_name}"):
{indented_call}
        pathlib.Path("inside_{case_name}.txt").write_text("inside")
    pathlib.Path("after_{case_name}.txt").write_text("after")
"""
    )

    result = pytester.runpytest("-q", "-p", "pytest_check")

    result.assert_outcomes(failed=1)
    assert (pytester.path / f"inside_{case_name}.txt").read_text() == "inside"
    assert not (pytester.path / f"after_{case_name}.txt").exists()


def test_expect_group_executes_all_direct_expect_calls_before_barrier(pytester):
    """A group should collect all direct expect_* failures before raising."""

    pytester.makepyfile(
        test_group=f"""
import importlib.util
import pathlib

import torch

helper_path = pathlib.Path({str(_HELPER_PATH)!r})
spec = importlib.util.spec_from_file_location("gsplat_helper_under_test", helper_path)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)


def test_group_collects_multiple_failures():
    with helper.expect_group("two failures"):
        helper.expect_close(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            rtol=0,
            atol=0,
            msg="first failure",
        )
        pathlib.Path("after_first.txt").write_text("after first")
        helper.expect_true(False, msg="second failure")
        pathlib.Path("after_second.txt").write_text("after second")
    pathlib.Path("after_group.txt").write_text("after group")
"""
    )

    result = pytester.runpytest("-q", "-p", "pytest_check")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Failed Checks: 2*"])
    assert (pytester.path / "after_first.txt").read_text() == "after first"
    assert (pytester.path / "after_second.txt").read_text() == "after second"
    assert not (pytester.path / "after_group.txt").exists()
