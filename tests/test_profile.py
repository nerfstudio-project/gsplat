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

"""Unit tests for ``gsplat.profile`` — the input-capture decorator.

The decorator reads its configuration from the environment at the moment it
wraps a target function, so each test builds a fresh decoration under a
freshly-reloaded copy of the module (to reset the process-wide
``_pending_captures`` / ``_next_capture_id`` state between cases).
"""

import contextlib
import glob
import importlib
import os
import re
import sys
import types
from pathlib import Path

import pytest
import torch

PROFILE_PATH = Path(__file__).resolve().parents[1] / "gsplat" / "profile.py"


@pytest.fixture
def load_profile(monkeypatch):
    """Return a freshly-imported copy of ``gsplat.profile``.

    Importing ``gsplat.profile`` via the real package would run ``gsplat``'s
    init (which pulls in CUDA-heavy modules unrelated to the decorator under
    test). Instead, we stub ``gsplat`` with a minimal package shell and supply
    a no-op ``gsplat.trace`` — the only symbol ``profile`` imports from it is
    ``trace_range``, used solely by ``main()`` which these tests don't exercise.
    """

    gsplat_pkg = types.ModuleType("gsplat")
    gsplat_pkg.__path__ = [str(PROFILE_PATH.parent)]
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_pkg)

    trace_module = types.ModuleType("gsplat.trace")

    @contextlib.contextmanager
    def _trace_range(*_args, **_kwargs):
        yield

    trace_module.trace_range = _trace_range
    monkeypatch.setitem(sys.modules, "gsplat.trace", trace_module)

    sys.modules.pop("gsplat.profile", None)
    module = importlib.import_module("gsplat.profile")
    return importlib.reload(module)


def _identity(x):
    return x


def _decorate(profile, envvar, target=None):
    """Decorate a dummy function with ``capture_inputs`` and return the wrapped fn."""
    fn = _identity if target is None else target
    return profile.capture_inputs(envvar=envvar)(fn)


def _saved_files(stem: str):
    """Collect all capture files that share the given stem prefix, sorted."""
    return sorted(glob.glob(f"{stem}_*_*.pt"))


def _call_index_from_path(path: str) -> int:
    """Extract the numeric call index from a capture filename."""
    match = re.search(r"_(\d+)\.pt$", path)
    assert match is not None, path
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Envvar handling
# ---------------------------------------------------------------------------


def test_no_envvar_returns_original_function(load_profile, monkeypatch):
    monkeypatch.delenv("MY_CAPTURE", raising=False)

    def fn(x):
        return x + 1

    wrapped = load_profile.capture_inputs(envvar="MY_CAPTURE")(fn)
    assert wrapped is fn


# ---------------------------------------------------------------------------
# Single-spec backward compatibility
# ---------------------------------------------------------------------------


def test_single_spec_captures_stop_calls(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:2")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    # Calls 0 and 1 are captured; the third call triggers SystemExit before we
    # see call index 2.
    wrapped(torch.tensor([0.0]))
    with pytest.raises(SystemExit):
        wrapped(torch.tensor([1.0]))

    files = _saved_files(stem)
    assert [_call_index_from_path(p) for p in files] == [0, 1]


def test_single_spec_start_stop(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:2:4")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    wrapped(torch.tensor([0.0]))  # call 0 — skipped
    wrapped(torch.tensor([1.0]))  # call 1 — skipped
    wrapped(torch.tensor([2.0]))  # call 2 — captured
    with pytest.raises(SystemExit):
        wrapped(torch.tensor([3.0]))  # call 3 — captured + exit

    assert [_call_index_from_path(p) for p in _saved_files(stem)] == [2, 3]


def test_single_spec_start_stop_step(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1:6:2")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        for i in range(6):
            wrapped(torch.tensor([float(i)]))

    assert [_call_index_from_path(p) for p in _saved_files(stem)] == [1, 3, 5]


# ---------------------------------------------------------------------------
# Multi-spec routing
# ---------------------------------------------------------------------------


def test_multi_spec_routes_calls_to_distinct_paths(load_profile, monkeypatch, tmp_path):
    cam_stem = str(tmp_path / "camera")
    lidar_stem = str(tmp_path / "lidar")
    monkeypatch.setenv("MY_CAPTURE", f"{cam_stem}:1,{lidar_stem}:1:2")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    wrapped(torch.tensor([0.0]))  # call 0 -> camera
    with pytest.raises(SystemExit):
        wrapped(torch.tensor([1.0]))  # call 1 -> lidar, then SystemExit

    cam_files = _saved_files(cam_stem)
    lidar_files = _saved_files(lidar_stem)
    assert len(cam_files) == 1
    assert len(lidar_files) == 1
    assert _call_index_from_path(cam_files[0]) == 0
    assert _call_index_from_path(lidar_files[0]) == 1


def test_multi_spec_padding_uses_global_max_index(load_profile, monkeypatch, tmp_path):
    a_stem = str(tmp_path / "a")
    b_stem = str(tmp_path / "b")
    # Max call index across both specs is 12 → all filenames should zero-pad
    # to 2 digits, including the index-0 file under ``a``.
    monkeypatch.setenv("MY_CAPTURE", f"{a_stem}:1,{b_stem}:12:13")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        for i in range(13):
            wrapped(torch.tensor([float(i)]))

    a_files = _saved_files(a_stem)
    b_files = _saved_files(b_stem)
    assert len(a_files) == 1 and a_files[0].endswith("_00.pt")
    assert len(b_files) == 1 and b_files[0].endswith("_12.pt")


def test_multi_spec_with_step_range(load_profile, monkeypatch, tmp_path):
    a_stem = str(tmp_path / "a")
    b_stem = str(tmp_path / "b")
    # a captures call 0 only; b captures calls 3 and 5 (range(3, 6, 2)).
    monkeypatch.setenv("MY_CAPTURE", f"{a_stem}:1,{b_stem}:3:6:2")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        for i in range(6):
            wrapped(torch.tensor([float(i)]))

    assert [_call_index_from_path(p) for p in _saved_files(a_stem)] == [0]
    assert [_call_index_from_path(p) for p in _saved_files(b_stem)] == [3, 5]


def test_multi_spec_tolerates_whitespace_and_empty_entries(
    load_profile, monkeypatch, tmp_path
):
    a_stem = str(tmp_path / "a")
    b_stem = str(tmp_path / "b")
    monkeypatch.setenv("MY_CAPTURE", f"  {a_stem}:1 , ,{b_stem}:1:2,")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    wrapped(torch.tensor([0.0]))
    with pytest.raises(SystemExit):
        wrapped(torch.tensor([1.0]))

    assert len(_saved_files(a_stem)) == 1
    assert len(_saved_files(b_stem)) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_overlapping_specs_raise(load_profile, monkeypatch, tmp_path):
    a_stem = str(tmp_path / "a")
    b_stem = str(tmp_path / "b")
    monkeypatch.setenv("MY_CAPTURE", f"{a_stem}:0:3,{b_stem}:2:4")
    with pytest.raises(ValueError, match="claimed by multiple specs"):
        _decorate(load_profile, "MY_CAPTURE")


def test_negative_range_values_raise(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:-1:3")
    with pytest.raises(ValueError, match="negative values"):
        _decorate(load_profile, "MY_CAPTURE")


def test_stop_less_than_start_raises(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:5:3")
    with pytest.raises(ValueError, match="must be >="):
        _decorate(load_profile, "MY_CAPTURE")


def test_empty_range_raises(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:0")
    with pytest.raises(ValueError, match="empty range"):
        _decorate(load_profile, "MY_CAPTURE")


def test_missing_range_ints_raise(load_profile, monkeypatch, tmp_path):
    monkeypatch.setenv("MY_CAPTURE", "just_a_path")
    with pytest.raises(ValueError, match="expected <path>"):
        _decorate(load_profile, "MY_CAPTURE")


def test_all_specs_blank_raises(load_profile, monkeypatch):
    monkeypatch.setenv("MY_CAPTURE", " , ,")
    with pytest.raises(ValueError, match="no specs provided"):
        _decorate(load_profile, "MY_CAPTURE")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_capture_dir_prepended_for_relative_paths(load_profile, monkeypatch, tmp_path):
    monkeypatch.setenv("GSPLAT_INPUT_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MY_CAPTURE", "cam:1,lidar:1:2")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    wrapped(torch.tensor([0.0]))
    with pytest.raises(SystemExit):
        wrapped(torch.tensor([1.0]))

    assert len(_saved_files(str(tmp_path / "cam"))) == 1
    assert len(_saved_files(str(tmp_path / "lidar"))) == 1


def test_capture_dir_ignored_for_absolute_paths(load_profile, monkeypatch, tmp_path):
    other_dir = tmp_path / "somewhere_else"
    other_dir.mkdir()
    monkeypatch.setenv("GSPLAT_INPUT_CAPTURE_DIR", str(tmp_path / "ignored"))
    stem = str(other_dir / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        wrapped(torch.tensor([0.0]))

    assert len(_saved_files(stem)) == 1


def test_default_extension_is_pt(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "noext")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        wrapped(torch.tensor([0.0]))

    files = _saved_files(stem)
    assert len(files) == 1 and files[0].endswith(".pt")


def test_custom_extension_preserved(load_profile, monkeypatch, tmp_path):
    base = str(tmp_path / "out.bin")
    monkeypatch.setenv("MY_CAPTURE", f"{base}:1")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        wrapped(torch.tensor([0.0]))

    files = sorted(glob.glob(str(tmp_path / "out_*_*.bin")))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Worker tag: RANK vs PID
# ---------------------------------------------------------------------------


def test_rank_tag_used_when_RANK_is_set(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        wrapped(torch.tensor([0.0]))

    files = sorted(glob.glob(f"{stem}_r3_*.pt"))
    assert len(files) == 1


def test_pid_tag_used_without_RANK(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1")
    wrapped = _decorate(load_profile, "MY_CAPTURE")

    with pytest.raises(SystemExit):
        wrapped(torch.tensor([0.0]))

    files = sorted(glob.glob(f"{stem}_p{os.getpid()}_*.pt"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Saved-content sanity & coexistence
# ---------------------------------------------------------------------------


def test_captured_payload_matches_arguments(load_profile, monkeypatch, tmp_path):
    stem = str(tmp_path / "out")
    monkeypatch.setenv("MY_CAPTURE", f"{stem}:1")
    sentinel = torch.tensor([1.0, 2.0, 3.0])

    def target(a, b=7):
        return a, b

    wrapped = load_profile.capture_inputs(envvar="MY_CAPTURE")(target)
    with pytest.raises(SystemExit):
        wrapped(sentinel, b=9)

    (path,) = _saved_files(stem)
    saved = torch.load(path, weights_only=False)
    assert set(saved.keys()) == {"a", "b"}
    assert torch.equal(saved["a"], sentinel)
    assert saved["b"] == 9


def test_exit_waits_for_all_pending_decorators(load_profile, monkeypatch, tmp_path):
    """SystemExit must only fire once every registered decorator is done."""
    stem_a = str(tmp_path / "a")
    stem_b = str(tmp_path / "b")
    monkeypatch.setenv("CAP_A", f"{stem_a}:1")
    monkeypatch.setenv("CAP_B", f"{stem_b}:1")

    fn_a = load_profile.capture_inputs(envvar="CAP_A")(lambda x: x)
    fn_b = load_profile.capture_inputs(envvar="CAP_B")(lambda x: x)

    # A finishes first; SystemExit should NOT fire while B is still pending.
    fn_a(torch.tensor([0.0]))
    assert len(_saved_files(stem_a)) == 1

    # Now B finishes → exit fires.
    with pytest.raises(SystemExit):
        fn_b(torch.tensor([0.0]))
    assert len(_saved_files(stem_b)) == 1
