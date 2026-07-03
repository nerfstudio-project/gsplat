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

import glob
import importlib
import math
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
    test). Instead, we stub ``gsplat`` with a minimal package shell so the
    submodule import resolves without triggering the full package init.
    """

    gsplat_pkg = types.ModuleType("gsplat")
    gsplat_pkg.__path__ = [str(PROFILE_PATH.parent)]
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_pkg)

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


# --- Override-parsing helpers (`_parse_override_value`, `_parse_input_override`)
# Reached via the ``load_profile`` fixture so the gsplat package init (which
# pulls in CUDA-heavy modules) is not triggered at test-collection time.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8", 8),
        ("8.5", 8.5),
        ("true", True),
        ("False", False),
        ("null", None),
        ("None", None),
        ("[1, 2, 3]", [1, 2, 3]),
        ("(1, 2)", (1, 2)),
        ('"pinhole"', "pinhole"),
        ("pinhole", "pinhole"),
        ("", ""),
    ],
)
def test_parse_override_value(raw, expected, load_profile):
    assert load_profile._parse_override_value(raw) == expected


def test_parse_override_value_nan(load_profile):
    assert math.isnan(load_profile._parse_override_value("nan"))


def test_parse_input_override(load_profile):
    assert load_profile._parse_input_override("tile_size=8") == ("tile_size", 8)
    assert load_profile._parse_input_override(" camera_model = ftheta ") == (
        "camera_model",
        "ftheta",
    )


@pytest.mark.parametrize("raw", ["tile_size", "=8", "camera.foo=1", "a[0]=1"])
def test_parse_input_override_rejects_bad_names(raw, load_profile):
    with pytest.raises(ValueError):
        load_profile._parse_input_override(raw)


@pytest.mark.parametrize(
    ("raw", "expected_type"),
    [
        ("mixedbatch", "RendererConfig_MixedBatch"),
        ("MixedBatch", "RendererConfig_MixedBatch"),
        ("MIXEDBATCH", "RendererConfig_MixedBatch"),
        ("parallelbatch", "RendererConfig_ParallelBatch"),
        ("ParallelBatch", "RendererConfig_ParallelBatch"),
        ("PARALLELBATCH", "RendererConfig_ParallelBatch"),
    ],
)
def test_parse_renderer_config_override_names(
    load_profile, monkeypatch, raw, expected_type
):
    rendering = types.ModuleType("gsplat.rendering")
    for type_name in {
        "RendererConfig_MixedBatch",
        "RendererConfig_ParallelBatch",
    }:
        setattr(rendering, type_name, type(type_name, (), {}))
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)

    value = load_profile._parse_renderer_config_override(raw)

    assert type(value).__name__ == expected_type


def test_parse_renderer_config_override_keeps_non_string(load_profile):
    value = object()

    assert load_profile._parse_renderer_config_override(value) is value


@pytest.mark.parametrize(
    "raw",
    [
        "default",
        "mixed",
        "mixed_batch",
        "RendererConfig_MixedBatch",
        "serialbatch",
        "parallelbathc",
        "futurebatch",
    ],
)
def test_parse_renderer_config_override_rejects_unknown(load_profile, raw):
    with pytest.raises(ValueError, match="renderer_config override"):
        load_profile._parse_renderer_config_override(raw)


def test_apply_channel_override_slices_and_tiles(load_profile):
    # Slice when the target count is below the source channel count, ...
    inputs = {"colors": torch.zeros(1, 5, 8), "sh_degree": None}
    load_profile._apply_channel_override(inputs, 3)
    assert inputs["colors"].shape[-1] == 3
    # ... tile-and-truncate when above it.
    inputs = {"colors": torch.zeros(1, 5, 3), "sh_degree": None}
    load_profile._apply_channel_override(inputs, 8)
    assert inputs["colors"].shape[-1] == 8


def test_apply_channel_override_tiles_by_wrapping_channels(load_profile):
    # Tiling above the source count repeats the channel block then truncates,
    # so channels wrap: [c0, c1, c2, c0, c1, c2, c0, c1] for cur=3 -> 8.
    colors = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3)
    inputs = {"colors": colors, "sh_degree": None}
    load_profile._apply_channel_override(inputs, 8)
    expected = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0])
    assert torch.equal(inputs["colors"].reshape(-1), expected)


def test_apply_channel_override_preserves_grad_leaf(load_profile):
    # The resized tensor must stay a leaf carrying requires_grad so backward
    # replay can clear `.grad` through `_clear_replay_grads()`.
    colors = torch.zeros(1, 5, 4, requires_grad=True)
    inputs = {"colors": colors, "sh_degree": None}
    load_profile._apply_channel_override(inputs, 8)
    out = inputs["colors"]
    assert out.is_leaf and out.requires_grad


def test_apply_channel_override_then_normalize_is_camera_shaped(load_profile):
    # End-to-end shape contract: --channels followed by the 2DGS normalize step
    # must leave colors camera-shaped [..., C, N, channels], which
    # rasterize_to_pixels_2dgs requires. This pins the ordering of the two
    # steps, which is the load-bearing part of the feature.
    n_gaussians, n_cameras, channels = 5, 2, 3
    inputs = {
        "means": torch.zeros(n_gaussians, 3),
        "viewmats": torch.zeros(n_cameras, 4, 4),
        "colors": torch.zeros(n_gaussians, 8),
        "sh_degree": None,
        "packed": False,
    }
    load_profile._apply_channel_override(inputs, channels)
    load_profile._normalize_2dgs_replay_inputs(inputs)
    assert inputs["colors"].shape == (n_cameras, n_gaussians, channels)


def test_apply_channel_override_collapses_sh_to_dc_band(load_profile):
    # SH input [N, K, D] collapses to its DC band [N, D] and clears sh_degree
    # so the rasterizer templates on the post-activation channel count. K=4 is
    # consistent with sh_degree=1 ((sh_degree+1)**2 == 4).
    inputs = {"colors": torch.zeros(1, 5, 4, 3), "sh_degree": 1}
    load_profile._apply_channel_override(inputs, 3)
    assert inputs["sh_degree"] is None
    assert inputs["colors"].shape == (1, 5, 3)


def test_apply_channel_override_keeps_extra_signals_that_fit(load_profile):
    # A width that holds the captured colors and extra signals keeps both;
    # the post-concat templated count equals --channels exactly.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "extra_signals": torch.zeros(1, 5, 2),
        "sh_degree": None,
    }
    load_profile._apply_channel_override(inputs, 10)
    assert inputs["colors"].shape[-1] == 8
    assert inputs["extra_signals"].shape[-1] == 2


def test_apply_channel_override_extra_signals_pay_first(load_profile):
    # A width below the captured total trims the extra signals from the tail
    # first; colors keep their native width while any extra signal remains.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "extra_signals": torch.zeros(1, 5, 2),
        "sh_degree": None,
    }
    load_profile._apply_channel_override(inputs, 9)
    assert inputs["colors"].shape[-1] == 8
    assert inputs["extra_signals"].shape[-1] == 1


def test_apply_channel_override_trims_colors_after_extra_signals(load_profile):
    # Once the extra signals are exhausted they are dropped entirely (with
    # their SH degree) and the color block pays for the remaining reduction.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "extra_signals": torch.zeros(1, 5, 2),
        "sh_degree": None,
    }
    load_profile._apply_channel_override(inputs, 6)
    assert inputs["colors"].shape[-1] == 6
    assert inputs["extra_signals"] is None
    assert inputs["extra_signals_sh_degree"] is None


def test_apply_channel_override_resizes_backgrounds(load_profile):
    # backgrounds is asserted against the pre-depth feature width, so it must
    # track the color resize or the wrapper's shape assert fires.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "backgrounds": torch.zeros(1, 8),
        "sh_degree": None,
    }
    load_profile._apply_channel_override(inputs, 3)
    assert inputs["backgrounds"].shape[-1] == 3


def test_apply_channel_override_accounts_for_depth_channel(load_profile):
    # A color+depth render_mode appends one depth channel onto the feature
    # tensor, so the color block is sized to channels - 1.
    inputs = {"colors": torch.zeros(1, 5, 8), "sh_degree": None, "render_mode": "RGB+D"}
    load_profile._apply_channel_override(inputs, 4)
    assert inputs["colors"].shape[-1] == 3


def test_apply_channel_override_rejects_depth_only_render_mode(load_profile):
    # Depth-only modes discard colors before the kernel, so --channels would be
    # a silent no-op; reject it instead.
    inputs = {"colors": torch.zeros(1, 5, 8), "sh_degree": None, "render_mode": "D"}
    with pytest.raises(ValueError):
        load_profile._apply_channel_override(inputs, 3)


def test_apply_channel_override_rejects_channels_too_small(load_profile):
    # Even with every extra signal dropped, at least one color channel must
    # fit next to the reserved depth channel.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "extra_signals": torch.zeros(1, 5, 2),
        "sh_degree": None,
        "render_mode": "RGB+D",
    }
    with pytest.raises(ValueError):
        load_profile._apply_channel_override(inputs, 1)


def test_apply_channel_override_depth_squeezes_out_extra_signals(load_profile):
    # The old refusal boundary: depth + one color still fits once the extra
    # signals pay, so --channels=2 now trims instead of raising.
    inputs = {
        "colors": torch.zeros(1, 5, 8),
        "extra_signals": torch.zeros(1, 5, 2),
        "sh_degree": None,
        "render_mode": "RGB+D",
    }
    load_profile._apply_channel_override(inputs, 2)
    assert inputs["colors"].shape[-1] == 1
    assert inputs["extra_signals"] is None


def test_apply_channel_override_rejects_malformed_sh(load_profile):
    with pytest.raises(ValueError):  # K=4 inconsistent with sh_degree=2 (needs 9)
        load_profile._apply_channel_override(
            {"colors": torch.zeros(1, 5, 4, 3), "sh_degree": 2}, 3
        )
    with pytest.raises(ValueError):  # already-activated [N, D] is not SH-shaped
        load_profile._apply_channel_override(
            {"colors": torch.zeros(5, 3), "sh_degree": 1}, 3
        )


def test_apply_channel_override_rejects_bad_input(load_profile):
    with pytest.raises(ValueError):  # colors is None
        load_profile._apply_channel_override({"colors": None}, 3)
    with pytest.raises(ValueError):  # target count must be >= 1
        load_profile._apply_channel_override({"colors": torch.zeros(1, 5, 3)}, 0)
    with pytest.raises(ValueError):  # zero-channel source must not divide by zero
        load_profile._apply_channel_override({"colors": torch.zeros(1, 5, 0)}, 3)
