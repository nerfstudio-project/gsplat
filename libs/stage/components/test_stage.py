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

from __future__ import annotations

import pytest
import torch

from gsplat_scene import GaussianScene
from gsplat_stage import Stage


def _make_scene(n: int = 10, id: str = "scene") -> GaussianScene:
    """Create a minimal GaussianScene with n Gaussians."""
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3)),
            "scales": torch.nn.Parameter(torch.randn(n, 3)),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(torch.randn(n)),
        }
    )
    return GaussianScene.from_splats(splats, id=id)


def _mock_render_fn(splats, **kwargs):
    """Returns splats and kwargs so the caller can inspect what was received."""
    return splats, kwargs


class TestStageBasics:
    def test_add_scene(self):
        stage = Stage()
        scene = _make_scene()
        stage.add_scene(scene, _mock_render_fn)
        assert stage.scene_ids() == ["scene"]
        assert stage.get_scene("scene") is scene

    def test_add_multiple_scenes(self):
        stage = Stage()
        s1 = _make_scene(5, id="ego")
        s2 = _make_scene(8, id="map")
        stage.add_scene(s1, _mock_render_fn)
        stage.add_scene(s2, _mock_render_fn)
        assert stage.scene_ids() == ["ego", "map"]
        assert stage.get_scene("ego") is s1
        assert stage.get_scene("map") is s2

    def test_get_scene_unknown_id_raises(self):
        stage = Stage()
        with pytest.raises(KeyError, match="not registered"):
            stage.get_scene("nope")

    def test_add_duplicate_id_raises(self):
        stage = Stage()
        stage.add_scene(_make_scene(id="ego"), _mock_render_fn)
        with pytest.raises(ValueError, match="already registered"):
            stage.add_scene(_make_scene(id="ego"), _mock_render_fn)


class TestRender:
    def test_returns_render_fn_output(self):
        stage = Stage()
        scene = _make_scene()
        stage.add_scene(scene, _mock_render_fn)
        splats_received, kwargs_received = stage.render("scene", foo="bar", baz=42)
        assert splats_received is scene.splats
        assert kwargs_received == {"foo": "bar", "baz": 42}

    def test_splats_passed_as_kwarg(self):
        """Verify splats arrives as a keyword arg, not positional."""
        call_log = {}

        def capture_fn(**kwargs):
            call_log.update(kwargs)
            return "ok"

        stage = Stage()
        scene = _make_scene()
        stage.add_scene(scene, capture_fn)
        result = stage.render("scene", camtoworlds="cam", Ks="ks")
        assert result == "ok"
        assert "splats" in call_log
        assert call_log["splats"] is scene.splats
        assert call_log["camtoworlds"] == "cam"

    def test_renders_correct_scene_from_multi(self):
        stage = Stage()
        s1 = _make_scene(5, id="ego")
        s2 = _make_scene(8, id="map")

        def fn_a(splats, **kwargs):
            return "a", splats["means"].shape[0]

        def fn_b(splats, **kwargs):
            return "b", splats["means"].shape[0]

        stage.add_scene(s1, fn_a)
        stage.add_scene(s2, fn_b)
        assert stage.render("ego") == ("a", 5)
        assert stage.render("map") == ("b", 8)

    def test_unknown_id_raises(self):
        stage = Stage()
        stage.add_scene(_make_scene(), _mock_render_fn)
        with pytest.raises(KeyError, match="not registered"):
            stage.render("nonexistent")

    def test_errors_on_empty(self):
        stage = Stage()
        with pytest.raises(KeyError, match="not registered"):
            stage.render("scene")


class TestReturnPassthrough:
    def test_two_tuple(self):
        """AV trainer pattern: render_fn returns (renders, alphas)."""

        def render_2(splats, **kwargs):
            return torch.zeros(1), torch.ones(1)

        stage = Stage()
        stage.add_scene(_make_scene(), render_2)
        r, a = stage.render("scene")
        assert r.shape == (1,)
        assert a.shape == (1,)

    def test_three_tuple(self):
        """Simple trainer pattern: render_fn returns (renders, alphas, info)."""

        def render_3(splats, **kwargs):
            return torch.zeros(1), torch.ones(1), {"radii": torch.zeros(10)}

        stage = Stage()
        stage.add_scene(_make_scene(), render_3)
        r, a, info = stage.render("scene")
        assert "radii" in info
