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

"""Tests for GaussianScene + Stage integration with 2DGS splats.

The libs/scene and libs/stage subpackages are optional (installed via
``libs/install.sh scene`` / ``libs/install.sh stage``). On environments
where they are not installed (e.g. upstream GitHub Actions
core_tests.yml on ubuntu-latest), this whole test module is skipped at
collection time via ``pytest.importorskip``.
"""

import torch
import pytest

gsplat_scene = pytest.importorskip(
    "gsplat_scene",
    reason="gsplat_scene not installed (install via libs/install.sh scene)",
)
gsplat_stage = pytest.importorskip(
    "gsplat_stage",
    reason="gsplat_stage not installed (install via libs/install.sh stage)",
)

GaussianScene = gsplat_scene.GaussianScene
Stage = gsplat_stage.Stage


def _make_2dgs_splats(n: int = 100, device: str = "cpu") -> torch.nn.ParameterDict:
    """Create a minimal 2DGS-compatible splat ParameterDict."""
    sh_degree = 3
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3, device=device)),
            "quats": torch.nn.Parameter(torch.randn(n, 4, device=device)),
            "scales": torch.nn.Parameter(torch.randn(n, 3, device=device)),
            "opacities": torch.nn.Parameter(torch.randn(n, device=device)),
            "sh0": torch.nn.Parameter(torch.randn(n, 1, 3, device=device)),
            "shN": torch.nn.Parameter(
                torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, device=device)
            ),
        }
    )


class TestFromSplats2DGS:
    def test_validate_passes(self):
        splats = _make_2dgs_splats()
        scene = GaussianScene.from_splats(splats, id="scene")
        scene.validate()
        assert scene.num_gaussians() == 100

    def test_splat_keys_preserved(self):
        splats = _make_2dgs_splats()
        scene = GaussianScene.from_splats(splats, id="scene")
        assert set(scene.splats.keys()) == {
            "means",
            "quats",
            "scales",
            "opacities",
            "sh0",
            "shN",
        }


class TestStage7TuplePassthrough:
    def test_splats_injected_and_7tuple_returned(self):
        splats = _make_2dgs_splats()
        scene = GaussianScene.from_splats(splats, id="scene")
        stage = Stage()

        received = {}

        def mock_render(splats=None, **kwargs):
            received["splats"] = splats
            received["kwargs"] = kwargs
            return (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                {"meta": True},
            )

        stage.add_scene(scene, mock_render)
        result = stage.render(scene.id, width=800, height=600)

        assert received["splats"] is scene.splats
        assert received["kwargs"] == {"width": 800, "height": 600}
        assert len(result) == 7
        assert result[6] == {"meta": True}


class TestGradientFlowThroughStage:
    def test_backward_propagates(self):
        splats = _make_2dgs_splats()
        scene = GaussianScene.from_splats(splats, id="scene")
        stage = Stage()

        def differentiable_render(splats=None, **kwargs):
            return splats["means"].sum()

        stage.add_scene(scene, differentiable_render)
        out = stage.render(scene.id)
        out.backward()

        assert scene.splats["means"].grad is not None
        assert scene.splats["means"].grad.abs().sum() > 0


class TestTopologyHookShapeConsistency:
    def test_duplicate_keeps_shapes_in_sync(self):
        splats = _make_2dgs_splats(n=50)
        scene = GaussianScene.from_splats(splats, id="scene")

        sel = torch.tensor([0, 5, 10])
        n_before = scene.num_gaussians()

        # Simulate what strategy ops do to splats
        for key in scene.splats:
            scene.splats[key] = torch.nn.Parameter(
                torch.cat([scene.splats[key], scene.splats[key][sel]], dim=0)
            )
        scene.on_duplicate(sel)

        assert scene.component_index.shape[0] == n_before + len(sel)
        assert scene.splats["means"].shape[0] == n_before + len(sel)
        assert scene.component_index.shape[0] == scene.splats["means"].shape[0]

    def test_remove_keeps_shapes_in_sync(self):
        splats = _make_2dgs_splats(n=50)
        scene = GaussianScene.from_splats(splats, id="scene")

        mask = torch.zeros(50, dtype=torch.bool)
        mask[10:20] = True  # remove 10 gaussians

        for key in scene.splats:
            scene.splats[key] = torch.nn.Parameter(scene.splats[key][~mask])
        scene.on_remove(mask)

        assert scene.component_index.shape[0] == 40
        assert scene.splats["means"].shape[0] == 40
        assert scene.component_index.shape[0] == scene.splats["means"].shape[0]
