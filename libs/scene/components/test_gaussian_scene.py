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

"""Tests for GaussianScene core behavior."""

from __future__ import annotations

import io

import pytest
import torch

from gsplat_scene import GaussianScene, Scene


def make_splats(num_gaussians: int = 4, offset: float = 0.0) -> torch.nn.ParameterDict:
    base = torch.arange(num_gaussians, dtype=torch.float32) + offset
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.stack([base, base + 10, base + 20], dim=1)
            ),
            "scales": torch.nn.Parameter(
                torch.stack([base + 30, base + 40, base + 50], dim=1)
            ),
            "quats": torch.nn.Parameter(
                torch.stack([base + 60, base + 70, base + 80, base + 90], dim=1)
            ),
            "opacities": torch.nn.Parameter(base + 100),
            "colors": torch.nn.Parameter(
                torch.stack([base + 110, base + 120, base + 130], dim=1)
            ),
        }
    )


def make_signal(num_gaussians: int = 4, offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "camera": (
            torch.arange(num_gaussians * 2, dtype=torch.float32).reshape(
                num_gaussians, 2
            )
            + offset
        ),
        "lidar": (
            torch.arange(num_gaussians * 3, dtype=torch.float32).reshape(
                num_gaussians, 3
            )
            + 100.0
            + offset
        ),
    }


def make_scene() -> GaussianScene:
    scene = GaussianScene.from_splats(make_splats(), id="scene", signal=make_signal())
    scene.component_names = ["scene", "road", "car"]
    scene.component_index = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    return scene


def test_scene_id_must_be_non_empty_string():
    with pytest.raises(ValueError):
        GaussianScene(id="")


def test_from_splats_requires_id_and_rejects_empty():
    with pytest.raises(TypeError):
        GaussianScene.from_splats(make_splats())  # id missing
    with pytest.raises(ValueError):
        GaussianScene.from_splats(torch.nn.ParameterDict(), id="scene")


def test_gaussian_scene_put_single_component():
    splats = make_splats()
    scene = GaussianScene(id="scene")
    scene.put("road", splats)

    assert scene.splats is splats
    assert scene.component_names == ["road"]
    torch.testing.assert_close(scene.component_index, torch.zeros(4, dtype=torch.long))

    road = scene.get("road")
    assert road["name"] == "road"
    assert road["index"] == 0
    torch.testing.assert_close(road["mask"], torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(road["splats"]["means"], scene.splats["means"])
    torch.testing.assert_close(road["splats"]["colors"], scene.splats["colors"])
    assert road["signal"] == {}


def test_gaussian_scene_from_splats_keeps_live_parameterdict_identity():
    splats = make_splats()
    scene = GaussianScene.from_splats(splats, id="scene")

    assert scene.splats is splats
    assert scene.component_names == ["scene"]
    torch.testing.assert_close(scene.component_index, torch.zeros(4, dtype=torch.long))


def test_put_extends_signal_rows_for_new_components():
    road_splats = make_splats()
    extra_splats = make_splats(offset=100.0)
    signal = make_signal()
    scene = GaussianScene.from_splats(road_splats, id="scene", signal=signal)

    scene.put("car", extra_splats)

    assert scene.num_gaussians() == 8
    for value in scene.signal.values():
        assert value.shape[0] == 8
    torch.testing.assert_close(scene.signal["camera"][:4], signal["camera"])
    torch.testing.assert_close(
        scene.signal["camera"][4:], torch.zeros(4, 2, dtype=torch.float32)
    )


def test_gaussian_scene_put_appends_second_component_and_keeps_indices_aligned():
    road_splats = make_splats()
    car_splats = make_splats(offset=100.0)

    scene = GaussianScene(id="scene")
    scene.put("road", road_splats)
    scene.put("car", car_splats)

    assert scene.component_names == ["road", "car"]
    torch.testing.assert_close(
        scene.component_index,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
    )

    road = scene.get("road")
    car = scene.get("car")
    torch.testing.assert_close(
        road["mask"],
        torch.tensor([True, True, True, True, False, False, False, False]),
    )
    torch.testing.assert_close(
        car["mask"],
        torch.tensor([False, False, False, False, True, True, True, True]),
    )
    torch.testing.assert_close(road["splats"]["means"], road_splats["means"])
    torch.testing.assert_close(car["splats"]["means"], car_splats["means"])
    assert road["signal"] == {}
    assert car["signal"] == {}


def test_put_preserves_requires_grad_when_appending_components():
    road_splats = make_splats()
    road_splats["colors"].requires_grad = False

    scene = GaussianScene(id="scene")
    scene.put("road", road_splats)
    scene.put("car", make_splats(offset=100.0))

    assert scene.splats["colors"].requires_grad is False
    assert scene.splats["means"].requires_grad is True


def test_gaussian_scene_state_dict_round_trip_preserves_scene():
    scene = make_scene()
    scene.splats["colors"].requires_grad = False

    state = scene.state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = GaussianScene.from_state_dict(torch.load(buffer, map_location="cpu"))

    assert restored.splats is not scene.splats
    assert restored.component_names == scene.component_names
    torch.testing.assert_close(restored.component_index, scene.component_index)
    torch.testing.assert_close(restored.signal["camera"], scene.signal["camera"])
    torch.testing.assert_close(restored.signal["lidar"], scene.signal["lidar"])
    torch.testing.assert_close(restored.splats["means"], scene.splats["means"])
    torch.testing.assert_close(restored.splats["colors"], scene.splats["colors"])
    assert restored.splats["colors"].requires_grad is False


def test_gaussian_scene_from_state_dict_defaults_component_names_when_absent():
    scene = GaussianScene.from_splats(make_splats(), id="scene")
    state = scene.state_dict()
    del state["component_names"]

    restored = GaussianScene.from_state_dict(state)

    assert restored.component_names == ["scene"]
    torch.testing.assert_close(
        restored.component_index, torch.zeros(4, dtype=torch.long)
    )


def test_gaussian_scene_from_state_dict_empty_splats_round_trip():
    scene = GaussianScene(id="empty")
    state = scene.state_dict()
    restored = GaussianScene.from_state_dict(state)
    assert restored.id == "empty"
    assert restored.num_gaussians() == 0
    assert restored.component_names == []


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gsplat::relocation is CUDA-only"
)
def test_strategy_ops_keep_scene_aligned():
    """Every strategy op must drive the matching on_* hook so sidecars stay in sync."""
    from gsplat.strategy import ops

    def fresh() -> tuple[GaussianScene, dict[str, torch.optim.Optimizer]]:
        splats = make_splats()
        splats["shN"] = torch.nn.Parameter(torch.randn(4, 15, 3))
        splats["opacities"].data = torch.tensor([-10.0, 5.0, -10.0, 5.0])
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.detach().cuda()) for k, v in splats.items()}
        )
        scene = GaussianScene.from_splats(
            splats, id="scene", signal={"w": torch.arange(4).unsqueeze(1).float()}
        )
        opts = {
            name: torch.optim.Adam([{"params": p, "name": name}], lr=1e-3)
            for name, p in scene.splats.items()
        }
        return scene, opts

    binoms = torch.ones(5, 5, device="cuda")

    scene, opts = fresh()
    ops.duplicate(
        scene.splats,
        opts,
        {},
        torch.tensor([True, False, True, False]).cuda(),
        scene=scene,
    )
    assert scene.num_gaussians() == 6
    assert scene.component_index.shape[0] == 6
    assert scene.signal["w"].shape[0] == 6
    assert scene.splats["shN"].shape == (6, 15, 3)

    scene, opts = fresh()
    ops.split(
        scene.splats,
        opts,
        {},
        torch.tensor([False, True, False, True]).cuda(),
        scene=scene,
    )
    assert scene.num_gaussians() == 6  # (4-2) + 2*2
    assert scene.component_index.shape[0] == 6
    assert scene.signal["w"].shape[0] == 6
    assert scene.splats["shN"].shape == (6, 15, 3)

    scene, opts = fresh()
    ops.remove(
        scene.splats,
        opts,
        {},
        torch.tensor([False, True, False, True]).cuda(),
        scene=scene,
    )
    assert scene.num_gaussians() == 2
    assert scene.component_index.shape[0] == 2
    assert scene.signal["w"].shape[0] == 2

    scene, opts = fresh()
    dead_mask = torch.sigmoid(scene.splats["opacities"]) <= 0.005
    ops.relocate(scene.splats, opts, {}, dead_mask, binoms, scene=scene)
    assert scene.num_gaussians() == 4
    assert scene.component_index.shape[0] == 4
    assert scene.signal["w"].shape[0] == 4

    scene, opts = fresh()
    ops.sample_add(scene.splats, opts, {}, n=3, binoms=binoms, scene=scene)
    assert scene.num_gaussians() == 7
    assert scene.component_index.shape[0] == 7
    assert scene.signal["w"].shape[0] == 7


def test_gaussian_scene_topology_hooks_keep_sidecars_in_sync():
    scene = make_scene()

    scene.on_duplicate(torch.tensor([True, False, True, False]))
    torch.testing.assert_close(
        scene.component_index, torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    )
    torch.testing.assert_close(
        scene.signal["camera"],
        torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [0.0, 1.0], [4.0, 5.0]]
        ),
    )

    scene = make_scene()
    scene.on_split(
        torch.tensor([False, True, False, True]),
        torch.tensor([True, False, True, False]),
    )
    torch.testing.assert_close(
        scene.component_index,
        torch.tensor([0, 2, 1, 1, 1, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        scene.signal["lidar"],
        torch.tensor(
            [
                [100.0, 101.0, 102.0],
                [106.0, 107.0, 108.0],
                [103.0, 104.0, 105.0],
                [109.0, 110.0, 111.0],
                [103.0, 104.0, 105.0],
                [109.0, 110.0, 111.0],
            ]
        ),
    )

    scene = make_scene()
    scene.on_remove(torch.tensor([False, True, False, True]))
    torch.testing.assert_close(
        scene.component_index, torch.tensor([0, 2], dtype=torch.long)
    )
    torch.testing.assert_close(
        scene.signal["camera"], torch.tensor([[0.0, 1.0], [4.0, 5.0]])
    )

    scene = make_scene()
    scene.on_relocate(
        torch.tensor([1, 3], dtype=torch.long), torch.tensor([0, 2], dtype=torch.long)
    )
    torch.testing.assert_close(
        scene.component_index, torch.tensor([0, 0, 2, 2], dtype=torch.long)
    )
    torch.testing.assert_close(
        scene.signal["camera"],
        torch.tensor([[0.0, 1.0], [0.0, 1.0], [4.0, 5.0], [4.0, 5.0]]),
    )

    scene = make_scene()
    scene.on_sample_add(torch.tensor([0, 2], dtype=torch.long))
    torch.testing.assert_close(
        scene.component_index,
        torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long),
    )
    torch.testing.assert_close(
        scene.signal["camera"],
        torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [0.0, 1.0], [4.0, 5.0]]
        ),
    )
