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

"""Checks for stage-level component collection APIs."""

from __future__ import annotations

import inspect

import pytest
import torch

from gsplat.scene import GaussianScene
from gsplat.stage import ComponentCollection, Stage


def _make_scene(n: int, id: str, offset: float = 0.0) -> GaussianScene:
    base = torch.arange(n, dtype=torch.float32) + offset
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.stack([base, base + 10.0, base + 20.0], dim=1)
            ),
            "scales": torch.nn.Parameter(torch.zeros(n, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1)
            ),
            "opacities": torch.nn.Parameter(torch.zeros(n)),
        }
    )
    return GaussianScene.from_splats(splats, id=id)


def test_component_collection_exports_public_name():
    collection = ComponentCollection(id="world")

    assert collection.id == "world"
    assert collection.members == []
    assert collection.render_fn is None


def test_component_collection_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        ComponentCollection(id="")


def test_component_collection_method_signatures():
    assert "scene" in inspect.signature(ComponentCollection.add_scene).parameters

    collect_sig = inspect.signature(ComponentCollection.collect)
    assert "t" in collect_sig.parameters
    assert collect_sig.parameters["t"].default == 0.0

    render_sig = inspect.signature(ComponentCollection.render)
    assert "t" in render_sig.parameters
    assert render_sig.parameters["t"].default == 0.0

    split_sig = inspect.signature(ComponentCollection.split_info)
    assert "key" in split_sig.parameters
    assert split_sig.parameters["key"].default == "means2d"


def test_stage_exposes_add_collection():
    signature = inspect.signature(Stage.add_collection)

    assert "collection" in signature.parameters
    assert "render_fn" in signature.parameters


def test_component_collection_collects_member_splats():
    first = _make_scene(2, "first", offset=0.0)
    second = _make_scene(3, "second", offset=100.0)
    collection = ComponentCollection(id="world")
    collection.add_scene(first)
    collection.add_scene(second)

    splats = collection.collect(t=0.0)

    assert [scene.id for scene in collection.members] == ["first", "second"]
    torch.testing.assert_close(
        splats["means"],
        torch.cat([first.splats["means"], second.splats["means"]], dim=0),
    )
    assert splats["means"] is not first.splats["means"]


def test_component_collection_collect_forwards_time_to_each_member():
    first = _make_scene(2, "first")
    second = _make_scene(3, "second")
    calls: dict[str, list[float]] = {"first": [], "second": []}

    def make_apply(scene: GaussianScene, name: str):
        def apply_transforms(*, t=None, time_sec=0.0):
            calls[name].append(t if t is not None else time_sec)
            return {key: value for key, value in scene.splats.items()}

        return apply_transforms

    first.apply_transforms = make_apply(first, "first")
    second.apply_transforms = make_apply(second, "second")
    collection = ComponentCollection(id="world")
    collection.add_scene(first)
    collection.add_scene(second)

    collection.collect(t=7.5)

    assert calls == {"first": [7.5], "second": [7.5]}


def test_component_collection_rejects_incompatible_member_schema():
    first = _make_scene(2, "first")
    second = _make_scene(3, "second")
    del second.splats["quats"]
    collection = ComponentCollection(id="world")
    collection.add_scene(first)

    with pytest.raises(ValueError, match="splat keys"):
        collection.add_scene(second)


def test_component_collection_render_uses_shared_render_fn():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    calls = {}

    def render_fn(*, splats, **kwargs):
        calls["splats"] = splats
        calls["kwargs"] = kwargs
        return splats["means"].shape[0]

    collection.render_fn = render_fn

    result = collection.render(t=5.0, camera="front")

    assert result == 5
    assert calls["kwargs"] == {"camera": "front"}
    assert calls["splats"]["means"].shape == (5, 3)


def test_stage_add_collection_and_render():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    stage = Stage()

    def render_fn(*, splats, **kwargs):
        return splats["means"].shape[0], kwargs["view"]

    stage.add_collection(collection, render_fn)

    assert stage.scene_ids() == ["world"]
    assert stage.render("world", t=10, view="front") == (5, "front")


def test_component_collection_split_info_slices_nonpacked_gradients():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    collection.collect(t=0.0)
    means2d = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    means2d.requires_grad_(True)
    means2d.absgrad = torch.full_like(means2d, 2.0)
    info = {
        "means2d": means2d,
        "radii": torch.arange(5, dtype=torch.float32).reshape(1, 5),
        "gaussian_ids": torch.arange(5).reshape(1, 5),
        "width": 64,
        "height": 48,
        "n_cameras": 1,
    }

    collection.step_pre_backward(info)
    means2d.sum().backward()
    first, second = collection.split_info(info)

    torch.testing.assert_close(first["means2d"].grad, torch.ones(1, 2, 2))
    torch.testing.assert_close(second["means2d"].grad, torch.ones(1, 3, 2))
    torch.testing.assert_close(first["means2d"].absgrad, torch.full((1, 2, 2), 2.0))
    torch.testing.assert_close(second["radii"], torch.tensor([[2.0, 3.0, 4.0]]))
    torch.testing.assert_close(second["gaussian_ids"], torch.tensor([[0, 1, 2]]))


def test_component_collection_split_info_uses_known_gaussian_axis():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    collection.collect(t=0.0)
    means2d = torch.arange(50, dtype=torch.float32).reshape(5, 5, 2)
    means2d.requires_grad_(True)
    depths = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    info = {
        "means2d": means2d,
        "depths": depths,
        "width": 64,
        "height": 48,
        "n_cameras": 5,
    }

    first, second = collection.split_info(info)

    assert first["means2d"].shape == (5, 2, 2)
    assert second["means2d"].shape == (5, 3, 2)
    torch.testing.assert_close(first["depths"], depths[:, :2])
    torch.testing.assert_close(second["depths"], depths[:, 2:])


def test_component_collection_split_info_leaves_unknown_metadata_shared():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    collection.collect(t=0.0)
    means2d = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    unknown = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
    info = {
        "means2d": means2d,
        "isect_offsets": unknown,
        "width": 64,
        "height": 48,
        "n_cameras": 1,
    }

    first, second = collection.split_info(info)

    assert first["isect_offsets"] is unknown
    assert second["isect_offsets"] is unknown


def test_component_collection_split_info_rebases_packed_gaussian_ids():
    collection = ComponentCollection(id="world")
    collection.add_scene(_make_scene(2, "first"))
    collection.add_scene(_make_scene(3, "second"))
    collection.collect(t=0.0)
    means2d = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    means2d.requires_grad_(True)
    info = {
        "means2d": means2d,
        "radii": torch.arange(5, dtype=torch.float32),
        "gaussian_ids": torch.tensor([0, 2, 4, 1, 3]),
    }

    collection.step_pre_backward(info)
    means2d.sum().backward()
    first, second = collection.split_info(info)

    torch.testing.assert_close(first["means2d"].grad, torch.ones(2, 2))
    torch.testing.assert_close(second["means2d"].grad, torch.ones(3, 2))
    assert first["means2d"].absgrad is None
    assert second["means2d"].absgrad is None
    torch.testing.assert_close(first["gaussian_ids"], torch.tensor([0, 1]))
    torch.testing.assert_close(second["gaussian_ids"], torch.tensor([0, 2, 1]))
