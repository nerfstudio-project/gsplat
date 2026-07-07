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

"""Checks for scene transform public APIs."""

from __future__ import annotations

import inspect

import pytest
import torch

from gsplat.scene import (
    GaussianComponent,
    GaussianScene,
    HIDDEN_OPACITY_LOGIT,
    IdentityOp,
    RigidTransformOp,
    Scene,
    TensorViews,
    TransformCtxView,
    TransformGraph,
    TransformOp,
)


class _OffsetOp(TransformOp):
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def state_dict(self) -> dict[str, object]:
        return {"type": self.__class__.__name__, "offset": self.offset}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "_OffsetOp":
        return cls(float(state["offset"]))

    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        del ctx, count

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        transformed = dict(collected)
        transformed["means"] = transformed["means"] + self.offset
        return transformed


def make_splats(num_gaussians: int = 2) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(num_gaussians, 3)),
            "scales": torch.nn.Parameter(torch.zeros(num_gaussians, 3)),
            "quats": torch.nn.Parameter(torch.zeros(num_gaussians, 4)),
            "opacities": torch.nn.Parameter(torch.zeros(num_gaussians)),
        }
    )


def test_transform_exports_public_names():
    graph = TransformGraph([IdentityOp()])
    assert isinstance(graph.ops[0], IdentityOp)
    assert isinstance(RigidTransformOp(), RigidTransformOp)
    assert HIDDEN_OPACITY_LOGIT < 0

    parent = {"means": torch.zeros(2, 3)}
    splat_views = TensorViews(parent, lambda: (0, 2))
    signal_views = TensorViews({}, lambda: (0, 0))
    ctx_buffer = {"poses": torch.zeros(1, 7)}
    ctx_view = TransformCtxView(ctx_buffer, {"poses": (0, 1)})
    component = GaussianComponent(
        component_idx=0,
        splats=splat_views,
        signal=signal_views,
        transform_ctx=ctx_view,
    )
    assert component.component_idx == 0
    assert component.transform_ctx is ctx_view


def test_gaussian_scene_put_accepts_ctx_keyword():
    base_signature = inspect.signature(Scene.put)
    signature = inspect.signature(GaussianScene.put)
    assert "ctx" in base_signature.parameters
    assert base_signature.parameters["ctx"].default is None
    assert "ctx" in signature.parameters
    assert signature.parameters["ctx"].default is None


def test_gaussian_scene_static_put_still_works():
    scene = GaussianScene(id="scene")
    assert scene.ctx_buffer == {}
    assert scene._graph is None
    assert scene._component_ctx == {}

    scene.put("static", make_splats())

    assert scene.component_names == ["static"]

    component = scene.get("static")
    assert component["name"] == "static"
    assert component["index"] == 0
    torch.testing.assert_close(component["splats"]["means"], scene.splats["means"])


def test_transform_ctx_view_setitem_updates_context_slice():
    ctx_buffer = {"poses": torch.zeros(1, 7)}
    ctx_view = TransformCtxView(ctx_buffer, {"poses": (0, 1)})
    value = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 2.0, 2.0]])

    ctx_view["poses"] = value

    torch.testing.assert_close(
        ctx_buffer["poses"],
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5]]),
    )


def test_transform_ctx_view_setitem_validates_context_slice():
    ctx_buffer = {
        "poses": torch.zeros(1, 7),
        "pose_times": torch.tensor([0.0, 1.0]),
    }
    ctx_view = TransformCtxView(ctx_buffer, {"poses": (0, 1), "pose_times": (0, 2)})

    with pytest.raises(ValueError, match="shape"):
        ctx_view["poses"] = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="finite"):
        ctx_view["poses"] = torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="sorted"):
        ctx_view["pose_times"] = torch.tensor([1.0, 0.0])


def test_transform_graph_round_trips_registered_parametric_op():
    graph = TransformGraph([_OffsetOp(3.5)])

    restored = TransformGraph.from_state_dict(graph.state_dict())

    assert isinstance(restored.ops[0], _OffsetOp)
    assert restored.ops[0].offset == 3.5
