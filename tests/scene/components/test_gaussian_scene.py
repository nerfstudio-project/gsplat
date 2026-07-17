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

from tests._cuda import cuda_is_available

from gsplat.scene import GaussianScene, HIDDEN_OPACITY_LOGIT, Scene
from gsplat.scene import IdentityOp, RigidTransformOp, TransformGraph, TransformOp


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


class _NoOpPoseTrackOp(TransformOp):
    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        del count
        if "poses" not in ctx or "pose_times" not in ctx:
            raise KeyError("expected pose track context")

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return collected


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


def test_gaussian_scene_from_splats_keeps_live_parameters():
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


def test_component_views_stay_live_after_later_put():
    road_splats = make_splats()
    car_splats = make_splats(offset=100.0)
    scene = GaussianScene(id="scene")
    scene.put("road", road_splats)
    original_road_means = road_splats["means"].detach().clone()
    road = scene.get("road")

    scene.put("car", car_splats)

    assert road.splats._parent is scene.splats
    torch.testing.assert_close(road.splats["means"], original_road_means)
    torch.testing.assert_close(scene.get("car").splats["means"], car_splats["means"])


def test_legacy_component_dict_returns_snapshot_copies():
    road_splats = make_splats()
    car_splats = make_splats(offset=100.0)
    scene = GaussianScene(id="scene")
    scene.put("road", road_splats)
    scene.put("car", car_splats)
    original_means = scene.splats["means"].detach().clone()

    snapshot = scene.get("road")["splats"]
    with torch.no_grad():
        snapshot["means"].add_(100.0)

    torch.testing.assert_close(scene.splats["means"], original_means)


def test_put_normalizes_trailing_singleton_opacities():
    splats = make_splats()
    splats["opacities"] = torch.nn.Parameter(splats["opacities"].reshape(-1, 1))
    original_opacities = splats["opacities"]

    scene = GaussianScene.from_splats(splats, id="scene")

    assert splats["opacities"] is original_opacities
    assert splats["opacities"].shape == (4, 1)
    assert scene.splats["opacities"].shape == (4,)


def test_failed_put_does_not_normalize_caller_opacities_before_rejecting():
    scene = GaussianScene.from_splats(make_splats(), id="scene")
    bad_splats = make_splats()
    bad_splats["opacities"] = torch.nn.Parameter(bad_splats["opacities"].reshape(-1, 1))
    original_opacities = bad_splats["opacities"]
    del bad_splats["colors"]

    with pytest.raises(ValueError, match="splat keys"):
        scene.put("bad", bad_splats)

    assert bad_splats["opacities"] is original_opacities
    assert bad_splats["opacities"].shape == (4, 1)


def test_put_rejects_invalid_opacity_shape():
    splats = make_splats()
    splats["opacities"] = torch.nn.Parameter(torch.randn(4, 2))

    with pytest.raises(ValueError, match="opacities"):
        GaussianScene.from_splats(splats, id="scene")


def test_put_rejects_reserved_splat_keys():
    splats = make_splats()
    splats["component_index"] = torch.nn.Parameter(torch.zeros(4))

    with pytest.raises(ValueError, match="reserved"):
        GaussianScene.from_splats(splats, id="scene")


def test_static_apply_transforms_returns_live_splat_tensors():
    scene = GaussianScene.from_splats(make_splats(), id="scene")

    world = scene.apply_transforms()

    assert set(world.keys()) == set(scene.splats.keys())
    torch.testing.assert_close(world["means"], scene.splats["means"])
    assert world["means"] is scene.splats["means"]


def test_put_rejects_transform_context_without_graph():
    scene = GaussianScene(id="scene")

    with pytest.raises(ValueError, match="set a transform graph"):
        scene.put(
            "car",
            make_splats(),
            ctx={
                "poses": torch.zeros(1, 7),
                "pose_times": torch.zeros(1),
            },
        )


def test_put_validates_transform_context_against_graph():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))

    with pytest.raises(KeyError, match="poses"):
        scene.put("car", make_splats(), ctx={})


def test_put_stores_component_transform_context():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    ctx = {
        "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
        "pose_times": torch.tensor([0.0]),
    }

    scene.put("car", make_splats(), ctx=ctx)

    component = scene.get("car")
    assert component.transform_ctx is not None
    torch.testing.assert_close(component.transform_ctx["poses"], ctx["poses"])
    torch.testing.assert_close(
        scene.ctx_buffer["pose_times"],
        torch.tensor([0.0]),
    )
    torch.testing.assert_close(
        scene._collect_gaussians()["pose_offsets"],
        torch.tensor([0], dtype=torch.long),
    )
    torch.testing.assert_close(
        scene._collect_gaussians()["pose_counts"],
        torch.tensor([1], dtype=torch.long),
    )


def test_put_rejects_rigid_pose_dtype_mismatches():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    ctx = {
        "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]).double(),
        "pose_times": torch.tensor([0], dtype=torch.long),
    }

    with pytest.raises(TypeError, match="must match splat dtype"):
        scene.put("car", make_splats(), ctx=ctx)


def test_put_casts_integer_pose_times_to_int64():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([0], dtype=torch.int32),
        },
    )

    assert scene.ctx_buffer["pose_times"].dtype == torch.long


def test_transform_context_round_trip_preserves_scene_state():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )

    state = scene.state_dict()
    assert "ctx_ranges" in state
    assert "component_ctx" not in state
    assert state["transform_graph"] == {"ops": [{"type": "RigidTransformOp"}]}

    restored = GaussianScene.from_state_dict(scene.state_dict())

    torch.testing.assert_close(restored.ctx_buffer["poses"], scene.ctx_buffer["poses"])
    torch.testing.assert_close(
        restored.ctx_buffer["pose_times"],
        scene.ctx_buffer["pose_times"],
    )
    assert restored._component_ctx == scene._component_ctx
    assert restored._component_count == scene._component_count
    assert restored._graph is not None
    assert isinstance(restored._graph.ops[0], RigidTransformOp)


def test_transform_context_round_trip_preserves_registered_custom_op():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([_NoOpPoseTrackOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )

    restored = GaussianScene.from_state_dict(scene.state_dict())

    assert restored._graph is not None
    assert isinstance(restored._graph.ops[0], _NoOpPoseTrackOp)


def test_apply_transforms_rejects_context_loaded_without_graph():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )
    state = scene.state_dict()
    state["transform_graph"] = None
    restored = GaussianScene.from_state_dict(state)

    with pytest.raises(RuntimeError, match="transform context but no transform graph"):
        restored.apply_transforms(t=123)


def test_from_state_dict_accepts_legacy_component_ctx_key():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )
    state = scene.state_dict()
    state["component_ctx"] = state.pop("ctx_ranges")

    restored = GaussianScene.from_state_dict(state)

    assert restored._component_ctx == scene._component_ctx
    assert restored._graph is not None
    assert isinstance(restored._graph.ops[0], RigidTransformOp)


def test_identity_graph_allows_static_scene_transform_path():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([IdentityOp()]))
    scene.put("static", make_splats())

    world = scene.apply_transforms(t=0.0)

    torch.testing.assert_close(world["means"], scene.splats["means"])


def test_identity_graph_rejects_pose_track_context():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([IdentityOp()]))

    with pytest.raises(ValueError, match="IdentityOp"):
        scene.put(
            "car",
            make_splats(),
            ctx={
                "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
                "pose_times": torch.tensor([0.0]),
            },
        )


def test_keep_mask_hides_component_opacities():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([IdentityOp()]))
    scene.put(
        "hidden",
        make_splats(),
        ctx={"keep_mask": torch.tensor(False)},
    )

    world = scene.apply_transforms(t=0.0)

    torch.testing.assert_close(
        world["opacities"],
        torch.full((4,), HIDDEN_OPACITY_LOGIT),
    )


def test_keep_mask_backfills_existing_components():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([IdentityOp()]))
    scene.put("visible", make_splats())
    scene.put(
        "hidden",
        make_splats(offset=10.0),
        ctx={"keep_mask": torch.tensor(False)},
    )

    torch.testing.assert_close(
        scene.ctx_buffer["keep_mask"],
        torch.tensor([True, False]),
    )
    assert scene._component_ctx["visible"]["keep_mask"] == (0, 1)
    assert scene._component_ctx["hidden"]["keep_mask"] == (1, 1)
    world = scene.apply_transforms(t=0.0)
    torch.testing.assert_close(
        world["opacities"],
        torch.cat(
            [
                scene.splats["opacities"][:4],
                torch.full((4,), HIDDEN_OPACITY_LOGIT),
            ]
        ),
    )


def test_failed_keep_mask_put_does_not_mutate_existing_context():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "visible",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([0.0]),
        },
    )
    original_ctx_buffer = {
        key: value.detach().clone() for key, value in scene.ctx_buffer.items()
    }
    original_component_ctx = {
        name: dict(ranges) for name, ranges in scene._component_ctx.items()
    }

    with pytest.raises(KeyError, match="poses"):
        scene.put(
            "hidden",
            make_splats(offset=10.0),
            ctx={"keep_mask": torch.tensor(False)},
        )

    assert scene.component_names == ["visible"]
    assert scene._component_ctx == original_component_ctx
    assert "keep_mask" not in scene.ctx_buffer
    assert set(scene.ctx_buffer) == set(original_ctx_buffer)
    for key, value in original_ctx_buffer.items():
        torch.testing.assert_close(scene.ctx_buffer[key], value)


def test_pose_context_quaternions_are_normalized_on_put():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 2.0, 2.0],
                    [4.0, 5.0, 6.0, 0.0, 3.0, 0.0, 4.0],
                ]
            ),
            "pose_times": torch.tensor([0.0, 1.0]),
        },
    )

    torch.testing.assert_close(
        scene.ctx_buffer["poses"][:, :3],
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(
        scene.ctx_buffer["poses"][:, 3:].norm(dim=-1),
        torch.ones(2),
    )


def test_pose_context_rejects_invalid_quaternions():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))

    with pytest.raises(ValueError, match="non-zero norm"):
        scene.put(
            "zero",
            make_splats(),
            ctx={
                "poses": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "pose_times": torch.tensor([0.0]),
            },
        )

    with pytest.raises(ValueError, match="finite"):
        scene.put(
            "nan",
            make_splats(),
            ctx={
                "poses": torch.tensor([[0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 1.0]]),
                "pose_times": torch.tensor([0.0]),
            },
        )


def test_pose_context_rejects_non_finite_translations_and_times():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))

    with pytest.raises(ValueError, match="finite"):
        scene.put(
            "bad_translation",
            make_splats(),
            ctx={
                "poses": torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                "pose_times": torch.tensor([0.0]),
            },
        )

    with pytest.raises(ValueError, match="pose_times.*finite"):
        scene.put(
            "bad_time",
            make_splats(),
            ctx={
                "poses": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                "pose_times": torch.tensor([0.0, float("nan")]),
            },
        )


def test_scalar_tensor_transform_time_keeps_opacity_shape():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([_NoOpPoseTrackOp()]))
    scene.put(
        "car",
        make_splats(num_gaussians=2),
        ctx={
            "poses": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "pose_times": torch.tensor([0.0, 1.0]),
        },
    )

    world = scene.apply_transforms(t=torch.tensor([0.5]))

    assert world["opacities"].shape == scene.splats["opacities"].shape
    torch.testing.assert_close(world["opacities"], scene.splats["opacities"])


def test_per_component_transform_time_keeps_visibility_per_component():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([_NoOpPoseTrackOp()]))
    pose_ctx = {
        "poses": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "pose_times": torch.tensor([0.0, 1.0]),
    }
    scene.put("first", make_splats(num_gaussians=2), ctx=pose_ctx)
    scene.put("second", make_splats(num_gaussians=2, offset=10.0), ctx=pose_ctx)

    world = scene.apply_transforms(t=torch.tensor([0.5, 3.0]))

    torch.testing.assert_close(
        world["opacities"],
        torch.cat(
            [
                scene.splats["opacities"][:2],
                torch.full((2,), HIDDEN_OPACITY_LOGIT),
            ]
        ),
    )


def test_transform_time_rejects_non_scalar_non_component_vector():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([_NoOpPoseTrackOp()]))
    scene.put(
        "car",
        make_splats(num_gaussians=2),
        ctx={
            "poses": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "pose_times": torch.tensor([0.0, 1.0]),
        },
    )

    with pytest.raises(ValueError, match="one value per component"):
        scene.apply_transforms(t=torch.tensor([0.0, 1.0]))


@pytest.mark.skipif(
    not cuda_is_available(), reason="scene rigid transforms are CUDA-only"
)
def test_rigid_transform_graph_applies_component_pose_tracks():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))

    road_splats = make_splats(num_gaussians=2)
    car_splats = make_splats(num_gaussians=2, offset=10.0)
    road_splats["means"].data = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    car_splats["means"].data = torch.tensor([[0.0, 10.0, 0.0], [1.0, 10.0, 0.0]])
    road_splats["quats"].data = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    car_splats["quats"].data = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    road_splats = torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(value.detach().cuda())
            for key, value in road_splats.items()
        }
    )
    car_splats = torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(value.detach().cuda())
            for key, value in car_splats.items()
        }
    )

    scene.put(
        "road",
        road_splats,
        ctx={
            "poses": torch.tensor(
                [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]],
                device="cuda",
            ),
            "pose_times": torch.tensor([0.0], device="cuda"),
        },
    )
    scene.put(
        "car",
        car_splats,
        ctx={
            "poses": torch.tensor(
                [[10.0, 20.0, 30.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5]],
                device="cuda",
            ),
            "pose_times": torch.tensor([0.0], device="cuda"),
        },
    )

    try:
        world = scene.apply_transforms(0.0)
    except RuntimeError as exc:
        if "geometry" in str(exc).lower() and "cuda" in str(exc).lower():
            pytest.skip(f"geometry CUDA extension unavailable: {exc}")
        raise

    expected_means = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [0.0, 20.0, 30.0],
            [0.0, 21.0, 30.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(world["means"], expected_means)
    torch.testing.assert_close(
        world["quats"],
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0**-0.5, 0.0, 0.0, 2.0**-0.5],
                [2.0**-0.5, 0.0, 0.0, 2.0**-0.5],
            ],
            device="cuda",
        ),
    )


def test_gaussian_scene_put_appends_second_component_and_keeps_indices_aligned():
    road_splats = make_splats()
    car_splats = make_splats(offset=100.0)

    scene = GaussianScene(id="scene")
    scene.put("road", road_splats)
    original_road_means = road_splats["means"].detach().clone()
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
    torch.testing.assert_close(road["splats"]["means"], original_road_means)
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


def test_from_state_dict_rejects_uncovered_ctx_rows():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )
    state = scene.state_dict()
    state["ctx_buffer"]["poses"] = torch.cat(
        [state["ctx_buffer"]["poses"], state["ctx_buffer"]["poses"]],
        dim=0,
    )

    with pytest.raises(ValueError, match="exactly cover"):
        GaussianScene.from_state_dict(state)


def test_from_state_dict_rejects_overlapping_ctx_ranges():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    for name, offset in (("first", 0.0), ("second", 10.0)):
        scene.put(
            name,
            make_splats(offset=offset),
            ctx={
                "poses": torch.tensor([[offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                "pose_times": torch.tensor([123], dtype=torch.long),
            },
        )
    state = scene.state_dict()
    state["ctx_ranges"]["second"]["poses"] = (0, 1)
    state["ctx_ranges"]["second"]["pose_times"] = (0, 1)

    with pytest.raises(ValueError, match="exactly cover"):
        GaussianScene.from_state_dict(state)


def test_from_state_dict_rejects_mismatched_pose_time_ranges():
    scene = GaussianScene(id="scene")
    scene.set_graph(TransformGraph([RigidTransformOp()]))
    scene.put(
        "car",
        make_splats(),
        ctx={
            "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            "pose_times": torch.tensor([123], dtype=torch.long),
        },
    )
    state = scene.state_dict()
    state["transform_graph"] = None
    state["ctx_buffer"]["pose_times"] = torch.tensor([123, 124], dtype=torch.long)
    state["ctx_ranges"]["car"]["pose_times"] = (0, 2)

    with pytest.raises(ValueError, match="poses and pose_times"):
        GaussianScene.from_state_dict(state)


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


def _make_optimizers(
    splats: torch.nn.ParameterDict,
) -> dict[str, torch.optim.Optimizer]:
    return {key: torch.optim.Adam([value], lr=1e-3) for key, value in splats.items()}


def _move_scene_to_device(scene: GaussianScene, device: torch.device) -> None:
    scene.splats = torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(
                value.detach().to(device),
                requires_grad=value.requires_grad,
            )
            for key, value in scene.splats.items()
        }
    )
    scene.signal = {key: value.to(device) for key, value in scene.signal.items()}
    scene.component_index = scene.component_index.to(device)


def _assert_scene_rows_aligned(scene: GaussianScene) -> None:
    n = scene.num_gaussians()
    assert scene.component_index.shape == (n,)
    assert all(value.shape[0] == n for value in scene.splats.values())
    assert all(value.shape[0] == n for value in scene.signal.values())


def test_strategy_ops_keep_scene_topology_sidecars_in_sync():
    from gsplat.strategy import ops

    scene = make_scene()
    optimizers = _make_optimizers(scene.splats)
    state = {"count": torch.arange(4, dtype=torch.float32)}

    ops.duplicate(
        scene.splats,
        optimizers,
        state,
        torch.tensor([True, False, True, False]),
        scene=scene,
    )

    _assert_scene_rows_aligned(scene)
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

    scene = make_scene()
    optimizers = _make_optimizers(scene.splats)
    state = {"count": torch.arange(4, dtype=torch.float32)}

    ops.split(
        scene.splats,
        optimizers,
        state,
        torch.tensor([False, True, False, True]),
        scene=scene,
    )

    _assert_scene_rows_aligned(scene)
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
    optimizers = _make_optimizers(scene.splats)
    state = {"count": torch.arange(4, dtype=torch.float32)}

    ops.remove(
        scene.splats,
        optimizers,
        state,
        torch.tensor([False, True, False, True]),
        scene=scene,
    )

    _assert_scene_rows_aligned(scene)
    torch.testing.assert_close(
        scene.component_index,
        torch.tensor([0, 2], dtype=torch.long),
    )
    torch.testing.assert_close(
        scene.signal["camera"],
        torch.tensor([[0.0, 1.0], [4.0, 5.0]]),
    )


@pytest.mark.skipif(
    not cuda_is_available(), reason="relocate/sample_add coverage needs CUDA"
)
def test_mcmc_strategy_ops_keep_scene_topology_sidecars_in_sync():
    from gsplat.strategy import MCMCStrategy, ops

    device = torch.device("cuda")
    scene = make_scene()
    _move_scene_to_device(scene, device)
    with torch.no_grad():
        scene.splats["opacities"].copy_(
            torch.logit(
                torch.tensor(
                    [0.001, 0.8, 0.001, 0.7],
                    device=device,
                    dtype=scene.splats["opacities"].dtype,
                )
            )
        )
        scene.splats["scales"].zero_()
    optimizers = _make_optimizers(scene.splats)
    binoms = MCMCStrategy().initialize_state()["binoms"].to(device)

    try:
        ops.relocate(
            scene.splats,
            optimizers,
            state={},
            mask=torch.tensor([True, False, True, False], device=device),
            binoms=binoms,
            scene=scene,
        )
        ops.sample_add(
            scene.splats,
            optimizers,
            state={},
            n=2,
            binoms=binoms,
            scene=scene,
        )
    except RuntimeError as exc:
        if "relocation" in str(exc).lower() or "cuda" in str(exc).lower():
            pytest.skip(f"relocation CUDA op unavailable: {exc}")
        raise

    _assert_scene_rows_aligned(scene)
    assert scene.num_gaussians() == 6

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
