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

import torch

from .transforms import (
    GaussianComponent,
    HIDDEN_OPACITY_LOGIT,
    TensorViews,
    TransformCtxView,
    TransformGraph,
)

from .base import Scene
from .schema import (
    RESERVED_SCENE_KEYS,
    normalize_splat_opacities,
    validate_splat_schema,
)


class GaussianScene(Scene):
    """Wrapper around Gaussian splat parameters and extra signal data."""

    _POSE_TIME_DTYPES = (torch.int64, torch.float32, torch.float64)

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.splats = torch.nn.ParameterDict()
        self.signal: dict[str, torch.Tensor] = {}
        self.component_names: list[str] = []
        self.component_index = torch.zeros(0, dtype=torch.long)
        self.ctx_buffer: dict[str, torch.Tensor] = {}
        self._graph: TransformGraph | None = None
        self._component_ctx: dict[str, dict[str, tuple[int, int]]] = {}
        self._component_count: dict[str, int] = {}
        self._ctx_range_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _validate_component_splat_shapes(
        self,
        component: torch.nn.ParameterDict,
        reference: torch.nn.ParameterDict | None = None,
    ) -> None:
        """Validate component splat tensor shape compatibility before append.

        The implementation should check that all component tensors have the
        same leading row count and, when ``reference`` is provided, that common
        splat keys are compatible with the already stored scene tensors.
        """
        validate_splat_schema(
            component,
            reference=reference,
            context="component splat",
        )

    @staticmethod
    def _normalize_opacities(
        splats: torch.nn.ParameterDict,
        count: int,
    ) -> torch.nn.ParameterDict:
        """Return splats with renderer-canonical 1-D opacity storage."""
        return normalize_splat_opacities(splats, count)

    def _validate_existing_signal_rows(self, count: int) -> None:
        """Validate that existing signal tensors can align with a component.

        The implementation should check that scene-owned signal tensors have
        the expected leading row count before a first component is installed.
        """
        for key, value in self.signal.items():
            if value.ndim == 0:
                raise ValueError(f"signal tensor {key!r} must have a leading row dim")
            if value.shape[0] != count:
                raise ValueError(
                    f"signal tensor {key!r} has leading dim {value.shape[0]}, "
                    f"expected {count}"
                )

    def _validate_transform_ctx(
        self,
        ctx: dict[str, torch.Tensor],
        count: int,
    ) -> None:
        """Validate transform context before it is stored on the scene.

        The implementation should check the component's transform context
        against the active transform graph and the component row count.
        """
        if self._graph is None:
            if len(ctx) != 0:
                raise ValueError(
                    "set a transform graph before adding transform context"
                )
            return
        self._graph.validate_ctx(ctx, count)

    def put(
        self,
        name: str,
        component: torch.nn.ParameterDict,
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Add a named component to the scene.

        Init-only: calling ``put()`` after optimizers have been created
        will orphan the old Parameter objects.  Pads existing signal
        rows for the new component to keep everything aligned.

        Args:
            name: Component name.
            component: Component splat parameters.
            ctx: Optional transform context consumed by the active transform graph.
        """
        if not name:
            raise ValueError("component name must not be empty")
        if name in self.component_names:
            raise ValueError(f"Component {name!r} already exists in scene")
        self._validate_component_splat_shapes(
            component, self.splats if len(self.splats) > 0 else None
        )
        count = component["means"].shape[0]
        component = self._normalize_opacities(component, count)
        if len(self.splats) == 0:
            self._validate_existing_signal_rows(count)
        if ctx is not None:
            ctx_key_collisions = sorted(set(ctx.keys()) & set(component.keys()))
            if ctx_key_collisions:
                raise ValueError(
                    "context keys must not conflict with splat keys: "
                    f"{ctx_key_collisions}"
                )
            if self._graph is None and len(ctx) != 0:
                raise ValueError(
                    "set a transform graph before adding transform context"
                )

        normalized_ctx = self._normalize_transform_ctx(ctx or {}, component["means"])
        normalized_ctx = self._normalize_keep_mask_ctx(
            normalized_ctx,
            component["means"],
        )
        self._validate_transform_ctx(normalized_ctx, count)
        if "keep_mask" in normalized_ctx and "keep_mask" not in self.ctx_buffer:
            self._backfill_existing_keep_mask_ctx(component["means"])
        ctx_ranges = self._prepare_ctx_ranges(normalized_ctx)

        if len(self.splats) == 0:
            self.splats = component
            self.component_names = [name]
            self.component_index = torch.zeros(
                count,
                device=self.splats["means"].device,
                dtype=torch.long,
            )
        else:
            for key in list(self.splats.keys()):
                self.splats[key] = torch.nn.Parameter(
                    torch.cat(
                        [self.splats[key].detach(), component[key].detach()],
                        dim=0,
                    ),
                    requires_grad=self.splats[key].requires_grad,
                )
            self.component_names.append(name)
            self.component_index = torch.cat(
                [
                    self.component_index,
                    torch.full(
                        (count,),
                        len(self.component_names) - 1,
                        device=self.component_index.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            for key, value in self.signal.items():
                pad = torch.zeros(
                    (count, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                self.signal[key] = torch.cat([value, pad], dim=0)

        for key, value in normalized_ctx.items():
            self.ctx_buffer[key] = (
                value
                if key not in self.ctx_buffer
                else torch.cat([self.ctx_buffer[key], value], dim=0)
            )
        self._component_ctx[name] = ctx_ranges
        self._component_count[name] = count
        self._invalidate_ctx_range_cache()
        self.validate()

    def set_graph(self, graph: TransformGraph) -> None:
        """Set the stateless scene transform graph."""
        for name in self.component_names:
            ctx = self._component_ctx_dict(name)
            graph.validate_ctx(ctx, self._component_row_count(name))
        self._graph = graph

    def _collect_gaussians(self) -> dict[str, torch.Tensor]:
        """Collect live splats plus transform context for graph execution."""
        collected: dict[str, torch.Tensor] = {
            key: value for key, value in self.splats.items()
        }
        collected.update(self.ctx_buffer)
        collected["component_index"] = self.component_index
        if "poses" in self.ctx_buffer and all(
            "poses" in self._component_ctx[name] for name in self.component_names
        ):
            collected["pose_offsets"] = self._ctx_offsets("poses")
            collected["pose_counts"] = self._ctx_counts("poses")
        return collected

    def apply_transforms(
        self,
        time_sec: torch.Tensor | float = 0.0,
        *,
        t: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return transient world-space splats for render time ``time_sec``."""
        if t is not None:
            time_sec = t
        if self._graph is None:
            if len(self.ctx_buffer) != 0:
                raise RuntimeError(
                    "GaussianScene has transform context but no transform graph; "
                    "rendering dynamic components without a graph would return "
                    "untransformed local splats."
                )
            return {key: value for key, value in self.splats.items()}

        collected = self._collect_gaussians()
        collected["t"] = self._time_tensor(time_sec)
        transformed = self._graph.apply(collected)
        result = self._finalize_transformed_splats(transformed)
        visible_mask = self._visible_mask(collected, collected["t"])
        if visible_mask is not None and "opacities" in result:
            result["opacities"] = torch.where(
                visible_mask,
                result["opacities"],
                result["opacities"].new_full(
                    result["opacities"].shape, HIDDEN_OPACITY_LOGIT
                ),
            )
        return result

    def _finalize_transformed_splats(
        self,
        transformed: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return a splat-only dict from transformed graph output."""
        return {key: transformed[key] for key in self.splats}

    def _normalize_transform_ctx(
        self,
        ctx: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Normalize user-provided component context into row-major tensors."""
        splat_keys = set(self.splats.keys())
        normalized: dict[str, torch.Tensor] = {}
        for key, value in ctx.items():
            if key in RESERVED_SCENE_KEYS:
                raise ValueError(f"context key {key!r} is reserved")
            if key in splat_keys:
                raise ValueError(f"context key {key!r} conflicts with a splat key")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"context value for {key!r} must be a torch.Tensor")
            if value.device != reference.device:
                raise ValueError(
                    f"context value for {key!r} must be on device {reference.device}, "
                    f"got {value.device}"
                )

            if key == "poses":
                rows = value.reshape(1, 7) if value.ndim == 1 else value
                if not rows.dtype.is_floating_point:
                    raise TypeError(
                        f"context value for {key!r} must have a floating dtype, "
                        f"got {rows.dtype}"
                    )
                if rows.dtype != reference.dtype:
                    raise TypeError(
                        f"context value for {key!r} dtype {rows.dtype} must match "
                        f"splat dtype {reference.dtype}"
                    )
                if rows.ndim != 2 or rows.shape[1] != 7:
                    raise ValueError(
                        "context value for 'poses' must have shape (7,) or (N, 7) "
                        "with layout [tx, ty, tz, qx, qy, qz, qw], "
                        f"got {tuple(rows.shape)}"
                    )
                rows = self._normalize_pose_rows(rows)
            elif key == "pose_times":
                rows = value.reshape(-1)
                if not rows.dtype.is_floating_point:
                    rows = rows.to(dtype=torch.long)
                elif rows.dtype not in self._POSE_TIME_DTYPES:
                    raise TypeError(
                        f"context value for {key!r} must have dtype int64, "
                        f"float32, or float64, got {rows.dtype}"
                    )
                if rows.dtype.is_floating_point and not bool(
                    torch.isfinite(rows).all().item()
                ):
                    raise ValueError("pose_times must be finite")
            elif key == "keep_mask":
                rows = value.reshape(1)
            else:
                rows = value.unsqueeze(0) if value.ndim == 1 else value
            normalized[key] = rows.contiguous()
        return normalized

    @staticmethod
    def _normalize_pose_rows(poses: torch.Tensor) -> torch.Tensor:
        """Return pose rows with finite, unit-norm ``xyzw`` quaternions."""
        if not bool(torch.isfinite(poses).all().item()):
            raise ValueError("pose rows must be finite")
        rotations = poses[:, 3:7]
        norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
        eps = torch.finfo(poses.dtype).eps
        if bool((norms <= eps).any().item()):
            raise ValueError("pose quaternions must have non-zero norm")
        return torch.cat([poses[:, :3], rotations / norms], dim=-1)

    def _normalize_keep_mask_ctx(
        self,
        ctx: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Normalize optional keep_mask as one bool row per component."""
        if "keep_mask" in ctx:
            keep = (
                ctx["keep_mask"]
                .reshape(1)
                .to(
                    device=reference.device,
                    dtype=torch.bool,
                )
            )
            ctx = dict(ctx)
            ctx["keep_mask"] = keep.contiguous()
        elif "keep_mask" in self.ctx_buffer:
            ctx = dict(ctx)
            ctx["keep_mask"] = torch.ones(
                (1,),
                device=self.ctx_buffer["keep_mask"].device,
                dtype=torch.bool,
            )
        return ctx

    def _backfill_existing_keep_mask_ctx(self, reference: torch.Tensor) -> None:
        """Add visible keep-mask rows for components created before keep_mask."""
        if not self.component_names:
            return
        self.ctx_buffer["keep_mask"] = torch.ones(
            (len(self.component_names),),
            device=reference.device,
            dtype=torch.bool,
        )
        for component_id, component_name in enumerate(self.component_names):
            self._component_ctx.setdefault(component_name, {})["keep_mask"] = (
                component_id,
                1,
            )

    def _prepare_ctx_ranges(
        self, ctx: dict[str, torch.Tensor]
    ) -> dict[str, tuple[int, int]]:
        """Return ranges this context will occupy once appended."""
        ranges: dict[str, tuple[int, int]] = {}
        for key, value in ctx.items():
            offset = self.ctx_buffer[key].shape[0] if key in self.ctx_buffer else 0
            ranges[key] = (offset, value.shape[0])
        return ranges

    def _component_ctx_dict(self, name: str) -> dict[str, torch.Tensor]:
        """Materialize one component's transform context for validation."""
        ctx: dict[str, torch.Tensor] = {}
        for key, (offset, count) in self._component_ctx.get(name, {}).items():
            ctx[key] = self.ctx_buffer[key][offset : offset + count]
        return ctx

    def _component_row_count(self, name: str) -> int:
        """Return Gaussian row count for a component."""
        if name in self._component_count:
            return self._component_count[name]
        component_id = self._component_id(name)
        return int((self.component_index == component_id).sum().item())

    def _ctx_offsets(self, key: str) -> torch.Tensor:
        """Return packed context offsets for every component."""
        return self._ctx_ranges(key)[0]

    def _ctx_counts(self, key: str) -> torch.Tensor:
        """Return packed context counts for every component."""
        return self._ctx_ranges(key)[1]

    def _ctx_ranges(
        self,
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached per-component context offsets, counts, and presence."""
        cached = self._ctx_range_cache.get(key)
        if cached is not None:
            return cached

        device = (
            self.ctx_buffer[key].device
            if key in self.ctx_buffer
            else self.component_index.device
        )
        offsets: list[int] = []
        counts: list[int] = []
        present: list[bool] = []
        for name in self.component_names:
            ranges = self._component_ctx.get(name, {})
            if key in ranges:
                offset, count = ranges[key]
                offsets.append(offset)
                counts.append(count)
                present.append(True)
            else:
                offsets.append(0)
                counts.append(0)
                present.append(False)

        tensors = (
            torch.tensor(offsets, device=device, dtype=torch.long),
            torch.tensor(counts, device=device, dtype=torch.long),
            torch.tensor(present, device=device, dtype=torch.bool),
        )
        self._ctx_range_cache[key] = tensors
        return tensors

    def _invalidate_ctx_range_cache(self) -> None:
        """Invalidate cached per-component context range tensors."""
        self._ctx_range_cache.clear()

    def _time_tensor(self, time_sec: torch.Tensor | float) -> torch.Tensor:
        """Return a scalar render time on the scene device."""
        if "means" in self.splats:
            reference = self.splats["means"]
            device = reference.device
            dtype = reference.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        if isinstance(time_sec, torch.Tensor):
            time_sec = time_sec.to(device=device)
            if (
                time_sec.numel() != 1
                and self.component_names
                and time_sec.numel() != len(self.component_names)
            ):
                raise ValueError(
                    "transform time tensor must be scalar or have one value "
                    "per component"
                )
            if time_sec.numel() == 1:
                return time_sec.reshape(())
            return time_sec
        return torch.tensor(time_sec, device=device, dtype=dtype)

    def _visible_mask(
        self,
        collected: dict[str, torch.Tensor],
        time_sec: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return per-Gaussian visibility for transform time validity."""
        if "poses" not in self.ctx_buffer and "keep_mask" not in self.ctx_buffer:
            return None

        component_count = len(self.component_names)
        if component_count == 0:
            return None
        component_visible = torch.ones(
            (component_count,), device=self.component_index.device, dtype=torch.bool
        )
        component_times = self._component_times(time_sec)
        pose_times = collected.get("pose_times")

        if pose_times is not None and "pose_times" in self.ctx_buffer:
            offsets, counts, present = self._ctx_ranges("pose_times")
            active = present & (counts > 1)
            if bool(active.any().item()):
                active_idx = torch.nonzero(active, as_tuple=True)[0]
                times = pose_times.reshape(-1)
                time_dtype = self._visible_time_dtype(times, component_times)
                first = times[offsets[active_idx]].to(dtype=time_dtype)
                last = times[(offsets[active_idx] + counts[active_idx] - 1)].to(
                    dtype=time_dtype
                )
                t = component_times.to(device=times.device, dtype=time_dtype)[
                    active_idx
                ]
                in_range = (t >= first) & (t <= last)
                component_visible[
                    active_idx.to(component_visible.device)
                ] &= in_range.to(component_visible.device)

        if "keep_mask" in self.ctx_buffer:
            offsets, counts, present = self._ctx_ranges("keep_mask")
            active = present & (counts > 0)
            if bool(active.any().item()):
                active_idx = torch.nonzero(active, as_tuple=True)[0]
                keep = self.ctx_buffer["keep_mask"].reshape(-1)[offsets[active_idx]]
                component_visible[active_idx.to(component_visible.device)] &= keep.to(
                    device=component_visible.device, dtype=torch.bool
                )

        return component_visible[self.component_index]

    def _component_times(self, time_sec: torch.Tensor) -> torch.Tensor:
        """Return one render time per component."""
        component_count = len(self.component_names)
        if time_sec.numel() == 1:
            return time_sec.reshape(()).expand(component_count)
        if time_sec.numel() != component_count:
            raise ValueError(
                "transform time tensor must be scalar or have one value per component"
            )
        return time_sec.reshape(component_count).to(device=self.component_index.device)

    @staticmethod
    def _visible_time_dtype(
        pose_times: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.dtype:
        """Return dtype for visibility checks using interpolation rules."""
        if pose_times.is_floating_point():
            if query_time.is_floating_point():
                return torch.promote_types(pose_times.dtype, query_time.dtype)
            return pose_times.dtype
        if query_time.is_floating_point():
            return torch.float64
        return torch.int64

    def _component_range(self, component_id: int) -> tuple[int, int] | None:
        """Return a contiguous range for a component, or ``None`` if scattered."""
        indices = torch.nonzero(
            self.component_index == component_id,
            as_tuple=False,
        ).flatten()
        if indices.numel() == 0:
            return (0, 0)
        start = int(indices[0].item())
        count = int(indices.numel())
        expected = torch.arange(
            start,
            start + count,
            device=indices.device,
            dtype=indices.dtype,
        )
        if torch.equal(indices, expected):
            return (start, count)
        return None

    def _component_indices(self, component_id: int) -> torch.Tensor:
        """Return current row indices for a component."""
        return torch.nonzero(
            self.component_index == component_id,
            as_tuple=False,
        ).flatten()

    def _validate_ctx_metadata(self) -> None:
        """Validate stored transform context ranges against context buffers."""
        collisions = sorted(set(self.ctx_buffer.keys()) & set(self.splats.keys()))
        if collisions:
            raise ValueError(
                f"context keys must not conflict with splat keys: {collisions}"
            )
        referenced_by_key: dict[str, list[tuple[int, int, str]]] = {
            key: [] for key in self.ctx_buffer
        }
        for name, ranges in self._component_ctx.items():
            if name not in self.component_names:
                raise ValueError(f"component ctx references unknown component {name!r}")
            if ("poses" in ranges) != ("pose_times" in ranges):
                raise ValueError(
                    f"component {name!r} must store poses and pose_times together"
                )
            if "poses" in ranges and ranges["poses"] != ranges["pose_times"]:
                raise ValueError(
                    f"component {name!r} poses and pose_times ranges must match"
                )
            for key, (offset, count) in ranges.items():
                if key not in self.ctx_buffer:
                    raise ValueError(f"component ctx references missing key {key!r}")
                if self.ctx_buffer[key].ndim == 0:
                    raise ValueError(f"context buffer {key!r} must have a row dim")
                if offset < 0 or count < 0:
                    raise ValueError(f"component ctx range for {key!r} is negative")
                if offset + count > self.ctx_buffer[key].shape[0]:
                    raise ValueError(
                        f"component ctx range for {key!r} exceeds buffer rows"
                    )
                referenced_by_key.setdefault(key, []).append((offset, count, name))

        for key, value in self.ctx_buffer.items():
            if value.ndim == 0:
                raise ValueError(f"context buffer {key!r} must have a row dim")
            cursor = 0
            for offset, count, name in sorted(referenced_by_key.get(key, [])):
                if offset != cursor:
                    raise ValueError(
                        f"context ranges for {key!r} must exactly cover buffer rows; "
                        f"component {name!r} starts at {offset}, expected {cursor}"
                    )
                cursor = offset + count
            if cursor != value.shape[0]:
                raise ValueError(
                    f"context ranges for {key!r} must exactly cover buffer rows; "
                    f"covered {cursor} of {value.shape[0]} rows"
                )

    @classmethod
    def from_splats(
        cls,
        splats: torch.nn.ParameterDict,
        id: str,
        signal: dict[str, torch.Tensor] | None = None,
    ) -> "GaussianScene":
        if len(splats) == 0 or "means" not in splats:
            raise ValueError(
                "from_splats requires a non-empty ParameterDict containing 'means'"
            )
        scene = cls(id)
        if signal is not None:
            device = splats["means"].device
            scene.signal = {k: v.to(device) for k, v in signal.items()}
        scene.put(id, splats)
        return scene

    def validate(self) -> None:
        # Init-only: the bounds checks call ``.item()`` on ``component_index``,
        # which forces a CUDA sync. Don't move into a per-step hot path.
        required_keys = ("means", "scales", "quats", "opacities")
        missing = [key for key in required_keys if key not in self.splats]
        if len(self.splats) > 0 and missing:
            raise ValueError(f"missing required splat keys: {missing}")
        if len(self.splats) > 0:
            validate_splat_schema(self.splats, context="scene splat")

        n = self.num_gaussians()
        if not all(v.shape[0] == n for v in self.splats.values()):
            raise ValueError(
                f"every splat tensor must have leading dim == num_gaussians: {n}"
            )
        if not all(v.shape[0] == n for v in self.signal.values()):
            raise ValueError(
                f"every signal tensor must have leading dim == num_gaussians: {n}"
            )
        if self.component_index.shape != (n,):
            raise ValueError(
                f"component_index shape {tuple(self.component_index.shape)} != ({n},)"
            )
        if len(self.splats) > 0:
            if len(self.component_names) == 0:
                raise ValueError("component_names must not be empty")
            if n > 0:
                if int(self.component_index.min().item()) < 0:
                    raise ValueError("component_index must be non-negative")
                if int(self.component_index.max().item()) >= len(self.component_names):
                    raise ValueError("component_index refers to an unknown component")
        self._validate_ctx_metadata()

    def num_gaussians(self) -> int:
        if "means" not in self.splats:
            return 0
        return self.splats["means"].shape[0]

    def _component_id(self, component: str | int) -> int:
        if isinstance(component, int):
            if component < 0 or component >= len(self.component_names):
                raise KeyError(f"Unknown component index: {component}")
            return component
        try:
            return self.component_names.index(component)
        except ValueError as exc:
            raise KeyError(f"Unknown component name: {component}") from exc

    def get(self, component: str | int) -> GaussianComponent:
        component_id = self._component_id(component)
        name = self.component_names[component_id]
        splats = TensorViews(
            self.splats,
            lambda component_id=component_id: self._component_range(component_id),
            lambda component_id=component_id: self._component_indices(component_id),
        )
        signal = TensorViews(
            self.signal,
            lambda component_id=component_id: self._component_range(component_id),
            lambda component_id=component_id: self._component_indices(component_id),
        )
        ctx_ranges = self._component_ctx.get(name, {})
        transform_ctx = (
            TransformCtxView(self.ctx_buffer, ctx_ranges)
            if len(ctx_ranges) != 0
            else None
        )
        return GaussianComponent(
            component_idx=component_id,
            splats=splats,
            signal=signal,
            transform_ctx=transform_ctx,
            name=name,
            mask_fn=lambda component_id=component_id: self.component_index
            == component_id,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "splats": self.splats.state_dict(),
            "splats_requires_grad": {
                key: bool(value.requires_grad) for key, value in self.splats.items()
            },
            "signal": {
                key: value.detach().clone() for key, value in self.signal.items()
            },
            "component_names": list(self.component_names),
            "component_index": self.component_index.detach().clone(),
            "ctx_buffer": {
                key: value.detach().clone() for key, value in self.ctx_buffer.items()
            },
            "ctx_ranges": {
                name: dict(ranges) for name, ranges in self._component_ctx.items()
            },
            "component_count": {
                name: self._component_row_count(name) for name in self.component_names
            },
            "transform_graph": (
                self._graph.state_dict() if self._graph is not None else None
            ),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GaussianScene":
        if "id" not in state:
            raise KeyError("state_dict missing required 'id' entry")

        scene = cls(state["id"])
        requires_grad_map: dict[str, bool] = state.get("splats_requires_grad", {})
        scene.splats = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(
                    value.clone(), requires_grad=requires_grad_map.get(key, True)
                )
                for key, value in state["splats"].items()
            }
        )
        if "means" in scene.splats:
            scene.splats = scene._normalize_opacities(
                scene.splats,
                scene.splats["means"].shape[0],
            )
        scene.signal = dict(state.get("signal", {}))  # already cloned in state_dict()
        scene.component_names = list(state.get("component_names", []))

        has_splats = "means" in scene.splats
        component_index = state.get("component_index")
        if component_index is None:
            n = scene.splats["means"].shape[0] if has_splats else 0
            device = scene.splats["means"].device if has_splats else torch.device("cpu")
            component_index = torch.zeros(n, device=device, dtype=torch.long)
        elif has_splats:
            # Align loaded `component_index` to the splats' device. Without
            # this, a checkpoint saved on CUDA + loaded with map_location="cpu"
            # (or the reverse) leaves these tensors on different devices, and
            # the next on_remove / on_duplicate / on_relocate crashes the
            # moment it indexes one with the other.
            component_index = component_index.to(scene.splats["means"].device)
        scene.component_index = component_index
        if has_splats and not scene.component_names:
            scene.component_names = [state["id"]]
        scene.ctx_buffer = {
            key: value.to(scene.splats["means"].device) if has_splats else value
            for key, value in state.get("ctx_buffer", {}).items()
        }
        ctx_ranges_state = state.get("ctx_ranges", state.get("component_ctx", {}))
        scene._component_ctx = {
            name: {
                key: (int(offset), int(count))
                for key, (offset, count) in ranges.items()
            }
            for name, ranges in ctx_ranges_state.items()
        }
        scene._refresh_component_metadata()
        graph_state = state.get("transform_graph")
        if graph_state is not None:
            scene.set_graph(TransformGraph.from_state_dict(graph_state))

        scene.validate()
        return scene

    def _refresh_component_metadata(self) -> None:
        """Refresh cached component row counts from ``component_index``."""
        self._invalidate_ctx_range_cache()
        if not self.component_names:
            self._component_count = {}
            return
        if self.component_index.numel() == 0:
            self._component_count = {name: 0 for name in self.component_names}
            return

        counts = torch.bincount(
            self.component_index,
            minlength=len(self.component_names),
        )
        counts_list = [int(count) for count in counts.detach().cpu().tolist()]
        self._component_count = {
            name: counts_list[idx] for idx, name in enumerate(self.component_names)
        }

    def _cat_signal(self, indices: torch.Tensor) -> None:
        # ``indices`` must be a LongTensor; bool masks would change semantics
        # (gather True rows vs gather positions).
        for key, value in self.signal.items():
            self.signal[key] = torch.cat([value, value[indices]], dim=0)

    def on_duplicate(self, sel: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [self.component_index, self.component_index[sel]], dim=0
        )
        self._cat_signal(sel)
        self._refresh_component_metadata()

    def on_split(self, sel: torch.Tensor, rest: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [
                self.component_index[rest],
                self.component_index[sel],
                self.component_index[sel],
            ],
            dim=0,
        )
        for key, value in self.signal.items():
            self.signal[key] = torch.cat(
                [value[rest], value[sel], value[sel]],
                dim=0,
            )
        self._refresh_component_metadata()

    def on_remove(self, remove_mask: torch.Tensor) -> None:
        keep = ~remove_mask
        self.component_index = self.component_index[keep]
        for key, value in self.signal.items():
            self.signal[key] = value[keep]
        self._refresh_component_metadata()

    def on_relocate(
        self, dead_indices: torch.Tensor, sampled_indices: torch.Tensor
    ) -> None:
        self.component_index[dead_indices] = self.component_index[sampled_indices]
        for key, value in self.signal.items():
            value[dead_indices] = value[sampled_indices]
        self._refresh_component_metadata()

    def on_sample_add(self, sampled_indices: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [self.component_index, self.component_index[sampled_indices]], dim=0
        )
        self._cat_signal(sampled_indices)
        self._refresh_component_metadata()

    def on_permute(self, order: torch.Tensor) -> None:
        self.component_index = self.component_index[order]
        for key, value in self.signal.items():
            self.signal[key] = value[order]
        self._refresh_component_metadata()
