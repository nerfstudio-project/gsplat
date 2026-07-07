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

"""Stage-level component collection API."""

from __future__ import annotations

from typing import Any, Callable

import torch

from gsplat.scene import GaussianScene
from gsplat.scene.components.schema import validate_splat_schema


class ComponentCollection:
    """One rasterizer dispatch over multiple member scenes."""

    _GAUSSIAN_AXIS_KEYS = {
        "means2d",
        "gradient_2dgs",
        "radii",
        "depths",
        "conics",
        "compensations",
        "gaussian_ids",
    }
    _VECTOR_TRAILING_DIMS = {
        "means2d": {2},
        "gradient_2dgs": {2},
        "radii": {2},
        "conics": {3},
    }

    def __init__(self, id: str) -> None:
        if not id:
            raise ValueError("ComponentCollection id must not be empty")
        self.id = id
        self._members: list[GaussianScene] = []
        self._member_spans: list[tuple[int, int]] | None = None
        self.render_fn: Callable | None = None

    @property
    def members(self) -> list[GaussianScene]:
        """Member scenes in registration / concatenation order."""
        return list(self._members)

    def add_scene(self, scene: GaussianScene) -> None:
        """Register a member scene after validating collection schema."""
        if any(member.id == scene.id for member in self._members):
            raise ValueError(f"Scene {scene.id!r} already registered in collection")
        if scene.num_gaussians() == 0:
            raise ValueError("ComponentCollection scenes must not be empty")
        if self._members:
            self._validate_schema(scene, self._members[0])
        self._members.append(scene)
        self._member_spans = None

    def collect(self, t: torch.Tensor | float = 0.0) -> dict[str, torch.Tensor]:
        """Return transient world-space splats for all members at time ``t``."""
        if not self._members:
            raise ValueError("ComponentCollection has no member scenes")

        if len(self._members) == 1:
            splats = self._members[0].apply_transforms(t=t)
            self._member_spans = [(0, splats["means"].shape[0])]
            return splats

        splat_dicts = [scene.apply_transforms(t=t) for scene in self._members]
        self._validate_collected_schema(splat_dicts)

        spans: list[tuple[int, int]] = []
        offset = 0
        for splats in splat_dicts:
            count = splats["means"].shape[0]
            spans.append((offset, count))
            offset += count
        self._member_spans = spans

        return {
            key: torch.cat([splats[key] for splats in splat_dicts], dim=0)
            for key in splat_dicts[0]
        }

    def render(self, t: torch.Tensor | float = 0.0, **kwargs) -> Any:
        """Render the collected member scenes with one shared render function."""
        if self.render_fn is None:
            raise RuntimeError("ComponentCollection has no render function")
        return self.render_fn(splats=self.collect(t=t), **kwargs)

    def step_pre_backward(self, info: dict, *, key: str = "means2d") -> None:
        """Prepare merged rasterizer info before backpropagation."""
        if key not in info:
            raise KeyError(f"Missing rasterizer info key {key!r}")
        info[key].retain_grad()

    def split_info(self, info: dict, *, key: str = "means2d") -> list[dict]:
        """Split merged rasterizer info into per-member local row spaces."""
        if self._member_spans is None:
            raise RuntimeError("Call collect() or render() before split_info()")
        if key not in info:
            raise KeyError(f"Missing rasterizer info key {key!r}")

        total = sum(count for _, count in self._member_spans)
        packed_ids = self._packed_gaussian_ids(info, key, total)
        split: list[dict] = []
        for start, count in self._member_spans:
            mask = (
                (packed_ids >= start) & (packed_ids < start + count)
                if packed_ids is not None
                else None
            )
            member_info: dict[str, Any] = {}
            for info_key, value in info.items():
                member_info[info_key] = self._split_info_value(
                    info_key,
                    value,
                    split_key=key,
                    total=total,
                    start=start,
                    count=count,
                    packed_mask=mask,
                )
            split.append(member_info)
        return split

    @staticmethod
    def _validate_schema(scene: GaussianScene, reference: GaussianScene) -> None:
        validate_splat_schema(
            scene.splats,
            reference=reference.splats,
            context="collection member splat",
        )

    @staticmethod
    def _validate_collected_schema(splat_dicts: list[dict[str, torch.Tensor]]) -> None:
        reference = splat_dicts[0]
        for splats in splat_dicts[1:]:
            validate_splat_schema(
                splats,
                reference=reference,
                context="collected splat",
            )

    @staticmethod
    def _packed_gaussian_ids(
        info: dict,
        key: str,
        total: int,
    ) -> torch.Tensor | None:
        gaussian_ids = info.get("gaussian_ids")
        value = info.get(key)
        if not isinstance(gaussian_ids, torch.Tensor):
            return None
        if not isinstance(value, torch.Tensor):
            return None
        if gaussian_ids.ndim != 1 or value.ndim == 0:
            return None
        if value.shape[0] == gaussian_ids.shape[0]:
            return gaussian_ids
        return None

    @classmethod
    def _gaussian_dim(
        cls,
        info_key: str,
        value: torch.Tensor,
        total: int,
        split_key: str,
    ) -> int | None:
        if info_key not in cls._GAUSSIAN_AXIS_KEYS and info_key != split_key:
            return None
        trailing_dims = cls._VECTOR_TRAILING_DIMS.get(info_key, set())
        if value.ndim >= 2 and value.shape[-1] in trailing_dims:
            return value.ndim - 2 if value.shape[-2] == total else None
        if value.ndim >= 1 and value.shape[-1] == total:
            return value.ndim - 1
        if value.ndim >= 1 and value.shape[0] == total:
            return 0
        return None

    def _split_info_value(
        self,
        info_key: str,
        value: Any,
        *,
        split_key: str,
        total: int,
        start: int,
        count: int,
        packed_mask: torch.Tensor | None,
    ) -> Any:
        if not isinstance(value, torch.Tensor):
            return value

        if (
            packed_mask is not None
            and value.ndim > 0
            and value.shape[0] == len(packed_mask)
        ):
            if info_key == split_key:
                return _GradSlice(value, packed_mask)
            sliced = value[packed_mask]
            return sliced - start if info_key == "gaussian_ids" else sliced

        dim = self._gaussian_dim(info_key, value, total, split_key)
        if dim is None:
            return value

        index: list[slice] = [slice(None)] * value.ndim
        index[dim] = slice(start, start + count)
        idx = tuple(index)
        if info_key == split_key:
            return _GradSlice(value, idx)
        sliced = value[idx]
        return sliced - start if info_key == "gaussian_ids" else sliced


class _GradSlice:
    """Per-member access to a merged rasterizer tensor and gradient slice."""

    def __init__(self, merged: torch.Tensor, idx) -> None:
        self._merged = merged
        self._idx = idx

    @property
    def value(self) -> torch.Tensor:
        """Return the selected forward tensor slice."""
        return self._merged[self._idx]

    def retain_grad(self) -> None:
        """Retain gradients on the merged rasterizer tensor."""
        self._merged.retain_grad()

    @property
    def grad(self) -> torch.Tensor | None:
        """Return the selected slice of ``merged.grad``."""
        if self._merged.grad is None:
            return None
        return self._merged.grad[self._idx]

    @property
    def absgrad(self) -> torch.Tensor | None:
        """Return the selected slice of ``merged.absgrad``."""
        absgrad = getattr(self._merged, "absgrad", None)
        if absgrad is None:
            return None
        return absgrad[self._idx]

    def __getitem__(self, key) -> torch.Tensor:
        return self.value[key]

    def __getattr__(self, name: str):
        return getattr(self.value, name)
