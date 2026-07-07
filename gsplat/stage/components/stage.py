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

from typing import Any, Callable

from gsplat.scene import GaussianScene

from .component_collection import ComponentCollection


class Stage:
    """Orchestrates GaussianScene(s) and their render functions.

    Stage pairs each scene with a render function and dispatches
    ``render(scene_id, **kwargs)`` calls, passing transient render-time splats as
    a keyword argument. Scenes are keyed by their ``id`` property.

    During training, a Stage typically holds a single scene.
    During inference, multiple scenes can be loaded for independent
    rendering.

    Note: today Stage is intentionally Gaussian-only — it forwards the
    ``scene.apply_transforms()`` result to the render fn as ``splats``.
    Optimizers still operate on persistent ``scene.splats``.
    """

    def __init__(self) -> None:
        self._scenes: dict[str, tuple[GaussianScene, Callable]] = {}
        self._collections: dict[str, ComponentCollection] = {}

    def add_scene(self, scene: GaussianScene, render_fn: Callable) -> None:
        """Register a scene and its render function, keyed by scene.id."""
        if scene.id in self._scenes or scene.id in self._collections:
            raise ValueError(f"Scene {scene.id!r} already registered on this Stage")
        self._scenes[scene.id] = (scene, render_fn)

    def add_collection(
        self,
        collection: ComponentCollection,
        render_fn: Callable,
    ) -> None:
        """Register a component collection and its shared render function."""
        if collection.id in self._scenes or collection.id in self._collections:
            raise ValueError(
                f"Collection {collection.id!r} already registered on this Stage"
            )
        collection.render_fn = render_fn
        self._collections[collection.id] = collection

    def scene_ids(self) -> list[str]:
        """Return the ids renderable through this stage."""
        return list(self._scenes.keys()) + list(self._collections.keys())

    def get_scene(self, scene_id: str) -> GaussianScene:
        """Return the registered ``GaussianScene`` for ``scene_id``."""
        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; "
                f"available scenes: {list(self._scenes.keys())}"
            )
        return self._scenes[scene_id][0]

    @staticmethod
    def _pop_render_time(kwargs: dict[str, Any]) -> Any:
        """Consume optional transform time from render kwargs."""
        time_sec = kwargs.pop("time_sec", None)
        t = kwargs.pop("t", None)
        if time_sec is not None and t is not None:
            raise ValueError("Pass only one of 't' or 'time_sec' to Stage.render")
        return t if t is not None else time_sec

    def render(self, scene_id: str, **kwargs) -> Any:
        """Render a scene by id.

        Args:
            scene_id: The id of the scene to render.
            **kwargs: Forwarded to the scene's render function.

        The render function is called with ``splats=...`` plus forwarded
        ``kwargs``. ``t``/``time_sec`` are consumed as transform time before
        dispatch; static scenes use the same path with identity behavior.
        Returns whatever the render function returns — caller is responsible
        for matching the return arity (e.g. 3-tuple for 3DGS, 7-tuple for 2DGS).
        """
        if scene_id in self._collections:
            render_time = self._pop_render_time(kwargs)
            collection = self._collections[scene_id]
            if render_time is None:
                return collection.render(**kwargs)
            return collection.render(t=render_time, **kwargs)

        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; " f"available: {self.scene_ids()}"
            )
        s, fn = self._scenes[scene_id]
        render_time = self._pop_render_time(kwargs)
        splats = s.apply_transforms(t=render_time if render_time is not None else 0.0)
        return fn(splats=splats, **kwargs)
