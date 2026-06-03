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

from gsplat_scene import GaussianScene


class Stage:
    """Orchestrates GaussianScene(s) and their render functions.

    Stage pairs each scene with a render function and dispatches
    ``render(scene_id, **kwargs)`` calls, passing
    ``splats=scene.splats`` as a keyword argument.
    Scenes are keyed by their ``id`` property.

    During training, a Stage typically holds a single scene.
    During inference, multiple scenes can be loaded for independent
    rendering.

    Note: today Stage is intentionally Gaussian-only — it forwards
    ``scene.splats`` to the render fn, which is a ``GaussianScene``-specific
    field. The annotations below reflect that. Generalizing to non-Gaussian
    scenes will require lifting ``splats`` (or its forwarded slot) onto
    ``Scene`` first.
    """

    def __init__(self) -> None:
        self._scenes: dict[str, tuple[GaussianScene, Callable]] = {}

    def add_scene(self, scene: GaussianScene, render_fn: Callable) -> None:
        """Register a scene and its render function, keyed by scene.id."""
        if scene.id in self._scenes:
            raise ValueError(f"Scene {scene.id!r} already registered on this Stage")
        self._scenes[scene.id] = (scene, render_fn)

    def scene_ids(self) -> list[str]:
        """Return the ids of all registered scenes."""
        return list(self._scenes.keys())

    def get_scene(self, scene_id: str) -> GaussianScene:
        """Return the registered ``GaussianScene`` for ``scene_id``."""
        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; "
                f"available: {list(self._scenes.keys())}"
            )
        return self._scenes[scene_id][0]

    def render(self, scene_id: str, **kwargs) -> Any:
        """Render a scene by id.

        Args:
            scene_id: The id of the scene to render.
            **kwargs: Forwarded to the scene's render function.

        The render_fn is called as ``render_fn(splats=s.splats, **kwargs)``.
        Returns whatever the render function returns — caller is responsible
        for matching the return arity (e.g. 3-tuple for 3DGS, 7-tuple for 2DGS).
        """
        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; "
                f"available: {list(self._scenes.keys())}"
            )
        s, fn = self._scenes[scene_id]
        return fn(splats=s.splats, **kwargs)
