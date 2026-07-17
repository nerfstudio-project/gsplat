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

"""Abstract base class for direction-dependent background representations."""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from .base import Scene


class BackgroundScene(Scene):
    """Abstract base for direction-dependent background representations.

    Subclasses must implement :meth:`sample`. The default :meth:`composite`
    implementation is correct for most cases but may be overridden.

    Topology hooks (``on_duplicate``, ``on_split``, …) are inherited as
    no-ops — background representations have no rows to permute or remove.

    Args:
        id: Non-empty string uniquely identifying this scene on a Stage
            (accepted via the inherited :class:`Scene` constructor).
    """

    @abstractmethod
    def sample(self, rays_d: Tensor) -> Tensor:
        """Return background RGB for each ray direction.

        Args:
            rays_d: World-space unit ray directions, shape ``[N, 3]``.

        Returns:
            RGB radiance, shape ``[N, 3]``. Values are non-negative; the
            concrete implementation decides whether to clamp to ``[0, 1]`` or
            allow HDR values.
        """

    def composite(
        self,
        gaussian_rgb: Tensor,  # [N, 3]
        opacity: Tensor,  # [N]
        rays_d: Tensor,  # [N, 3]
        is_training: bool = False,
    ) -> Tensor:
        """Over-composite the background behind the Gaussian render.

        Default formula (sRGB space)::

            final = gaussian_rgb + sample(rays_d) * (1 - opacity)

        Args:
            gaussian_rgb: Accumulated Gaussian color from the rasterizer ``[N, 3]``.
            opacity: Accumulated Gaussian alpha from the rasterizer; ``[N]`` or
                ``[N, 1]`` are both accepted.
            rays_d: World-space ray directions matching the raster output ``[N, 3]``.
            is_training: Enables per-step gradient tracking in subclasses that
                need it (e.g. :class:`EnvMapBackground`); the side-effect-free
                default path ignores it.

        Returns:
            Composited RGB, same shape as ``gaussian_rgb``.
        """
        return self._blend_background(gaussian_rgb, self.sample(rays_d), opacity)

    def _blend_background(
        self,
        gaussian_rgb: Tensor,
        bg_rgb: Tensor,
        opacity: Tensor,
    ) -> Tensor:
        """Over-composite pre-sampled background radiance behind the render.

        Shared by :meth:`composite` and subclasses that sample/track before
        blending (e.g. :class:`EnvMapBackground`).
        """
        weight = 1.0 - opacity.reshape(-1, 1)
        return gaussian_rgb + bg_rgb * weight

    # -- Scene ABC contract --

    def put(
        self,
        name: str,
        component: object,
        ctx: dict[str, Tensor] | None = None,
    ) -> None:
        """Store a named tensor or parameter (e.g. ``"textures"``).

        Background representations do not support transform context; a
        non-``None`` ``ctx`` is rejected.
        """
        if ctx is not None:
            raise ValueError("BackgroundScene does not support transform context")
        setattr(self, name, component)

    def get(self, component: str) -> object:
        """Retrieve a previously stored named value."""
        return getattr(self, component)
