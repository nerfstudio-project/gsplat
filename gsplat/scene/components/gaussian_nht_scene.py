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

from typing import Optional

import torch

from gsplat.nht import NHTParams

from .gaussian_scene import GaussianScene


class GaussianNHTScene(GaussianScene):
    """A ``GaussianScene`` whose splats carry harmonic features for NHT.

    Identical container to :class:`GaussianScene` (same ``splats``
    ParameterDict, same topology hooks used by densification strategies);
    the distinct type exists so that :func:`gsplat.experimental.render.functional.render_scene`
    can dispatch to the NHT feature-rasterization + deferred-shading path
    instead of the standard SH/RGB path. ``splats["features"]`` (rather than
    ``"colors"``/``"sh0"``/``"shN"``) holds the per-Gaussian harmonic
    features consumed by the deferred shader.

    ``nht_params`` is mutable rather than frozen at construction time: it
    typically depends on the deferred shader module's current
    ``ray_dir_scale``, which callers re-derive on every render call.
    """

    def __init__(self, id: str, nht_params: Optional[NHTParams] = None) -> None:
        super().__init__(id)
        self.nht_params = nht_params

    @classmethod
    def from_splats(
        cls,
        splats: torch.nn.ParameterDict,
        id: str,
        nht_params: Optional[NHTParams] = None,
        signal: dict[str, torch.Tensor] | None = None,
    ) -> "GaussianNHTScene":
        if len(splats) == 0 or "means" not in splats or "features" not in splats:
            raise ValueError(
                "GaussianNHTScene.from_splats requires a non-empty ParameterDict "
                "containing 'means' and 'features'"
            )
        scene = cls(id, nht_params=nht_params)
        if signal is not None:
            device = splats["means"].device
            scene.signal = {k: v.to(device) for k, v in signal.items()}
        scene.put(id, splats)
        return scene
