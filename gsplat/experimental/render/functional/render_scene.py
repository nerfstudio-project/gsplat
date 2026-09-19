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

"""Render dispatcher for the ``experimental`` package.

Dispatches to the Inference rasteriser for ``GaussianInferenceScene``, and to
the differentiable NHT rasteriser for ``GaussianNHTScene``.
"""

from __future__ import annotations

from typing import Any, Optional

from gsplat.scene import GaussianInferenceScene, GaussianNHTScene

from .gaussian_inference import (
    rasterize_gaussian_inference_scene,
)
from .gaussian_nht import (
    rasterize_gaussian_nht_scene,
)
from ..types import RenderReturn


def render_scene(
    scene,
    *,
    out: Optional[RenderReturn] = None,
    **request: Any,
) -> RenderReturn:
    """Render a ``GaussianInferenceScene`` or a ``GaussianNHTScene``.

    Raises:
        TypeError: If *scene* is neither a :class:`GaussianInferenceScene`
            nor a :class:`GaussianNHTScene`.
    """
    if isinstance(scene, GaussianNHTScene):
        ret = rasterize_gaussian_nht_scene(scene, out=out, **request)
        ret.metadata["render_path"] = "nht"
        return ret

    if isinstance(scene, GaussianInferenceScene):
        ret = rasterize_gaussian_inference_scene(scene, out=out, **request)
        ret.metadata["render_path"] = "inference"
        return ret

    raise TypeError(
        f"render_scene requires a GaussianInferenceScene or GaussianNHTScene; "
        f"got {type(scene).__name__}"
    )
