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
"""Public-facing NHT (Neural Harmonic Textures) rendering parameters.

The actual NHT raster dispatch lives in :mod:`gsplat.nht._wrapper` (the
``rasterize_to_pixels_eval3d_nht_extra`` entry point); :func:`gsplat.rendering.rasterization`
forwards ``nht_params`` through ``rasterize_to_pixels_eval3d_extra`` without
any NHT-specific branches of its own. This module is therefore deliberately
small: it just defines the ``NHTParams`` dataclass that users pass in.
"""

from dataclasses import dataclass


@dataclass
class NHTParams:
    """Container for NHT-only rasterization parameters.

    Grouping these into a single optional kwarg of ``rasterization`` keeps the
    main signature change to one parameter and makes downstream code that
    doesn't use NHT effectively untouched.

    Attributes:
        enabled: Master switch. When False, ``rasterization()`` ignores the
            NHTParams and behaves like a standard 3DGS call.
        center_ray_mode: If True, the NHT kernel uses a single per-camera
            center ray for the view-direction output instead of per-pixel
            rays. Matches the ``deferred_opt_center_ray_encoding`` config
            in ``simple_trainer_nht.py``.
        ray_dir_scale: Multiplier applied to the normalized ray direction
            before mapping into the tiny-cuda-nn ``[0, 1]`` range
            (``(v * scale + 1) / 2``). Set by the deferred-shader module
            (see ``DeferredShaderModule.ray_dir_scale``).
    """

    enabled: bool = True
    center_ray_mode: bool = False
    ray_dir_scale: float = 1.0


__all__ = ["NHTParams"]
