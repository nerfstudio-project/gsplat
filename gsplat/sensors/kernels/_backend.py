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

"""Load native sensor registrations, preferring prebuilt then JIT build.

Environment variables:

- ``GSPLAT_SENSORS_FORCE_JIT``: When set to ``"1"``, skip the prebuilt
  ``gsplat_sensors_hip`` wheel import and fall straight through to the JIT
  build path in :mod:`.hip.build`. Useful when iterating on the native
  sources locally so every interpreter restart picks up edits without
  uninstalling/reinstalling the wheel.
"""

from __future__ import annotations

from gsplat._lazy_backend import make_lazy_backend


def _build():
    # Deferred so importing this module does not import .hip.build.
    from .hip.build import build_and_load_sensors_hip

    return build_and_load_sensors_hip()


_get_backend, __getattr__ = make_lazy_backend(
    module_name=__name__,
    public_name="_SENSORS_HIP",
    prebuilt_module="gsplat_sensors_hip",
    jit_loader=_build,
    FORCE_JIT_env="GSPLAT_SENSORS_FORCE_JIT",
)

__all__ = ["_SENSORS_HIP"]
