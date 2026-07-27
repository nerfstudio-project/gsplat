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

from .components.base import Scene
from .components.gaussian_scene import GaussianScene
from .components.gaussian_inference_scene import GaussianInferenceScene
from .sh_compression import SHCompressionMode

__all__ = [
    "functional",
    "Scene",
    "GaussianScene",
    "GaussianInferenceScene",
    "SHCompressionMode",
]


def __getattr__(name: str):
    if name == "functional":
        from importlib import import_module

        module = import_module(f"{__name__}.functional")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
