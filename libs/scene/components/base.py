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

from abc import ABC, abstractmethod

import torch


class Scene(ABC):
    """Minimal base interface for scene-like containers.

    Subclasses must pass a non-empty string ``id`` that uniquely
    identifies this scene on a Stage.

    Topology hooks (``on_duplicate``, ``on_split``, etc.) are called by
    strategy ops when Gaussians are added, removed, or relocated.
    Override them to keep sidecar data in sync with scene components.
    The defaults are no-ops.
    """

    def __init__(self, id: str) -> None:
        self.id = id

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("Scene id must be a non-empty string")
        self._id = value

    @abstractmethod
    def put(self, name: str, component: object) -> None:
        """Add a named component to the scene."""

    @abstractmethod
    def get(self, component: str) -> object:
        """Return a component from the scene."""

    # -- Topology hooks (no-op defaults) --

    def on_duplicate(self, sel: torch.Tensor) -> None:
        pass

    def on_split(self, sel: torch.Tensor, rest: torch.Tensor) -> None:
        pass

    def on_remove(self, remove_mask: torch.Tensor) -> None:
        pass

    def on_relocate(
        self, dead_indices: torch.Tensor, sampled_indices: torch.Tensor
    ) -> None:
        pass

    def on_sample_add(self, sampled_indices: torch.Tensor) -> None:
        pass

    def on_permute(self, order: torch.Tensor) -> None:
        pass
