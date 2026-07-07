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

"""Gaussian component view handle."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Callable

import torch

from .tensor_views import TensorViews
from .transform_ctx_view import TransformCtxView


@dataclass(frozen=True)
class GaussianComponent(Mapping[str, object]):
    """View handle for a named Gaussian scene component."""

    component_idx: int
    splats: TensorViews
    signal: TensorViews
    transform_ctx: TransformCtxView | None = None
    name: str | None = None
    mask_fn: Callable[[], torch.Tensor] | None = None

    @property
    def id(self) -> int:
        """Component id within its scene."""
        return self.component_idx

    def _mask(self) -> torch.Tensor | None:
        if self.mask_fn is None:
            return None
        return self.mask_fn()

    def _snapshot(self, views: TensorViews) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in views.items()}

    def __getitem__(self, key: str) -> object:
        """Return legacy dict-style fields for compatibility."""
        if key == "name":
            return self.name
        if key == "index":
            return self.component_idx
        if key == "mask":
            return self._mask()
        if key == "splats":
            return self._snapshot(self.splats)
        if key == "signal":
            return self._snapshot(self.signal)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("name", "index", "mask", "splats", "signal"))

    def __len__(self) -> int:
        return 5
