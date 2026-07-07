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

"""Component tensor view helpers."""

from __future__ import annotations

from collections.abc import Iterator, KeysView, Mapping
from typing import Callable

import torch


class TensorViews:
    """Named tensor views into a flat scene-owned tensor dictionary."""

    def __init__(
        self,
        parent: Mapping[str, torch.Tensor],
        range_fn: Callable[[], tuple[int, int] | None],
        indices_fn: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        self._parent = parent
        self._range_fn = range_fn
        self._indices_fn = indices_fn

    def _range(self) -> tuple[int, int] | None:
        return self._range_fn()

    def _indices(self) -> torch.Tensor:
        if self._indices_fn is None:
            range_ = self._range()
            if range_ is None:
                raise ValueError("TensorViews requires indices for non-contiguous rows")
            offset, count = range_
            return torch.arange(offset, offset + count)
        return self._indices_fn()

    @property
    def offset(self) -> int:
        """Current first row of this view in the parent buffer."""
        range_ = self._range()
        if range_ is not None:
            return range_[0]
        indices = self._indices()
        if indices.numel() == 0:
            return 0
        return int(indices[0].item())

    @property
    def count(self) -> int:
        """Current row count of this view in the parent buffer."""
        range_ = self._range()
        if range_ is not None:
            return range_[1]
        return int(self._indices().numel())

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return a named tensor view from this component."""
        range_ = self._range()
        if range_ is None:
            return self._parent[key][self._indices()]
        offset, count = range_
        return self._parent[key][offset : offset + count]

    def __contains__(self, key: object) -> bool:
        return key in self._parent

    def keys(self) -> KeysView[str]:
        """Return tensor names available through this component view."""
        return self._parent.keys()

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor_view)`` pairs for this component."""
        for key in self._parent:
            yield key, self[key]

    def mergeable(self, other: "TensorViews") -> bool:
        """Return whether two views are currently adjacent slices."""
        if self._parent is not other._parent:
            return False
        self_range = self._range()
        other_range = other._range()
        if self_range is None or other_range is None:
            return False
        return self_range[0] + self_range[1] == other_range[0]

    def merge(self, other: "TensorViews") -> "TensorViews":
        """Return a transient wider view over two currently adjacent views."""
        if not self.mergeable(other):
            raise ValueError("Cannot merge non-adjacent TensorViews")
        self_range = self._range()
        other_range = other._range()
        assert self_range is not None and other_range is not None
        start = self_range[0]
        count = self_range[1] + other_range[1]
        return TensorViews(self._parent, lambda: (start, count))
