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

"""Base transform operation."""

from __future__ import annotations

from typing import Any, ClassVar

import torch


class TransformOp:
    """Base class for one stateless scene transform operation."""

    _registry: ClassVar[dict[str, type["TransformOp"]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        TransformOp._registry[cls.__name__] = cls

    def state_dict(self) -> dict[str, object]:
        """Return serializable metadata for this transform op."""
        return {"type": self.__class__.__name__}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TransformOp":
        """Restore this op from ``state_dict()`` metadata."""
        del state
        return cls()

    @classmethod
    def build_from_state_dict(cls, state: dict[str, Any] | str) -> "TransformOp":
        """Construct a registered transform op from serialized metadata."""
        if isinstance(state, str):
            op_type = state
            op_state: dict[str, Any] = {"type": state}
        else:
            op_type = state["type"]
            op_state = state
        op_cls = cls._registry.get(op_type)
        if op_cls is None:
            raise ValueError(f"Unknown transform op type {op_type!r}")
        return op_cls.from_state_dict(op_state)

    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        """Validate component context before graph execution."""
        raise NotImplementedError("TransformOp.validate_ctx is not implemented")

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply this op to collected tensors."""
        raise NotImplementedError("TransformOp.apply is not implemented")
