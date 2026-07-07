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

"""Transform graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .transform_op import TransformOp


@dataclass(frozen=True)
class TransformGraph:
    """Stateless sequence of scene transform operations."""

    ops: list[TransformOp]

    def state_dict(self) -> dict[str, object]:
        """Return serializable transform graph metadata."""
        return {"ops": [op.state_dict() for op in self.ops]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TransformGraph":
        """Restore a transform graph from ``state_dict()`` metadata."""
        from .identity_op import IdentityOp
        from .rigid_transform_op import RigidTransformOp

        # Import built-ins so TransformOp.__init_subclass__ registers them.
        del IdentityOp, RigidTransformOp
        ops: list[TransformOp] = []
        for item in state.get("ops", []):
            ops.append(TransformOp.build_from_state_dict(item))
        return cls(ops)

    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        """Validate component context against every op in this graph."""
        for op in self.ops:
            op.validate_ctx(ctx, count)

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply every op in order to collected splats/context."""
        transformed = collected
        for op in self.ops:
            transformed = op.apply(transformed)
        return transformed
