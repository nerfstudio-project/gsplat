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

"""Identity transform op."""

from __future__ import annotations

import torch

from .transform_op import TransformOp


class IdentityOp(TransformOp):
    """No-op transform used for world-space scenes."""

    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        """Validate mask-only identity context."""
        del count
        unsupported = sorted(set(ctx.keys()) - {"keep_mask"})
        if unsupported:
            raise ValueError(
                "IdentityOp only supports mask context; unsupported context keys: "
                f"{unsupported}"
            )
        if "keep_mask" in ctx and ctx["keep_mask"].reshape(-1).shape != (1,):
            raise ValueError("IdentityOp keep_mask context must contain one value")

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return collected tensors unchanged."""
        return collected
