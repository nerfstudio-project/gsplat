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

from enum import IntEnum


class SHCompressionMode(IntEnum):
    """Scene packer SH layout modes.

    These control how SH coefficients are stored in GaussianInferenceScene.colors_packed.
    PACKED modes are simple layout transforms (flatten + optional fp16 cast),
    distinct from the render kernel's COMPRESS modes which apply full YCoCg quantization.
    """

    NONE = 0  # [N, 16, 3] float16 — raw coefficients
    PACKED_32B = 1  # [N, 48] float32 — flattened
    PACKED_16B = 2  # [N, 48] float16 — flattened + quantized to fp16


SH_COMPRESSION_MAP = {
    "none": SHCompressionMode.NONE,
    "32b": SHCompressionMode.PACKED_32B,
    "16b": SHCompressionMode.PACKED_16B,
}
SH_COMPRESSION_MODE_VALUES = frozenset(SHCompressionMode)

__all__ = [
    "SHCompressionMode",
    "SH_COMPRESSION_MAP",
    "SH_COMPRESSION_MODE_VALUES",
]
