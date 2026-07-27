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

"""CPU unit tests for ``gsplat._helper.ensure_shape``."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import pytest
import torch

from gsplat._helper import ensure_shape


class Expect(Enum):
    """Expected ``ensure_shape`` outcome for one shape-coercion case."""

    SAME = auto()
    EXPAND = auto()
    RAISE = auto()


@dataclass(frozen=True)
class EnsureShapeCase:
    """One CPU-only ``ensure_shape`` shape-coercion scenario."""

    name: str
    input_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    expect: Expect


ENSURE_SHAPE_CASES = (
    EnsureShapeCase("exact", (2, 3, 4), (2, 3, 4), Expect.SAME),
    EnsureShapeCase(
        "missing-leading-dim",
        (4, 5, 6),
        (1, 4, 5, 6),
        Expect.EXPAND,
    ),
    EnsureShapeCase("size-1", (1, 5, 1), (4, 5, 6), Expect.EXPAND),
    EnsureShapeCase("non-broadcastable", (2, 5), (3, 5), Expect.RAISE),
    EnsureShapeCase(
        "broadcast-to-larger",
        (2, 4, 5, 6),
        (1, 4, 5, 6),
        Expect.RAISE,
    ),
)


@pytest.mark.parametrize(
    "case",
    ENSURE_SHAPE_CASES,
    ids=[case.name for case in ENSURE_SHAPE_CASES],
)
def test_ensure_shape(case: EnsureShapeCase):
    tensor = torch.arange(
        torch.Size(case.input_shape).numel(),
        dtype=torch.float32,
    ).reshape(case.input_shape)

    if case.expect is Expect.RAISE:
        with pytest.raises(ValueError, match=case.name):
            ensure_shape(case.name, tensor, case.target_shape)
        return

    result = ensure_shape(case.name, tensor, case.target_shape)

    if case.expect is Expect.SAME:
        assert result is tensor
        return

    assert case.expect is Expect.EXPAND
    assert result.shape == case.target_shape
    torch.testing.assert_close(result, tensor.expand(case.target_shape))
