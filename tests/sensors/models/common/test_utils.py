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

import pytest
import torch

from gsplat.sensors.models.common import (
    compute_scaled_resolution,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

pytestmark = [pytest.mark.wheel_smoke]


def test_compute_scaled_resolution():
    """Verify isotropic, anisotropic, and explicit-override resolution scaling."""
    assert compute_scaled_resolution((100, 80), 0.5) == (50, 40)
    assert compute_scaled_resolution((100, 80), (0.5, 0.25)) == (50, 20)
    assert compute_scaled_resolution((100, 80), 0.5, (12, 13)) == (12, 13)


def test_quaternion_order_helpers():
    """Verify wxyz_to_xyzw permutes correctly and xyzw_to_wxyz is its inverse."""
    q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(wxyz_to_xyzw(q), torch.tensor([2.0, 3.0, 4.0, 1.0]))
    assert torch.equal(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
