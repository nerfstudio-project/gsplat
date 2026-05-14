# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gsplat_sensors.models.common import (
    compute_scaled_resolution,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


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
