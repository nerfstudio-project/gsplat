# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gsplat_sensors.models import ImageFrame


def test_frame_properties(pinhole_model, static_pose):
    """Verify is_rolling_shutter and frame_duration_us for a frame with distinct timestamps."""
    frame = ImageFrame("cam", pinhole_model, static_pose, 10, 20, torch.zeros(3, 4, 1))
    assert frame.is_rolling_shutter
    assert frame.frame_duration_us == 10
