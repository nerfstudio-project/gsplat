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

import torch

from gsplat.sensors.models import ImageFrame


def test_frame_properties(pinhole_model, static_pose):
    """Verify is_rolling_shutter and frame_duration_us for a frame with distinct timestamps."""
    frame = ImageFrame("cam", pinhole_model, static_pose, 10, 20, torch.zeros(3, 4, 1))
    assert frame.is_rolling_shutter
    assert frame.frame_duration_us == 10
