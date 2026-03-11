# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_test_data(
    data_path: Optional[str] = None,
    device="cuda",
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),
    scene_grid: int = 1,
):
    """Load the test data."""
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz")
    data = np.load(data_path)
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    Ks = torch.from_numpy(data["Ks"]).float().to(device)
    means = torch.from_numpy(data["means3d"]).float().to(device)
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)
    C = len(viewmats)

    # crop
    aabb = torch.tensor(scene_crop, device=device)
    edges = aabb[3:] - aabb[:3]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, colors = means[sel], colors[sel]

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)
    colors = colors.repeat(repeats**2, 1)

    # create gaussian attributes
    N = len(means)
    # Generate scales in range [1e-4, 0.02] to avoid numerical instability
    # Gradient of 1/scale is -1/scale², which explodes for extremely small scales
    min_scale = 1e-4
    max_scale = 0.02
    scales = torch.rand((N, 3), device=device) * (max_scale - min_scale) + min_scale

    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)
    opacities = torch.rand((N,), device=device)

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height
