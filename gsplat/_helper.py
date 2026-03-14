# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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


def expand_named_params(named_params):
    """
    Expand a list of (name, value) tuples into pytest.param objects with IDs.

    Args:
        named_params: List of (name, value) tuples where name is the test ID
                     and value is the parameter value.

    Returns:
        List of pytest.param objects with IDs set.

    Example:
        >>> ROLLING_SHUTTER_TYPES = [
        ...     ("L2R", RollingShutterType.ROLLING_LEFT_TO_RIGHT),
        ...     ("R2L", RollingShutterType.ROLLING_RIGHT_TO_LEFT),
        ... ]
        >>> @pytest.mark.parametrize("rs_type", expand_named_params(ROLLING_SHUTTER_TYPES))
    """
    import pytest

    return [pytest.param(value, id=name) for name, value in named_params]


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


def get_inlier_abserror_mask(actual, expected, *, quantile=None, atol=None, rtol=None):
    """
    Create mask for inliers based on error thresholds.

    Combines quantile-based filtering with absolute/relative tolerance checks.
    Uses the same condition as torch.testing.assert_close.

    Args:
        actual: Actual tensor (e.g., CUDA implementation)
        expected: Expected tensor (e.g., reference implementation)
        quantile: Quantile threshold in [0, 1] (e.g., 0.99 = mask out worst 1%). Optional.
        atol: Absolute tolerance. Optional.
        rtol: Relative tolerance (relative to expected). Optional. Requires atol if specified.

    Returns:
        Boolean mask same shape as inputs, True for inliers (values within all specified thresholds)
    """
    # Validate arguments
    assert (
        rtol is None or atol is not None
    ), "If rtol is specified, atol must also be specified"

    abs_diff = (actual - expected).abs()

    # Build mask by combining conditions
    mask = torch.ones_like(abs_diff, dtype=torch.bool)

    # Apply quantile threshold if specified
    if quantile is not None:
        quantile_threshold = torch.quantile(abs_diff, quantile).item()
        mask = mask & (abs_diff <= quantile_threshold)

    # Apply atol/rtol threshold if specified
    if atol is not None:
        rtol_val = rtol if rtol is not None else 0
        # Relative tolerance (element-wise, depends on expected values)
        threshold = atol + rtol_val * expected.abs()
        mask = mask & (abs_diff <= threshold)

    return mask


def assert_shape(name: str, t: torch.Tensor, shape: tuple):
    """
    Check if the shape of a tensor matches a given shape.

    Args:
        name: Name of the tensor
        t: Tensor to check
        shape: Shape to check against
    """
    if t.ndim != len(shape):
        raise ValueError(
            f"{name} must have rank {len(shape)} like {shape}, got {t.shape}"
        )

    try:
        torch.broadcast_shapes(t.shape, shape)
        return True
    except Exception:
        raise ValueError(f"{name} must have shape {shape}, got {t.shape}")


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
):
    # rtol, atol = 0,0

    torch.testing.assert_close(
        actual,
        expected,
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
    )


def assert_mismatch_ratio(actual, expected, *, max=1e-5):
    """
    Assert that the mismatch ratio is less than a given tolerance.
    """
    if max is None:
        max = 1e-5

    # max=0

    assert actual.shape == expected.shape, f"{actual.shape=} {expected.shape=}"

    mismatch = (actual != expected).sum().item()
    total = expected.numel()
    mismatch_ratio = mismatch / total if total > 0 else 1
    assert (
        mismatch_ratio <= max
    ), f"Too many validity mismatches: {mismatch}/{total} ({mismatch_ratio*100:.2f}%) "
