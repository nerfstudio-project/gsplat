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

"""Unit tests for AV trainer Gaussian initialization."""

from __future__ import annotations

import pytest
import torch

from gsplat.init_utils import knn_scale_init

# Skip cleanly (don't error collection) when the examples package/extra is
# unavailable, as on a minimal CPU core_tests env. It can be absent because:
#   - the build-tree pytest config omits the repo root from sys.path (only the
#     installed wheel adds it), so `examples` may not import
#   - av_trainer pulls in matplotlib, which lives only in the "examples" extra
init_gaussians = pytest.importorskip("examples.av_trainer").init_gaussians


def test_init_gaussians_multi_point_knn_isotropic_scales_and_identity_quats():
    """Multi-point clouds get kNN log-scales expanded isotropically and identity quats."""
    device = "cpu"
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    colors = torch.full((points.shape[0], 3), 0.5, device=device)

    params = init_gaussians(points, colors, device=device, sh_degree=0)

    expected_log_scales = knn_scale_init(points, k=3)
    expected_scales = expected_log_scales[:, None].expand(-1, 3)
    assert torch.allclose(params["scales"], expected_scales, atol=1e-6)
    assert torch.allclose(
        params["scales"],
        params["scales"][:, :1].expand_as(params["scales"]),
        atol=1e-6,
    )

    expected_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(
        points.shape[0], -1
    )
    assert torch.allclose(params["quats"], expected_quats, atol=1e-6)


def test_init_gaussians_singleton_fixed_scale_and_identity_quat():
    """A single point gets the fixed -5.0 log-scale and identity quaternion."""
    device = "cpu"
    points = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    colors = torch.tensor([[0.5, 0.5, 0.5]], device=device)

    params = init_gaussians(points, colors, device=device, sh_degree=0)

    assert torch.allclose(params["scales"], torch.full((1, 3), -5.0, device=device))
    assert torch.allclose(
        params["quats"], torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    )
