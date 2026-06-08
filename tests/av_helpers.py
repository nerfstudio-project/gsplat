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

"""Shared helpers for AV trainer tests.

``av_trainer`` is imported behind a try/except so that this helper module is
safely importable from upstream environments (e.g. GitHub Actions
``core_tests.yml`` on ubuntu-latest) that do not install av_trainer's
optional dependencies (imageio, etc.). When the import fails, ``av_trainer``
is set to ``None`` and consuming tests are expected to skip themselves
(see ``tests/conftest.py::av_train_env`` and
``tests/test_av_scene_wrapper.py`` for the canonical patterns).
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples"))
try:
    av_trainer = importlib.import_module("av_trainer")
except ImportError:
    av_trainer = None


def make_av_splats() -> torch.nn.ParameterDict:
    """Minimal 2-Gaussian ParameterDict for AV trainer tests."""
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
            ),
            "scales": torch.nn.Parameter(torch.zeros(2, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            ),
            "opacities": torch.nn.Parameter(torch.zeros(2)),
            "colors": torch.nn.Parameter(
                torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
            ),
        }
    )


def make_av_scene(
    is_test: np.ndarray | None = None,
    height: int = 8,
    width: int = 8,
) -> SimpleNamespace:
    """Minimal 1-frame, 1-camera scene for AV trainer tests."""
    if is_test is None:
        is_test = np.array([False])
    image = torch.zeros(height, width, 3)
    return SimpleNamespace(
        images=image.unsqueeze(0).unsqueeze(0),
        viewmats=torch.eye(4).view(1, 1, 4, 4),
        Ks=torch.eye(3).view(1, 1, 3, 3),
        lidar_points=torch.zeros(0, 4),
        lidar_frame_indices=torch.zeros(0, dtype=torch.long),
        lidar_by_frame={},
        n_frames=1,
        n_cams=1,
        H=height,
        W=width,
        camera_names=["cam0"],
        is_test=is_test,
    )
