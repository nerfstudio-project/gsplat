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

"""Coordinate-frame and interval helpers for NCore dataset loading.

Ported from 3dgrut-internal/threedgrut/datasets/ncoreUtils.py.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt


class HalfClosedInterval:
    """Represents a half-closed interval [start, end)."""

    def __init__(self, start: int, end: int) -> None:
        assert start <= end
        self.start = start
        self.end = end

    def cover_range(self, sorted_samples: np.ndarray) -> range:
        """Return the index range covering samples within [start, end)."""
        cover_range_start = int(np.searchsorted(sorted_samples, self.start, side="left"))
        cover_range_stop = int(np.searchsorted(sorted_samples, self.end, side="left"))
        return range(cover_range_start, cover_range_stop)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, int):
            return self.start <= item < self.end
        elif isinstance(item, HalfClosedInterval):
            return (self.start <= item.start) and (item.end <= self.end)
        raise TypeError(f"Expected int or HalfClosedInterval, got {type(item).__name__}")


class FrameConversion:
    """Converts poses and points between canonical 3D frames.

    Encodes a combined axis-permutation, origin-shift, and uniform scale as a
    single 4x4 matrix, where matrix[3,3] = 1/scale.
    """

    def __init__(self, matrix: npt.NDArray[np.float32]) -> None:
        assert matrix.shape == (4, 4)
        assert matrix.dtype == np.float32
        assert matrix[3, 3] > 0.0
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0)
        self.matrix = matrix

    @classmethod
    def from_origin_scale_axis(
        cls,
        target_origin: npt.NDArray[np.float32],
        target_scale: float,
        target_axis: List[int],
    ) -> "FrameConversion":
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = -target_origin
        matrix[3, 3] = 1.0 / target_scale
        assert len(np.unique(target_axis)) == 3
        matrix = matrix[target_axis + [3]]
        return cls(matrix=matrix)

    @property
    def target_scale(self) -> float:
        return 1.0 / float(self.matrix[3, 3])

    def get_transformation_matrices(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        T = self.matrix.copy()
        T *= self.target_scale
        inv_s = float(self.matrix[3, 3])
        S = np.zeros((4, 4), dtype=np.float32)
        np.fill_diagonal(S, [inv_s, inv_s, inv_s, 1.0])
        return T, S

    def transform_poses(self, T_poses_source: np.ndarray) -> np.ndarray:
        T_poses = T_poses_source.reshape((-1, 4, 4))
        T, S = self.get_transformation_matrices()
        T_poses = T @ T_poses @ S
        return T_poses.squeeze()
