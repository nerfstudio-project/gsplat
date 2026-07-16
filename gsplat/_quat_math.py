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
"""Extension-free quaternion math shared across gsplat's Python fallbacks.

Pure-torch implementations of the geometry-convention quaternion helpers
(mirroring the device templates in
``gsplat/geometry/kernels/cuda/csrc/quaternion.cuh``) that must stay
importable without compiling any CUDA extension: CPU fallbacks call them
directly, and the geometry kernels module reuses them as its
differentiable references.

The rolling-shutter slerp (``gsplat/cuda/_math.py``, glm convention) and
the sensor mean-pose slerp (``gsplat/sensors/kernels/cameras/ops.py``,
wxyz order) deliberately do NOT live here — each mirrors a different
kernel and must keep its own convention for parity.
"""

import torch

from torch import Tensor

from .constants import SLERP_SMALL_ANGLE_DOT_THRESHOLD


def quat_slerp_batched(q1: Tensor, q2: Tensor, t: Tensor | float) -> Tensor:
    """Differentiable SLERP matching ``gsplat_geometry::quat_slerp_pair_fwd``.

    Quaternions are ``[..., 4]`` in xyzw order; ``t`` is a Python float or a
    tensor broadcastable against ``[..., 1]``. Semantics mirror the CUDA
    helper exactly: hemisphere flip on a negative dot, dot clamped into
    ``[-1, 1]``, a normalized linear blend above
    ``SLERP_SMALL_ANGLE_DOT_THRESHOLD``, and exact ``sin``-ratio weights
    (no renormalization) below it.
    """
    d = (q1 * q2).sum(dim=-1, keepdim=True)
    q2e = torch.where(d < 0, -q2, q2)
    c_raw = (q1 * q2e).sum(dim=-1, keepdim=True)
    c = c_raw.clamp(min=-1.0, max=1.0)
    use_lerp = c > SLERP_SMALL_ANGLE_DOT_THRESHOLD
    om = 1.0 - t
    r = om * q1 + t * q2e
    y_lerp = r / r.norm(dim=-1, keepdim=True)
    theta = torch.acos(c)
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    y_slerp = w1 * q1 + w2 * q2e
    return torch.where(use_lerp.expand_as(q1), y_lerp, y_slerp)
