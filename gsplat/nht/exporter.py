# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NHT / deferred-shading model export.

Writes per-primitive feature vectors (instead of SH coefficients) into a
standard PLY file, plus a companion ``.deferred.pt`` with the deferred
shading network weights.
"""

import os
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch


def export_splats_nht(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    deferred_module: Optional[Dict[str, Any]] = None,
    save_to: Optional[str] = None,
) -> bytes:
    """Export an NHT Gaussian Splat model to a PLY file.

    Features are stored as ``f_nht_0 .. f_nht_{D-1}`` properties.  When
    *deferred_module* is provided (state-dict or config+weights dict) and
    *save_to* is set, the module is saved as a companion ``.deferred.pt``
    file next to the PLY.

    Args:
        means: Splat means, shape ``(N, 3)``.
        scales: Splat log-scales, shape ``(N, 3)``.
        quats: Splat quaternions, shape ``(N, 4)``.
        opacities: Splat logit-opacities, shape ``(N,)``.
        features: Per-primitive feature vectors, shape ``(N, D)``.
        deferred_module: Optional dict to persist alongside the PLY.
        save_to: If given, write the PLY (and companion) to this path.

    Returns:
        The binary PLY bytes.
    """
    total = means.shape[0]
    assert means.shape == (total, 3)
    assert scales.shape == (total, 3)
    assert quats.shape == (total, 4)
    assert opacities.shape == (total,)
    assert features.ndim == 2 and features.shape[0] == total

    invalid = (
        torch.isnan(means).any(dim=1)
        | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities)
        | torch.isinf(opacities)
        | torch.isnan(features).any(dim=1)
        | torch.isinf(features).any(dim=1)
    )
    valid = ~invalid
    means = means[valid]
    scales = scales[valid]
    quats = quats[valid]
    opacities = opacities[valid]
    features = features[valid]

    num_splats = means.shape[0]
    buf = BytesIO()

    buf.write(b"ply\n")
    buf.write(b"format binary_little_endian 1.0\n")
    buf.write(f"element vertex {num_splats}\n".encode())
    buf.write(b"property float x\n")
    buf.write(b"property float y\n")
    buf.write(b"property float z\n")
    for j in range(features.shape[1]):
        buf.write(f"property float f_nht_{j}\n".encode())
    buf.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buf.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buf.write(f"property float rot_{i}\n".encode())
    buf.write(b"end_header\n")

    splat_data = torch.cat(
        [means, features, opacities.unsqueeze(1), scales, quats], dim=1
    ).to(torch.float32)

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    buf.write(splat_data.detach().cpu().numpy().astype(float_dtype).tobytes())

    data = buf.getvalue()

    if save_to:
        with open(save_to, "wb") as f:
            f.write(data)
        if deferred_module is not None:
            base, _ = os.path.splitext(save_to)
            torch.save(deferred_module, base + ".deferred.pt")

    return data
