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

"""NHT / deferred-shading model export.

Writes per-primitive feature vectors (instead of SH coefficients) into a
standard PLY file, plus a companion ``.deferred.pt`` with the deferred
shading network weights.

Both the per-primitive features and the tcnn MLP backbone weights are stored
as ``float16``. This matches the precision actually consumed by the NHT
rasterization kernels (which always cast features and run the fused MLP in
half precision), so the conversion is lossless w.r.t. inference and halves
the on-disk footprint of the largest tensors.
"""

import os
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch


# Header marker written into the PLY ``comment`` line so consumers can detect
# the fp16-encoded ``f_nht_*`` payload without parsing every property type.
NHT_PLY_FP16_COMMENT = (
    "NHT_F16 f_nht_* are float16 values stored as raw uint16 (little-endian) bits"
)


def cast_state_dict_to_fp16(
    state_dict: Dict[str, Any],
    *,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copy of ``state_dict`` with float32 tensors cast to float16.

    Args:
        state_dict: Mapping of parameter names to tensors (or nested values).
        prefix: If given, only keys starting with this prefix are downcast.
            Useful to limit conversion to the tcnn backbone (where fp16 is
            the native compute dtype) while keeping fp32 readout heads intact.

    Non-tensor values and tensors that are not ``torch.float32`` are passed
    through unchanged.
    """
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        match = prefix is None or k.startswith(prefix)
        if match and torch.is_tensor(v) and v.dtype == torch.float32:
            out[k] = v.detach().to(torch.float16)
        else:
            out[k] = v
    return out


def cast_state_dict_to_fp32(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`cast_state_dict_to_fp16`: upcast fp16 tensors to fp32.

    Modules created with default settings (both ``tinycudann`` networks and
    ``torch.nn.Linear``) hold fp32 master weights, so callers that load a
    state-dict produced by :func:`cast_state_dict_to_fp16` may want to upcast
    before :meth:`torch.nn.Module.load_state_dict`. (PyTorch's ``load_state_dict``
    already handles dtype mismatches via ``copy_``, but explicit upcasting
    keeps any downstream code that inspects the dict dtype consistent.)
    """
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.dtype == torch.float16:
            out[k] = v.detach().to(torch.float32)
        else:
            out[k] = v
    return out


def _cast_deferred_payload_to_fp16(payload: Any) -> Any:
    """Recursively downcast tensors inside the deferred-shader payload.

    Recognizes the conventional ``{"state_dict": ..., "ema": ..., "config": ...}``
    layout produced by the NHT trainer. Only the tcnn ``backbone.*`` weights
    are converted to fp16 — that's the only part of the network that actually
    runs in half precision. The auxiliary ``torch.nn.Linear`` heads keep fp32
    storage to remain bit-exact with their runtime behaviour.
    """
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in ("state_dict", "ema") and isinstance(v, dict):
                out[k] = cast_state_dict_to_fp16(v, prefix="backbone.")
            else:
                out[k] = v
        return out
    return payload


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

    Features are stored as ``f_nht_0 .. f_nht_{D-1}`` properties of type
    ``ushort``: each 16-bit value holds the raw IEEE-754 ``float16`` bit
    pattern of the corresponding feature channel. The PLY header carries a
    ``comment`` line (see :data:`NHT_PLY_FP16_COMMENT`) to make this convention
    discoverable. Compared to fp32 storage this halves the size of the feature
    payload (which dominates the file for typical NHT scenes) while remaining
    lossless w.r.t. the rasterization kernels, which always cast features to
    ``__half`` before use.

    When ``deferred_module`` is provided (state-dict or config+weights dict)
    and ``save_to`` is set, the module is saved as a companion ``.deferred.pt``
    file next to the PLY. tcnn backbone weights inside the payload are
    downcast to fp16 to match their on-device storage; readout heads remain
    fp32.

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
    feature_dim = features.shape[1]
    n_scales = scales.shape[1]
    n_quats = quats.shape[1]

    buf = BytesIO()

    buf.write(b"ply\n")
    buf.write(b"format binary_little_endian 1.0\n")
    buf.write(f"comment {NHT_PLY_FP16_COMMENT}\n".encode())
    buf.write(f"element vertex {num_splats}\n".encode())
    buf.write(b"property float x\n")
    buf.write(b"property float y\n")
    buf.write(b"property float z\n")
    for j in range(feature_dim):
        buf.write(f"property ushort f_nht_{j}\n".encode())
    buf.write(b"property float opacity\n")
    for i in range(n_scales):
        buf.write(f"property float scale_{i}\n".encode())
    for i in range(n_quats):
        buf.write(f"property float rot_{i}\n".encode())
    buf.write(b"end_header\n")

    # Build a single packed structured array with mixed fp32 / uint16 fields.
    # numpy's default ``align=False`` keeps the layout dense, matching PLY's
    # binary little-endian convention.
    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]
    dtype_fields += [(f"f_nht_{j}", "<u2") for j in range(feature_dim)]
    dtype_fields.append(("opacity", "<f4"))
    dtype_fields += [(f"scale_{i}", "<f4") for i in range(n_scales)]
    dtype_fields += [(f"rot_{i}", "<f4") for i in range(n_quats)]
    arr = np.empty(num_splats, dtype=np.dtype(dtype_fields))

    means_np = means.detach().to(torch.float32).cpu().numpy()
    arr["x"] = means_np[:, 0]
    arr["y"] = means_np[:, 1]
    arr["z"] = means_np[:, 2]

    # Cast features through fp16 then re-interpret the bits as uint16 so they
    # can be written via a standard PLY ``ushort`` property.
    features_fp16 = features.detach().to(torch.float16).contiguous().cpu().numpy()
    features_bits = features_fp16.view(np.uint16)
    for j in range(feature_dim):
        arr[f"f_nht_{j}"] = features_bits[:, j]

    arr["opacity"] = opacities.detach().to(torch.float32).cpu().numpy()
    scales_np = scales.detach().to(torch.float32).cpu().numpy()
    for i in range(n_scales):
        arr[f"scale_{i}"] = scales_np[:, i]
    quats_np = quats.detach().to(torch.float32).cpu().numpy()
    for i in range(n_quats):
        arr[f"rot_{i}"] = quats_np[:, i]

    buf.write(arr.tobytes())

    data = buf.getvalue()

    if save_to:
        with open(save_to, "wb") as f:
            f.write(data)
        if deferred_module is not None:
            base, _ = os.path.splitext(save_to)
            torch.save(
                _cast_deferred_payload_to_fp16(deferred_module),
                base + ".deferred.pt",
            )

    return data
