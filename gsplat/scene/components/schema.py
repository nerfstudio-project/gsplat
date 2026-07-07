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

"""Shared splat schema validation utilities."""

from __future__ import annotations

from collections.abc import Mapping

import torch

RESERVED_SCENE_KEYS = frozenset(
    {
        "component_index",
        "pose_offsets",
        "pose_counts",
        "t",
    }
)


def validate_splat_schema(
    splats: Mapping[str, torch.Tensor],
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
    context: str = "splat tensor",
) -> int:
    """Validate leading row alignment and optional reference compatibility.

    Opacities are allowed in either renderer-canonical ``(N,)`` form or the
    legacy trailing-singleton ``(N, 1)`` form.  Reference comparisons use the
    canonical shape so callers can normalize opacities after validation without
    mutating user-owned ``ParameterDict`` inputs before an error is raised.
    """
    if len(splats) == 0 or "means" not in splats:
        raise ValueError(f"{context}s must not be empty and must contain 'means'")

    reserved = sorted(set(splats.keys()) & RESERVED_SCENE_KEYS)
    if reserved:
        raise ValueError(f"{context} keys are reserved for scene metadata: {reserved}")

    count = splats["means"].shape[0]
    for key, value in splats.items():
        if value.ndim == 0:
            raise ValueError(f"{context} {key!r} must have a leading row dim")
        if value.shape[0] != count:
            raise ValueError(
                f"{context} {key!r} has leading dim {value.shape[0]}, "
                f"expected {count}"
            )
        if key == "opacities" and _canonical_shape(key, value.shape, count) is None:
            raise ValueError(
                f"{context} 'opacities' must have shape ({count},) or "
                f"({count}, 1), got {tuple(value.shape)}"
            )

    if reference is None or len(reference) == 0:
        return count

    splat_keys = set(splats.keys())
    reference_keys = set(reference.keys())
    if splat_keys != reference_keys:
        missing = sorted(reference_keys - splat_keys)
        extra = sorted(splat_keys - reference_keys)
        raise ValueError(
            f"{context} keys must match reference; missing={missing}, extra={extra}"
        )

    reference_count = reference["means"].shape[0]
    for key, value in splats.items():
        ref = reference[key]
        shape = _canonical_shape(key, value.shape, count)
        ref_shape = _canonical_shape(key, ref.shape, reference_count)
        if shape is None or ref_shape is None:
            raise ValueError(f"{context} {key!r} has invalid shape")
        if shape[1:] != ref_shape[1:]:
            raise ValueError(
                f"{context} {key!r} trailing shape {tuple(shape[1:])} "
                f"does not match reference shape {tuple(ref_shape[1:])}"
            )
        if value.dtype != ref.dtype:
            raise TypeError(
                f"{context} {key!r} dtype {value.dtype} does not match "
                f"reference dtype {ref.dtype}"
            )
        if value.device != ref.device:
            raise ValueError(
                f"{context} {key!r} device {value.device} does not match "
                f"reference device {ref.device}"
            )

    return count


def normalize_splat_opacities(
    splats: torch.nn.ParameterDict,
    count: int,
) -> torch.nn.ParameterDict:
    """Return splats with opacities normalized to ``(N,)`` without mutating input."""
    if "opacities" not in splats or splats["opacities"].shape == (count,):
        return splats

    opacities = splats["opacities"]
    if opacities.shape != (count, 1):
        raise ValueError(
            f"splat tensor 'opacities' must have shape ({count},) or "
            f"({count}, 1), got {tuple(opacities.shape)}"
        )

    normalized = torch.nn.ParameterDict(
        {
            key: (
                torch.nn.Parameter(
                    value.reshape(count),
                    requires_grad=value.requires_grad,
                )
                if key == "opacities"
                else value
            )
            for key, value in splats.items()
        }
    )
    return normalized


def _canonical_shape(
    key: str,
    shape: torch.Size,
    count: int,
) -> tuple[int, ...] | None:
    if key != "opacities":
        return tuple(shape)
    if tuple(shape) == (count,) or tuple(shape) == (count, 1):
        return (count,)
    return None
