# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device/dtype tensor helpers shared by the sensorlib kernel op layers.

Lives in ``common`` so each kernel op layer can share one device-transfer policy
without importing from another (the kernel layers must not depend on one another).
"""

from __future__ import annotations

import torch
from torch import Tensor


def raise_or_target_device(tensor: Tensor, allow_device_transfer: bool) -> torch.device:
    """Return a CUDA device for the given tensor, raising if transfer is disallowed.

    Args:
        tensor: Input tensor whose device is inspected.
        allow_device_transfer: If False, raises when the tensor is not already on CUDA.
    """
    if tensor.device.type == "cuda":
        return tensor.device
    if not allow_device_transfer:
        raise RuntimeError(
            "CPU inputs require allow_device_transfer=True before gsplat_sensors ops launch."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat_sensors ops require CUDA")
    return torch.device("cuda")


def to_dev(
    tensor: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> Tensor:
    """Move a tensor to the target device and dtype, ensuring contiguity.

    Args:
        tensor: Source tensor.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: If False, raises when the tensor needs a device or
            dtype transfer before the kernel can launch.
    """
    target_device = device
    if device.type == "cuda" and device.index is None and tensor.device.type == "cuda":
        target_device = tensor.device
    needs_transfer = tensor.device != target_device or tensor.dtype != dtype
    if needs_transfer and not allow_device_transfer:
        raise RuntimeError(
            f"Tensor on {tensor.device} (dtype={tensor.dtype}) requires transfer "
            f"to {target_device} (dtype={dtype}). Set allow_device_transfer=True to allow "
            "implicit conversion before gsplat_sensors ops launch."
        )
    if not needs_transfer and tensor.is_contiguous():
        return tensor
    converted = (
        tensor.to(device=target_device, dtype=dtype) if needs_transfer else tensor
    )
    return converted if converted.is_contiguous() else converted.contiguous()


def zero_like(shape: tuple[int, ...], reference: Tensor) -> Tensor:
    """Allocate a zero tensor with the same device and dtype as ``reference``."""
    return torch.zeros(shape, device=reference.device, dtype=reference.dtype)


def timestamp_bounds(
    start_timestamp_us: int | None, end_timestamp_us: int | None
) -> tuple[int, int]:
    """Normalize optional timestamp arguments to a (start, end) int pair.

    Args:
        start_timestamp_us: Start timestamp in microseconds, or None.
        end_timestamp_us: End timestamp in microseconds, or None.

    Returns:
        ``(start_timestamp_us, end_timestamp_us)`` as ints, or ``(0, 0)`` when
        both are None. Raises ValueError if exactly one is None.
    """
    if start_timestamp_us is None and end_timestamp_us is None:
        return 0, 0
    if start_timestamp_us is None or end_timestamp_us is None:
        raise ValueError(
            "start_timestamp_us and end_timestamp_us must be provided together"
        )
    return int(start_timestamp_us), int(end_timestamp_us)


__all__ = ["raise_or_target_device", "timestamp_bounds", "to_dev", "zero_like"]
