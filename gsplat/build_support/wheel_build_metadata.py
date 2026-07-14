# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate build-dependent dependencies for compiled gsplat wheels."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from packaging.version import Version


_WHEEL_METADATA_STATES = frozenset(
    {"wheel", "editable", "metadata_wheel", "metadata_editable"}
)


def _build_torch_cuda_version() -> str | None:
    """Return the CUDA toolkit version used by build-environment Torch."""

    import torch

    return torch.version.cuda


def cupy_requirement_for_cuda(cuda_version: object) -> str:
    """Return the prebuilt CuPy distribution for a dotted CUDA version."""

    if not isinstance(cuda_version, str):
        raise RuntimeError(
            "cannot select CuPy because build-environment Torch has no CUDA version"
        )
    match = re.fullmatch(r"(\d+)\.\d+(?:\.\d+)?", cuda_version)
    if match is None:
        raise RuntimeError(
            "cannot select CuPy from malformed build-environment Torch CUDA "
            f"version {cuda_version!r}"
        )
    return f"cupy-cuda{int(match.group(1))}x"


class WheelBuildMetadataProvider:
    """Add build-compatible Torch and CuPy dependencies to wheel metadata."""

    def __init__(self) -> None:
        self._build_state: str | None = None

    @staticmethod
    def _check_settings(settings: Mapping[str, Any]) -> None:
        """Reject configuration because this provider has no tunable fields."""

        if settings:
            raise RuntimeError("wheel metadata provider accepts no settings")

    def build_state(self, build_state: str) -> None:
        """Record which packaging artifact scikit-build-core is producing."""

        self._build_state = build_state

    def dynamic_metadata(
        self,
        settings: Mapping[str, Any],
        _project: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the state-dependent addition to project dependencies."""

        self._check_settings(settings)
        if self._build_state is None:
            raise RuntimeError("scikit-build-core did not provide the build state")

        # An sdist keeps the broad static Torch floor and remains portable
        # across CUDA majors. Wheel metadata is resolved later, inside the
        # actual build environment.
        if self._build_state == "sdist":
            return {"dependencies": []}
        if self._build_state not in _WHEEL_METADATA_STATES:
            raise RuntimeError(f"unsupported build state: {self._build_state}")

        # CUDA variants use PEP 440 local labels such as +cu130. The public
        # release is the ABI identity the wheel matrix promises; leaving the
        # local label out lets a matching CUDA variant satisfy this pin.
        public_torch_version = Version(metadata.version("torch")).public
        cupy_requirement = cupy_requirement_for_cuda(_build_torch_cuda_version())
        return {
            "dependencies": [
                f"torch=={public_torch_version}",
                f'{cupy_requirement}; extra == "png"',
            ]
        }

    @staticmethod
    def dynamic_wheel(settings: Mapping[str, Any]) -> dict[str, bool]:
        """Declare that wheel dependencies may extend sdist dependencies."""

        WheelBuildMetadataProvider._check_settings(settings)
        return {"dependencies": True}
