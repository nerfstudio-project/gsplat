# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate build-dependent requirements for compiled gsplat wheels.

This module is also executable so pre-wheel environment provisioning can use
the same CUDA-to-CuPy mapping as the wheel metadata provider.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from importlib import metadata
import re
import sys
from typing import Any

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover - bootstrap may start with pip alone.
    from pip._vendor.packaging.version import Version


_WHEEL_METADATA_STATES = frozenset(
    {"wheel", "editable", "metadata_wheel", "metadata_editable"}
)


def cupy_requirement_for_cuda(cuda_version: str | None) -> str:
    """Return the binary CuPy distribution matching ``cuda_version``'s major.

    CuPy names its binary distributions by CUDA major, while both Torch and
    NVCC report dotted toolkit versions. Requiring a complete dotted version
    catches CPU-only Torch builds and malformed configuration values instead
    of silently selecting an unrelated CuPy package.
    """

    if cuda_version is None:
        raise RuntimeError(
            "cannot select CuPy because the build Torch does not report a CUDA version"
        )
    if not isinstance(cuda_version, str):
        raise RuntimeError(
            f"cannot select CuPy from non-string CUDA version {cuda_version!r}"
        )

    match = re.fullmatch(r"([1-9][0-9]*)\.[0-9]+(?:\.[0-9]+)*", cuda_version)
    if match is None:
        raise RuntimeError(
            f"cannot select CuPy from malformed CUDA version {cuda_version!r}"
        )
    return f"cupy-cuda{match.group(1)}x"


def _build_torch_cuda_version() -> str | None:
    """Return the CUDA toolkit version reported by the build environment's Torch."""

    # Import Torch lazily: sdist metadata is intentionally usable without the
    # heavyweight wheel-build dependency installed.
    torch = importlib.import_module("torch")
    return torch.version.cuda


def cupy_requirement_for_build_torch() -> str:
    """Return the CuPy requirement matching the build environment's Torch."""

    return cupy_requirement_for_cuda(_build_torch_cuda_version())


class WheelBuildMetadataProvider:
    """Add build Torch and CUDA-matched CuPy requirements to wheel metadata."""

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

        # An sdist records the broad compatibility floor and marks
        # Requires-Dist as wheel-dynamic. Querying Torch here would make source
        # distribution creation unnecessarily require a heavyweight build dep.
        if self._build_state == "sdist":
            return {"dependencies": []}

        if self._build_state not in _WHEEL_METADATA_STATES:
            raise RuntimeError(f"unsupported build state: {self._build_state}")

        # CUDA variants use PEP 440 local labels such as +cu130. The public
        # release is the ABI identity the wheel matrix promises; leaving the
        # local label out also lets a matching CUDA variant satisfy this pin.
        public_version = Version(metadata.version("torch")).public

        # Core metadata is prepared before CMake configures the build, so the
        # build environment's Torch is the only authoritative CUDA selector at
        # this point. GSplatTorchCuda.cmake later rejects a compiler whose CUDA
        # major differs from Torch, which makes this the compiled CUDA major for
        # every successful wheel. Runtime validation still catches environments
        # where either package is replaced after installation.
        cupy_requirement = cupy_requirement_for_build_torch()
        return {
            "dependencies": [
                f"torch=={public_version}",
                f'{cupy_requirement}; extra == "png"',
            ]
        }

    @staticmethod
    def dynamic_wheel(settings: Mapping[str, Any]) -> dict[str, bool]:
        """Declare that wheel dependencies may extend sdist dependencies."""

        WheelBuildMetadataProvider._check_settings(settings)
        return {"dependencies": True}


def _main(argv: list[str]) -> int:
    """Implement the small command-line interface used by build tooling."""

    try:
        if len(argv) == 3 and argv[1] == "--cupy-requirement":
            requirement = cupy_requirement_for_cuda(argv[2])
        elif len(argv) == 2 and argv[1] == "--cupy-requirement-for-build-torch":
            requirement = cupy_requirement_for_build_torch()
        else:
            raise SystemExit(
                f"usage: {argv[0]} "
                "(--cupy-requirement <CUDA-major.minor[.patch]> | "
                "--cupy-requirement-for-build-torch)"
            )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(requirement)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
