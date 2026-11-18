#!/usr/bin/env bash
# Not part of upstream — reconstructed equivalent of the "path.sh" referenced
# in BUILD_AND_TEST.md (never actually committed to this branch).
#
# Exports ROCM_PATH/HIP_PATH and puts the pip-installed ROCm SDK's headers
# onto CPATH/CPLUS_INCLUDE_PATH so JIT/AOT builds find hip/hip_runtime.h etc.
#
# Usage: source ./path.sh   (must be *sourced*, not executed)

if ! command -v rocm-sdk >/dev/null 2>&1; then
    echo "path.sh: 'rocm-sdk' not found on PATH. Activate your venv with the" >&2
    echo "         rocm-sdk-core/-devel packages installed first." >&2
    return 1 2>/dev/null || exit 1
fi

ROCM_SDK_ROOT="$(rocm-sdk path --root)"
ROCM_SDK_BIN="$(rocm-sdk path --bin)"

export ROCM_HOME="${ROCM_SDK_ROOT}"
export ROCM_PATH="${ROCM_SDK_ROOT}"
export HIP_PATH="${ROCM_SDK_ROOT}"

export PATH="${ROCM_SDK_BIN}:${PATH}"
export CPATH="${ROCM_SDK_ROOT}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ROCM_SDK_ROOT}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ROCM_SDK_ROOT}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ROCM_SDK_ROOT}/lib:${LD_LIBRARY_PATH:-}"

echo "path.sh: ROCM_PATH=${ROCM_PATH}"
echo "path.sh: hipcc -> $(command -v hipcc || echo 'NOT FOUND')"
