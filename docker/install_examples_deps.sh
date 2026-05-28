#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# ============================================================================
# Install gsplat example-script dependencies, backend-aware.
#
# Called from the Dockerfile after `pip install -e .`. Does the right thing
# based on $GSPLAT_BACKEND ({cuda, rocm_radeon, rocm_instinct}).
#
#   CUDA: full upstream examples/requirements.txt.
#   ROCm: ROCm-safe subset + fused-ssim + patched fused-bilagrid + (Instinct)
#         a Common.h patch for PyTorch >= 2.13's removal of c10::hip alias.
#
# Mirrors the per-variant setup_and_build.sh recipes from the original
# Radeon and Instinct translations.
#
# Why a script and not inline RUN: BuildKit RUN heredocs (`RUN <<EOF`) are
# silently no-op'd by the classic Docker builder, which made the bilagrid
# patch path get skipped on classic builds. A regular shell script works
# under both builders.
# ============================================================================
set -euo pipefail

GSPLAT_REPO_ROOT="${GSPLAT_REPO_ROOT:-/opt/gsplat}"
cd "$GSPLAT_REPO_ROOT"

case "${GSPLAT_BACKEND:?GSPLAT_BACKEND must be set}" in
  cuda)
    echo "[gsplat-docker] Installing CUDA examples deps (ppisp excluded — optional, brittle nvcc build)..."
    # --no-build-isolation: several deps (fused-ssim, fused-bilagrid, ppisp)
    # have setup.py files that import torch — pip's build-isolation venv
    # doesn't have it, causing ModuleNotFoundError. Building against the
    # parent env's torch (already in the base image) is the right fix.
    #
    # ppisp is filtered out: NVIDIA's ISP pipeline is optional, its nvcc
    # build is brittle (requires specific CUDA dev paths). simple_trainer
    # default doesn't need it. Install separately afterwards (non-fatal).
    grep -vE 'nv-tlabs/ppisp' examples/requirements.txt > /tmp/req-cuda.txt
    pip install --no-cache-dir --no-build-isolation -r /tmp/req-cuda.txt
    rm -f /tmp/req-cuda.txt

    echo "[gsplat-docker] Attempting optional ppisp install (non-fatal)..."
    pip install --no-cache-dir --no-build-isolation \
        "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0" \
        || echo "[gsplat-docker] ppisp install failed — skipping (it's optional)"
    ;;

  rocm_radeon|rocm_instinct)
    # Note: `at::cuda::getCurrentCUDAStream` and `at::cuda::OptionalCUDAGuard`
    # are missing on PyTorch 2.8 ROCm release (PyTorch 2.13 added them back).
    # Handled at compile time by the force-included compat shim
    # `arch/rocm/gsplat/cuda/include/torch_rocm_compat.h` (wired in
    # `_rocm_flags()` in gsplat/cuda/build.py via `-include`). No source-file
    # sed patches needed.

    # Patch PyTorch 2.8's cpp_extension.py: hipify can return hipified_path=None
    # for files that don't need hipifying (already .hip or pure HIP source).
    # PyTorch 2.8 adds that None to the sources set, then crashes downstream
    # with `TypeError: expected str, bytes or os.PathLike object, not NoneType`.
    # PyTorch 2.13 fixed this by falling back to the original source path when
    # hipified_path is None — port that one-line fix here. No-op on 2.13+.
    CPP_EXT="$(python -c 'import torch.utils.cpp_extension as e; print(e.__file__)')"
    if [ -f "$CPP_EXT" ] && grep -q 'hipified_sources.add(hipify_result\[s_abs\].hipified_path if s_abs in hipify_result else s_abs)' "$CPP_EXT"; then
        echo "[gsplat-docker] Patching $CPP_EXT for PyTorch 2.8 hipify None-path bug..."
        sed -i 's|hipified_sources.add(hipify_result\[s_abs\].hipified_path if s_abs in hipify_result else s_abs)|hipified_sources.add(hipify_result[s_abs].hipified_path if (s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None) else s_abs)|' "$CPP_EXT"
        echo "[gsplat-docker] Patch applied."
    else
        echo "[gsplat-docker] PyTorch hipify None-path bug not present (already fixed or different version) — skipping patch."
    fi

    echo "[gsplat-docker] Installing ROCm-safe subset of examples deps..."
    pip install --no-cache-dir \
        viser "imageio[ffmpeg]" "numpy<2.0.0" scipy scikit-learn tqdm \
        "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow piexif \
        tensorboard tensorly pyyaml matplotlib splines plyfile

    pip install --no-cache-dir \
        "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"

    echo "[gsplat-docker] Installing fused-ssim (upstream supports ROCm)..."
    pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/rahul-goel/fused-ssim.git"

    echo "[gsplat-docker] Building patched fused-bilagrid for ROCm..."
    if [ "$GSPLAT_BACKEND" = "rocm_radeon" ]; then
        BILA_SHIM="$GSPLAT_REPO_ROOT/arch/radeon/gsplat/cuda/include/hip_shim"
    else
        BILA_SHIM="$GSPLAT_REPO_ROOT/arch/instinct/gsplat/cuda/include/hip_shim"
    fi

    rm -rf /tmp/fused-bilagrid
    git clone --quiet https://github.com/harry7557558/fused-bilagrid.git /tmp/fused-bilagrid
    cd /tmp/fused-bilagrid

    # Patch (a): wave64 demands uint64_t shuffle masks (ROCm 7.x __shfl_sync).
    for f in fused_bilagrid/uniform_sample_backward_v1.cu \
             fused_bilagrid/uniform_sample_backward_v2.cu; do
        [ -f "$f" ] || continue
        sed -i -E \
          -e 's/const unsigned mask =/const uint64_t mask =/g' \
          -e 's/unsigned mask = __activemask/uint64_t mask = __activemask/g' \
          -e 's/~0u\b/~0ull/g' \
          "$f"
    done

    # Patch (b): HIP-aware setup.py (drops --use_fast_math, adds shim include).
    cat > setup.py <<PY
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
IS_ROCM = torch.version.hip is not None
SHIM = "$BILA_SHIM"
nvcc_args = ["-O3", "-ffast-math", f"-I{SHIM}"] if IS_ROCM else ["-O3", "--use_fast_math"]
cxx_args = ["-O3"] + ([f"-I{SHIM}"] if IS_ROCM else [])
setup(
    name="fused_bilagrid", packages=['fused_bilagrid'],
    ext_modules=[CUDAExtension(name="fused_bilagrid_cuda",
        sources=["fused_bilagrid/sample_forward.cu","fused_bilagrid/sample_backward.cu",
                 "fused_bilagrid/uniform_sample.cu","fused_bilagrid/tv_loss_forward.cu",
                 "fused_bilagrid/tv_loss_backward.cu","fused_bilagrid/ext.cpp"],
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args})],
    cmdclass={'build_ext': BuildExtension})
PY

    pip install --no-cache-dir --no-build-isolation .
    cd "$GSPLAT_REPO_ROOT"
    rm -rf /tmp/fused-bilagrid

    # Instinct-only: PyTorch >= 2.13 removed the c10::hip::HIPCachingAllocator
    # alias (only c10::cuda::CUDACachingAllocator survives as the masquerade).
    if [ "$GSPLAT_BACKEND" = "rocm_instinct" ]; then
        COMMON_H="$GSPLAT_REPO_ROOT/arch/instinct/gsplat/cuda/include/Common.h"
        if [ -f "$COMMON_H" ] && grep -q "c10::hip::HIPCachingAllocator" "$COMMON_H"; then
            sed -i "s|c10::hip::HIPCachingAllocator|c10::cuda::CUDACachingAllocator|g" "$COMMON_H"
            echo "[gsplat-docker] Patched Common.h: c10::hip::HIPCachingAllocator -> c10::cuda::CUDACachingAllocator"
        fi
    fi
    ;;

  *)
    echo "[gsplat-docker] Unknown GSPLAT_BACKEND='$GSPLAT_BACKEND'" >&2
    echo "  Valid: cuda | rocm_radeon | rocm_instinct" >&2
    exit 1
    ;;
esac

echo "[gsplat-docker] examples deps install complete (backend=$GSPLAT_BACKEND)"
