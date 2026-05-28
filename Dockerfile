# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# syntax=docker/dockerfile:1.7
# ============================================================================
# gsplat tri-target Dockerfile
#
# ONE file, three GPU backends — pick at build time via GSPLAT_BACKEND:
#
#   docker build --build-arg GSPLAT_BACKEND=cuda          -t gsplat:cuda     .
#   docker build --build-arg GSPLAT_BACKEND=rocm_radeon   -t gsplat:radeon   .
#   docker build --build-arg GSPLAT_BACKEND=rocm_instinct -t gsplat:instinct .
#
# Optional overrides:
#   --build-arg PYTORCH_ROCM_ARCH='gfx1100'        (Radeon: default gfx1100;gfx1200)
#   --build-arg PYTORCH_ROCM_ARCH='gfx950'         (Instinct: default gfx942;gfx950)
#   --build-arg BASE_IMAGE_CUDA=...                (override the CUDA base image)
#   --build-arg BASE_IMAGE_ROCM=...                (override the ROCm base image)
#
# Run:
#   # CUDA (NVIDIA Container Toolkit required on host)
#   docker run --rm -it --gpus all --shm-size=8G gsplat:cuda
#
#   # ROCm Radeon / Instinct (KFD + DRI passthrough)
#   docker run --rm -it \
#       --device /dev/kfd --device /dev/dri \
#       --group-add video --group-add render \
#       --security-opt seccomp=unconfined \
#       --shm-size=8G \
#       gsplat:radeon
#
# First import on ROCm triggers a one-time JIT compile (~140s Radeon / ~167s
# Instinct). On CUDA the extension is built AOT during `pip install -e .`, so
# first import is instant. The active backend is printed at startup:
#   [gsplat.build] GSPLAT_BACKEND = rocm_radeon
# ============================================================================

ARG GSPLAT_BACKEND=cuda
ARG BASE_IMAGE_CUDA=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# Released ROCm + PyTorch 2.8.0 + Python 3.12 + Ubuntu 24.04. Frozen, ~30 GB
# (vs ~70 GB for rocm/pytorch-nightly:latest, which bundles benchmark git
# clones and tracks PyTorch main). Override with --build-arg if you need
# the bleeding edge.
ARG BASE_IMAGE_ROCM=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# ---- per-backend base image stages ---------------------------------------
FROM ${BASE_IMAGE_CUDA} AS base-cuda
FROM ${BASE_IMAGE_ROCM} AS base-rocm_radeon
FROM ${BASE_IMAGE_ROCM} AS base-rocm_instinct

# ---- final stage: the alias resolves to one of the three above -----------
FROM base-${GSPLAT_BACKEND} AS final

ARG GSPLAT_BACKEND
ENV GSPLAT_BACKEND=${GSPLAT_BACKEND}

# Optional user override of the per-backend default arch list
ARG PYTORCH_ROCM_ARCH=""
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

# CUDA arch list for the AOT build (only used when GSPLAT_BACKEND=cuda).
# Default covers Turing through Hopper (7.5=RTX20/T4/RTX6000, 8.0=A100,
# 8.6=RTX30/A40, 8.9=RTX40/L40, 9.0=H100). Override with --build-arg if you
# want a smaller fatbin for a single GPU model.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

WORKDIR /opt/gsplat

# System deps:
# - git, ca-certificates, build-essential, ninja-build: needed by
#   torch.utils.cpp_extension JIT and pip git+ installs.
# - libgl1, libglib2.0-0: required at runtime by opencv-python (used by
#   examples/datasets/colmap.py for image loading). Without these, you get
#   `ImportError: libGL.so.1: cannot open shared object file` on first
#   `import cv2`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy the merged tri-target repo. (.dockerignore excludes scratch dirs.)
COPY . /opt/gsplat/

# Optional Python deps used by the test suite. Some ROCm tests still skip
# (nerfacc and PLAS are CUDA-only); these install cleanly on all backends.
RUN pip install --no-cache-dir pytest-timeout pytorch_msssim

# Build/install gsplat. CUDA does the AOT extension build via setup.py.
# ROCm sets BUILD_NO_CUDA=1 so setup.py skips AOT — the extension compiles
# JIT on first import (avoids the merged-tree layout's hipify quirks in AOT).
RUN set -eux; \
    # Populate the upstream glm submodule. All three backends now consume the
    # same submodule (CUDA: -I directly; ROCm: mirrored to /tmp via
    # _isolate_glm_for_rocm), so init unconditionally.
    git submodule update --init --recursive || true ; \
    case "$GSPLAT_BACKEND" in \
      cuda) \
        ARCHS="${PYTORCH_ROCM_ARCH}" ; \
        BNC="" ; \
        ;; \
      rocm_radeon) \
        ARCHS="${PYTORCH_ROCM_ARCH:-gfx1100;gfx1200}" ; \
        BNC="1" ; \
        ;; \
      rocm_instinct) \
        ARCHS="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}" ; \
        BNC="1" ; \
        ;; \
      *) echo "[gsplat-docker] Unknown GSPLAT_BACKEND='$GSPLAT_BACKEND'" >&2 ; \
         echo "  Valid: cuda | rocm_radeon | rocm_instinct" >&2 ; \
         exit 1 ;; \
    esac ; \
    echo "[gsplat-docker] backend=$GSPLAT_BACKEND archs='$ARCHS' BUILD_NO_CUDA='$BNC'" ; \
    PYTORCH_ROCM_ARCH="$ARCHS" \
    BUILD_NO_CUDA="$BNC" \
        pip install -e . --no-build-isolation ; \
    # Persist the resolved arch list in the image env so later `pip install`
    # reruns inside the container pick up the same defaults.
    if [ -n "$ARCHS" ]; then \
        echo "export PYTORCH_ROCM_ARCH='$ARCHS'" > /etc/profile.d/gsplat-arch.sh ; \
    fi

# ============================================================================
# Example-script dependencies + fused-* C++ extensions (backend-aware).
#
# Logic lives in docker/install_examples_deps.sh (so it works in BOTH the
# classic Docker builder AND BuildKit — RUN heredocs are silently no-op'd
# by the classic builder, which would skip the bilagrid patches).
#
# CUDA: full upstream examples/requirements.txt.
# ROCm: ROCm-safe subset + fused-ssim + patched fused-bilagrid (uint64 masks
#       + HIP-aware setup.py with the per-variant cooperative_groups shim
#       on the include path) + (Instinct only) Common.h patch for PyTorch
#       >= 2.13's removal of the c10::hip::HIPCachingAllocator alias.
# Skipped on ROCm: nvidia-ncore, ppisp, nerfacc (NVIDIA-only, no ROCm port).
# Mirrors the per-variant setup_and_build.sh recipes from the original
# Radeon and Instinct translations.
# ============================================================================
RUN chmod +x /opt/gsplat/docker/install_examples_deps.sh \
 && GSPLAT_REPO_ROOT=/opt/gsplat /opt/gsplat/docker/install_examples_deps.sh

# Stable JIT cache location (mountable as a host volume to persist compiles).
ENV TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions

# Pre-warm the JIT extension on CUDA only. CUDA already AOT-built the .so in
# step #5, so this is a fast verification import.
#
# On ROCm we deliberately SKIP the pre-warm. `build.py` auto-detects the
# current GPU's arch and sets PYTORCH_ROCM_ARCH to just that one arch
# (~5-6x faster compile vs the 11-arch PyTorch default). Detection requires
# a visible GPU, which `docker build` does not expose — so pre-warming here
# would either fall back to the multi-arch fat list or guess wrong. Instead
# the JIT compile runs on first `docker run ... import gsplat`, when KFD/DRI
# are mounted and torch.cuda.get_device_properties() returns the real gfx.
# Result lands in $TORCH_EXTENSIONS_DIR and is cached for subsequent runs.
RUN if [ "$GSPLAT_BACKEND" = "cuda" ]; then \
        python -c "import gsplat; print('[gsplat-docker] gsplat ready:', gsplat.__file__)" ; \
    else \
        echo "[gsplat-docker] ROCm: JIT compile deferred to first runtime import (auto-detects GPU arch)." ; \
    fi

# Friendly default: drop into a shell with PYTHONPATH set so `python -c
# "import gsplat"` works from anywhere.
ENV PYTHONPATH=/opt/gsplat:${PYTHONPATH}
WORKDIR /opt/gsplat

CMD ["bash"]
