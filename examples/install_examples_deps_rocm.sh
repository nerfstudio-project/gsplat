#!/usr/bin/env bash
# ROCm-specific install helper for examples/requirements.txt.
#
# examples/requirements.txt is upstream's file and is left unmodified so it
# still works unchanged for CUDA users. On ROCm it cannot be installed
# as-is via a plain `pip install -r examples/requirements.txt` because:
#
#   1. `torch==2.9.1` / `torchvision==0.24.1` are pinned to exact versions.
#      A plain `pip install -r` would force-reinstall these from PyPI,
#      silently replacing the matched ROCm PyTorch build (built/shipped in
#      the ROCm container image) with an incompatible stock CUDA/CPU build.
#      -> Skip them; they must already be correctly installed.
#   2. `nvidia-ncore` is NVIDIA-only and unused by `simple_trainer.py`'s
#      default training path.
#   3. `fused-bilagrid` and `ppisp` are unused unless you pass
#      `--post_processing bilateral_grid` / `--post_processing ppisp`
#      (both default to unused, `post_processing` defaults to `None`).
#      `fused-bilagrid`'s CUDA kernels are not HIP-ported and will fail to
#      build on ROCm.
#   4. `fused-ssim` (git+https://github.com/rahul-goel/fused-ssim) DOES
#      support ROCm natively as of commit a7c48d6 -- its own setup.py
#      checks `torch.version.hip` and configures a HIP-compatible build
#      automatically. The reason a naive `pip install -r requirements.txt`
#      fails on ROCm is NOT a missing HIP port -- it's that pip's default
#      build isolation creates a fresh venv for the build step, which
#      installs a stock (non-ROCm) `torch` there, so `torch.version.hip`
#      reads as None during the build and the CUDA-only architecture-
#      detection path runs instead. `--no-build-isolation` (building
#      against the real, already-installed ROCm torch) fixes this
#      completely -- no source patch needed. Measured on Radeon RX 7900
#      XTX (gfx1100): fused_ssim's native kernel is ~12x faster per call
#      than gsplat's own pure-PyTorch `torch_ssim_loss` fallback (0.27ms
#      vs 3.3ms/iter for a representative garden-scene tile), and cuts a
#      real 800-step simple_trainer.py training run from ~22.2s to
#      ~16.3s wall time (see migration_report.md / optimizer_report.md for
#      full evidence). This should apply equally to Instinct (CDNA) since
#      the detection is generic ROCm/HIP, not architecture-specific --
#      not yet re-confirmed on Instinct as of this writing because that
#      host was unreachable when this was found; treat as expected-but-
#      unverified there until re-tested.
#
# Usage:
#   bash examples/install_examples_deps_rocm.sh
# Run from anywhere; paths are resolved relative to this script's location.

set -eu

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_SCRIPT_DIR"

echo "[install_examples_deps_rocm] Installing ROCm-safe subset of examples/requirements.txt..."
# NOTE: fused-ssim (git+https://github.com/rahul-goel/fused-ssim, listed in
# requirements.txt) MUST also be filtered out of this plain `pip install -r`
# pass, not just handled by the explicit --no-build-isolation install below.
# Without this exclusion, pip's default (isolated) build environment for this
# entry hits the exact "CUDA_HOME environment variable is not set" failure
# this script's own comment above (#4) documents as already root-caused and
# solved -- because the isolated build venv pulls a stock (non-ROCm) torch,
# so fused-ssim's setup.py sees torch.version.hip == None during that build
# and takes the CUDA-only architecture-detection branch. Since `set -eu` is
# active, that failure aborts the whole script before line 57's correct
# --no-build-isolation install is ever reached, silently reproducing this
# exact previously-fixed bug on any host where fused-ssim is present in
# requirements.txt (confirmed on real Instinct MI350X hardware).
grep -vE '^torch==|^torchvision==|^nvidia-ncore|fused-bilagrid|nv-tlabs/ppisp|rahul-goel/fused-ssim' requirements.txt \
    > /tmp/requirements_rocm_filtered.txt
python3 -m pip install -r /tmp/requirements_rocm_filtered.txt
rm -f /tmp/requirements_rocm_filtered.txt

echo "[install_examples_deps_rocm] Installing fused-ssim (native ROCm support, needs --no-build-isolation)..."
python3 -m pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

echo "[install_examples_deps_rocm] Done. Skipped (unused by simple_trainer.py's default path):"
echo "  - nvidia-ncore (NVIDIA-only)"
echo "  - fused-bilagrid, ppisp (post_processing defaults to None; fused-bilagrid also lacks a HIP port)"
echo "[install_examples_deps_rocm] torch/torchvision were left untouched (must already be the matched ROCm build)."
