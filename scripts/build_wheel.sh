#!/usr/bin/env bash
# Build an installable wheel from this gsplat fork.
#
#   bash scripts/build_wheel.sh                 # on a node that has nvcc
#   sbatch --account=<acct> --partition=<part> --gres=gpu:1 \
#          --cpus-per-task=32 scripts/build_wheel.sh
#
# Environment overrides:
#   TORCH_CUDA_ARCH_LIST   default: 8.0   (A100: 8.0, H100: 9.0, "8.0+PTX" for forward compat)
#   MAX_JOBS               default: nproc
#   LOAD_MODULES           default: 1     (cluster module stack; set 0 to skip)
set -euo pipefail

# sbatch spools the script elsewhere; prefer the submission directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

if [[ "${LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export UV_NO_BUILD_ISOLATION=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# --no-build-isolation compiles against the active interpreter, so the
# build must run inside an environment that has torch, setuptools and ninja.
ENV="${GS_WHEEL_ENV:-$REPO_ROOT/envs/gs}"
if [[ ! -f "$ENV/bin/activate" ]]; then
    echo "error: no virtualenv at $ENV (set GS_WHEEL_ENV)" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV/bin/activate"

# glm headers are needed to compile the CUDA sources.
if [[ ! -f gsplat/cuda/csrc/third_party/glm/glm/glm.hpp ]]; then
    echo "== initialising glm submodule =="
    git submodule update --init gsplat/cuda/csrc/third_party/glm
fi

echo "== environment =="
python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    from torch.utils.cpp_extension import CUDA_HOME
    print("CUDA_HOME:", CUDA_HOME)
except Exception as exc:  # pragma: no cover - diagnostics only
    print("CUDA_HOME: <unavailable>", exc)
PY
echo "arch list: ${TORCH_CUDA_ARCH_LIST}"

echo "== building wheel =="
T0=$(date +%s)
uv build --wheel --no-build-isolation --out-dir dist
echo "build took $(( $(date +%s) - T0 ))s"

echo "== artifacts =="
ls -la dist/*.whl
