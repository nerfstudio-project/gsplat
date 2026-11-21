#!/bin/bash
#SBATCH --job-name=gsplat-build
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#
# Editable install of gsplat (AOT CUDA compile) for the environment in $ENV.
#
# Submit with your own allocation, from the repository root (create logs/ first):
#   mkdir -p logs
#   sbatch --account=<acct> --partition=<gpu-part> [--qos=<qos>] examples/build_gsplat.sh
#
# Overrides: ENV=<venv>  TORCH_CUDA_ARCH_LIST=<arch list>  MAX_JOBS=<n>

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV="${ENV:-$REPO/envs/gs}"
cd "$REPO"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

source "$ENV/bin/activate"

echo "== host $(hostname) start $(date +%T)"
echo "== repo $REPO"
echo "== env  $ENV"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "== nvcc $(which nvcc 2>/dev/null || echo '<not found>')"
nvcc --version 2>/dev/null | tail -2 || true

# Build in place so object files persist in build/ and a resubmission resumes
# (pip/uv build in a throw-away temp dir, which would restart the compile).
echo "== build CUDA extensions (in-place, resumable) =="
T0=$(date +%s)
python setup.py build_ext --inplace
echo "build took $(( $(date +%s) - T0 ))s"

echo "== editable install (metadata/link only, reuses built objects) =="
python setup.py develop --no-deps

echo "== built extensions =="
find "$REPO/gsplat" -name "*.so" | sort

echo "== import + CUDA op check =="
python - <<"EOF"
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "ext loaded:", _C is not None)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
s = torch.ones(1, 3, device="cuda")
covar, preci = gsplat.quat_scale_to_covar_preci(q, s)
print("quat_scale_to_covar_preci ->", tuple(covar.shape), covar.dtype)
EOF

echo "== tests (test_basic.py) =="
python -m pytest tests/test_basic.py -x -q 2>&1 | tail -12

echo "== done $(date +%T) =="
