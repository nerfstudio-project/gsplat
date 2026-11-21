#!/bin/bash
#SBATCH --job-name=gsplat-build
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#
# Same as examples/build_gsplat.sh, but for an out-of-tree source copy and a
# separate environment - useful when validating a new torch/CUDA stack without
# touching the working checkout.
#
#   mkdir -p logs
#   sbatch --account=<acct> --partition=<gpu-part> [--qos=<qos>] examples/build_gs214.sh
#
# Overrides: SRC=<source copy>  ENV=<venv>  TORCH_CUDA_ARCH_LIST=<arch list>

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SRC="${SRC:-$REPO}"
ENV="${ENV:-$REPO/envs/gs214}"
cd "$SRC"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

source "$ENV/bin/activate"

echo "== host $(hostname) start $(date +%T)"
echo "== src  $SRC"
echo "== env  $ENV"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "== nvcc $(which nvcc 2>/dev/null || echo '<not found>')"
nvcc --version 2>/dev/null | tail -2 || true

echo "== build (in-place, resumable) =="
T0=$(date +%s)
python setup.py build_ext --inplace
echo "build took $(( $(date +%s) - T0 ))s"

echo "== editable link =="
python setup.py develop --no-deps

echo "== import + CUDA op check =="
python - <<"PY"
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "ext loaded:", _C is not None)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
s = torch.ones(1, 3, device="cuda")
print("op ->", tuple(gsplat.quat_scale_to_covar_preci(q, s)[0].shape))
PY

echo "== tests =="
python -m pytest tests/test_basic.py -x -q -k "quat_scale_to_covar_preci or test_projection or test_isect or rasterize_to_pixels" 2>&1 | tail -12

echo "== done $(date +%T) =="
