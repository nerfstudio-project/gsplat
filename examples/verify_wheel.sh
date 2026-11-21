#!/bin/bash
#SBATCH --job-name=verify-wheel
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#
# Release check: import gsplat from site-packages only (no source tree on
# sys.path, no nvcc needed) and execute one kernel.
#
#   mkdir -p logs
#   ENV=/path/to/venv-with-wheel \
#   sbatch --account=<acct> --partition=<gpu-part> [--qos=<qos>] examples/verify_wheel.sh
#
# Overrides: ENV=<venv containing the installed wheel>

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV="${ENV:-$REPO/envs/gs}"

# Run outside the source tree so the import must come from site-packages.
cd /

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

source "$ENV/bin/activate"

python - <<'PY'
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "from", gsplat.__file__)
print("ext loaded:", _C is not None)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
s = torch.ones(1, 3, device="cuda")
covar, preci = gsplat.quat_scale_to_covar_preci(q, s)
print("quat_scale_to_covar_preci ->", tuple(covar.shape), covar.dtype)
print("covar:", covar.flatten().tolist())
PY
