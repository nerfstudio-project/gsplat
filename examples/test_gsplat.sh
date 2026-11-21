#!/bin/bash
#SBATCH --job-name=gsplat-test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#
# Core-kernel test subset (fast smoke of the native extension).
#
#   mkdir -p logs
#   sbatch --account=<acct> --partition=<gpu-part> [--qos=<qos>] examples/test_gsplat.sh
#
# Overrides: ENV=<venv>  TESTS=<pytest -k expression>

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV="${ENV:-$REPO/envs/gs}"
TESTS="${TESTS:-quat_scale_to_covar_preci or test_projection or test_isect or rasterize_to_pixels}"
cd "$REPO"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

source "$ENV/bin/activate"

echo "== host $(hostname) $(date +%T)"
python -c "import gsplat, torch; print('gsplat', gsplat.__version__, '| ext:', gsplat.cuda._backend._C is not None, '|', torch.cuda.get_device_name(0))"

python -m pytest tests/test_basic.py -x -q -k "$TESTS" 2>&1 | tail -25

echo "== done $(date +%T) =="
