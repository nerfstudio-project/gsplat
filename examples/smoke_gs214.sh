#!/bin/bash
#SBATCH --job-name=gsplat-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#
# Same smoke run as examples/smoke.sh, but from an out-of-tree source copy and a
# separate environment - intended for stack-comparison runs.
#
#   mkdir -p logs
#   DATA_DIR=/path/to/colmap_scene \
#   sbatch --account=<acct> --partition=<gpu-part> [--qos=<qos>] examples/smoke_gs214.sh
#
# Overrides: DATA_DIR (required)  SRC=<source copy>  ENV=<venv>  MAX_STEPS=<n>
#            RESULT_DIR=<dir>

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SRC="${SRC:-$REPO}"
ENV="${ENV:-$REPO/envs/gs214}"
: "${DATA_DIR:?set DATA_DIR to a COLMAP scene directory (e.g. .../mip360/garden)}"
MAX_STEPS="${MAX_STEPS:-2000}"
DATA_FACTOR="${DATA_FACTOR:-4}"
RESULT_DIR="${RESULT_DIR:-results/smoke}"
cd "$SRC"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.6 2>/dev/null || true
    module load gcc/12.2.0 2>/dev/null || true
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

source "$ENV/bin/activate"

echo "== host $(hostname) start $(date +%T)"
echo "== data $DATA_DIR"
python -c "import gsplat, torch; print('torch', torch.__version__, '| gsplat', gsplat.__version__, '|', torch.cuda.get_device_name(0))"

srun timeout -s TERM 600 python examples/simple_trainer.py default \
    --data_dir "$DATA_DIR" \
    --data_factor "$DATA_FACTOR" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$MAX_STEPS" \
    --disable_viewer \
    --result_dir "$RESULT_DIR"

echo "== stats =="
cat "$RESULT_DIR"/stats/val_step*.json 2>/dev/null || echo "(no eval stats found)"
echo "== done $(date +%T) =="
