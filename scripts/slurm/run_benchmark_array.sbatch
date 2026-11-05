#!/bin/bash
# Run one EgoStream config as a Slurm job array: one GPU and one process per rank.
#
#   sbatch --array=0-19 scripts/slurm/run_benchmark_array.sbatch
#
# Environment variables (override on the sbatch command line or with --export):
#   PROJECT_DIR   repository root (default: directory of this script/../..)
#   VENV_DIR      virtualenv to activate (default: $PROJECT_DIR/.venv)
#   CONFIG_PATH   run config (default: configs/qwen3vl_8b/t4_r11_prune_intra_cos.json)
#   RUNNER        run_benchmark_qwen3vl.py | run_benchmark_internvl35.py | run_benchmark_molmo2.py
#   HF_HOME       Hugging Face cache (models must be cached when HF_HUB_OFFLINE=1)
#
#SBATCH --job-name=egostream
#SBATCH --output=logs/egostream_%A_%a.out
#SBATCH --error=logs/egostream_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-19

set -eEo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen3vl_8b/t4_r11_prune_intra_cos.json}"
RUNNER="${RUNNER:-run_benchmark_qwen3vl.py}"

# Site-specific modules go here, e.g.:
# module purge && module load gcc/12.2.0 cuda/12.2 python/3.11.7

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_DIR"
mkdir -p logs

RANK="${SLURM_ARRAY_TASK_ID:-0}"
WORLD_SIZE="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "rank $RANK / world size $WORLD_SIZE | config $CONFIG_PATH | runner $RUNNER"
python "$RUNNER" --config "$CONFIG_PATH" --rank "$RANK" --world_size "$WORLD_SIZE"
