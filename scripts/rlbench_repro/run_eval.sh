#!/bin/bash
# Usage: bash scripts/rlbench_repro/run_eval.sh <run_id> [gpu_id]
# Runs the released BridgeVLA RLBench checkpoint on all 18 tasks.
# Default protocol: 25 episodes per task, max 25 decisions per episode.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${1:?usage: run_eval.sh <run_id> [gpu_id]}"
GPU_ID="${2:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$REPO_ROOT/env.sh"
set -euo pipefail

cd "$REPO_ROOT/finetune/RLBench"

MODEL_FOLDER="$REPO_ROOT/data/bridgevla_ckpt/bridgevla/rlbench"
EVAL_DATAFOLDER="$REPO_ROOT/data/RLBench_TRAIN_DATA"
LOG_NAME="rlbench_repro/run_${RUN_ID}"

export TF_CPP_MIN_LOG_LEVEL=3
export BITSANDBYTES_NOWELCOME=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

echo "=== BridgeVLA RLBench eval run ${RUN_ID} on GPU ${GPU_ID} ==="
python eval.py \
  --model-folder "$MODEL_FOLDER" \
  --eval-datafolder "$EVAL_DATAFOLDER" \
  --tasks all \
  --eval-episodes "$EVAL_EPISODES" \
  --episode-length "$EPISODE_LENGTH" \
  --log-name "$LOG_NAME" \
  --device "$GPU_ID" \
  --headless \
  --model-name model_80.pth
