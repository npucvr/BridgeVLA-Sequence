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

set -euo pipefail
: "${BRIDGEVLA_START_XVFB:=1}"
export BRIDGEVLA_START_XVFB
source "$REPO_ROOT/scripts/bridgevla_runtime.sh"

cd "$REPO_ROOT/finetune/RLBench"
echo "[runner] host=$(hostname -s 2>/dev/null || hostname) pwd=$(pwd) gpu=${GPU_ID}"

MODEL_FOLDER="${MODEL_FOLDER:-$REPO_ROOT/data/bridgevla_ckpt/bridgevla/rlbench}"
# The released TRAIN_DATA contains the 100 training demonstrations.  The
# paper evaluates on the separate held-out EVAL_DATA split (25 episodes).
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-$REPO_ROOT/data/RLBench_EVAL_DATA}"
RESULT_LOG_DIR="${RESULT_LOG_DIR:-rlbench_repro}"
LOG_NAME="${LOG_NAME:-$RESULT_LOG_DIR/run_${RUN_ID}}"

if [ ! -d "$EVAL_DATAFOLDER" ]; then
  echo "Evaluation data directory does not exist: $EVAL_DATAFOLDER" >&2
  echo "Download LPY/BridgeVLA_RLBench_EVAL_DATA and extract episodes 0-24." >&2
  exit 2
fi
if [[ "$EVAL_DATAFOLDER" == *"RLBench_TRAIN_DATA"* ]]; then
  echo "WARNING: evaluating on RLBench_TRAIN_DATA; use the held-out RLBench_EVAL_DATA split for paper comparison." >&2
fi

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
