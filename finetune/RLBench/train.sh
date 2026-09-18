#!/usr/bin/env bash
# RLBench fine-tuning launcher.
#
# Single-node, two-GPU example:
#   CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 bash train.sh ...
#
# Multi-node runs can additionally set NNODES, NODE_RANK, MASTER_ADDR and
# MASTER_PORT.  The actual runtime/bootstrap is shared with other entrypoints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29567}"

if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer, got '$GPUS_PER_NODE'" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#visible_gpus[@]}" -ne "$GPUS_PER_NODE" ]]; then
    echo "GPUS_PER_NODE=$GPUS_PER_NODE does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
fi

export BRIDGEVLA_EXPECTED_HOSTNAME="${BRIDGEVLA_EXPECTED_HOSTNAME:-}"
source "$REPO_ROOT/scripts/bridgevla_runtime.sh"
cd "$SCRIPT_DIR"

echo "[bridgevla-train] host=$(hostname -s 2>/dev/null || hostname) pwd=$(pwd)"
echo "[bridgevla-train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} gpus_per_node=$GPUS_PER_NODE nnodes=$NNODES node_rank=$NODE_RANK"

if [[ "$NNODES" == "1" ]]; then
  exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    train.py "$@"
fi

exec torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py "$@"
