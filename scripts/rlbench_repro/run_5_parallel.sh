#!/bin/bash
# Run the 5 RLBench eval repeats in parallel on GPU_IDS, then aggregate.
# With fewer than five IDs, repeats are scheduled in waves (e.g. GPU_IDS=0,2).
# DISPLAY_IDS can map those slots to isolated Xvfb displays (e.g. :1.0,:2.0).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/bridgevla_rlbench_parallel_logs}"
MODEL_FOLDER="${MODEL_FOLDER:-$REPO_ROOT/data/bridgevla_ckpt/bridgevla/rlbench}"
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-$REPO_ROOT/data/RLBench_EVAL_DATA}"
RESULT_LOG_DIR="${RESULT_LOG_DIR:-rlbench_repro}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
DISPLAY_IDS="${DISPLAY_IDS:-}"
export EVAL_DATAFOLDER MODEL_FOLDER RESULT_LOG_DIR GPU_IDS DISPLAY_IDS

cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/bridgevla_runtime.sh"
PYTHON="${PYTHON:-$(command -v python)}"
export PYTHON

mkdir -p "$LOG_DIR"
: > "$LOG_DIR/status.txt"
echo "[launcher] eval data: $EVAL_DATAFOLDER" | tee -a "$LOG_DIR/status.txt"

# Fail early if the caller accidentally points at the 100-demo training split.
if ! "$PYTHON" "$SCRIPT_DIR/verify_eval_data.py" \
    --data-folder "$EVAL_DATAFOLDER" --exact-episodes >> "$LOG_DIR/status.txt" 2>&1; then
  echo "[launcher] evaluation data verification failed" | tee -a "$LOG_DIR/status.txt"
  exit 2
fi

echo "[launcher] eval data verified" | tee -a "$LOG_DIR/status.txt"
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
if [ "${#gpu_ids[@]}" -eq 0 ]; then
  echo "[launcher] GPU_IDS is empty" | tee -a "$LOG_DIR/status.txt"
  exit 2
fi
echo "[launcher] GPU_IDS=$GPU_IDS (${#gpu_ids[@]} concurrent slots)" | tee -a "$LOG_DIR/status.txt"
display_ids=()
if [ -n "$DISPLAY_IDS" ]; then
  IFS=',' read -r -a display_ids <<< "$DISPLAY_IDS"
  echo "[launcher] DISPLAY_IDS=$DISPLAY_IDS" | tee -a "$LOG_DIR/status.txt"
fi

rc=0
wave_pids=()
wave_runs=()
wait_wave() {
  local i run_id pid wait_rc
  for i in "${!wave_pids[@]}"; do
    run_id="${wave_runs[$i]}"
    pid="${wave_pids[$i]}"
    if wait "$pid"; then
      echo "[launcher] run ${run_id} OK $(date)" >> "$LOG_DIR/status.txt"
    else
      wait_rc=$?
      echo "[launcher] run ${run_id} FAILED (exit ${wait_rc}) $(date)" >> "$LOG_DIR/status.txt"
      rc=1
    fi
  done
  wave_pids=()
  wave_runs=()
}

for run_id in 1 2 3 4 5; do
  slot=$(( (run_id - 1) % ${#gpu_ids[@]} ))
  gpu_id="${gpu_ids[$slot]}"
  display_id=""
  if [ "${#display_ids[@]}" -gt 0 ]; then
    display_slot=$(( (run_id - 1) % ${#display_ids[@]} ))
    display_id="${display_ids[$display_slot]}"
  fi
  echo "[launcher] starting run ${run_id} on GPU ${gpu_id}${display_id:+ display ${display_id}} $(date)" | tee -a "$LOG_DIR/status.txt"
  if [ -n "$display_id" ]; then
    DISPLAY="$display_id" bash "$SCRIPT_DIR/run_eval.sh" "$run_id" "$gpu_id" > "$LOG_DIR/run_${run_id}.log" 2>&1 &
  else
    bash "$SCRIPT_DIR/run_eval.sh" "$run_id" "$gpu_id" > "$LOG_DIR/run_${run_id}.log" 2>&1 &
  fi
  wave_pids+=("$!")
  wave_runs+=("$run_id")
  if [ "${#wave_pids[@]}" -ge "${#gpu_ids[@]}" ] || [ "$run_id" -eq 5 ]; then
    wait_wave
  fi
done

"$PYTHON" "$SCRIPT_DIR/aggregate_runs.py" \
  --model-folder "$MODEL_FOLDER" \
  --log-dir "$RESULT_LOG_DIR" \
  --runs 1,2,3,4,5 > "$LOG_DIR/aggregate.txt" 2>&1
cat "$LOG_DIR/aggregate.txt" >> "$LOG_DIR/status.txt"
"$PYTHON" "$SCRIPT_DIR/compare_to_paper.py" \
  --ours "$LOG_DIR/aggregate.txt" >> "$LOG_DIR/status.txt" 2>&1
echo "[launcher] all done $(date)" >> "$LOG_DIR/status.txt"
exit $rc
