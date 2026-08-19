#!/bin/bash
# Run the 5 RLBench eval repeats in parallel on GPUs 0-4, then aggregate.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/bridgevla_rlbench_parallel_logs}"
MODEL_FOLDER="$REPO_ROOT/data/bridgevla_ckpt/bridgevla/rlbench"
PYTHON="${PYTHON:-/home/sunguodong/.conda/envs/bridgevla/bin/python}"

mkdir -p "$LOG_DIR"
: > "$LOG_DIR/status.txt"

pids=()
for run_id in 1 2 3 4 5; do
  gpu_id=$((run_id - 1))
  echo "[launcher] starting run ${run_id} on GPU ${gpu_id} $(date)" | tee -a "$LOG_DIR/status.txt"
  bash "$SCRIPT_DIR/run_eval.sh" "$run_id" "$gpu_id" > "$LOG_DIR/run_${run_id}.log" 2>&1 &
  pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
  run_id=$((i + 1))
  pid=${pids[$i]}
  if wait "$pid"; then
    echo "[launcher] run ${run_id} OK $(date)" >> "$LOG_DIR/status.txt"
  else
    echo "[launcher] run ${run_id} FAILED (exit $?) $(date)" >> "$LOG_DIR/status.txt"
    rc=1
  fi
done

"$PYTHON" "$SCRIPT_DIR/aggregate_runs.py" \
  --model-folder "$MODEL_FOLDER" \
  --runs 1,2,3,4,5 > "$LOG_DIR/aggregate.txt" 2>&1
cat "$LOG_DIR/aggregate.txt" >> "$LOG_DIR/status.txt"
"$PYTHON" "$SCRIPT_DIR/compare_to_paper.py" \
  --ours "$LOG_DIR/aggregate.txt" >> "$LOG_DIR/status.txt" 2>&1
echo "[launcher] all done $(date)" >> "$LOG_DIR/status.txt"
exit $rc
