#!/bin/bash
# Wait until eval data is ready and a GPU + enough CPU RAM are available,
# then run the released model_80 checkpoint five times and aggregate.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="${LOG_FILE:-/tmp/bridgevla_rlbench_repro_5runs.log}"
PYTHON="${PYTHON:-/home/sunguodong/.conda/envs/bridgevla/bin/python}"
MIN_CPU_AVAIL_MB="${MIN_CPU_AVAIL_MB:-51200}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$REPO_ROOT/env.sh" >/dev/null 2>&1
EXTRACT_LOG="${EXTRACT_LOG:-/tmp/bridgevla_extract_eval_data.log}"
echo "[scheduler] start $(date)" >> "$LOG_FILE"

wait_for_extract() {
  while ! grep -q '\[ALL-DONE\]' "$EXTRACT_LOG" 2>/dev/null; do
    echo "[scheduler] extraction still running, waiting 120s $(date)" >> "$LOG_FILE"
    sleep 120
  done
  echo "[scheduler] extraction finished $(date)" >> "$LOG_FILE"
}

wait_for_data() {
  while true; do
    if "$PYTHON" "$SCRIPT_DIR/verify_eval_data.py" \
        --data-folder "$REPO_ROOT/data/RLBench_TRAIN_DATA" >> "$LOG_FILE" 2>&1; then
      echo "[scheduler] eval data verified $(date)" >> "$LOG_FILE"
      return 0
    fi
    echo "[scheduler] data not ready yet, waiting 120s $(date)" >> "$LOG_FILE"
    sleep 120
  done
}

wait_for_resources() {
  while true; do
    local avail_mb
    avail_mb=$(free -m | awk '/^Mem:/ {print $7}')
    local gpu="" g procs mem util
    for g in 0 1 2 3 4 5 6 7; do
      procs=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
      read -r mem util < <(nvidia-smi -i "$g" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
      if [ "$procs" -eq 0 ] && [ "${mem:-999999}" -lt 5000 ] && [ "${util:-999999}" -lt 20 ]; then
        gpu="$g"
        break
      fi
    done
    if [ "${avail_mb:-0}" -ge "$MIN_CPU_AVAIL_MB" ] && [ -n "$gpu" ]; then
      echo "[scheduler] resources ready: cpu_avail=${avail_mb}MB gpu=${gpu} $(date)" >> "$LOG_FILE"
      echo "$gpu"
      return 0
    fi
    echo "[scheduler] waiting resources cpu_avail=${avail_mb:-?}MB free_gpu=${gpu:-none} $(date)" >> "$LOG_FILE"
    sleep 300
  done
}

wait_for_extract
wait_for_data
GPU_ID="$(wait_for_resources)"

for run_id in 1 2 3 4 5; do
  echo "[scheduler] ===== run ${run_id}/5 on GPU ${GPU_ID} $(date) =====" >> "$LOG_FILE"
  if bash "$SCRIPT_DIR/run_eval.sh" "$run_id" "$GPU_ID" >> "$LOG_FILE" 2>&1; then
    echo "[scheduler] run ${run_id} finished OK $(date)" >> "$LOG_FILE"
  else
    echo "[scheduler] run ${run_id} finished with exit code $? $(date)" >> "$LOG_FILE"
  fi
done

echo "[scheduler] all runs done $(date)" >> "$LOG_FILE"
"$PYTHON" "$SCRIPT_DIR/aggregate_runs.py" \
  --model-folder "$REPO_ROOT/data/bridgevla_ckpt/bridgevla/rlbench" \
  --runs 1,2,3,4,5 > /tmp/bridgevla_rlbench_aggregate.txt 2>&1
cat /tmp/bridgevla_rlbench_aggregate.txt >> "$LOG_FILE"
"$PYTHON" "$SCRIPT_DIR/compare_to_paper.py" \
  --ours /tmp/bridgevla_rlbench_aggregate.txt >> "$LOG_FILE" 2>&1
echo "[scheduler] aggregation done $(date)" >> "$LOG_FILE"
