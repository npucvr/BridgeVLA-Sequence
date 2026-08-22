#!/bin/bash
# Extract the 25 held-out evaluation episodes (0-24) for each RLBench task.
# The published EVAL_DATA files are gzip-compressed despite their .tar.xz
# suffix. Do not point this at RLBench_TRAIN_DATA: that archive contains the
# 100 demonstrations used for fine-tuning, not the paper's held-out set.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/RLBench_EVAL_DATA}"
LOG_FILE="${LOG_FILE:-/tmp/bridgevla_extract_eval_data.log}"
PARALLEL="${PARALLEL:-4}"

TASKS=(
  close_jar
  reach_and_drag
  insert_onto_square_peg
  meat_off_grill
  open_drawer
  place_cups
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  light_bulb_in
  slide_block_to_color_target
  place_shape_in_shape_sorter
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
)

have_episodes_0_24() {
  local task="$1" ep
  for ep in $(seq 0 24); do
    [ -d "$DATA_DIR/$task/all_variations/episodes/episode$ep" ] || return 1
  done
  return 0
}

extract_one() {
  local task="$1"
  local tar_file="$DATA_DIR/${task}.tar.xz"

  if have_episodes_0_24 "$task"; then
    echo "[skip] $task already has episodes 0-24"
    return 0
  fi

  echo "[start] $task $(date +%H:%M:%S)"
  # Most EVAL_DATA archives preserve the generator's absolute-path prefix:
  # mnt/hdfs/lpy/RLBench/peract_dataset/peract_18tasks/all_variations_128/
  # eval_upload/<task>/.... One released archive has a ./<task>/ prefix, so
  # detect both layouts and strip only the prefix components.
  archive_prefix="mnt/hdfs/lpy/RLBench/peract_dataset/peract_18tasks/all_variations_128/eval_upload"
  strip_components=8
  first_entry="$(tar -tzf "$tar_file" | sed -n '1p')"
  if [[ "$first_entry" == "./${task}/"* ]]; then
    archive_prefix="."
    strip_components=1
  elif [[ "$first_entry" != "${archive_prefix}/${task}/"* ]]; then
    echo "[FAIL] $task unsupported archive prefix: $first_entry"
    return 1
  fi
  if nice -n 19 ionice -c3 tar -xzf "$tar_file" -C "$DATA_DIR" --strip-components="$strip_components" --wildcards \
      "${archive_prefix}/${task}/all_variations/episodes/episode[0-9]/*" \
      "${archive_prefix}/${task}/all_variations/episodes/episode1[0-9]/*" \
      "${archive_prefix}/${task}/all_variations/episodes/episode2[0-4]/*"; then
    if have_episodes_0_24 "$task"; then
      echo "[done]  $task $(date +%H:%M:%S)"
    else
      echo "[WARN]  $task extraction finished but episodes 0-24 incomplete"
    fi
  else
    echo "[FAIL]  $task $(date +%H:%M:%S)"
  fi
}

for task in "${TASKS[@]}"; do
  extract_one "$task" >> "$LOG_FILE" 2>&1 &
  while [ "$(jobs -pr | wc -l)" -ge "$PARALLEL" ]; do
    wait -n 2>/dev/null || true
  done
done
wait
echo "[ALL-DONE] $(date +%H:%M:%S)" >> "$LOG_FILE"
