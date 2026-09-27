#!/usr/bin/env bash
# Copy BridgeVLA raw RLBench data to a shared, resumable namespace.
# The command never deletes files from the destination.
set -euo pipefail

source_root=${1:?usage: $0 SOURCE_DATA_ROOT DESTINATION_ROOT [ITEM ...]}
destination_root=${2:?usage: $0 SOURCE_DATA_ROOT DESTINATION_ROOT [ITEM ...]}
shift 2
sync_jobs=${BRIDGEVLA_SYNC_JOBS:-1}

[[ "$sync_jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "BRIDGEVLA_SYNC_JOBS must be a positive integer, got: $sync_jobs" >&2
  exit 2
}

if (($# == 0)); then
  # Legacy replay and retired encoded_v1 are intentionally not shared-data
  # dependencies.  Future encoded_train_v2 data will be rebuilt from these
  # raw RLBench sources.
  items=(RLBench_TRAIN_DATA RLBench_EVAL_DATA)
else
  items=("$@")
fi

[[ -d "$source_root" ]] || {
  echo "source data root does not exist: $source_root" >&2
  exit 1
}
mkdir -p "$destination_root"

for item in "${items[@]}"; do
  case "$item" in
    RLBench_TRAIN_DATA|RLBench_EVAL_DATA) ;;
    *)
      echo "unsupported data item: $item" >&2
      exit 2
      ;;
  esac
  [[ -d "$source_root/$item" ]] || {
    echo "missing source item: $source_root/$item" >&2
    exit 1
  }
  source_item="$source_root/$item"
  destination_item="$destination_root/$item"
  mkdir -p "$destination_item"
  echo "[rsync] $item jobs=$sync_jobs"
  if ((sync_jobs == 1)); then
    rsync -a --partial --info=progress2 \
      "$source_item/" "$destination_item/"
    continue
  fi

  # The raw RLBench tree has independent task directories.  Parallelize only
  # this one-time source-to-NFS migration; training and evaluation still use
  # the finished NFS tree through a single episode-aware reader.
  rsync -a --partial --info=stats2 --exclude='*/' \
    "$source_item/" "$destination_item/"
  mapfile -t task_dirs < <(
    find "$source_item" -mindepth 1 -maxdepth 1 -type d ! -name '.cache' -printf '%f\n' | sort
  )
  if ((${#task_dirs[@]})); then
    printf '%s\0' "${task_dirs[@]}" | xargs -0 -r -n 1 -P "$sync_jobs" \
      bash -c '
        set -euo pipefail
        source_item=$1
        destination_item=$2
        task=$3
        echo "[rsync-task] $task"
        rsync -a --partial --info=stats2 "$source_item/$task/" "$destination_item/$task/"
      ' _ "$source_item" "$destination_item"
  fi
done

echo "[rsync] completed: $destination_root"
