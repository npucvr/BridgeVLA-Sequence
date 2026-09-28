#!/usr/bin/env bash
# Copy packed RLBench task archives from /share into shared /remote_databuffer.
# Archive transfer avoids the NFS small-file metadata bottleneck of expanded trees.
set -euo pipefail

source_root=${1:?usage: $0 SOURCE_ARCHIVE_ROOT DESTINATION_ARCHIVE_ROOT [TRAIN|EVAL|ALL]}
destination_root=${2:?usage: $0 SOURCE_ARCHIVE_ROOT DESTINATION_ARCHIVE_ROOT [TRAIN|EVAL|ALL]}
mode=${3:-ALL}
copy_jobs=${BRIDGEVLA_ARCHIVE_JOBS:-6}

[[ "$copy_jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "BRIDGEVLA_ARCHIVE_JOBS must be a positive integer, got: $copy_jobs" >&2
  exit 2
}

case "$mode" in
  TRAIN) splits=(RLBench_TRAIN_DATA) ;;
  EVAL) splits=(RLBench_EVAL_DATA) ;;
  ALL) splits=(RLBench_TRAIN_DATA RLBench_EVAL_DATA) ;;
  *)
    echo "mode must be TRAIN, EVAL, or ALL, got: $mode" >&2
    exit 2
    ;;
esac

[[ -d "$source_root" ]] || {
  echo "source archive root does not exist: $source_root" >&2
  exit 1
}
mkdir -p "$destination_root"

for split in "${splits[@]}"; do
  source_item="$source_root/$split"
  destination_item="$destination_root/$split"
  [[ -d "$source_item" ]] || {
    echo "missing source split: $source_item" >&2
    exit 1
  }
  mkdir -p "$destination_item"

  mapfile -t archives < <(
    find "$source_item" -mindepth 1 -maxdepth 1 -type f \( -name '*.tar.xz' -o -name '*.tar.gz' -o -name '*.tgz' \) -printf '%f\n' | sort
  )
  if ((${#archives[@]} == 0)); then
    echo "no archives under $source_item" >&2
    exit 1
  fi

  echo "[archives] $split jobs=$copy_jobs count=${#archives[@]}"
  printf '%s\0' "${archives[@]}" | xargs -0 -r -n 1 -P "$copy_jobs" \
    bash -c '
      set -euo pipefail
      source_item=$1
      destination_item=$2
      name=$3
      src="$source_item/$name"
      dst="$destination_item/$name"
      src_bytes=$(stat -c %s "$src")
      if [[ -f "$dst" ]]; then
        dst_bytes=$(stat -c %s "$dst")
        if [[ "$src_bytes" == "$dst_bytes" ]]; then
          echo "[skip] $name size=$src_bytes"
          exit 0
        fi
        echo "[resume] $name src=$src_bytes dst=$dst_bytes"
      else
        echo "[copy] $name size=$src_bytes"
      fi
      # Large sequential copies; --partial keeps interrupted transfers resumable.
      rsync -a --partial --info=stats2 "$src" "$dst.partial"
      dst_bytes=$(stat -c %s "$dst.partial")
      if [[ "$src_bytes" != "$dst_bytes" ]]; then
        echo "[fail] $name size mismatch src=$src_bytes dst=$dst_bytes" >&2
        exit 1
      fi
      mv -f "$dst.partial" "$dst"
      echo "[done] $name size=$dst_bytes"
    ' _ "$source_item" "$destination_item"
done

echo "[archives] completed: $destination_root"
