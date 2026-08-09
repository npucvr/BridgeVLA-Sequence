#!/usr/bin/env bash

# Intentionally do NOT use "set -e":
# if one worker fails, we still wait for the other workers and summarize
# whatever results were produced.
set -uo pipefail

###############################################################################
# Configuration
###############################################################################

EXPECTED_ENV="bridgevla_rlbench"

BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Sequence}"

MODEL_FOLDER="${MODEL_FOLDER:-/data2/local_userdata/lizhe/VLA/BridgeVLA/checkpoints/bridgevla/rlbench}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"

EVAL_DATA="${EVAL_DATA:-/data2/local_userdata/lizhe/VLA/BridgeVLA/datasets/rlbench/eval}"

START_EPISODE="${START_EPISODE:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"

# Limit CPU oversubscription when many GPU workers run together.
CPU_THREADS="${CPU_THREADS:-4}"

# Unique by default, so eval_results.csv from different runs never get mixed.
RUN_NAME="${RUN_NAME:-parallel18_$(date +%Y%m%d_%H%M%S)}"

RLbench_DIR="$BRIDGEVLA_ROOT/finetune/RLBench"
EVAL_PY="$RLbench_DIR/eval.py"
RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"

###############################################################################
# Official BridgeVLA 18 RLBench tasks
###############################################################################

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

###############################################################################
# Usage
###############################################################################

if (( $# == 0 )); then
    echo "Usage:"
    echo "  $0 GPU_ID [GPU_ID ...]"
    echo
    echo "Examples:"
    echo "  $0 0"
    echo "  $0 0 1 2 3"
    echo "  $0 1 3 5 7"
    echo "  $0 0,1,2,3"
    echo
    echo "Optional environment variables:"
    echo "  RUN_NAME=my_eval"
    echo "  EVAL_EPISODES=25"
    echo "  EPISODE_LENGTH=25"
    echo "  CPU_THREADS=4"
    exit 2
fi

###############################################################################
# Environment checks
###############################################################################

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "ERROR: expected conda environment '$EXPECTED_ENV'."
    echo "Current environment: ${CONDA_DEFAULT_ENV:-<none>}"
    echo
    echo "Run:"
    echo "  conda activate $EXPECTED_ENV"
    exit 2
fi

export BRIDGEVLA_ROOT
export USE_TF=0
export USE_TORCH=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

# Avoid huge core dump files if a simulator worker crashes.
ulimit -c 0 2>/dev/null || true

if [[ ! -f "$EVAL_PY" ]]; then
    echo "ERROR: eval.py not found:"
    echo "  $EVAL_PY"
    exit 2
fi

if [[ ! -f "$MODEL_FOLDER/$MODEL_NAME" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "  $MODEL_FOLDER/$MODEL_NAME"
    exit 2
fi

for cfg in exp_cfg.yaml mvt_cfg.yaml; do
    if [[ ! -f "$MODEL_FOLDER/$cfg" ]]; then
        echo "ERROR: checkpoint config not found:"
        echo "  $MODEL_FOLDER/$cfg"
        exit 2
    fi
done

if [[ ! -d "$EVAL_DATA" ]]; then
    echo "ERROR: evaluation dataset not found:"
    echo "  $EVAL_DATA"
    exit 2
fi

if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
    echo "ERROR: COPPELIASIM_ROOT is not set."
    exit 2
fi

if [[ ! -x "$COPPELIASIM_ROOT/coppeliaSim" ]]; then
    echo "ERROR: CoppeliaSim executable not found:"
    echo "  $COPPELIASIM_ROOT/coppeliaSim"
    exit 2
fi

###############################################################################
# Parse GPU list
###############################################################################

GPU_STRING="$*"
GPU_STRING="${GPU_STRING//,/ }"

read -r -a GPUS <<< "$GPU_STRING"

if (( ${#GPUS[@]} == 0 )); then
    echo "ERROR: no GPUs specified."
    exit 2
fi

declare -A SEEN_GPUS=()

GPU_COUNT="$(nvidia-smi -L | wc -l)"

for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid GPU id: $gpu"
        exit 2
    fi

    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
        echo "ERROR: duplicate GPU id: $gpu"
        exit 2
    fi
    SEEN_GPUS["$gpu"]=1

    if (( gpu >= GPU_COUNT )); then
        echo "ERROR: GPU $gpu does not exist. nvidia-smi reports $GPU_COUNT GPUs."
        exit 2
    fi
done

# More than 18 GPUs cannot help because there are only 18 tasks.
if (( ${#GPUS[@]} > ${#TASKS[@]} )); then
    echo "WARNING: ${#GPUS[@]} GPUs specified for ${#TASKS[@]} tasks."
    echo "Only the first ${#TASKS[@]} GPUs will be used."
    GPUS=("${GPUS[@]:0:${#TASKS[@]}}")
fi

NWORKERS="${#GPUS[@]}"
NTASKS="${#TASKS[@]}"

###############################################################################
# Validate evaluation dataset
###############################################################################

echo
echo "============================================================"
echo "Checking evaluation dataset"
echo "============================================================"

DATASET_ERROR=0

for task in "${TASKS[@]}"; do
    EP_DIR="$EVAL_DATA/$task/all_variations/episodes"

    if [[ ! -d "$EP_DIR" ]]; then
        echo "MISSING: $EP_DIR"
        DATASET_ERROR=1
        continue
    fi

    N_EP="$(
        find "$EP_DIR" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name 'episode[0-9]*' \
        | wc -l
    )"

    REQUIRED_EP=$(( START_EPISODE + EVAL_EPISODES ))

    printf "%-40s %2d episodes\n" "$task" "$N_EP"

    if (( N_EP < REQUIRED_EP )); then
        echo "ERROR: $task has only $N_EP episodes; need at least $REQUIRED_EP."
        DATASET_ERROR=1
    fi
done

if (( DATASET_ERROR != 0 )); then
    exit 2
fi

###############################################################################
# Check that BridgeVLA comes from Sequence and TensorFlow is not loaded
###############################################################################

echo
echo "============================================================"
echo "BridgeVLA / TensorFlow preflight"
echo "============================================================"

python - <<'PY'
import os
import bridgevla

root = os.path.realpath(os.environ["BRIDGEVLA_ROOT"])
pkg = os.path.realpath(bridgevla.__file__)

print("BRIDGEVLA_ROOT :", root)
print("bridgevla      :", pkg)

if not pkg.startswith(root + os.sep):
    raise SystemExit(
        "ERROR: bridgevla is not imported from BRIDGEVLA_ROOT."
    )

from bridgevla.utils.rvt_utils import TensorboardManager

with open("/proc/self/maps", "r") as f:
    maps = f.read()

if "libtensorflow" in maps:
    raise SystemExit(
        "ERROR: libtensorflow was loaded by rvt_utils. "
        "Do not start parallel evaluation until TensorFlow is removed "
        "from this RLBench environment."
    )

print("libtensorflow  : NOT loaded")
print("Preflight      : OK")
PY

if (( $? != 0 )); then
    exit 2
fi

###############################################################################
# Prepare run directory
###############################################################################

if [[ -e "$RUN_ROOT" ]]; then
    echo
    echo "ERROR: run directory already exists:"
    echo "  $RUN_ROOT"
    echo
    echo "Choose another RUN_NAME to avoid mixing/duplicating CSV rows."
    exit 2
fi

mkdir -p "$RUN_ROOT"

ASSIGNMENTS="$RUN_ROOT/assignments.tsv"

printf "worker\tgpu\ttasks\n" > "$ASSIGNMENTS"

###############################################################################
# Display assignment
###############################################################################

echo
echo "============================================================"
echo "BridgeVLA parallel RLBench evaluation"
echo "============================================================"
echo "Run name        : $RUN_NAME"
echo "Run root        : $RUN_ROOT"
echo "Checkpoint      : $MODEL_FOLDER/$MODEL_NAME"
echo "Eval data       : $EVAL_DATA"
echo "Episodes/task   : $EVAL_EPISODES"
echo "Episode length  : $EPISODE_LENGTH"
echo "Workers         : $NWORKERS"
echo "Physical GPUs   : ${GPUS[*]}"
echo

for ((i = 0; i < NWORKERS; i++)); do
    gpu="${GPUS[$i]}"
    worker="worker_${i}_gpu${gpu}"

    worker_tasks=()

    # Round-robin task assignment.
    for ((j = i; j < NTASKS; j += NWORKERS)); do
        worker_tasks+=("${TASKS[$j]}")
    done

    printf "%-18s GPU %-3s : %s\n" \
        "$worker" \
        "$gpu" \
        "${worker_tasks[*]}"

    printf "%s\t%s\t%s\n" \
        "$worker" \
        "$gpu" \
        "${worker_tasks[*]}" \
        >> "$ASSIGNMENTS"
done

###############################################################################
# Launch workers
###############################################################################

echo
echo "============================================================"
echo "Launching workers"
echo "============================================================"

PIDS=()
WORKER_NAMES=()

for ((i = 0; i < NWORKERS; i++)); do
    gpu="${GPUS[$i]}"
    worker="worker_${i}_gpu${gpu}"
    worker_dir="$RUN_ROOT/$worker"

    worker_tasks=()

    for ((j = i; j < NTASKS; j += NWORKERS)); do
        worker_tasks+=("${TASKS[$j]}")
    done

    mkdir -p "$worker_dir"

    (
        cd "$RLbench_DIR" || exit 1

        echo "Worker          : $worker"
        echo "Physical GPU    : $gpu"
        echo "CUDA device     : 0"
        echo "Tasks           : ${worker_tasks[*]}"
        echo "Start time      : $(date)"
        echo

        CUDA_VISIBLE_DEVICES="$gpu" \
        xvfb-run \
            --auto-servernum \
            --server-args='-screen 0 1024x768x24 -ac' \
            python -u eval.py \
                --model-folder "$MODEL_FOLDER" \
                --model-name "$MODEL_NAME" \
                --eval-datafolder "$EVAL_DATA" \
                --tasks "${worker_tasks[@]}" \
                --start-episode "$START_EPISODE" \
                --eval-episodes "$EVAL_EPISODES" \
                --episode-length "$EPISODE_LENGTH" \
                --log-name "$RUN_NAME/$worker" \
                --device 0 \
                --headless

        rc=$?

        echo
        echo "End time        : $(date)"
        echo "Exit code       : $rc"

        exit "$rc"

    ) > "$worker_dir/stdout.log" 2>&1 &

    pid=$!

    PIDS+=("$pid")
    WORKER_NAMES+=("$worker")

    echo "Started $worker on physical GPU $gpu, PID=$pid"
    echo "  log: $worker_dir/stdout.log"
done

###############################################################################
# Wait for all workers
###############################################################################

echo
echo "============================================================"
echo "All workers launched. Waiting..."
echo "============================================================"
echo

FAILED=0

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    worker="${WORKER_NAMES[$i]}"

    if wait "$pid"; then
        echo "[DONE] $worker"
    else
        rc=$?
        FAILED=1

        echo
        echo "[FAILED] $worker, exit code=$rc"
        echo "Last 40 log lines:"
        echo "------------------------------------------------------------"
        tail -n 40 "$RUN_ROOT/$worker/stdout.log" || true
        echo "------------------------------------------------------------"
        echo
    fi
done

###############################################################################
# Aggregate worker CSV files
###############################################################################

echo
echo "============================================================"
echo "Aggregating results"
echo "============================================================"

python - "$RUN_ROOT" "$MODEL_NAME" <<'PY'
import csv
import glob
import os
import statistics
import sys

run_root = sys.argv[1]
model_name = sys.argv[2]
model_stem = os.path.splitext(model_name)[0]

expected_tasks = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

pattern = os.path.join(
    run_root,
    "worker_*",
    model_stem,
    "eval_results.csv",
)

csv_paths = sorted(glob.glob(pattern))

results = {}
duplicates = []

for path in csv_paths:
    worker = os.path.basename(
        os.path.dirname(
            os.path.dirname(path)
        )
    )

    if "_gpu" in worker:
        gpu = worker.rsplit("_gpu", 1)[1]
    else:
        gpu = "unknown"

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task = (row.get("task") or "").strip()
            value = (row.get("success rate") or "").strip()

            if not task or not value:
                continue

            try:
                success_rate = float(value)
            except ValueError:
                print(
                    f"WARNING: cannot parse success rate "
                    f"for {task}: {value!r}"
                )
                continue

            if task in results:
                duplicates.append(task)
                continue

            length_text = (row.get("length") or "").strip()
            transitions_text = (
                row.get("total_transitions") or ""
            ).strip()

            results[task] = {
                "success_rate": success_rate,
                "length": length_text,
                "total_transitions": transitions_text,
                "gpu": gpu,
                "worker": worker,
                "source_csv": path,
            }

summary_csv = os.path.join(run_root, "summary.csv")
summary_txt = os.path.join(run_root, "summary.txt")

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task",
            "success_rate",
            "length",
            "total_transitions",
            "gpu",
            "worker",
        ],
    )
    writer.writeheader()

    for task in expected_tasks:
        if task not in results:
            continue

        r = results[task]

        writer.writerow(
            {
                "task": task,
                "success_rate": r["success_rate"],
                "length": r["length"],
                "total_transitions": r["total_transitions"],
                "gpu": r["gpu"],
                "worker": r["worker"],
            }
        )

lines = []

lines.append(
    f"{'Task':40s} {'Success rate':>12s} {'GPU':>5s}"
)
lines.append("-" * 62)

for task in expected_tasks:
    if task in results:
        r = results[task]
        lines.append(
            f"{task:40s} "
            f"{r['success_rate']:11.2f}% "
            f"{r['gpu']:>5s}"
        )
    else:
        lines.append(
            f"{task:40s} {'MISSING':>12s} {'-':>5s}"
        )

lines.append("-" * 62)

available_rates = [
    results[t]["success_rate"]
    for t in expected_tasks
    if t in results
]

if available_rates:
    mean_rate = statistics.mean(available_rates)

    if len(available_rates) == len(expected_tasks):
        lines.append(
            f"{'MEAN (18 tasks)':40s} "
            f"{mean_rate:11.2f}%"
        )
    else:
        lines.append(
            f"{'PARTIAL MEAN':40s} "
            f"{mean_rate:11.2f}%"
        )

missing = [
    t for t in expected_tasks
    if t not in results
]

if missing:
    lines.append("")
    lines.append("Missing tasks:")
    for task in missing:
        lines.append(f"  - {task}")

if duplicates:
    lines.append("")
    lines.append("Duplicate task results detected:")
    for task in sorted(set(duplicates)):
        lines.append(f"  - {task}")

text = "\n".join(lines)

print()
print(text)
print()
print("Summary CSV :", summary_csv)
print("Summary TXT :", summary_txt)

with open(summary_txt, "w") as f:
    f.write(text)
    f.write("\n")

if missing or duplicates:
    sys.exit(2)
PY

SUMMARY_RC=$?

###############################################################################
# Final status
###############################################################################

echo

if (( FAILED != 0 )); then
    echo "============================================================"
    echo "Evaluation finished with one or more FAILED workers."
    echo "Inspect:"
    echo "  $RUN_ROOT/worker_*/stdout.log"
    echo "============================================================"
    exit 1
fi

if (( SUMMARY_RC != 0 )); then
    echo "============================================================"
    echo "Workers exited, but the 18-task summary is incomplete."
    echo "Inspect:"
    echo "  $RUN_ROOT"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "SUCCESS: all 18 RLBench tasks finished."
echo
echo "Results:"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/summary.txt"
echo
echo "Worker logs:"
echo "  $RUN_ROOT/worker_*/stdout.log"
echo "============================================================"
