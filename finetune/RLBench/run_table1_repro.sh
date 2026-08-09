#!/usr/bin/env bash

set -uo pipefail

###############################################################################
# BridgeVLA Table-1 RLBench reproduction
#
# Public-paper protocol:
#   18 tasks
#   25 trials / task
#   max 25 action steps / trial
#   5 complete evaluation repetitions
#
# Each repetition runs "--tasks all" in one Python/CoppeliaSim process.
###############################################################################

EXPECTED_ENV="bridgevla_rlbench"

BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Sequence}"

MODEL_FOLDER="${MODEL_FOLDER:-/data2/local_userdata/lizhe/VLA/BridgeVLA/checkpoints/bridgevla/rlbench}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"

EVAL_DATA="${EVAL_DATA:-/data2/local_userdata/lizhe/VLA/BridgeVLA/datasets/rlbench/eval}"

N_REPEATS=5
START_EPISODE=0
EVAL_EPISODES=25
EPISODE_LENGTH=25

CPU_THREADS="${CPU_THREADS:-4}"

RUN_NAME="${RUN_NAME:-table1_repro_$(date +%Y%m%d_%H%M%S)}"

RLBENCH_DIR="$BRIDGEVLA_ROOT/finetune/RLBench"
EVAL_PY="$RLBENCH_DIR/eval.py"

MODEL_STEM="${MODEL_NAME%.pth}"
RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"

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
    echo "  $0 0 1"
    echo "  $0 0 1 2 3 4"
    echo "  $0 1 3 5 7"
    echo "  $0 0,1,2,3,4"
    echo
    echo "Only 5 complete evaluation repetitions exist,"
    echo "so at most 5 GPUs are used concurrently."
    exit 2
fi

###############################################################################
# Environment
###############################################################################

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
    echo "ERROR: activate $EXPECTED_ENV first."
    echo "Current env: ${CONDA_DEFAULT_ENV:-<none>}"
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

ulimit -c 0 2>/dev/null || true

###############################################################################
# Basic checks
###############################################################################

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

for f in exp_cfg.yaml mvt_cfg.yaml; do
    if [[ ! -f "$MODEL_FOLDER/$f" ]]; then
        echo "ERROR: missing checkpoint config:"
        echo "  $MODEL_FOLDER/$f"
        exit 2
    fi
done

if [[ ! -d "$EVAL_DATA" ]]; then
    echo "ERROR: eval data not found:"
    echo "  $EVAL_DATA"
    exit 2
fi

if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
    echo "ERROR: COPPELIASIM_ROOT is not set."
    exit 2
fi

if [[ ! -x "$COPPELIASIM_ROOT/coppeliaSim" ]]; then
    echo "ERROR: CoppeliaSim not found:"
    echo "  $COPPELIASIM_ROOT/coppeliaSim"
    exit 2
fi

###############################################################################
# Parse GPUs
###############################################################################

GPU_STRING="$*"
GPU_STRING="${GPU_STRING//,/ }"

read -r -a GPUS <<< "$GPU_STRING"

declare -A SEEN=()

GPU_COUNT=$(nvidia-smi -L | wc -l)

for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid GPU id: $gpu"
        exit 2
    fi

    if (( gpu >= GPU_COUNT )); then
        echo "ERROR: GPU $gpu does not exist."
        exit 2
    fi

    if [[ -n "${SEEN[$gpu]:-}" ]]; then
        echo "ERROR: duplicate GPU id: $gpu"
        exit 2
    fi

    SEEN["$gpu"]=1
done

if (( ${#GPUS[@]} > N_REPEATS )); then
    echo "NOTE: ${#GPUS[@]} GPUs provided, but Table 1 uses only 5 repetitions."
    echo "Using the first 5 GPUs: ${GPUS[*]:0:5}"
    GPUS=("${GPUS[@]:0:5}")
fi

NWORKERS=${#GPUS[@]}

###############################################################################
# Dataset check: episodes 0-24 must exist for every task
###############################################################################

echo
echo "============================================================"
echo "Checking RLBench Table-1 evaluation data"
echo "============================================================"

DATA_ERROR=0

for task in "${TASKS[@]}"; do
    ep_root="$EVAL_DATA/$task/all_variations/episodes"

    if [[ ! -d "$ep_root" ]]; then
        echo "MISSING: $ep_root"
        DATA_ERROR=1
        continue
    fi

    missing=0

    for ((ep = 0; ep < EVAL_EPISODES; ep++)); do
        if [[ ! -d "$ep_root/episode$ep" ]]; then
            echo "MISSING: $ep_root/episode$ep"
            missing=1
            DATA_ERROR=1
        fi
    done

    if (( missing == 0 )); then
        printf "%-40s episodes 0-24 OK\n" "$task"
    fi
done

if (( DATA_ERROR != 0 )); then
    exit 2
fi

###############################################################################
# BridgeVLA/TensorFlow preflight
###############################################################################

echo
echo "============================================================"
echo "BridgeVLA preflight"
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
        "ERROR: bridgevla is not imported from BRIDGEVLA_ROOT"
    )

from bridgevla.utils.rvt_utils import TensorboardManager

with open("/proc/self/maps", "r") as f:
    maps = f.read()

if "libtensorflow" in maps:
    raise SystemExit(
        "ERROR: libtensorflow is loaded. "
        "The RLBench/CoppeliaSim OpenSSL conflict may return."
    )

print("libtensorflow  : NOT loaded")
print("Preflight      : OK")
PY

if (( $? != 0 )); then
    exit 2
fi

###############################################################################
# Create run directory
###############################################################################

if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: run directory already exists:"
    echo "  $RUN_ROOT"
    exit 2
fi

mkdir -p "$RUN_ROOT"

###############################################################################
# Save reproduction metadata
###############################################################################

{
    echo "BridgeVLA Table-1 reproduction"
    echo
    echo "date=$(date)"
    echo "hostname=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "bridgevla_root=$BRIDGEVLA_ROOT"
    echo "model=$MODEL_FOLDER/$MODEL_NAME"
    echo "eval_data=$EVAL_DATA"
    echo "n_repeats=$N_REPEATS"
    echo "tasks=18"
    echo "eval_episodes=$EVAL_EPISODES"
    echo "start_episode=$START_EPISODE"
    echo "episode_length=$EPISODE_LENGTH"
    echo "gpus=${GPUS[*]}"
    echo
    echo "git_commit=$(git -C "$BRIDGEVLA_ROOT" rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git -C "$BRIDGEVLA_ROOT" branch --show-current 2>/dev/null || true)"
} > "$RUN_ROOT/protocol.txt"

git -C "$BRIDGEVLA_ROOT" diff \
    > "$RUN_ROOT/code_diff.patch" 2>/dev/null || true

python -m pip freeze \
    > "$RUN_ROOT/pip_freeze.txt" 2>/dev/null || true

nvidia-smi \
    > "$RUN_ROOT/nvidia_smi_start.txt" 2>&1 || true

###############################################################################
# Display schedule
###############################################################################

echo
echo "============================================================"
echo "BridgeVLA Table-1 reproduction"
echo "============================================================"
echo "Run root       : $RUN_ROOT"
echo "Model          : $MODEL_NAME"
echo "Tasks/run      : 18"
echo "Episodes/task  : 25"
echo "Steps/episode  : 25 max"
echo "Repetitions    : 5"
echo "Total trials   : 2250"
echo "GPUs           : ${GPUS[*]}"
echo

for ((rep = 1; rep <= N_REPEATS; rep++)); do
    worker=$(( (rep - 1) % NWORKERS ))
    gpu="${GPUS[$worker]}"

    printf "Repetition %d -> physical GPU %s\n" "$rep" "$gpu"
done

###############################################################################
# One complete Table-1 repetition
###############################################################################

run_one_repetition() {
    local rep="$1"
    local gpu="$2"

    local rep_name
    local rep_dir
    local rc

    rep_name=$(printf "run_%02d" "$rep")
    rep_dir="$RUN_ROOT/$rep_name"

    mkdir -p "$rep_dir"

    {
        echo "repetition=$rep"
        echo "gpu=$gpu"
        echo "start=$(date)"
    } > "$rep_dir/meta.txt"

    echo "[$(date)] START $rep_name on GPU $gpu"

    (
        cd "$RLBENCH_DIR" || exit 1

        CUDA_VISIBLE_DEVICES="$gpu" \
        xvfb-run \
            --auto-servernum \
            --server-args='-screen 0 1024x768x24 -ac' \
            python -u eval.py \
                --model-folder "$MODEL_FOLDER" \
                --model-name "$MODEL_NAME" \
                --eval-datafolder "$EVAL_DATA" \
                --tasks all \
                --start-episode "$START_EPISODE" \
                --eval-episodes "$EVAL_EPISODES" \
                --episode-length "$EPISODE_LENGTH" \
                --log-name "$RUN_NAME/$rep_name" \
                --device 0 \
                --headless

    ) > "$rep_dir/stdout.log" 2>&1

    rc=$?

    echo "$rc" > "$rep_dir/exit_code.txt"

    {
        echo "end=$(date)"
        echo "exit_code=$rc"
    } >> "$rep_dir/meta.txt"

    if (( rc == 0 )); then
        echo "[$(date)] DONE  $rep_name on GPU $gpu"
    else
        echo "[$(date)] FAIL  $rep_name on GPU $gpu, rc=$rc"
    fi

    return "$rc"
}

###############################################################################
# Worker loop
#
# Each physical GPU runs at most one CoppeliaSim/eval process at a time.
###############################################################################

worker_loop() {
    local worker="$1"
    local gpu="$2"
    local failed=0

    for ((rep = worker + 1; rep <= N_REPEATS; rep += NWORKERS)); do
        if ! run_one_repetition "$rep" "$gpu"; then
            failed=1

            rep_name=$(printf "run_%02d" "$rep")

            echo
            echo "Last 40 lines from failed $rep_name:"
            tail -n 40 "$RUN_ROOT/$rep_name/stdout.log" || true
            echo
        fi
    done

    return "$failed"
}

###############################################################################
# Launch GPU workers
###############################################################################

echo
echo "============================================================"
echo "Launching evaluation repetitions"
echo "============================================================"

PIDS=()
WORKER_NAMES=()

for ((worker = 0; worker < NWORKERS; worker++)); do
    gpu="${GPUS[$worker]}"

    worker_loop "$worker" "$gpu" &

    PIDS+=("$!")
    WORKER_NAMES+=("gpu$gpu")

    echo "Started worker on physical GPU $gpu, PID=${PIDS[-1]}"
done

###############################################################################
# Wait
###############################################################################

FAILED=0

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[WORKER DONE] ${WORKER_NAMES[$i]}"
    else
        echo "[WORKER FAILED] ${WORKER_NAMES[$i]}"
        FAILED=1
    fi
done

###############################################################################
# Aggregate the 5 complete evaluation repetitions
###############################################################################

echo
echo "============================================================"
echo "Aggregating Table-1 statistics"
echo "============================================================"

python - "$RUN_ROOT" "$MODEL_STEM" <<'PY'
import csv
import os
import statistics
import sys

run_root = sys.argv[1]
model_stem = sys.argv[2]

tasks = [
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

n_runs = 5

all_runs = {}
errors = []

for run_idx in range(1, n_runs + 1):
    run_name = f"run_{run_idx:02d}"

    csv_path = os.path.join(
        run_root,
        run_name,
        model_stem,
        "eval_results.csv",
    )

    if not os.path.isfile(csv_path):
        errors.append(f"{run_name}: missing {csv_path}")
        continue

    run_results = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task = (row.get("task") or "").strip()
            value = (row.get("success rate") or "").strip()

            if not task:
                continue

            if task in run_results:
                errors.append(
                    f"{run_name}: duplicate task {task}"
                )
                continue

            try:
                value = float(value)
            except ValueError:
                errors.append(
                    f"{run_name}: invalid success rate "
                    f"for {task}: {value!r}"
                )
                continue

            run_results[task] = value

    missing = [t for t in tasks if t not in run_results]
    extra = [t for t in run_results if t not in tasks]

    if missing:
        errors.append(
            f"{run_name}: missing tasks: {', '.join(missing)}"
        )

    if extra:
        errors.append(
            f"{run_name}: unexpected tasks: {', '.join(extra)}"
        )

    if not missing and not extra:
        all_runs[run_idx] = run_results

if errors:
    print()
    print("Aggregation errors:")
    for e in errors:
        print("  -", e)
    print()

if len(all_runs) != n_runs:
    print(
        f"ERROR: only {len(all_runs)}/{n_runs} "
        "complete repetitions are available."
    )
    sys.exit(2)

summary_csv = os.path.join(
    run_root,
    "table1_summary.csv",
)

summary_txt = os.path.join(
    run_root,
    "table1_summary.txt",
)

per_run_csv = os.path.join(
    run_root,
    "table1_per_run.csv",
)

###############################################################################
# Per-run table
###############################################################################

with open(per_run_csv, "w", newline="") as f:
    fieldnames = [
        "task",
        "run_01",
        "run_02",
        "run_03",
        "run_04",
        "run_05",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for task in tasks:
        row = {"task": task}

        for run_idx in range(1, n_runs + 1):
            row[f"run_{run_idx:02d}"] = (
                all_runs[run_idx][task]
            )

        writer.writerow(row)

###############################################################################
# Mean +/- sample standard deviation
#
# The paper does not explicitly define the +/- estimator.
# Table-1 values are consistent with sample standard deviation across
# five evaluation repetitions, so statistics.stdev (ddof=1) is used.
###############################################################################

summary_rows = []

for task in tasks:
    values = [
        all_runs[run_idx][task]
        for run_idx in range(1, n_runs + 1)
    ]

    mean = statistics.mean(values)
    std = statistics.stdev(values)

    summary_rows.append(
        {
            "task": task,
            "run_01": values[0],
            "run_02": values[1],
            "run_03": values[2],
            "run_04": values[3],
            "run_05": values[4],
            "mean_success_rate": mean,
            "std_success_rate": std,
        }
    )

with open(summary_csv, "w", newline="") as f:
    fieldnames = [
        "task",
        "run_01",
        "run_02",
        "run_03",
        "run_04",
        "run_05",
        "mean_success_rate",
        "std_success_rate",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(summary_rows)

###############################################################################
# Overall Avg. SR
###############################################################################

run_avgs = {}

for run_idx in range(1, n_runs + 1):
    run_avgs[run_idx] = statistics.mean(
        all_runs[run_idx][task]
        for task in tasks
    )

task_means = [
    row["mean_success_rate"]
    for row in summary_rows
]

overall_avg_sr = statistics.mean(task_means)

###############################################################################
# Human-readable Table-1 style output
###############################################################################

lines = []

header = (
    f"{'Task':38s} "
    f"{'R1':>6s} "
    f"{'R2':>6s} "
    f"{'R3':>6s} "
    f"{'R4':>6s} "
    f"{'R5':>6s} "
    f"{'Mean +/- Std':>18s}"
)

lines.append(header)
lines.append("-" * len(header))

for row in summary_rows:
    lines.append(
        f"{row['task']:38s} "
        f"{row['run_01']:6.1f} "
        f"{row['run_02']:6.1f} "
        f"{row['run_03']:6.1f} "
        f"{row['run_04']:6.1f} "
        f"{row['run_05']:6.1f} "
        f"{row['mean_success_rate']:6.1f} +/- "
        f"{row['std_success_rate']:.1f}"
    )

lines.append("-" * len(header))

for run_idx in range(1, n_runs + 1):
    lines.append(
        f"Run {run_idx:02d} Avg. SR:"
        f"{run_avgs[run_idx]:9.2f}%"
    )

lines.append("")
lines.append(
    f"TABLE-1 Avg. SR (18-task mean): "
    f"{overall_avg_sr:.2f}%"
)

text = "\n".join(lines)

print()
print(text)
print()

with open(summary_txt, "w") as f:
    f.write(text)
    f.write("\n")

print("Saved:")
print(" ", per_run_csv)
print(" ", summary_csv)
print(" ", summary_txt)

PY

SUMMARY_RC=$?

###############################################################################
# Final report
###############################################################################

echo

if (( FAILED != 0 )); then
    echo "============================================================"
    echo "One or more evaluation repetitions FAILED."
    echo "Inspect:"
    echo "  $RUN_ROOT/run_*/stdout.log"
    echo "============================================================"
    exit 1
fi

if (( SUMMARY_RC != 0 )); then
    echo "============================================================"
    echo "Evaluation processes ended, but summary is incomplete."
    echo "Inspect:"
    echo "  $RUN_ROOT"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "SUCCESS: Table-1 protocol completed."
echo
echo "Main result:"
echo "  $RUN_ROOT/table1_summary.txt"
echo
echo "CSV:"
echo "  $RUN_ROOT/table1_summary.csv"
echo
echo "Per-run results:"
echo "  $RUN_ROOT/table1_per_run.csv"
echo
echo "Raw logs:"
echo "  $RUN_ROOT/run_*/stdout.log"
echo "============================================================"
