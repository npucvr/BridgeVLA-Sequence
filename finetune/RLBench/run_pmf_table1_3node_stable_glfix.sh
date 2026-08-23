#!/usr/bin/env bash

# BridgeVLA + Probabilistic Motion Filter
# RLBench Table-1, three-node dynamic evaluation.
#
# Fixed evaluation protocol:
#   18 tasks
#   25 trials per task
#   max 25 action steps per trial
#   5 complete repetitions
#
# Nodes:
#   server07  : controller/local, Ubuntu 24.04, /data2/local_userdata/lizhe
#   server112 : remote, Ubuntu 22.04, /data2/local_userdata/lizhe
#   server108 : remote, Ubuntu 22.04, /data/local_userdata/lizhe
#
# Important runtime policy:
#   - All workers claim repetitions from one atomic queue on shared /remote_userdata.
#   - Hugging Face is forced offline; all model files must exist in the local HF cache.
#   - server07 uses the system Ubuntu xvfb-run.
#   - server112/server108 use user-space Ubuntu 22.04 xvfb-run extracted from .deb files.
#   - Conda Xvfb is NOT used.
#   - Ubuntu 22 workers preload Conda libGL.so.1 only for the eval Python process.
#     This keeps libGL/libGLX/libGLdispatch on one consistent GLVND frontend while
#     leaving the system Mesa GLX vendor driver / llvmpipe renderer unchanged.
#   - Every Python entry uses $CONDA_PREFIX/bin/python explicitly; PATH cannot
#     accidentally select /usr/bin/python on remote nodes.
#   - Remote nodes complete preflight before a common START barrier is released.
#   - Ctrl-C on the controller creates a shared ABORT marker and force-stops
#     local/remote workers, eval.py processes, and their Xvfb descendants.
#
# Recommended controller usage on server07:
#   LOCAL_GPUS="0" \
#   SERVER112_GPUS="6 7" \
#   SERVER108_GPUS="0 1" \
#   bash finetune/RLBench/run_pmf_table1_3node_stable_glfix.sh

# LOCAL_GPUS="0" SERVER112_GPUS="0 1 2 3" \bash finetune/RLBench/run_pmf_table1_3node_stable_glfix.sh

set -uo pipefail

# Fail fast on any shell syntax error before launching remote workers.
# This is especially useful because Bash can otherwise execute earlier top-level
# commands before it reaches a malformed later heredoc/aggregation block.
if [[ "${SCRIPT_MODE:-controller}" == "controller" ]]; then
    if ! bash -n "$0"; then
        echo "ERROR: launcher syntax check failed: $0" >&2
        exit 2
    fi
fi

###############################################################################
# Fixed paper evaluation protocol
###############################################################################

EXPECTED_ENV="bridgevla_rlbench"
EXPECTED_BRANCH="lizhe/prob-motion-filter"

# Commit used by the successful 2026-08-23 headless GLVND diagnosis.
# A mismatch is a warning, not an automatic override or checkout.
DIAGNOSED_COMMIT="${DIAGNOSED_COMMIT:-a8656fe6c2b7d53d4bf8061e5c98f384d2058499}"

BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Sequence}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"

N_REPEATS=5
START_EPISODE=0
EVAL_EPISODES=25
EPISODE_LENGTH=25

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
# PMF / runtime configuration
###############################################################################

PMF_ENABLED="${PMF_ENABLED:-1}"
PMF_PRIOR_VAR="${PMF_PRIOR_VAR:-9e-4}"
PMF_OBSERVATION_VAR="${PMF_OBSERVATION_VAR:-1e-4}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
EXISTING_RUN_ROOT="${EXISTING_RUN_ROOT:-}"
CPU_THREADS="${CPU_THREADS:-4}"

if [[ "$PMF_ENABLED" != "0" && "$PMF_ENABLED" != "1" ]]; then
    echo "ERROR: PMF_ENABLED must be 0 or 1."
    exit 2
fi

if [[ "$AGGREGATE_ONLY" != "0" && "$AGGREGATE_ONLY" != "1" ]]; then
    echo "ERROR: AGGREGATE_ONLY must be 0 or 1."
    exit 2
fi

if [[ "$PMF_ENABLED" == "1" ]]; then
    PMF_ENABLED_TEXT="true"
    PMF_ENABLED_LABEL="yes"
    EVALUATION_VARIANT="BridgeVLA + PMF"
    DEFAULT_RUN_PREFIX="pmf"
else
    PMF_ENABLED_TEXT="false"
    PMF_ENABLED_LABEL="no"
    EVALUATION_VARIANT="BridgeVLA baseline"
    DEFAULT_RUN_PREFIX="baseline"
fi

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_PREFIX}_table1_3node_glfix_$(date +%Y%m%d_%H%M%S)}"
SHARED_EVAL_ROOT="${SHARED_EVAL_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/multinode_eval}"
REMOTE_CONDA_SH="${REMOTE_CONDA_SH:-/home/lizhe/miniconda3/etc/profile.d/conda.sh}"
REMOTE_READY_TIMEOUT="${REMOTE_READY_TIMEOUT:-180}"

# Controller aliases configured in ~/.ssh/config on server07.
SERVER112_HOST="${SERVER112_HOST:-server112}"
SERVER108_HOST="${SERVER108_HOST:-server108}"

# Per-node local storage roots.
SERVER07_DATA_ROOT="${SERVER07_DATA_ROOT:-/data2/local_userdata/lizhe}"
SERVER112_DATA_ROOT="${SERVER112_DATA_ROOT:-/data2/local_userdata/lizhe}"
SERVER108_DATA_ROOT="${SERVER108_DATA_ROOT:-/data/local_userdata/lizhe}"

# Native Ubuntu xvfb-run locations.
SERVER07_XVFB_RUN="${SERVER07_XVFB_RUN:-/usr/bin/xvfb-run}"
SERVER112_XVFB_RUN="${SERVER112_XVFB_RUN:-/data2/local_userdata/lizhe/tools/xvfb_jammy/usr/bin/xvfb-run}"
SERVER108_XVFB_RUN="${SERVER108_XVFB_RUN:-/data/local_userdata/lizhe/tools/xvfb_jammy/usr/bin/xvfb-run}"

# GLVND frontend policy from the 2026-08-23 diagnosis:
#   server07: keep the known-good baseline unchanged.
#   Ubuntu 22 workers: preload the Conda libGL frontend for the eval Python
#   process so libGL/libGLX/libGLdispatch are selected consistently.
SERVER07_GL_PRELOAD="${SERVER07_GL_PRELOAD:-}"
SERVER112_GL_PRELOAD="${SERVER112_GL_PRELOAD:-/home/lizhe/miniconda3/envs/bridgevla_rlbench/lib/libGL.so.1}"
SERVER108_GL_PRELOAD="${SERVER108_GL_PRELOAD:-/home/lizhe/miniconda3/envs/bridgevla_rlbench/lib/libGL.so.1}"

# Internal variables set by the controller for worker processes.
SCRIPT_MODE="${SCRIPT_MODE:-controller}"
WORKER_TAG="${WORKER_TAG:-local}"
NODE_DATA_ROOT="${NODE_DATA_ROOT:-$SERVER07_DATA_ROOT}"
XVFB_RUN="${XVFB_RUN:-$SERVER07_XVFB_RUN}"
NODE_GL_PRELOAD="${NODE_GL_PRELOAD:-$SERVER07_GL_PRELOAD}"

# Derive node-local model/data/cache paths from NODE_DATA_ROOT.
MODEL_FOLDER="${MODEL_FOLDER:-$NODE_DATA_ROOT/VLA/BridgeVLA/checkpoints/bridgevla/rlbench}"
EVAL_DATA="${EVAL_DATA:-$NODE_DATA_ROOT/VLA/BridgeVLA/datasets/rlbench/eval}"

# Deliberately ignore inherited HF_HOME from a copied Conda environment unless
# NODE_HF_HOME is explicitly supplied. This prevents /data2 vs /data mistakes.
export HF_HOME="${NODE_HF_HOME:-$NODE_DATA_ROOT/huggingface_cache}"
export HF_HUB_CACHE="${NODE_HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

RLBENCH_DIR="$BRIDGEVLA_ROOT/finetune/RLBench"
EVAL_PY="$RLBENCH_DIR/eval.py"
SCRIPT_PATH="$RLBENCH_DIR/$(basename "$0")"
MODEL_STEM="${MODEL_NAME%.pth}"

# eval.py writes full outputs locally under MODEL_FOLDER/eval.
LOCAL_RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"

# Shared queue, logs, metadata, and copied result CSVs.
RUN_ROOT="$SHARED_EVAL_ROOT/$RUN_NAME"
QUEUE_ROOT="$RUN_ROOT/queue"
PENDING_DIR="$QUEUE_ROOT/pending"
RUNNING_DIR="$QUEUE_ROOT/running"
DONE_DIR="$QUEUE_ROOT/done"
FAILED_DIR="$QUEUE_ROOT/failed"
ASSIGNMENT_DIR="$QUEUE_ROOT/assignments"
NODE_STATUS_DIR="$QUEUE_ROOT/nodes"
READY_FILE="$QUEUE_ROOT/QUEUE_READY"
START_FILE="$QUEUE_ROOT/START"
ASSIGNMENT_LOG="$RUN_ROOT/assignments.tsv"
ABORT_FILE="$QUEUE_ROOT/ABORT"
ABORTED_INFO="$RUN_ROOT/ABORTED.txt"

NODE_NAME="$(hostname -s 2>/dev/null || hostname)"
NODE_NAME="${NODE_NAME//[^A-Za-z0-9_.-]/_}"
WORKER_TAG="${WORKER_TAG//[^A-Za-z0-9_.-]/_}"

###############################################################################
# Helpers
###############################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$NODE_NAME] $*"
}

shell_quote() {
    printf '%q' "$1"
}

# Recursively signal descendants of one shell/process. This is used only for
# processes started by this launcher, never as a system-wide pkill.
kill_descendants() {
    local parent_pid="$1"
    local sig="${2:-TERM}"
    local child

    while read -r child; do
        [[ "$child" =~ ^[0-9]+$ ]] || continue
        kill_descendants "$child" "$sig"
        kill "-$sig" "$child" 2>/dev/null || true
    done < <(ps -o pid= --ppid "$parent_pid" 2>/dev/null | awk '{print $1}')
}

worker_signal_abort() {
    local sig="${1:-TERM}"

    # Avoid re-entering the handler while descendants are being terminated.
    trap - INT TERM

    mkdir -p "$QUEUE_ROOT" 2>/dev/null || true
    : > "$ABORT_FILE" 2>/dev/null || true
    log "Received $sig; force-stopping worker group and descendants."

    kill_descendants "$BASHPID" TERM
    sleep 2
    kill_descendants "$BASHPID" KILL
    exit 130
}

parse_gpu_spec() {
    local spec="$1"
    local gpu
    local gpu_count

    spec="${spec//,/ }"
    read -r -a PARSED_GPUS <<< "$spec"

    if (( ${#PARSED_GPUS[@]} == 0 )); then
        echo "ERROR: no GPU IDs were provided on $NODE_NAME."
        return 2
    fi

    gpu_count="$(nvidia-smi -L | wc -l)"
    declare -A seen=()

    for gpu in "${PARSED_GPUS[@]}"; do
        if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
            echo "ERROR: invalid GPU id on $NODE_NAME: $gpu"
            return 2
        fi
        if (( gpu >= gpu_count )); then
            echo "ERROR: GPU $gpu does not exist on $NODE_NAME (GPU count=$gpu_count)."
            return 2
        fi
        if [[ -n "${seen[$gpu]:-}" ]]; then
            echo "ERROR: duplicate GPU id on $NODE_NAME: $gpu"
            return 2
        fi
        seen["$gpu"]=1
    done
}

validate_pmf_parameters() {
    "$CONDA_PREFIX/bin/python" - "$PMF_PRIOR_VAR" "$PMF_OBSERVATION_VAR" <<'PY'
import sys
q = float(sys.argv[1])
r = float(sys.argv[2])
if q <= 0:
    raise SystemExit("ERROR: PMF_PRIOR_VAR must be > 0")
if r <= 0:
    raise SystemExit("ERROR: PMF_OBSERVATION_VAR must be > 0")
k = q / (q + r)
print(f"PMF prior variance       : {q:g}")
print(f"PMF observation variance : {r:g}")
print(f"PMF gain K               : {k:.8f}")
PY
}

aggregate_results() {
    local aggregate_run_root="$1"

    if [[ -z "${CONDA_PREFIX:-}" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "ERROR: explicit Conda Python is missing or not executable." >&2
        return 2
    fi

    echo
    echo "============================================================"
    echo "Aggregating Table-1 statistics"
    echo "============================================================"

    "$CONDA_PREFIX/bin/python" - "$aggregate_run_root" "$MODEL_STEM" <<'PY'
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
    csv_path = os.path.join(run_root, run_name, model_stem, "eval_results.csv")

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
                errors.append(f"{run_name}: duplicate task {task}")
                continue
            try:
                value = float(value)
            except ValueError:
                errors.append(f"{run_name}: invalid success rate for {task}: {value!r}")
                continue
            run_results[task] = value

    missing = [t for t in tasks if t not in run_results]
    extra = [t for t in run_results if t not in tasks]
    if missing:
        errors.append(f"{run_name}: missing tasks: {', '.join(missing)}")
    if extra:
        errors.append(f"{run_name}: unexpected tasks: {', '.join(extra)}")
    if not missing and not extra:
        all_runs[run_idx] = run_results

if errors:
    print()
    print("Aggregation errors:")
    for e in errors:
        print("  -", e)
    print()

if len(all_runs) != n_runs:
    print(f"ERROR: only {len(all_runs)}/{n_runs} complete repetitions are available.")
    sys.exit(2)

per_run_csv = os.path.join(run_root, "table1_per_run.csv")
summary_csv = os.path.join(run_root, "table1_summary.csv")
summary_txt = os.path.join(run_root, "table1_summary.txt")

with open(per_run_csv, "w", newline="") as f:
    fieldnames = ["task", "run_01", "run_02", "run_03", "run_04", "run_05"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for task in tasks:
        row = {"task": task}
        for run_idx in range(1, n_runs + 1):
            row[f"run_{run_idx:02d}"] = all_runs[run_idx][task]
        writer.writerow(row)

summary_rows = []
for task in tasks:
    values = [all_runs[run_idx][task] for run_idx in range(1, n_runs + 1)]
    summary_rows.append(
        {
            "task": task,
            "run_01": values[0],
            "run_02": values[1],
            "run_03": values[2],
            "run_04": values[3],
            "run_05": values[4],
            "mean_success_rate": statistics.mean(values),
            "std_success_rate": statistics.stdev(values),
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
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

run_avgs = {
    run_idx: statistics.mean(all_runs[run_idx][task] for task in tasks)
    for run_idx in range(1, n_runs + 1)
}

overall_avg_sr = statistics.mean(row["mean_success_rate"] for row in summary_rows)

header = (
    f"{'Task':38s} "
    f"{'R1':>6s} "
    f"{'R2':>6s} "
    f"{'R3':>6s} "
    f"{'R4':>6s} "
    f"{'R5':>6s} "
    f"{'Mean +/- Std':>18s}"
)

lines = [header, "-" * len(header)]
for row in summary_rows:
    lines.append(
        f"{row['task']:38s} "
        f"{row['run_01']:6.1f} "
        f"{row['run_02']:6.1f} "
        f"{row['run_03']:6.1f} "
        f"{row['run_04']:6.1f} "
        f"{row['run_05']:6.1f} "
        f"{row['mean_success_rate']:6.1f} +/- {row['std_success_rate']:.1f}"
    )

lines.append("-" * len(header))
for run_idx in range(1, n_runs + 1):
    lines.append(f"Run {run_idx:02d} Avg. SR: {run_avgs[run_idx]:.2f}%")

lines.append("")
lines.append(f"TABLE-1 Avg. SR (18-task mean): {overall_avg_sr:.2f}%")
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
}

validate_eval_data() {
    local data_error=0
    local task
    local ep_root
    local missing
    local ep

    echo
    echo "============================================================"
    echo "Checking RLBench evaluation data on $NODE_NAME"
    echo "============================================================"

    for task in "${TASKS[@]}"; do
        ep_root="$EVAL_DATA/$task/all_variations/episodes"
        if [[ ! -d "$ep_root" ]]; then
            echo "MISSING: $ep_root"
            data_error=1
            continue
        fi

        missing=0
        for ((ep = START_EPISODE; ep < START_EPISODE + EVAL_EPISODES; ep++)); do
            if [[ ! -d "$ep_root/episode$ep" ]]; then
                echo "MISSING: $ep_root/episode$ep"
                missing=1
                data_error=1
            fi
        done

        if (( missing == 0 )); then
            printf "%-40s episodes 0-24 OK\n" "$task"
        fi
    done

    return "$data_error"
}

bridgevla_preflight() {
    echo
    echo "============================================================"
    echo "BridgeVLA preflight on $NODE_NAME"
    echo "============================================================"

    "$CONDA_PREFIX/bin/python" - <<'PY'
import os
import bridgevla

root = os.path.realpath(os.environ["BRIDGEVLA_ROOT"])
pkg = os.path.realpath(bridgevla.__file__)
print("BRIDGEVLA_ROOT :", root)
print("bridgevla      :", pkg)

if not pkg.startswith(root + os.sep):
    raise SystemExit("ERROR: bridgevla is not imported from BRIDGEVLA_ROOT")

from bridgevla.utils.rvt_utils import TensorboardManager

with open("/proc/self/maps", "r") as f:
    maps = f.read()

if "libtensorflow" in maps:
    raise SystemExit(
        "ERROR: libtensorflow is loaded. The RLBench/CoppeliaSim OpenSSL conflict may return."
    )

print("libtensorflow  : NOT loaded")
print("Preflight      : OK")
PY
}

hf_offline_preflight() {
    echo
    echo "============================================================"
    echo "Hugging Face offline preflight on $NODE_NAME"
    echo "============================================================"

    "$CONDA_PREFIX/bin/python" - <<'PY'
import os
from huggingface_hub import constants
from transformers import AutoConfig
from transformers.utils import hub

print("HF_HOME                :", constants.HF_HOME)
print("HF_HUB_CACHE           :", constants.HF_HUB_CACHE)
print("HF_HUB_OFFLINE env     :", os.environ.get("HF_HUB_OFFLINE"))
print("HF_HUB_OFFLINE parsed  :", constants.HF_HUB_OFFLINE)
print("Transformers offline   :", hub.is_offline_mode())

if not constants.HF_HUB_OFFLINE:
    raise SystemExit("ERROR: huggingface_hub offline mode is not active")
if not hub.is_offline_mode():
    raise SystemExit("ERROR: transformers offline mode is not active")

cfg = AutoConfig.from_pretrained(
    "google/paligemma-3b-pt-224",
    local_files_only=True,
)
print("PaliGemma local config : OK")
print("Offline preflight      : OK")
PY
}

xvfb_preflight() {
    local xvfb_bin_dir
    local test_log
    local rc

    echo
    echo "============================================================"
    echo "Native Xvfb preflight on $NODE_NAME"
    echo "============================================================"

    if [[ ! -x "$XVFB_RUN" ]]; then
        echo "ERROR: xvfb-run is missing or not executable:"
        echo "  $XVFB_RUN"
        return 2
    fi

    xvfb_bin_dir="$(dirname "$XVFB_RUN")"
    if [[ ! -x "$xvfb_bin_dir/Xvfb" ]]; then
        echo "ERROR: matching Xvfb is missing:"
        echo "  $xvfb_bin_dir/Xvfb"
        return 2
    fi

    if [[ ! -x "$xvfb_bin_dir/xauth" && ! -x /usr/bin/xauth ]]; then
        echo "ERROR: xauth is not available for xvfb-run."
        return 2
    fi

    echo "xvfb-run : $XVFB_RUN"
    echo "Xvfb     : $xvfb_bin_dir/Xvfb"

    test_log="/tmp/bridgevla_xvfb_preflight_${USER}_${NODE_NAME}_$$.log"
    PATH="$xvfb_bin_dir:$PATH:/usr/bin:/bin" \
    timeout 15 "$XVFB_RUN" \
        -a \
        -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
        bash -c 'test -n "$DISPLAY" && echo "DISPLAY=$DISPLAY"' \
        > "$test_log" 2>&1
    rc=$?

    if (( rc != 0 )); then
        echo "ERROR: native xvfb-run preflight failed, rc=$rc"
        cat "$test_log" || true
        rm -f "$test_log"
        return 2
    fi

    cat "$test_log"
    rm -f "$test_log"
    echo "Native Xvfb preflight: OK"
}

validate_node_environment() {
    local gpu_spec="$1"
    local f

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first on $NODE_NAME."
        echo "Current env: ${CONDA_DEFAULT_ENV:-<none>}"
        return 2
    fi

    if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "ERROR: Conda Python is missing or not executable:"
        echo "  $CONDA_PREFIX/bin/python"
        return 2
    fi

    if [[ -n "$NODE_GL_PRELOAD" && ! -r "$NODE_GL_PRELOAD" ]]; then
        echo "ERROR: configured GLVND preload library is missing or unreadable:"
        echo "  $NODE_GL_PRELOAD"
        return 2
    fi

    if [[ ! -d "$BRIDGEVLA_ROOT/.git" ]]; then
        echo "ERROR: not a git repository: $BRIDGEVLA_ROOT"
        return 2
    fi

    CURRENT_BRANCH="$(git -C "$BRIDGEVLA_ROOT" branch --show-current 2>/dev/null || true)"
    CURRENT_COMMIT="$(git -C "$BRIDGEVLA_ROOT" rev-parse HEAD 2>/dev/null || true)"

    if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
        echo "ERROR: wrong branch on $NODE_NAME."
        echo "Expected: $EXPECTED_BRANCH"
        echo "Current : ${CURRENT_BRANCH:-<detached/unknown>}"
        return 2
    fi

    if [[ "$CURRENT_COMMIT" != "$DIAGNOSED_COMMIT" ]]; then
        echo "WARNING: current commit differs from the commit used for the successful headless diagnosis."
        echo "Diagnosed: $DIAGNOSED_COMMIT"
        echo "Current  : $CURRENT_COMMIT"
        echo
    fi

    if [[ ! -f "$EVAL_PY" ]]; then
        echo "ERROR: eval.py not found: $EVAL_PY"
        return 2
    fi

    if [[ ! -f "$MODEL_FOLDER/$MODEL_NAME" ]]; then
        echo "ERROR: checkpoint not found: $MODEL_FOLDER/$MODEL_NAME"
        return 2
    fi

    for f in exp_cfg.yaml mvt_cfg.yaml; do
        if [[ ! -f "$MODEL_FOLDER/$f" ]]; then
            echo "ERROR: missing checkpoint config: $MODEL_FOLDER/$f"
            return 2
        fi
    done

    if [[ ! -d "$EVAL_DATA" ]]; then
        echo "ERROR: eval data not found: $EVAL_DATA"
        return 2
    fi

    if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
        echo "ERROR: COPPELIASIM_ROOT is not set on $NODE_NAME."
        return 2
    fi

    if [[ ! -x "$COPPELIASIM_ROOT/coppeliaSim" ]]; then
        echo "ERROR: CoppeliaSim executable not found:"
        echo "  $COPPELIASIM_ROOT/coppeliaSim"
        return 2
    fi

    if [[ -e "$LOCAL_RUN_ROOT" ]]; then
        echo "ERROR: local eval directory already exists on $NODE_NAME:"
        echo "  $LOCAL_RUN_ROOT"
        echo "Choose a new RUN_NAME to prevent eval_results.csv from being appended."
        return 2
    fi

    if [[ "$PMF_ENABLED" == "1" ]]; then
        validate_pmf_parameters || return 2
    else
        echo "PMF: disabled"
    fi
    parse_gpu_spec "$gpu_spec" || return 2
    validate_eval_data || return 2
    hf_offline_preflight || return 2
    xvfb_preflight || return 2
    bridgevla_preflight || return 2

    return 0
}

write_node_metadata() {
    mkdir -p "$NODE_STATUS_DIR"

    {
        echo "worker_tag=$WORKER_TAG"
        echo "node=$NODE_NAME"
        echo "date=$(date)"
        echo "hostname=$(hostname)"
        echo "os=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' 2>/dev/null || true)"
        echo "conda_env=${CONDA_DEFAULT_ENV:-}"
        echo "bridgevla_root=$BRIDGEVLA_ROOT"
        echo "git_branch=${CURRENT_BRANCH:-}"
        echo "git_commit=${CURRENT_COMMIT:-}"
        echo "node_data_root=$NODE_DATA_ROOT"
        echo "model=$MODEL_FOLDER/$MODEL_NAME"
        echo "eval_data=$EVAL_DATA"
        echo "local_run_root=$LOCAL_RUN_ROOT"
        echo "shared_run_root=$RUN_ROOT"
        echo "hf_home=$HF_HOME"
        echo "hf_hub_cache=$HF_HUB_CACHE"
        echo "hf_hub_offline=$HF_HUB_OFFLINE"
        echo "transformers_offline=$TRANSFORMERS_OFFLINE"
        echo "xvfb_run=$XVFB_RUN"
        echo "conda_python=$CONDA_PREFIX/bin/python"
        echo "gl_preload=${NODE_GL_PRELOAD:-<none>}"
        echo "coppeliasim_root=${COPPELIASIM_ROOT:-}"
    } > "$RUN_ROOT/node_${WORKER_TAG}_${NODE_NAME}.txt"

    "$CONDA_PREFIX/bin/python" -m pip freeze > "$RUN_ROOT/pip_freeze_${WORKER_TAG}_${NODE_NAME}.txt" 2>/dev/null || true
    nvidia-smi > "$RUN_ROOT/nvidia_smi_${WORKER_TAG}_${NODE_NAME}_start.txt" 2>&1 || true
}

###############################################################################
# Queue operations
###############################################################################

claim_next_repetition() {
    local gpu="$1"
    local rep
    local rep_name
    local pending
    local running
    local assignment

    for ((rep = 1; rep <= N_REPEATS; rep++)); do
        rep_name="$(printf 'run_%02d' "$rep")"
        pending="$PENDING_DIR/${rep_name}.job"
        running="$RUNNING_DIR/${rep_name}.${NODE_NAME}.gpu${gpu}.job"

        # rename on the shared filesystem is atomic. Exactly one worker wins.
        if mv "$pending" "$running" 2>/dev/null; then
            assignment="$ASSIGNMENT_DIR/${rep_name}.tsv"
            printf "%s\t%s\t%s\t%s\n" \
                "$(date '+%Y-%m-%d %H:%M:%S')" \
                "$NODE_NAME" \
                "$gpu" \
                "$rep" \
                > "$assignment"

            printf "%s|%s\n" "$rep" "$running"
            return 0
        fi
    done

    return 1
}

copy_result_to_shared() {
    local rep_name="$1"
    local local_rep_dir="$LOCAL_RUN_ROOT/$rep_name"
    local shared_rep_dir="$RUN_ROOT/$rep_name"
    local result_csv="$local_rep_dir/$MODEL_STEM/eval_results.csv"

    if [[ ! -f "$result_csv" ]]; then
        echo "ERROR: eval.py returned successfully but result CSV is missing:"
        echo "  $result_csv"
        return 1
    fi

    mkdir -p "$shared_rep_dir/$MODEL_STEM"
    cp -f "$result_csv" "$shared_rep_dir/$MODEL_STEM/eval_results.csv" || return 1

    if [[ -f "$local_rep_dir/eval_config.yaml" ]]; then
        cp -f "$local_rep_dir/eval_config.yaml" "$shared_rep_dir/eval_config.yaml" || true
    fi

    return 0
}

run_one_repetition() {
    local rep="$1"
    local gpu="$2"
    local rep_name
    local shared_rep_dir
    local local_rep_dir
    local xvfb_bin_dir
    local rc
    local -a pmf_args=()

    rep_name="$(printf 'run_%02d' "$rep")"
    shared_rep_dir="$RUN_ROOT/$rep_name"
    local_rep_dir="$LOCAL_RUN_ROOT/$rep_name"
    xvfb_bin_dir="$(dirname "$XVFB_RUN")"

    if [[ "$PMF_ENABLED" == "1" ]]; then
        pmf_args+=(
            --pmf-enabled
            --pmf-prior-var "$PMF_PRIOR_VAR"
            --pmf-observation-var "$PMF_OBSERVATION_VAR"
        )
    fi

    mkdir -p "$shared_rep_dir"

    {
        echo "repetition=$rep"
        echo "node=$NODE_NAME"
        echo "worker_tag=$WORKER_TAG"
        echo "gpu=$gpu"
        echo "start=$(date)"
        echo "local_eval_dir=$local_rep_dir"
        echo "shared_rep_dir=$shared_rep_dir"
        echo "xvfb_run=$XVFB_RUN"
        echo "hf_home=$HF_HOME"
        echo "hf_hub_offline=$HF_HUB_OFFLINE"
        echo "transformers_offline=$TRANSFORMERS_OFFLINE"
        echo "conda_python=$CONDA_PREFIX/bin/python"
        echo "gl_preload=${NODE_GL_PRELOAD:-<none>}"
        echo "pmf_enabled=$PMF_ENABLED"
    } > "$shared_rep_dir/meta.txt"

    log "START $rep_name on physical GPU $gpu"

    (
        cd "$RLBENCH_DIR" || exit 1

        # xvfb-run invokes Xvfb/xauth by name, so prepend only the matching
        # native Ubuntu Xvfb directory. Keep the existing Conda PATH ahead of
        # /usr/bin to avoid selecting the system Python on Ubuntu 22.
        #
        # GLVND fix: NODE_GL_PRELOAD is injected only into the eval command
        # launched inside Xvfb. It is intentionally NOT preloaded into Xvfb or
        # xauth themselves. server07 leaves NODE_GL_PRELOAD empty.
        eval_env=(
            "CUDA_VISIBLE_DEVICES=$gpu"
            "HF_HOME=$HF_HOME"
            "HF_HUB_CACHE=$HF_HUB_CACHE"
            "HF_HUB_OFFLINE=1"
            "TRANSFORMERS_OFFLINE=1"
        )

        if [[ -n "$NODE_GL_PRELOAD" ]]; then
            eval_env+=("LD_PRELOAD=$NODE_GL_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}")
        fi

        PATH="$xvfb_bin_dir:$PATH:/usr/bin:/bin" \
        "$XVFB_RUN" \
            -a \
            -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
            env "${eval_env[@]}" \
            "$CONDA_PREFIX/bin/python" -u eval.py \
                --model-folder "$MODEL_FOLDER" \
                --model-name "$MODEL_NAME" \
                --eval-datafolder "$EVAL_DATA" \
                --tasks all \
                --start-episode "$START_EPISODE" \
                --eval-episodes "$EVAL_EPISODES" \
                --episode-length "$EPISODE_LENGTH" \
                --log-name "$RUN_NAME/$rep_name" \
                --device 0 \
                --headless \
                "${pmf_args[@]}"

    ) > "$shared_rep_dir/stdout.log" 2>&1

    rc=$?

    if (( rc == 0 )); then
        if ! copy_result_to_shared "$rep_name"; then
            rc=90
        fi
    fi

    echo "$rc" > "$shared_rep_dir/exit_code.txt"

    {
        echo "end=$(date)"
        echo "exit_code=$rc"
    } >> "$shared_rep_dir/meta.txt"

    if (( rc == 0 )); then
        log "DONE  $rep_name on physical GPU $gpu"
    else
        log "FAIL  $rep_name on physical GPU $gpu, rc=$rc"
    fi

    return "$rc"
}

worker_loop() {
    local gpu="$1"
    local failed=0
    local claim
    local rep
    local running_marker
    local rep_name
    local marker_name

    while true; do
        if [[ -f "$ABORT_FILE" ]]; then
            log "GPU $gpu: ABORT marker detected; worker exits."
            break
        fi

        if ! claim="$(claim_next_repetition "$gpu")"; then
            log "GPU $gpu: queue empty, worker exits."
            break
        fi

        IFS='|' read -r rep running_marker <<< "$claim"
        rep_name="$(printf 'run_%02d' "$rep")"
        marker_name="$(basename "$running_marker")"
        log "GPU $gpu claimed $rep_name"

        if run_one_repetition "$rep" "$gpu"; then
            mv "$running_marker" "$DONE_DIR/$marker_name" 2>/dev/null || true
        else
            failed=1
            mv "$running_marker" "$FAILED_DIR/$marker_name" 2>/dev/null || true
            echo
            echo "Last 60 lines from failed $rep_name:"
            tail -n 60 "$RUN_ROOT/$rep_name/stdout.log" || true
            echo
        fi
    done

    return "$failed"
}

launch_worker_group() {
    local gpu_spec="$1"
    local -a gpus
    local -a pids=()
    local -a worker_names=()
    local gpu
    local pid
    local i
    local failed=0

    parse_gpu_spec "$gpu_spec" || return 2
    gpus=("${PARSED_GPUS[@]}")

    for gpu in "${gpus[@]}"; do
        worker_loop "$gpu" &
        pid=$!
        pids+=("$pid")
        worker_names+=("${NODE_NAME}_gpu${gpu}")
        log "Started dynamic queue worker on physical GPU $gpu, PID=$pid"
    done

    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "[WORKER DONE] ${worker_names[$i]}"
        else
            log "[WORKER FAILED] ${worker_names[$i]}"
            failed=1
        fi
    done

    return "$failed"
}

###############################################################################
# Worker-group mode: used by server112/server108 over SSH
###############################################################################

run_worker_mode() {
    local gpu_spec="${WORKER_GPUS:-}"
    local ready_marker
    local worker_failed=0

    if [[ -z "$gpu_spec" ]]; then
        echo "ERROR: WORKER_GPUS is empty in worker mode."
        exit 2
    fi

    WORKER_PID_FILE="$NODE_STATUS_DIR/${WORKER_TAG}.pid"
    mkdir -p "$NODE_STATUS_DIR"
    printf "%s\n" "$BASHPID" > "$WORKER_PID_FILE"

    # The controller can terminate this PID over SSH. The trap then tears down
    # the full worker/eval/xvfb process tree on this remote node.
    trap 'worker_signal_abort INT' INT
    trap 'worker_signal_abort TERM' TERM
    trap 'rm -f "${WORKER_PID_FILE:-}" 2>/dev/null || true' EXIT

    while [[ ! -f "$READY_FILE" ]]; do
        if [[ -f "$ABORT_FILE" ]]; then
            log "ABORT marker detected before queue ready; worker exits."
            exit 130
        fi
        sleep 1
    done

    validate_node_environment "$gpu_spec" || exit 2
    write_node_metadata

    ready_marker="$NODE_STATUS_DIR/${WORKER_TAG}.ready"
    printf "worker_tag=%s\nnode=%s\ngpus=%s\nready=%s\n" \
        "$WORKER_TAG" "$NODE_NAME" "$gpu_spec" "$(date)" \
        > "$ready_marker"

    log "$WORKER_TAG preflight passed; waiting for common START barrier."

    while [[ ! -f "$START_FILE" ]]; do
        if [[ -f "$ABORT_FILE" ]]; then
            log "ABORT marker detected before START; worker exits."
            exit 130
        fi
        sleep 0.2
    done

    if [[ -f "$ABORT_FILE" ]]; then
        log "ABORT marker detected at START; worker exits."
        exit 130
    fi

    log "START barrier released. Launching GPU workers: $gpu_spec"
    launch_worker_group "$gpu_spec" || worker_failed=1

    nvidia-smi > "$RUN_ROOT/nvidia_smi_${WORKER_TAG}_${NODE_NAME}_end.txt" 2>&1 || true

    if (( worker_failed != 0 )); then
        exit 1
    fi
    exit 0
}

if [[ "$SCRIPT_MODE" == "worker" ]]; then
    run_worker_mode
fi

if [[ "$SCRIPT_MODE" != "controller" ]]; then
    echo "ERROR: invalid SCRIPT_MODE=$SCRIPT_MODE"
    exit 2
fi

if [[ "$AGGREGATE_ONLY" == "1" ]]; then
    if [[ -z "$EXISTING_RUN_ROOT" ]]; then
        echo "ERROR: EXISTING_RUN_ROOT is required when AGGREGATE_ONLY=1"
        exit 2
    fi

    if [[ ! -d "$EXISTING_RUN_ROOT" ]]; then
        echo "ERROR: EXISTING_RUN_ROOT does not exist: $EXISTING_RUN_ROOT"
        exit 2
    fi

    missing_results=0
    for ((rep = 1; rep <= N_REPEATS; rep++)); do
        result_csv="$EXISTING_RUN_ROOT/$(printf 'run_%02d' "$rep")/$MODEL_STEM/eval_results.csv"
        if [[ ! -f "$result_csv" ]]; then
            echo "ERROR: missing aggregation input: $result_csv"
            missing_results=1
        fi
    done

    if (( missing_results != 0 )); then
        exit 2
    fi

    aggregate_results "$EXISTING_RUN_ROOT"
    exit $?
fi

###############################################################################
# Controller GPU selection
###############################################################################

LOCAL_GPU_SPEC="${LOCAL_GPUS:-}"
SERVER112_GPU_SPEC="${SERVER112_GPUS:-}"
SERVER108_GPU_SPEC="${SERVER108_GPUS:-}"

if [[ -z "$LOCAL_GPU_SPEC" && -z "$SERVER112_GPU_SPEC" && -z "$SERVER108_GPU_SPEC" ]]; then
    echo "Usage on controller ($NODE_NAME):"
    echo "  LOCAL_GPUS=\"0\" SERVER112_GPUS=\"6 7\" SERVER108_GPUS=\"0 1\" $0"
    exit 2
fi

###############################################################################
# Controller preflight
###############################################################################

if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    WORKER_TAG="server07"
    NODE_DATA_ROOT="$SERVER07_DATA_ROOT"
    MODEL_FOLDER="$NODE_DATA_ROOT/VLA/BridgeVLA/checkpoints/bridgevla/rlbench"
    EVAL_DATA="$NODE_DATA_ROOT/VLA/BridgeVLA/datasets/rlbench/eval"
    HF_HOME="$NODE_DATA_ROOT/huggingface_cache"
    HF_HUB_CACHE="$HF_HOME/hub"
    XVFB_RUN="$SERVER07_XVFB_RUN"
    NODE_GL_PRELOAD="$SERVER07_GL_PRELOAD"
    LOCAL_RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"
    export WORKER_TAG NODE_DATA_ROOT MODEL_FOLDER EVAL_DATA HF_HOME HF_HUB_CACHE XVFB_RUN NODE_GL_PRELOAD LOCAL_RUN_ROOT

    validate_node_environment "$LOCAL_GPU_SPEC" || exit 2
else
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first on controller."
        exit 2
    fi
    if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "ERROR: Conda Python is missing or not executable: $CONDA_PREFIX/bin/python"
        exit 2
    fi
    CURRENT_BRANCH="$(git -C "$BRIDGEVLA_ROOT" branch --show-current 2>/dev/null || true)"
    CURRENT_COMMIT="$(git -C "$BRIDGEVLA_ROOT" rev-parse HEAD 2>/dev/null || true)"
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: shared run directory already exists:"
    echo "  $RUN_ROOT"
    echo "Choose a new RUN_NAME."
    exit 2
fi

###############################################################################
# Initialize the shared queue exactly once on controller
###############################################################################

mkdir -p \
    "$PENDING_DIR" \
    "$RUNNING_DIR" \
    "$DONE_DIR" \
    "$FAILED_DIR" \
    "$ASSIGNMENT_DIR" \
    "$NODE_STATUS_DIR"

for ((rep = 1; rep <= N_REPEATS; rep++)); do
    rep_name="$(printf 'run_%02d' "$rep")"
    : > "$PENDING_DIR/${rep_name}.job"
done

{
    echo "$EVALUATION_VARIANT RLBench Table-1 three-node stable GLVND evaluation"
    echo
    echo "date=$(date)"
    echo "controller_hostname=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "bridgevla_root=$BRIDGEVLA_ROOT"
    echo "git_branch=${CURRENT_BRANCH:-}"
    echo "git_commit=${CURRENT_COMMIT:-}"
    echo "model_name=$MODEL_NAME"
    echo "n_repeats=$N_REPEATS"
    echo "tasks=18"
    echo "eval_episodes=$EVAL_EPISODES"
    echo "start_episode=$START_EPISODE"
    echo "episode_length=$EPISODE_LENGTH"
    echo "pmf_enabled=$PMF_ENABLED_TEXT"
    echo "pmf_prior_var=$PMF_PRIOR_VAR"
    echo "pmf_observation_var=$PMF_OBSERVATION_VAR"
    echo "hf_hub_offline=1"
    echo "transformers_offline=1"
    echo "local_gpus=$LOCAL_GPU_SPEC"
    echo "server112_host=$SERVER112_HOST"
    echo "server112_gpus=$SERVER112_GPU_SPEC"
    echo "server108_host=$SERVER108_HOST"
    echo "server108_gpus=$SERVER108_GPU_SPEC"
    echo "server07_data_root=$SERVER07_DATA_ROOT"
    echo "server112_data_root=$SERVER112_DATA_ROOT"
    echo "server108_data_root=$SERVER108_DATA_ROOT"
    echo "server07_xvfb_run=$SERVER07_XVFB_RUN"
    echo "server112_xvfb_run=$SERVER112_XVFB_RUN"
    echo "server108_xvfb_run=$SERVER108_XVFB_RUN"
    echo "server07_gl_preload=${SERVER07_GL_PRELOAD:-<none>}"
    echo "server112_gl_preload=$SERVER112_GL_PRELOAD"
    echo "server108_gl_preload=$SERVER108_GL_PRELOAD"
    echo "shared_run_root=$RUN_ROOT"
    echo "scheduling=three_node_atomic_shared_queue"
} > "$RUN_ROOT/protocol.txt"

git -C "$BRIDGEVLA_ROOT" diff > "$RUN_ROOT/code_diff.patch" 2>/dev/null || true

if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    write_node_metadata
fi

: > "$READY_FILE"

###############################################################################
# Display schedule
###############################################################################

echo
echo "============================================================"
echo "$EVALUATION_VARIANT RLBench Table-1 three-node stable GLVND evaluation"
echo "============================================================"
echo "Shared run root : $RUN_ROOT"
echo "Model           : $MODEL_NAME"
echo "Tasks/run       : 18"
echo "Episodes/task   : 25"
echo "Steps/episode   : 25 max"
echo "Repetitions     : 5"
echo "Total trials    : 2250"
echo "PMF enabled     : $PMF_ENABLED_LABEL"
echo "PMF prior var   : $PMF_PRIOR_VAR"
echo "PMF obs var     : $PMF_OBSERVATION_VAR"
echo "HF mode         : strict offline"
echo "Controller      : $NODE_NAME"
echo "Local GPUs      : ${LOCAL_GPU_SPEC:-<none>}"
echo "server112 GPUs  : ${SERVER112_GPU_SPEC:-<none>}"
echo "server108 GPUs  : ${SERVER108_GPU_SPEC:-<none>}"
echo "server07 xvfb   : $SERVER07_XVFB_RUN"
echo "server112 xvfb  : $SERVER112_XVFB_RUN"
echo "server108 xvfb  : $SERVER108_XVFB_RUN"
echo "server07 GL     : ${SERVER07_GL_PRELOAD:-<baseline/no preload>}"
echo "server112 GL    : preload $SERVER112_GL_PRELOAD"
echo "server108 GL    : preload $SERVER108_GL_PRELOAD"
echo "Python          : explicit \$CONDA_PREFIX/bin/python"
echo "Scheduling      : three-node dynamic shared queue"
echo "Initial queue   : run_01 run_02 run_03 run_04 run_05"
echo

TOTAL_WORKERS=0
for spec in "$LOCAL_GPU_SPEC" "$SERVER112_GPU_SPEC" "$SERVER108_GPU_SPEC"; do
    if [[ -n "$spec" ]]; then
        normalized="${spec//,/ }"
        read -r -a tmp_gpus <<< "$normalized"
        TOTAL_WORKERS=$((TOTAL_WORKERS + ${#tmp_gpus[@]}))
    fi
done

echo "Requested workers: $TOTAL_WORKERS"
if (( TOTAL_WORKERS > N_REPEATS )); then
    echo "NOTE: extra workers will see an empty queue and exit without loading the model."
fi

###############################################################################
# Remote launch helpers
###############################################################################

REMOTE_PIDS=()
REMOTE_TAGS=()
REMOTE_LOGS=()
LOCAL_GROUP_PID=""
CONTROLLER_ABORTING=0

request_remote_abort() {
    local tag="$1"
    local host="$2"
    local pid_file="$NODE_STATUS_DIR/${tag}.pid"
    local remote_pid=""
    local remote_cmd

    if [[ -f "$pid_file" ]]; then
        read -r remote_pid < "$pid_file" || true
    fi

    remote_cmd=": > $(shell_quote "$ABORT_FILE"); "
    if [[ "$remote_pid" =~ ^[0-9]+$ ]]; then
        remote_cmd+="kill -TERM $remote_pid 2>/dev/null || true; "
    fi

    # Safety fallback: only match eval.py processes carrying this unique RUN_NAME.
    remote_cmd+="pkill -TERM -f -- $(shell_quote "--log-name $RUN_NAME/") 2>/dev/null || true"

    timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "$remote_cmd" \
        >/dev/null 2>&1 || true
}

controller_signal_abort() {
    local sig="${1:-INT}"
    local pid

    if (( CONTROLLER_ABORTING != 0 )); then
        echo >&2
        echo "Second interrupt received: sending SIGKILL to launcher descendants." >&2
        kill_descendants "$BASHPID" KILL
        exit 130
    fi
    CONTROLLER_ABORTING=1

    # A second Ctrl-C becomes an immediate hard kill while cleanup is running.
    trap 'echo "Second interrupt: SIGKILL" >&2; kill_descendants "$BASHPID" KILL; exit 130' INT TERM

    echo >&2
    echo "============================================================" >&2
    echo "Ctrl-C / $sig received: FORCE STOPPING this three-node run" >&2
    echo "RUN_NAME: $RUN_NAME" >&2
    echo "============================================================" >&2

    mkdir -p "$QUEUE_ROOT" 2>/dev/null || true
    : > "$ABORT_FILE" 2>/dev/null || true
    {
        echo "aborted=$(date)"
        echo "signal=$sig"
        echo "controller=$(hostname)"
        echo "run_name=$RUN_NAME"
    } > "$ABORTED_INFO" 2>/dev/null || true

    # Ask remote worker shells to terminate their complete process trees.
    if [[ -n "${SERVER112_GPU_SPEC:-}" ]]; then
        request_remote_abort "server112" "$SERVER112_HOST"
    fi
    if [[ -n "${SERVER108_GPU_SPEC:-}" ]]; then
        request_remote_abort "server108" "$SERVER108_HOST"
    fi

    # Stop controller-owned local/SSH children.
    if [[ -n "${LOCAL_GROUP_PID:-}" ]]; then
        kill -TERM "$LOCAL_GROUP_PID" 2>/dev/null || true
    fi
    for pid in "${REMOTE_PIDS[@]:-}"; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done

    kill_descendants "$BASHPID" TERM
    sleep 2
    kill_descendants "$BASHPID" KILL

    echo "Forced shutdown requested on all participating nodes." >&2
    echo "Aborted run metadata: $ABORTED_INFO" >&2
    exit 130
}

# From this point onward one Ctrl-C on server07 stops the entire distributed run.
trap 'controller_signal_abort INT' INT
trap 'controller_signal_abort TERM' TERM

launch_remote_group() {
    local tag="$1"
    local host="$2"
    local gpu_spec="$3"
    local data_root="$4"
    local xvfb_run="$5"
    local gl_preload="$6"
    local remote_cmd
    local launch_log
    local pid

    [[ -n "$gpu_spec" ]] || return 0

    launch_log="$RUN_ROOT/remote_${tag}_launcher.log"

    remote_cmd="source $(shell_quote "$REMOTE_CONDA_SH"); "
    remote_cmd+="conda activate $(shell_quote "$EXPECTED_ENV"); "
    remote_cmd+="cd $(shell_quote "$BRIDGEVLA_ROOT"); "
    remote_cmd+="SCRIPT_MODE=worker "
    remote_cmd+="WORKER_TAG=$(shell_quote "$tag") "
    remote_cmd+="WORKER_GPUS=$(shell_quote "$gpu_spec") "
    remote_cmd+="RUN_NAME=$(shell_quote "$RUN_NAME") "
    remote_cmd+="SHARED_EVAL_ROOT=$(shell_quote "$SHARED_EVAL_ROOT") "
    remote_cmd+="BRIDGEVLA_ROOT=$(shell_quote "$BRIDGEVLA_ROOT") "
    remote_cmd+="MODEL_NAME=$(shell_quote "$MODEL_NAME") "
    remote_cmd+="NODE_DATA_ROOT=$(shell_quote "$data_root") "
    remote_cmd+="MODEL_FOLDER=$(shell_quote "$data_root/VLA/BridgeVLA/checkpoints/bridgevla/rlbench") "
    remote_cmd+="EVAL_DATA=$(shell_quote "$data_root/VLA/BridgeVLA/datasets/rlbench/eval") "
    remote_cmd+="NODE_HF_HOME=$(shell_quote "$data_root/huggingface_cache") "
    remote_cmd+="NODE_HF_HUB_CACHE=$(shell_quote "$data_root/huggingface_cache/hub") "
    remote_cmd+="XVFB_RUN=$(shell_quote "$xvfb_run") "
    remote_cmd+="NODE_GL_PRELOAD=$(shell_quote "$gl_preload") "
    remote_cmd+="PMF_ENABLED=$(shell_quote "$PMF_ENABLED") "
    remote_cmd+="PMF_PRIOR_VAR=$(shell_quote "$PMF_PRIOR_VAR") "
    remote_cmd+="PMF_OBSERVATION_VAR=$(shell_quote "$PMF_OBSERVATION_VAR") "
    remote_cmd+="CPU_THREADS=$(shell_quote "$CPU_THREADS") "
    remote_cmd+="bash $(shell_quote "$SCRIPT_PATH")"

    log "Launching remote group $tag on $host: GPUs $gpu_spec"
    ssh "$host" "$remote_cmd" > "$launch_log" 2>&1 &
    pid=$!

    REMOTE_PIDS+=("$pid")
    REMOTE_TAGS+=("$tag")
    REMOTE_LOGS+=("$launch_log")
}

if ! command -v ssh >/dev/null 2>&1; then
    echo "ERROR: ssh command not found on controller."
    exit 2
fi

launch_remote_group \
    "server112" "$SERVER112_HOST" "$SERVER112_GPU_SPEC" \
    "$SERVER112_DATA_ROOT" "$SERVER112_XVFB_RUN" "$SERVER112_GL_PRELOAD"

launch_remote_group \
    "server108" "$SERVER108_HOST" "$SERVER108_GPU_SPEC" \
    "$SERVER108_DATA_ROOT" "$SERVER108_XVFB_RUN" "$SERVER108_GL_PRELOAD"

###############################################################################
# Start local group in the background, but keep it behind the START barrier
###############################################################################

LOCAL_GROUP_STATUS_FILE="$NODE_STATUS_DIR/server07.local_group_status"

if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    (
        trap 'worker_signal_abort INT' INT
        trap 'worker_signal_abort TERM' TERM
        while [[ ! -f "$START_FILE" ]]; do
            if [[ -f "$ABORT_FILE" ]]; then
                log "ABORT marker detected before local START; group exits."
                exit 130
            fi
            sleep 0.2
        done
        if [[ -f "$ABORT_FILE" ]]; then
            log "ABORT marker detected at local START; group exits."
            exit 130
        fi
        log "START barrier released. Launching local GPU workers: $LOCAL_GPU_SPEC"
        if launch_worker_group "$LOCAL_GPU_SPEC"; then
            echo 0 > "$LOCAL_GROUP_STATUS_FILE"
            exit 0
        else
            echo 1 > "$LOCAL_GROUP_STATUS_FILE"
            exit 1
        fi
    ) &
    LOCAL_GROUP_PID=$!
fi

###############################################################################
# Wait for every requested remote node to finish preflight
###############################################################################

wait_for_remote_ready() {
    local tag="$1"
    local pid="$2"
    local launch_log="$3"
    local waited=0
    local ready_marker="$NODE_STATUS_DIR/${tag}.ready"

    while [[ ! -f "$ready_marker" ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: remote group $tag exited before becoming ready."
            echo "Inspect: $launch_log"
            tail -n 100 "$launch_log" || true
            return 2
        fi

        if (( waited >= REMOTE_READY_TIMEOUT )); then
            echo "ERROR: remote group $tag did not become ready within ${REMOTE_READY_TIMEOUT}s."
            echo "Inspect: $launch_log"
            tail -n 100 "$launch_log" || true
            return 2
        fi

        sleep 1
        waited=$((waited + 1))
    done

    log "Remote node $tag preflight is ready."
}

for i in "${!REMOTE_PIDS[@]}"; do
    if ! wait_for_remote_ready "${REMOTE_TAGS[$i]}" "${REMOTE_PIDS[$i]}" "${REMOTE_LOGS[$i]}"; then
        echo "ERROR: preflight failed; aborting all launched worker groups." >&2
        controller_signal_abort TERM
    fi
done

###############################################################################
# Release the common START barrier
###############################################################################

if [[ -f "$ABORT_FILE" ]]; then
    echo "ERROR: ABORT marker exists; START barrier will not be released." >&2
    controller_signal_abort TERM
fi
: > "$START_FILE"
log "Released common START barrier for all requested nodes."

###############################################################################
# Wait for local and remote worker groups
###############################################################################

CONTROLLER_FAILED=0

if [[ -n "$LOCAL_GROUP_PID" ]]; then
    if wait "$LOCAL_GROUP_PID"; then
        log "[LOCAL GROUP DONE] server07"
    else
        log "[LOCAL GROUP FAILED] server07"
        CONTROLLER_FAILED=1
    fi
fi

for i in "${!REMOTE_PIDS[@]}"; do
    if wait "${REMOTE_PIDS[$i]}"; then
        log "[REMOTE GROUP DONE] ${REMOTE_TAGS[$i]}"
    else
        log "[REMOTE GROUP FAILED] ${REMOTE_TAGS[$i]}"
        CONTROLLER_FAILED=1
        echo
        echo "Last 100 lines from ${REMOTE_TAGS[$i]} launcher:"
        tail -n 100 "${REMOTE_LOGS[$i]}" || true
        echo
    fi
done

###############################################################################
# Build deterministic assignment log
###############################################################################

printf "time\tnode\tgpu\trepetition\n" > "$ASSIGNMENT_LOG"
for assignment in "$ASSIGNMENT_DIR"/run_*.tsv; do
    [[ -e "$assignment" ]] || continue
    cat "$assignment" >> "$ASSIGNMENT_LOG"
done

###############################################################################
# Aggregate the 5 complete repetitions
###############################################################################

aggregate_results "$RUN_ROOT"

SUMMARY_RC=$?

nvidia-smi > "$RUN_ROOT/nvidia_smi_${NODE_NAME}_controller_end.txt" 2>&1 || true

echo

if (( CONTROLLER_FAILED != 0 )); then
    echo "============================================================"
    echo "Evaluation finished with one or more FAILED worker groups."
    echo "Inspect:"
    echo "  $RUN_ROOT/run_*/stdout.log"
    echo "  $RUN_ROOT/remote_*_launcher.log"
    echo "============================================================"
    exit 1
fi

if (( SUMMARY_RC != 0 )); then
    echo "============================================================"
    echo "Evaluation workers finished, but the Table-1 summary is incomplete."
    echo "Inspect:"
    echo "  $RUN_ROOT"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "SUCCESS: $EVALUATION_VARIANT Table-1 three-node evaluation finished."
echo
echo "Shared results:"
echo "  $RUN_ROOT/table1_per_run.csv"
echo "  $RUN_ROOT/table1_summary.csv"
echo "  $RUN_ROOT/table1_summary.txt"
echo
echo "Dynamic assignment log:"
echo "  $ASSIGNMENT_LOG"
echo
echo "Run logs:"
echo "  $RUN_ROOT/run_*/stdout.log"
echo
echo "Full eval outputs remain on the node that executed each run."
echo "============================================================"
