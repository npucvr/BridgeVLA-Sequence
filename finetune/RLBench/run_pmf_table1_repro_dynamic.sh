#!/usr/bin/env bash

# BridgeVLA + Probabilistic Motion Filter
# RLBench Table-1 evaluation protocol:
#   18 tasks
#   25 trials / task
#   max 25 action steps / trial
#   5 complete evaluation repetitions
#
# Multi-node dynamic scheduling:
#   - server07 runs this script in controller mode.
#   - controller launches remote workers through SSH.
#   - every physical GPU owns one worker loop.
#   - repetitions are claimed atomically from a queue on shared /remote_userdata.
#   - model/checkpoint/RLBench data/HF cache remain local under each node's /data2.
#   - eval.py writes locally; the small result CSV is copied back to shared RUN_ROOT.
#
# Recommended controller usage on server07:
#   LOCAL_GPUS="0 1" REMOTE_GPUS="0 1" \
#   bash finetune/RLBench/run_pmf_table1_repro_multinode.sh
#
# Backward-compatible local GPU syntax:
#   REMOTE_GPUS="0 1" \
#   bash finetune/RLBench/run_pmf_table1_repro_multinode.sh 0 1
#
# Optional:
#   RUN_NAME=my_pmf_eval \
#   REMOTE_HOST=server112 \
#   PMF_PRIOR_VAR=9e-4 \
#   PMF_OBSERVATION_VAR=1e-4 \
#   LOCAL_GPUS="0 1" \
#   REMOTE_GPUS="0 1 2" \
#   bash finetune/RLBench/run_pmf_table1_repro_multinode.sh

set -uo pipefail

###############################################################################
# Fixed paper evaluation protocol
###############################################################################

EXPECTED_ENV="bridgevla_rlbench"
EXPECTED_BRANCH="lizhe/prob-motion-filter"
REVIEWED_COMMIT="f6cb28c8bfffc77808f2829da11f2e54b2ee9ed3"

BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Sequence}"

MODEL_FOLDER="${MODEL_FOLDER:-/data2/local_userdata/lizhe/VLA/BridgeVLA/checkpoints/bridgevla/rlbench}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"

EVAL_DATA="${EVAL_DATA:-/data2/local_userdata/lizhe/VLA/BridgeVLA/datasets/rlbench/eval}"

N_REPEATS=5
START_EPISODE=0
EVAL_EPISODES=25
EPISODE_LENGTH=25

###############################################################################
# PMF / runtime configuration
###############################################################################

PMF_PRIOR_VAR="${PMF_PRIOR_VAR:-9e-4}"
PMF_OBSERVATION_VAR="${PMF_OBSERVATION_VAR:-1e-4}"
CPU_THREADS="${CPU_THREADS:-4}"

RUN_NAME="${RUN_NAME:-pmf_table1_$(date +%Y%m%d_%H%M%S)}"

# Shared coordination/results location. This path MUST be visible on both nodes.
SHARED_EVAL_ROOT="${SHARED_EVAL_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/multinode_eval}"

# SSH alias configured on server07.
REMOTE_HOST="${REMOTE_HOST:-server112}"
REMOTE_CONDA_SH="${REMOTE_CONDA_SH:-/home/lizhe/miniconda3/etc/profile.d/conda.sh}"
REMOTE_READY_TIMEOUT="${REMOTE_READY_TIMEOUT:-120}"

# Internal mode. Users normally run only controller mode.
SCRIPT_MODE="${SCRIPT_MODE:-controller}"

# Make HF cache deterministic on both nodes.
export HF_HOME="${HF_HOME:-/data2/local_userdata/lizhe/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

RLBENCH_DIR="$BRIDGEVLA_ROOT/finetune/RLBench"
EVAL_PY="$RLBENCH_DIR/eval.py"
SCRIPT_PATH="$RLBENCH_DIR/$(basename "$0")"

MODEL_STEM="${MODEL_NAME%.pth}"

# eval.py always writes below MODEL_FOLDER/eval, which is local /data2 on each node.
LOCAL_RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"

# Queue, metadata, logs, and copied CSVs live on shared storage.
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

NODE_NAME="$(hostname -s 2>/dev/null || hostname)"
NODE_NAME="${NODE_NAME//[^A-Za-z0-9_.-]/_}"

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
# Helpers
###############################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$NODE_NAME] $*"
}

shell_quote() {
    printf '%q' "$1"
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
    python - "$PMF_PRIOR_VAR" "$PMF_OBSERVATION_VAR" <<'PY'
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

    python - <<'PY'
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
        "ERROR: libtensorflow is loaded. "
        "The RLBench/CoppeliaSim OpenSSL conflict may return."
    )

print("libtensorflow  : NOT loaded")
print("Preflight      : OK")
PY
}

validate_node_environment() {
    local gpu_spec="$1"
    local f

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first on $NODE_NAME."
        echo "Current env: ${CONDA_DEFAULT_ENV:-<none>}"
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

    if [[ "$CURRENT_COMMIT" != "$REVIEWED_COMMIT" ]]; then
        echo "WARNING: current commit differs from reviewed PMF V1 commit."
        echo "Reviewed : $REVIEWED_COMMIT"
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
        echo "ERROR: CoppeliaSim executable not found: $COPPELIASIM_ROOT/coppeliaSim"
        return 2
    fi

    if [[ -e "$LOCAL_RUN_ROOT" ]]; then
        echo "ERROR: local eval directory already exists on $NODE_NAME:"
        echo "  $LOCAL_RUN_ROOT"
        echo "Choose a new RUN_NAME to prevent eval_results.csv from being appended."
        return 2
    fi

    validate_pmf_parameters || return 2
    parse_gpu_spec "$gpu_spec" || return 2
    validate_eval_data || return 2
    bridgevla_preflight || return 2

    return 0
}

write_node_metadata() {
    mkdir -p "$NODE_STATUS_DIR"

    {
        echo "node=$NODE_NAME"
        echo "date=$(date)"
        echo "hostname=$(hostname)"
        echo "conda_env=${CONDA_DEFAULT_ENV:-}"
        echo "bridgevla_root=$BRIDGEVLA_ROOT"
        echo "git_branch=${CURRENT_BRANCH:-}"
        echo "git_commit=${CURRENT_COMMIT:-}"
        echo "model=$MODEL_FOLDER/$MODEL_NAME"
        echo "eval_data=$EVAL_DATA"
        echo "local_run_root=$LOCAL_RUN_ROOT"
        echo "shared_run_root=$RUN_ROOT"
        echo "hf_home=$HF_HOME"
        echo "hf_hub_cache=$HF_HUB_CACHE"
        echo "coppeliasim_root=${COPPELIASIM_ROOT:-}"
    } > "$RUN_ROOT/node_${NODE_NAME}.txt"

    python -m pip freeze > "$RUN_ROOT/pip_freeze_${NODE_NAME}.txt" 2>/dev/null || true
    nvidia-smi > "$RUN_ROOT/nvidia_smi_${NODE_NAME}_start.txt" 2>&1 || true
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

    # stdout from this function is machine-readable by worker_loop.
    for ((rep = 1; rep <= N_REPEATS; rep++)); do
        rep_name="$(printf 'run_%02d' "$rep")"
        pending="$PENDING_DIR/${rep_name}.job"
        running="$RUNNING_DIR/${rep_name}.${NODE_NAME}.gpu${gpu}.job"

        # rename within the same shared filesystem is atomic. Exactly one
        # worker, even on another server, can move a pending job successfully.
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
    local rc

    rep_name="$(printf 'run_%02d' "$rep")"
    shared_rep_dir="$RUN_ROOT/$rep_name"
    local_rep_dir="$LOCAL_RUN_ROOT/$rep_name"

    mkdir -p "$shared_rep_dir"

    {
        echo "repetition=$rep"
        echo "node=$NODE_NAME"
        echo "gpu=$gpu"
        echo "start=$(date)"
        echo "local_eval_dir=$local_rep_dir"
        echo "shared_rep_dir=$shared_rep_dir"
    } > "$shared_rep_dir/meta.txt"

    log "START $rep_name on physical GPU $gpu"

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
                --headless \
                --pmf-enabled \
                --pmf-prior-var "$PMF_PRIOR_VAR" \
                --pmf-observation-var "$PMF_OBSERVATION_VAR"

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
            echo "Last 40 lines from failed $rep_name:"
            tail -n 40 "$RUN_ROOT/$rep_name/stdout.log" || true
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
# Remote worker mode (started automatically by controller through SSH)
###############################################################################

run_remote_worker_mode() {
    local gpu_spec="${WORKER_GPUS:-}"

    if [[ -z "$gpu_spec" ]]; then
        echo "ERROR: WORKER_GPUS is empty in worker mode."
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

    while [[ ! -f "$READY_FILE" ]]; do
        sleep 1
    done

    validate_node_environment "$gpu_spec" || exit 2
    write_node_metadata

    printf "node=%s\ngpus=%s\nready=%s\n" \
        "$NODE_NAME" "$gpu_spec" "$(date)" \
        > "$NODE_STATUS_DIR/remote.ready"

    log "Remote node preflight passed; waiting for controller START barrier."

    while [[ ! -f "$START_FILE" ]]; do
        sleep 1
    done

    log "START barrier released. Launching remote GPU workers: $gpu_spec"

    REMOTE_FAILED=0
    launch_worker_group "$gpu_spec" || REMOTE_FAILED=1

    nvidia-smi > "$RUN_ROOT/nvidia_smi_${NODE_NAME}_end.txt" 2>&1 || true

    if (( REMOTE_FAILED != 0 )); then
        exit 1
    fi

    exit 0
}

if [[ "$SCRIPT_MODE" == "worker" ]]; then
    run_remote_worker_mode
fi

if [[ "$SCRIPT_MODE" != "controller" ]]; then
    echo "ERROR: invalid SCRIPT_MODE=$SCRIPT_MODE"
    exit 2
fi

###############################################################################
# Controller usage / GPU selection
###############################################################################

if [[ -n "${LOCAL_GPUS:-}" ]]; then
    LOCAL_GPU_SPEC="$LOCAL_GPUS"
elif (( $# > 0 )); then
    LOCAL_GPU_SPEC="$*"
else
    LOCAL_GPU_SPEC=""
fi

REMOTE_GPU_SPEC="${REMOTE_GPUS:-}"

if [[ -z "$LOCAL_GPU_SPEC" && -z "$REMOTE_GPU_SPEC" ]]; then
    echo "Usage on controller ($NODE_NAME):"
    echo "  LOCAL_GPUS=\"0 1\" REMOTE_GPUS=\"0 1\" $0"
    echo
    echo "or keep positional local GPUs:"
    echo "  REMOTE_GPUS=\"0 1\" $0 0 1"
    echo
    echo "REMOTE_HOST defaults to: $REMOTE_HOST"
    exit 2
fi

###############################################################################
# Controller preflight
###############################################################################

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

if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    validate_node_environment "$LOCAL_GPU_SPEC" || exit 2
else
    # Controller still needs the conda/code environment for final aggregation.
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first."
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
# Initialize shared run directory / queue exactly once on controller
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
    echo "BridgeVLA + PMF RLBench Table-1 multi-node evaluation"
    echo
    echo "date=$(date)"
    echo "controller_hostname=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "bridgevla_root=$BRIDGEVLA_ROOT"
    echo "git_branch=${CURRENT_BRANCH:-}"
    echo "git_commit=${CURRENT_COMMIT:-}"
    echo "model=$MODEL_FOLDER/$MODEL_NAME"
    echo "eval_data=$EVAL_DATA"
    echo "n_repeats=$N_REPEATS"
    echo "tasks=18"
    echo "eval_episodes=$EVAL_EPISODES"
    echo "start_episode=$START_EPISODE"
    echo "episode_length=$EPISODE_LENGTH"
    echo "pmf_enabled=true"
    echo "pmf_prior_var=$PMF_PRIOR_VAR"
    echo "pmf_observation_var=$PMF_OBSERVATION_VAR"
    echo "local_gpus=$LOCAL_GPU_SPEC"
    echo "remote_host=$REMOTE_HOST"
    echo "remote_gpus=$REMOTE_GPU_SPEC"
    echo "shared_run_root=$RUN_ROOT"
    echo "local_eval_root=$LOCAL_RUN_ROOT"
    echo "scheduling=multi_node_atomic_shared_queue"
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
echo "BridgeVLA + PMF RLBench Table-1 multi-node evaluation"
echo "============================================================"
echo "Shared run root : $RUN_ROOT"
echo "Model           : $MODEL_NAME"
echo "Tasks/run       : 18"
echo "Episodes/task   : 25"
echo "Steps/episode   : 25 max"
echo "Repetitions     : 5"
echo "Total trials    : 2250"
echo "PMF prior var   : $PMF_PRIOR_VAR"
echo "PMF obs var     : $PMF_OBSERVATION_VAR"
echo "Controller      : $NODE_NAME"
echo "Local GPUs      : ${LOCAL_GPU_SPEC:-<none>}"
echo "Remote host     : ${REMOTE_HOST:-<none>}"
echo "Remote GPUs     : ${REMOTE_GPU_SPEC:-<none>}"
echo "Scheduling      : multi-node dynamic shared queue"
echo "Initial queue   : run_01 run_02 run_03 run_04 run_05"
echo

TOTAL_WORKERS=0
if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    parse_gpu_spec "$LOCAL_GPU_SPEC" || exit 2
    TOTAL_WORKERS=$((TOTAL_WORKERS + ${#PARSED_GPUS[@]}))
fi
if [[ -n "$REMOTE_GPU_SPEC" ]]; then
    tmp_remote_spec="${REMOTE_GPU_SPEC//,/ }"
    read -r -a tmp_remote_gpus <<< "$tmp_remote_spec"
    TOTAL_WORKERS=$((TOTAL_WORKERS + ${#tmp_remote_gpus[@]}))
fi

if (( TOTAL_WORKERS > N_REPEATS )); then
    echo "NOTE: $TOTAL_WORKERS workers requested for only $N_REPEATS repetitions."
    echo "Extra workers will see an empty queue and exit without loading the model."
    echo
fi

###############################################################################
# Launch remote worker group first, then release a common START barrier
###############################################################################

REMOTE_SSH_PID=""
REMOTE_LAUNCH_LOG="$RUN_ROOT/remote_${REMOTE_HOST}_launcher.log"
CONTROLLER_FAILED=0

if [[ -n "$REMOTE_GPU_SPEC" ]]; then
    if ! command -v ssh >/dev/null 2>&1; then
        echo "ERROR: ssh command not found on controller."
        exit 2
    fi

    remote_cmd="source $(shell_quote "$REMOTE_CONDA_SH"); "
    remote_cmd+="conda activate $(shell_quote "$EXPECTED_ENV"); "
    remote_cmd+="cd $(shell_quote "$BRIDGEVLA_ROOT"); "
    remote_cmd+="SCRIPT_MODE=worker "
    remote_cmd+="WORKER_GPUS=$(shell_quote "$REMOTE_GPU_SPEC") "
    remote_cmd+="RUN_NAME=$(shell_quote "$RUN_NAME") "
    remote_cmd+="SHARED_EVAL_ROOT=$(shell_quote "$SHARED_EVAL_ROOT") "
    remote_cmd+="BRIDGEVLA_ROOT=$(shell_quote "$BRIDGEVLA_ROOT") "
    remote_cmd+="MODEL_FOLDER=$(shell_quote "$MODEL_FOLDER") "
    remote_cmd+="MODEL_NAME=$(shell_quote "$MODEL_NAME") "
    remote_cmd+="EVAL_DATA=$(shell_quote "$EVAL_DATA") "
    remote_cmd+="PMF_PRIOR_VAR=$(shell_quote "$PMF_PRIOR_VAR") "
    remote_cmd+="PMF_OBSERVATION_VAR=$(shell_quote "$PMF_OBSERVATION_VAR") "
    remote_cmd+="CPU_THREADS=$(shell_quote "$CPU_THREADS") "
    remote_cmd+="HF_HOME=$(shell_quote "$HF_HOME") "
    remote_cmd+="HF_HUB_CACHE=$(shell_quote "$HF_HUB_CACHE") "
    remote_cmd+="bash $(shell_quote "$SCRIPT_PATH")"

    log "Launching remote worker group on $REMOTE_HOST: GPUs $REMOTE_GPU_SPEC"
    ssh "$REMOTE_HOST" "$remote_cmd" > "$REMOTE_LAUNCH_LOG" 2>&1 &
    REMOTE_SSH_PID=$!

    # Do not let local workers claim the queue before the remote node has
    # completed its preflight; this gives both nodes a fair start.
    ready_waited=0
    while [[ ! -f "$NODE_STATUS_DIR/remote.ready" ]]; do
        if ! kill -0 "$REMOTE_SSH_PID" 2>/dev/null; then
            echo "ERROR: remote launcher exited before becoming ready."
            echo "Inspect: $REMOTE_LAUNCH_LOG"
            tail -n 80 "$REMOTE_LAUNCH_LOG" || true
            exit 2
        fi

        if (( ready_waited >= REMOTE_READY_TIMEOUT )); then
            echo "ERROR: remote node did not become ready within ${REMOTE_READY_TIMEOUT}s."
            echo "Inspect: $REMOTE_LAUNCH_LOG"
            tail -n 80 "$REMOTE_LAUNCH_LOG" || true
            kill "$REMOTE_SSH_PID" 2>/dev/null || true
            exit 2
        fi

        sleep 1
        ready_waited=$((ready_waited + 1))
    done

    log "Remote node preflight is ready."
fi

: > "$START_FILE"
log "Released common START barrier."

###############################################################################
# Launch controller-local worker group
###############################################################################

LOCAL_FAILED=0

if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    echo
    echo "============================================================"
    echo "Launching local evaluation workers on $NODE_NAME"
    echo "============================================================"

    launch_worker_group "$LOCAL_GPU_SPEC" || LOCAL_FAILED=1
fi

###############################################################################
# Wait for remote worker group
###############################################################################

REMOTE_FAILED=0

if [[ -n "$REMOTE_SSH_PID" ]]; then
    if wait "$REMOTE_SSH_PID"; then
        log "[REMOTE GROUP DONE] $REMOTE_HOST"
    else
        log "[REMOTE GROUP FAILED] $REMOTE_HOST"
        REMOTE_FAILED=1
        echo
        echo "Last 80 lines from remote launcher:"
        tail -n 80 "$REMOTE_LAUNCH_LOG" || true
        echo
    fi
fi

if (( LOCAL_FAILED != 0 || REMOTE_FAILED != 0 )); then
    CONTROLLER_FAILED=1
fi

###############################################################################
# Build deterministic assignment log from one-file-per-repetition records
###############################################################################

printf "time\tnode\tgpu\trepetition\n" > "$ASSIGNMENT_LOG"
for assignment in "$ASSIGNMENT_DIR"/run_*.tsv; do
    [[ -e "$assignment" ]] || continue
    cat "$assignment" >> "$ASSIGNMENT_LOG"
done

###############################################################################
# Aggregate the 5 complete repetitions
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
                errors.append(f"{run_name}: duplicate task {task}")
                continue

            try:
                value = float(value)
            except ValueError:
                errors.append(
                    f"{run_name}: invalid success rate for {task}: {value!r}"
                )
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
    print(
        f"ERROR: only {len(all_runs)}/{n_runs} complete repetitions are available."
    )
    sys.exit(2)

per_run_csv = os.path.join(run_root, "table1_per_run.csv")
summary_csv = os.path.join(run_root, "table1_summary.csv")
summary_txt = os.path.join(run_root, "table1_summary.txt")

with open(per_run_csv, "w", newline="") as f:
    fieldnames = [
        "task",
        "run_01",
        "run_02",
        "run_03",
        "run_04",
        "run_05",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for task in tasks:
        row = {"task": task}

        for run_idx in range(1, n_runs + 1):
            row[f"run_{run_idx:02d}"] = all_runs[run_idx][task]

        writer.writerow(row)

summary_rows = []

for task in tasks:
    values = [
        all_runs[run_idx][task]
        for run_idx in range(1, n_runs + 1)
    ]

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
    run_idx: statistics.mean(
        all_runs[run_idx][task]
        for task in tasks
    )
    for run_idx in range(1, n_runs + 1)
}

overall_avg_sr = statistics.mean(
    row["mean_success_rate"]
    for row in summary_rows
)

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
        f"{row['mean_success_rate']:6.1f} +/- "
        f"{row['std_success_rate']:.1f}"
    )

lines.append("-" * len(header))

for run_idx in range(1, n_runs + 1):
    lines.append(
        f"Run {run_idx:02d} Avg. SR: {run_avgs[run_idx]:.2f}%"
    )

lines.append("")
lines.append(
    f"TABLE-1 Avg. SR (18-task mean): {overall_avg_sr:.2f}%"
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

nvidia-smi > "$RUN_ROOT/nvidia_smi_${NODE_NAME}_end.txt" 2>&1 || true

echo

if (( CONTROLLER_FAILED != 0 )); then
    echo "============================================================"
    echo "Evaluation finished with one or more FAILED worker groups."
    echo "Inspect:"
    echo "  $RUN_ROOT/run_*/stdout.log"
    echo "  $REMOTE_LAUNCH_LOG"
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
echo "SUCCESS: BridgeVLA + PMF Table-1 multi-node evaluation finished."
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
echo "Local full eval outputs remain on the node that executed each run:"
echo "  $MODEL_FOLDER/eval/$RUN_NAME/run_XX"
echo "============================================================"
