#!/usr/bin/env bash

# BridgeVLA PMF diagnostics, three-node dynamic (mode, task) queue.
# Runtime/GL policy mirrors run_pmf_table1_3node_stable_glfix.sh.

set -uo pipefail

if [[ "${SCRIPT_MODE:-controller}" == "controller" ]]; then
    if ! bash -n "$0"; then
        echo "ERROR: launcher syntax check failed: $0" >&2
        exit 2
    fi
fi

EXPECTED_ENV="bridgevla_rlbench"
EXPECTED_BRANCH="lizhe/prob-motion-filter"
REFERENCE_COMMIT="${REFERENCE_COMMIT:-a5b64dc132175f8517cdccc0570747c9da227749}"

BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Sequence}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"
START_EPISODE="${START_EPISODE:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"

RLBENCH_TASKS=(
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

DIAG_TASKS="${DIAG_TASKS:-insert_onto_square_peg stack_blocks put_item_in_drawer light_bulb_in push_buttons close_jar}"
DIAG_MODES="${DIAG_MODES:-shadow pmf}"
PMF_PRIOR_VAR="${PMF_PRIOR_VAR:-9e-4}"
PMF_OBSERVATION_VAR="${PMF_OBSERVATION_VAR:-1e-4}"
CPU_THREADS="${CPU_THREADS:-4}"

RUN_NAME="${RUN_NAME:-pmf_diagnostics_3node_glfix_$(date +%Y%m%d_%H%M%S)}"
SHARED_EVAL_ROOT="${SHARED_EVAL_ROOT:-/remote_userdata/lizhe/VLA/BridgeVLA/multinode_eval}"
REMOTE_CONDA_SH="${REMOTE_CONDA_SH:-/home/lizhe/miniconda3/etc/profile.d/conda.sh}"
REMOTE_READY_TIMEOUT="${REMOTE_READY_TIMEOUT:-180}"

SERVER112_HOST="${SERVER112_HOST:-server112}"
SERVER108_HOST="${SERVER108_HOST:-server108}"
SERVER07_DATA_ROOT="${SERVER07_DATA_ROOT:-/data2/local_userdata/lizhe}"
SERVER112_DATA_ROOT="${SERVER112_DATA_ROOT:-/data2/local_userdata/lizhe}"
SERVER108_DATA_ROOT="${SERVER108_DATA_ROOT:-/data/local_userdata/lizhe}"
SERVER07_XVFB_RUN="${SERVER07_XVFB_RUN:-/usr/bin/xvfb-run}"
SERVER112_XVFB_RUN="${SERVER112_XVFB_RUN:-/data2/local_userdata/lizhe/tools/xvfb_jammy/usr/bin/xvfb-run}"
SERVER108_XVFB_RUN="${SERVER108_XVFB_RUN:-/data/local_userdata/lizhe/tools/xvfb_jammy/usr/bin/xvfb-run}"
SERVER07_GL_PRELOAD="${SERVER07_GL_PRELOAD:-}"
SERVER112_GL_PRELOAD="${SERVER112_GL_PRELOAD:-/home/lizhe/miniconda3/envs/bridgevla_rlbench/lib/libGL.so.1}"
SERVER108_GL_PRELOAD="${SERVER108_GL_PRELOAD:-/home/lizhe/miniconda3/envs/bridgevla_rlbench/lib/libGL.so.1}"

SCRIPT_MODE="${SCRIPT_MODE:-controller}"
WORKER_TAG="${WORKER_TAG:-local}"
NODE_DATA_ROOT="${NODE_DATA_ROOT:-$SERVER07_DATA_ROOT}"
XVFB_RUN="${XVFB_RUN:-$SERVER07_XVFB_RUN}"
NODE_GL_PRELOAD="${NODE_GL_PRELOAD:-$SERVER07_GL_PRELOAD}"
MODEL_FOLDER="${MODEL_FOLDER:-$NODE_DATA_ROOT/VLA/BridgeVLA/checkpoints/bridgevla/rlbench}"
EVAL_DATA="${EVAL_DATA:-$NODE_DATA_ROOT/VLA/BridgeVLA/datasets/rlbench/eval}"

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
ANALYZER_PY="$RLBENCH_DIR/analyze_pmf_diagnostics.py"
SCRIPT_PATH="$RLBENCH_DIR/$(basename "$0")"
MODEL_STEM="${MODEL_NAME%.pth}"
LOCAL_RUN_ROOT="$MODEL_FOLDER/eval/$RUN_NAME"

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
ABORT_FILE="$QUEUE_ROOT/ABORT"
ABORTED_INFO="$RUN_ROOT/ABORTED.txt"
ASSIGNMENT_LOG="$RUN_ROOT/assignments.tsv"

NODE_NAME="$(hostname -s 2>/dev/null || hostname)"
NODE_NAME="${NODE_NAME//[^A-Za-z0-9_.-]/_}"
WORKER_TAG="${WORKER_TAG//[^A-Za-z0-9_.-]/_}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$NODE_NAME] $*"
}

shell_quote() {
    printf '%q' "$1"
}

validate_diagnostics_config() {
    local value
    declare -A allowed_tasks=()
    declare -A seen_tasks=()
    declare -A seen_modes=()

    if [[ ! "$START_EPISODE" =~ ^[0-9]+$ ]]; then
        echo "ERROR: START_EPISODE must be a non-negative integer."
        return 2
    fi
    if [[ ! "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: EVAL_EPISODES must be a positive integer."
        return 2
    fi
    if [[ ! "$EPISODE_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: EPISODE_LENGTH must be a positive integer."
        return 2
    fi

    for value in "${RLBENCH_TASKS[@]}"; do
        allowed_tasks["$value"]=1
    done
    read -r -a DIAG_TASK_ARRAY <<< "$DIAG_TASKS"
    if (( ${#DIAG_TASK_ARRAY[@]} == 0 )); then
        echo "ERROR: DIAG_TASKS must not be empty."
        return 2
    fi
    for value in "${DIAG_TASK_ARRAY[@]}"; do
        if [[ -z "${allowed_tasks[$value]:-}" ]]; then
            echo "ERROR: diagnostics task is not in RLBENCH_TASKS: $value"
            return 2
        fi
        if [[ -n "${seen_tasks[$value]:-}" ]]; then
            echo "ERROR: duplicate diagnostics task: $value"
            return 2
        fi
        seen_tasks["$value"]=1
    done

    read -r -a DIAG_MODE_ARRAY <<< "$DIAG_MODES"
    if (( ${#DIAG_MODE_ARRAY[@]} == 0 )); then
        echo "ERROR: DIAG_MODES must not be empty."
        return 2
    fi
    for value in "${DIAG_MODE_ARRAY[@]}"; do
        if [[ "$value" != "shadow" && "$value" != "pmf" ]]; then
            echo "ERROR: diagnostics mode must be shadow or pmf: $value"
            return 2
        fi
        if [[ -n "${seen_modes[$value]:-}" ]]; then
            echo "ERROR: duplicate diagnostics mode: $value"
            return 2
        fi
        seen_modes["$value"]=1
    done

    TOTAL_JOBS=$((${#DIAG_TASK_ARRAY[@]} * ${#DIAG_MODE_ARRAY[@]}))
}

validate_diagnostics_config || exit 2

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
print(f"PMF prior variance       : {q:g}")
print(f"PMF observation variance : {r:g}")
print(f"PMF gain K               : {q / (q + r):.8f}")
PY
}

validate_eval_data() {
    local task
    local ep_root
    local ep
    local failed=0
    echo
    echo "Checking RLBench diagnostics data on $NODE_NAME"
    for task in "${DIAG_TASK_ARRAY[@]}"; do
        ep_root="$EVAL_DATA/$task/all_variations/episodes"
        if [[ ! -d "$ep_root" ]]; then
            echo "MISSING: $ep_root"
            failed=1
            continue
        fi
        for ((ep = START_EPISODE; ep < START_EPISODE + EVAL_EPISODES; ep++)); do
            if [[ ! -d "$ep_root/episode$ep" ]]; then
                echo "MISSING: $ep_root/episode$ep"
                failed=1
            fi
        done
    done
    return "$failed"
}

hf_offline_preflight() {
    "$CONDA_PREFIX/bin/python" - <<'PY'
from huggingface_hub import constants
from transformers import AutoConfig
from transformers.utils import hub
if not constants.HF_HUB_OFFLINE:
    raise SystemExit("ERROR: huggingface_hub offline mode is not active")
if not hub.is_offline_mode():
    raise SystemExit("ERROR: transformers offline mode is not active")
AutoConfig.from_pretrained("google/paligemma-3b-pt-224", local_files_only=True)
print("Hugging Face strict offline preflight: OK")
PY
}

xvfb_preflight() {
    local xvfb_bin_dir
    local test_log
    xvfb_bin_dir="$(dirname "$XVFB_RUN")"
    if [[ ! -x "$XVFB_RUN" || ! -x "$xvfb_bin_dir/Xvfb" ]]; then
        echo "ERROR: native xvfb-run/Xvfb is missing: $XVFB_RUN"
        return 2
    fi
    if [[ ! -x "$xvfb_bin_dir/xauth" && ! -x /usr/bin/xauth ]]; then
        echo "ERROR: xauth is not available for xvfb-run."
        return 2
    fi
    test_log="/tmp/bridgevla_diag_xvfb_${USER}_${NODE_NAME}_$$.log"
    PATH="$xvfb_bin_dir:$PATH:/usr/bin:/bin" timeout 15 "$XVFB_RUN" \
        -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
        bash -c 'test -n "$DISPLAY"' > "$test_log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        cat "$test_log" || true
        rm -f "$test_log"
        return 2
    fi
    rm -f "$test_log"
    echo "Native Xvfb preflight: OK"
}

bridgevla_preflight() {
    "$CONDA_PREFIX/bin/python" - <<'PY'
import os
import bridgevla
root = os.path.realpath(os.environ["BRIDGEVLA_ROOT"])
package = os.path.realpath(bridgevla.__file__)
if not package.startswith(root + os.sep):
    raise SystemExit("ERROR: bridgevla is not imported from BRIDGEVLA_ROOT")
from bridgevla.utils.rvt_utils import TensorboardManager
with open("/proc/self/maps") as maps_file:
    if "libtensorflow" in maps_file.read():
        raise SystemExit("ERROR: libtensorflow is loaded")
print("BridgeVLA preflight: OK")
PY
}

validate_node_environment() {
    local gpu_spec="$1"
    local config
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first on $NODE_NAME."
        return 2
    fi
    if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "ERROR: explicit Conda Python is missing: $CONDA_PREFIX/bin/python"
        return 2
    fi
    if [[ -n "$NODE_GL_PRELOAD" && ! -r "$NODE_GL_PRELOAD" ]]; then
        echo "ERROR: GL preload is missing: $NODE_GL_PRELOAD"
        return 2
    fi
    CURRENT_BRANCH="$(git -C "$BRIDGEVLA_ROOT" branch --show-current 2>/dev/null || true)"
    CURRENT_COMMIT="$(git -C "$BRIDGEVLA_ROOT" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
        echo "ERROR: wrong branch on $NODE_NAME: $CURRENT_BRANCH"
        return 2
    fi
    if [[ "$CURRENT_COMMIT" != "$REFERENCE_COMMIT" ]]; then
        echo "WARNING: current commit differs from diagnostics reference commit."
        echo "Reference: $REFERENCE_COMMIT"
        echo "Current  : $CURRENT_COMMIT"
    fi
    for config in "$EVAL_PY" "$ANALYZER_PY" "$MODEL_FOLDER/$MODEL_NAME" \
        "$MODEL_FOLDER/exp_cfg.yaml" "$MODEL_FOLDER/mvt_cfg.yaml"; do
        if [[ ! -f "$config" ]]; then
            echo "ERROR: required file is missing: $config"
            return 2
        fi
    done
    if [[ ! -d "$EVAL_DATA" ]]; then
        echo "ERROR: eval data not found: $EVAL_DATA"
        return 2
    fi
    if [[ -z "${COPPELIASIM_ROOT:-}" || ! -x "$COPPELIASIM_ROOT/coppeliaSim" ]]; then
        echo "ERROR: valid COPPELIASIM_ROOT is required on $NODE_NAME."
        return 2
    fi
    if [[ -e "$LOCAL_RUN_ROOT" ]]; then
        echo "ERROR: local diagnostics directory already exists: $LOCAL_RUN_ROOT"
        return 2
    fi
    validate_pmf_parameters || return 2
    parse_gpu_spec "$gpu_spec" || return 2
    validate_eval_data || return 2
    hf_offline_preflight || return 2
    xvfb_preflight || return 2
    bridgevla_preflight || return 2
}

write_node_metadata() {
    mkdir -p "$NODE_STATUS_DIR"
    {
        echo "worker_tag=$WORKER_TAG"
        echo "node=$NODE_NAME"
        echo "date=$(date)"
        echo "git_branch=${CURRENT_BRANCH:-}"
        echo "git_commit=${CURRENT_COMMIT:-}"
        echo "node_data_root=$NODE_DATA_ROOT"
        echo "model=$MODEL_FOLDER/$MODEL_NAME"
        echo "eval_data=$EVAL_DATA"
        echo "local_run_root=$LOCAL_RUN_ROOT"
        echo "shared_run_root=$RUN_ROOT"
        echo "hf_home=$HF_HOME"
        echo "hf_hub_cache=$HF_HUB_CACHE"
        echo "xvfb_run=$XVFB_RUN"
        echo "gl_preload=${NODE_GL_PRELOAD:-<none>}"
        echo "conda_python=$CONDA_PREFIX/bin/python"
    } > "$RUN_ROOT/node_${WORKER_TAG}_${NODE_NAME}.txt"
    "$CONDA_PREFIX/bin/python" -m pip freeze > "$RUN_ROOT/pip_freeze_${WORKER_TAG}_${NODE_NAME}.txt" 2>/dev/null || true
    nvidia-smi > "$RUN_ROOT/nvidia_smi_${WORKER_TAG}_${NODE_NAME}_start.txt" 2>&1 || true
}

claim_next_job() {
    local gpu="$1"
    local pending
    local job_id
    local running
    local mode
    local task
    local assignment
    for pending in "$PENDING_DIR"/job_*.job; do
        [[ -f "$pending" ]] || continue
        job_id="$(basename "$pending" .job)"
        running="$RUNNING_DIR/${job_id}.${NODE_NAME}.gpu${gpu}.job"
        if mv "$pending" "$running" 2>/dev/null; then
            IFS=$'\t' read -r mode task < "$running"
            assignment="$ASSIGNMENT_DIR/${job_id}.tsv"
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date '+%Y-%m-%d %H:%M:%S')" "$NODE_NAME" "$gpu" \
                "$job_id" "$mode" "$task" > "$assignment"
            printf "%s|%s|%s|%s\n" "$job_id" "$mode" "$task" "$running"
            return 0
        fi
    done
    return 1
}

copy_job_results() {
    local mode="$1"
    local task="$2"
    local local_job_dir="$LOCAL_RUN_ROOT/$mode/$task"
    local local_model_dir="$local_job_dir/$MODEL_STEM"
    local shared_job_dir="$RUN_ROOT/jobs/$mode/$task"
    local shared_model_dir="$shared_job_dir/$MODEL_STEM"
    local filename
    for filename in eval_results.csv pmf_diagnostics_steps.csv pmf_diagnostics_episodes.csv; do
        if [[ ! -f "$local_model_dir/$filename" ]]; then
            echo "ERROR: job result is missing: $local_model_dir/$filename"
            return 1
        fi
    done
    mkdir -p "$shared_model_dir"
    for filename in eval_results.csv pmf_diagnostics_steps.csv pmf_diagnostics_episodes.csv; do
        cp -f "$local_model_dir/$filename" "$shared_model_dir/$filename" || return 1
    done
    if [[ -f "$local_job_dir/eval_config.yaml" ]]; then
        cp -f "$local_job_dir/eval_config.yaml" "$shared_job_dir/eval_config.yaml" || return 1
    else
        echo "ERROR: eval_config.yaml is missing: $local_job_dir/eval_config.yaml"
        return 1
    fi
}

run_one_job() {
    local job_id="$1"
    local mode="$2"
    local task="$3"
    local gpu="$4"
    local local_job_dir="$LOCAL_RUN_ROOT/$mode/$task"
    local shared_job_dir="$RUN_ROOT/jobs/$mode/$task"
    local xvfb_bin_dir
    local rc
    local -a pmf_args=()
    xvfb_bin_dir="$(dirname "$XVFB_RUN")"
    if [[ "$mode" == "pmf" ]]; then
        pmf_args+=(--pmf-enabled)
    fi
    mkdir -p "$shared_job_dir"
    {
        echo "job_id=$job_id"
        echo "mode=$mode"
        echo "task=$task"
        echo "node=$NODE_NAME"
        echo "gpu=$gpu"
        echo "start=$(date)"
        echo "git_commit=${CURRENT_COMMIT:-}"
        echo "local_eval_dir=$local_job_dir"
        echo "shared_eval_dir=$shared_job_dir"
    } > "$shared_job_dir/meta.txt"
    log "START $job_id mode=$mode task=$task on physical GPU $gpu"
    (
        cd "$RLBENCH_DIR" || exit 1
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
        "$XVFB_RUN" -a \
            -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
            env "${eval_env[@]}" \
            "$CONDA_PREFIX/bin/python" -u eval.py \
                --model-folder "$MODEL_FOLDER" \
                --model-name "$MODEL_NAME" \
                --eval-datafolder "$EVAL_DATA" \
                --tasks "$task" \
                --start-episode "$START_EPISODE" \
                --eval-episodes "$EVAL_EPISODES" \
                --episode-length "$EPISODE_LENGTH" \
                --log-name "$RUN_NAME/$mode/$task" \
                --device 0 \
                --headless \
                --pmf-diagnostics-enabled \
                --pmf-diagnostics-mode "$mode" \
                --pmf-prior-var "$PMF_PRIOR_VAR" \
                --pmf-observation-var "$PMF_OBSERVATION_VAR" \
                "${pmf_args[@]}"
    ) > "$shared_job_dir/stdout.log" 2>&1
    rc=$?
    if (( rc == 0 )) && ! copy_job_results "$mode" "$task"; then
        rc=90
    fi
    {
        echo "end=$(date)"
        echo "exit_code=$rc"
    } >> "$shared_job_dir/meta.txt"
    echo "$rc" > "$shared_job_dir/exit_code.txt"
    if (( rc == 0 )); then
        log "DONE  $job_id mode=$mode task=$task on physical GPU $gpu"
    else
        log "FAIL  $job_id mode=$mode task=$task on physical GPU $gpu, rc=$rc"
    fi
    return "$rc"
}

worker_loop() {
    local gpu="$1"
    local failed=0
    local claim
    local job_id
    local mode
    local task
    local running_marker
    local marker_name
    while true; do
        if [[ -f "$ABORT_FILE" ]]; then
            log "GPU $gpu: ABORT marker detected; worker exits."
            break
        fi
        if ! claim="$(claim_next_job "$gpu")"; then
            log "GPU $gpu: queue empty, worker exits."
            break
        fi
        IFS='|' read -r job_id mode task running_marker <<< "$claim"
        marker_name="$(basename "$running_marker")"
        log "GPU $gpu claimed $job_id ($mode, $task)"
        if run_one_job "$job_id" "$mode" "$task" "$gpu"; then
            mv "$running_marker" "$DONE_DIR/$marker_name" 2>/dev/null || true
        else
            failed=1
            mv "$running_marker" "$FAILED_DIR/$marker_name" 2>/dev/null || true
            tail -n 60 "$RUN_ROOT/jobs/$mode/$task/stdout.log" || true
        fi
    done
    return "$failed"
}

launch_worker_group() {
    local gpu_spec="$1"
    local -a gpus
    local -a pids=()
    local gpu
    local pid
    local failed=0
    parse_gpu_spec "$gpu_spec" || return 2
    gpus=("${PARSED_GPUS[@]}")
    for gpu in "${gpus[@]}"; do
        worker_loop "$gpu" &
        pid=$!
        pids+=("$pid")
        log "Started dynamic queue worker on physical GPU $gpu, PID=$pid"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

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
    trap 'worker_signal_abort INT' INT
    trap 'worker_signal_abort TERM' TERM
    trap 'rm -f "${WORKER_PID_FILE:-}" 2>/dev/null || true' EXIT
    while [[ ! -f "$READY_FILE" ]]; do
        [[ ! -f "$ABORT_FILE" ]] || exit 130
        sleep 1
    done
    validate_node_environment "$gpu_spec" || exit 2
    write_node_metadata
    ready_marker="$NODE_STATUS_DIR/${WORKER_TAG}.ready"
    printf "worker_tag=%s\nnode=%s\ngpus=%s\nready=%s\n" \
        "$WORKER_TAG" "$NODE_NAME" "$gpu_spec" "$(date)" > "$ready_marker"
    log "$WORKER_TAG preflight passed; waiting for common START barrier."
    while [[ ! -f "$START_FILE" ]]; do
        [[ ! -f "$ABORT_FILE" ]] || exit 130
        sleep 0.2
    done
    [[ ! -f "$ABORT_FILE" ]] || exit 130
    launch_worker_group "$gpu_spec" || worker_failed=1
    nvidia-smi > "$RUN_ROOT/nvidia_smi_${WORKER_TAG}_${NODE_NAME}_end.txt" 2>&1 || true
    (( worker_failed == 0 )) || exit 1
    exit 0
}

if [[ "$SCRIPT_MODE" == "worker" ]]; then
    run_worker_mode
fi
if [[ "$SCRIPT_MODE" != "controller" ]]; then
    echo "ERROR: invalid SCRIPT_MODE=$SCRIPT_MODE"
    exit 2
fi

LOCAL_GPU_SPEC="${LOCAL_GPUS:-}"
SERVER112_GPU_SPEC="${SERVER112_GPUS:-}"
SERVER108_GPU_SPEC="${SERVER108_GPUS:-}"
if [[ -z "$LOCAL_GPU_SPEC" && -z "$SERVER112_GPU_SPEC" && -z "$SERVER108_GPU_SPEC" ]]; then
    echo "ERROR: at least one of LOCAL_GPUS, SERVER112_GPUS, SERVER108_GPUS is required."
    exit 2
fi

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
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "ERROR: activate $EXPECTED_ENV first on controller."
        exit 2
    fi
    CURRENT_BRANCH="$(git -C "$BRIDGEVLA_ROOT" branch --show-current 2>/dev/null || true)"
    CURRENT_COMMIT="$(git -C "$BRIDGEVLA_ROOT" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
        echo "ERROR: wrong branch on controller: $CURRENT_BRANCH"
        exit 2
    fi
    validate_pmf_parameters || exit 2
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: shared run directory already exists: $RUN_ROOT"
    exit 2
fi

mkdir -p "$PENDING_DIR" "$RUNNING_DIR" "$DONE_DIR" "$FAILED_DIR" \
    "$ASSIGNMENT_DIR" "$NODE_STATUS_DIR"
job_index=0
for task in "${DIAG_TASK_ARRAY[@]}"; do
    for mode in "${DIAG_MODE_ARRAY[@]}"; do
        job_index=$((job_index + 1))
        job_id="$(printf 'job_%03d' "$job_index")"
        printf "%s\t%s\n" "$mode" "$task" > "$PENDING_DIR/$job_id.job"
    done
done

{
    echo "BridgeVLA PMF diagnostics three-node stable GLVND evaluation"
    echo "date=$(date)"
    echo "controller_hostname=$(hostname)"
    echo "git_branch=${CURRENT_BRANCH:-}"
    echo "git_commit=${CURRENT_COMMIT:-}"
    echo "model_name=$MODEL_NAME"
    echo "diag_tasks=$DIAG_TASKS"
    echo "diag_modes=$DIAG_MODES"
    echo "start_episode=$START_EPISODE"
    echo "eval_episodes=$EVAL_EPISODES"
    echo "episode_length=$EPISODE_LENGTH"
    echo "pmf_prior_var=$PMF_PRIOR_VAR"
    echo "pmf_observation_var=$PMF_OBSERVATION_VAR"
    echo "local_gpus=$LOCAL_GPU_SPEC"
    echo "server112_gpus=$SERVER112_GPU_SPEC"
    echo "server108_gpus=$SERVER108_GPU_SPEC"
    echo "shared_run_root=$RUN_ROOT"
    echo "scheduling=three_node_atomic_mode_task_queue"
} > "$RUN_ROOT/protocol.txt"
git -C "$BRIDGEVLA_ROOT" diff > "$RUN_ROOT/code_diff.patch" 2>/dev/null || true
if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    write_node_metadata
fi
: > "$READY_FILE"

echo
echo "============================================================"
echo "BridgeVLA PMF diagnostics three-node stable GLVND evaluation"
echo "============================================================"
echo "Shared run root : $RUN_ROOT"
echo "Number of tasks : ${#DIAG_TASK_ARRAY[@]}"
echo "Number of modes : ${#DIAG_MODE_ARRAY[@]}"
echo "Total jobs      : $TOTAL_JOBS"
echo "Episodes/job    : $EVAL_EPISODES"
echo "Steps/episode   : $EPISODE_LENGTH max"
echo "PMF prior var   : $PMF_PRIOR_VAR"
echo "PMF obs var     : $PMF_OBSERVATION_VAR"
echo "HF mode         : strict offline"
echo "Local GPUs      : ${LOCAL_GPU_SPEC:-<none>}"
echo "server112 GPUs  : ${SERVER112_GPU_SPEC:-<none>}"
echo "server108 GPUs  : ${SERVER108_GPU_SPEC:-<none>}"

TOTAL_WORKERS=0
for spec in "$LOCAL_GPU_SPEC" "$SERVER112_GPU_SPEC" "$SERVER108_GPU_SPEC"; do
    if [[ -n "$spec" ]]; then
        normalized="${spec//,/ }"
        read -r -a tmp_gpus <<< "$normalized"
        TOTAL_WORKERS=$((TOTAL_WORKERS + ${#tmp_gpus[@]}))
    fi
done
echo "Requested workers: $TOTAL_WORKERS"
if (( TOTAL_WORKERS > TOTAL_JOBS )); then
    echo "NOTE: extra workers will see an empty queue and exit without loading the model."
fi

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
    remote_cmd+="pkill -TERM -f -- $(shell_quote "--log-name $RUN_NAME/") 2>/dev/null || true"
    timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "$remote_cmd" >/dev/null 2>&1 || true
}

controller_signal_abort() {
    local sig="${1:-INT}"
    local pid
    if (( CONTROLLER_ABORTING != 0 )); then
        kill_descendants "$BASHPID" KILL
        exit 130
    fi
    CONTROLLER_ABORTING=1
    trap 'kill_descendants "$BASHPID" KILL; exit 130' INT TERM
    mkdir -p "$QUEUE_ROOT" 2>/dev/null || true
    : > "$ABORT_FILE" 2>/dev/null || true
    {
        echo "aborted=$(date)"
        echo "signal=$sig"
        echo "controller=$(hostname)"
        echo "run_name=$RUN_NAME"
    } > "$ABORTED_INFO" 2>/dev/null || true
    [[ -z "${SERVER112_GPU_SPEC:-}" ]] || request_remote_abort server112 "$SERVER112_HOST"
    [[ -z "${SERVER108_GPU_SPEC:-}" ]] || request_remote_abort server108 "$SERVER108_HOST"
    [[ -z "${LOCAL_GROUP_PID:-}" ]] || kill -TERM "$LOCAL_GROUP_PID" 2>/dev/null || true
    for pid in "${REMOTE_PIDS[@]:-}"; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    kill_descendants "$BASHPID" TERM
    sleep 2
    kill_descendants "$BASHPID" KILL
    exit 130
}

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
    remote_cmd+="SCRIPT_MODE=worker WORKER_TAG=$(shell_quote "$tag") "
    remote_cmd+="WORKER_GPUS=$(shell_quote "$gpu_spec") RUN_NAME=$(shell_quote "$RUN_NAME") "
    remote_cmd+="SHARED_EVAL_ROOT=$(shell_quote "$SHARED_EVAL_ROOT") BRIDGEVLA_ROOT=$(shell_quote "$BRIDGEVLA_ROOT") "
    remote_cmd+="MODEL_NAME=$(shell_quote "$MODEL_NAME") NODE_DATA_ROOT=$(shell_quote "$data_root") "
    remote_cmd+="MODEL_FOLDER=$(shell_quote "$data_root/VLA/BridgeVLA/checkpoints/bridgevla/rlbench") "
    remote_cmd+="EVAL_DATA=$(shell_quote "$data_root/VLA/BridgeVLA/datasets/rlbench/eval") "
    remote_cmd+="NODE_HF_HOME=$(shell_quote "$data_root/huggingface_cache") "
    remote_cmd+="NODE_HF_HUB_CACHE=$(shell_quote "$data_root/huggingface_cache/hub") "
    remote_cmd+="XVFB_RUN=$(shell_quote "$xvfb_run") NODE_GL_PRELOAD=$(shell_quote "$gl_preload") "
    remote_cmd+="DIAG_TASKS=$(shell_quote "$DIAG_TASKS") DIAG_MODES=$(shell_quote "$DIAG_MODES") "
    remote_cmd+="START_EPISODE=$(shell_quote "$START_EPISODE") EVAL_EPISODES=$(shell_quote "$EVAL_EPISODES") "
    remote_cmd+="EPISODE_LENGTH=$(shell_quote "$EPISODE_LENGTH") PMF_PRIOR_VAR=$(shell_quote "$PMF_PRIOR_VAR") "
    remote_cmd+="PMF_OBSERVATION_VAR=$(shell_quote "$PMF_OBSERVATION_VAR") CPU_THREADS=$(shell_quote "$CPU_THREADS") "
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
launch_remote_group server112 "$SERVER112_HOST" "$SERVER112_GPU_SPEC" \
    "$SERVER112_DATA_ROOT" "$SERVER112_XVFB_RUN" "$SERVER112_GL_PRELOAD"
launch_remote_group server108 "$SERVER108_HOST" "$SERVER108_GPU_SPEC" \
    "$SERVER108_DATA_ROOT" "$SERVER108_XVFB_RUN" "$SERVER108_GL_PRELOAD"

LOCAL_GROUP_STATUS_FILE="$NODE_STATUS_DIR/server07.local_group_status"
if [[ -n "$LOCAL_GPU_SPEC" ]]; then
    (
        trap 'worker_signal_abort INT' INT
        trap 'worker_signal_abort TERM' TERM
        while [[ ! -f "$START_FILE" ]]; do
            [[ ! -f "$ABORT_FILE" ]] || exit 130
            sleep 0.2
        done
        [[ ! -f "$ABORT_FILE" ]] || exit 130
        if launch_worker_group "$LOCAL_GPU_SPEC"; then
            echo 0 > "$LOCAL_GROUP_STATUS_FILE"
        else
            echo 1 > "$LOCAL_GROUP_STATUS_FILE"
            exit 1
        fi
    ) &
    LOCAL_GROUP_PID=$!
fi

wait_for_remote_ready() {
    local tag="$1"
    local pid="$2"
    local launch_log="$3"
    local waited=0
    local ready_marker="$NODE_STATUS_DIR/${tag}.ready"
    while [[ ! -f "$ready_marker" ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: remote group $tag exited before becoming ready."
            tail -n 100 "$launch_log" || true
            return 2
        fi
        if (( waited >= REMOTE_READY_TIMEOUT )); then
            echo "ERROR: remote group $tag preflight timed out."
            return 2
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

for i in "${!REMOTE_PIDS[@]}"; do
    wait_for_remote_ready "${REMOTE_TAGS[$i]}" "${REMOTE_PIDS[$i]}" "${REMOTE_LOGS[$i]}" \
        || controller_signal_abort TERM
done
[[ ! -f "$ABORT_FILE" ]] || controller_signal_abort TERM
: > "$START_FILE"
log "Released common START barrier for all requested nodes."

CONTROLLER_FAILED=0
if [[ -n "$LOCAL_GROUP_PID" ]]; then
    wait "$LOCAL_GROUP_PID" || CONTROLLER_FAILED=1
fi
for i in "${!REMOTE_PIDS[@]}"; do
    if ! wait "${REMOTE_PIDS[$i]}"; then
        CONTROLLER_FAILED=1
        tail -n 100 "${REMOTE_LOGS[$i]}" || true
    fi
done

printf "time\tnode\tgpu\tjob_id\tmode\ttask\n" > "$ASSIGNMENT_LOG"
for assignment in "$ASSIGNMENT_DIR"/job_*.tsv; do
    [[ -f "$assignment" ]] || continue
    cat "$assignment" >> "$ASSIGNMENT_LOG"
done

FAILED_JOB_COUNT=0
for failed_job in "$FAILED_DIR"/job_*.job; do
    [[ -f "$failed_job" ]] || continue
    FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
    echo "FAILED JOB: $(basename "$failed_job") $(cat "$failed_job")"
done

"$CONDA_PREFIX/bin/python" "$ANALYZER_PY" --run-root "$RUN_ROOT"
ANALYZER_RC=$?
nvidia-smi > "$RUN_ROOT/nvidia_smi_${NODE_NAME}_controller_end.txt" 2>&1 || true

if (( CONTROLLER_FAILED != 0 || FAILED_JOB_COUNT != 0 || ANALYZER_RC != 0 )); then
    echo "ERROR: diagnostics finished with failed/partial jobs. Results remain in $RUN_ROOT"
    exit 1
fi

echo "SUCCESS: PMF diagnostics finished and aggregated."
echo "Shared results: $RUN_ROOT"
echo "Assignments   : $ASSIGNMENT_LOG"
