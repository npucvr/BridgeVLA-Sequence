#!/usr/bin/env bash
# gbw____
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data2/local_userdata/gaobowen/VLA/BridgeVLA-Sequence}"
CONDA_BASE="${CONDA_BASE:-/home/gaobowen/anaconda3}"
CONDA_ENV="${CONDA_ENV:-bridgevla_plus_rlbench}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/datasets/colosseum_eval}"
MODEL_FOLDER="${MODEL_FOLDER:-$PROJECT_ROOT/data/ckpt/colosseum}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/colosseum_a0_official}"
# gbw____
# manifest 是固定的数据/episode 清单，不应随 A0/A1 的结果目录复制一份；
# 两种方法必须读取同一个已验证 manifest。
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/colosseum_a0_official/manifest.json}"
# ____
DEVICE="${DEVICE:-0}"
# gbw____
# The paper-comparable scope is variation 1--14.  Variation 0 is retained in
# the manifest as a separately selectable clean sanity check.  The parser
# below accepts both ``1-14`` and comma-separated values.
# ____
REPEATS="${REPEATS:-1}"
REPEAT_START="${REPEAT_START:-0}"
START_EPISODE="${START_EPISODE:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
VARIATION_FILTER="${VARIATION_FILTER:-1-14}"
TASK_FILTER="${TASK_FILTER:-}"
# gbw____
# A0/A1 共用同一个官方评测入口。默认值保持 A0；A1 通过环境变量传入，
# 不在脚本内部改写 filter mode，确保普通 eval 和 workaround 子进程一致。
FILTER_MODE="${FILTER_MODE:-none}"
FILTER_DIAGNOSTICS="${FILTER_DIAGNOSTICS:-0}"
FILTER_DIAGNOSTICS_SHADOW_RAW="${FILTER_DIAGNOSTICS_SHADOW_RAW:-0}"
FILT3R_STAGES="${FILT3R_STAGES:-1}"
# ____
# gbw____
# README 中针对 variation 1/6 的三个已知异常单元采用“收集 25 个有效
# trial”的官方 workaround：只排除仿真/数据错误，保留任务失败 reward。
# 默认打开以得到 paper-comparable 结果；设为
# 0 时则保留严格 episode0--24 的原始结果，便于诊断数据/仿真异常。
OFFICIAL_WORKAROUND="${OFFICIAL_WORKAROUND:-1}"
OFFICIAL_WORKAROUND_MAX_ATTEMPTS="${OFFICIAL_WORKAROUND_MAX_ATTEMPTS:-250}"
OFFICIAL_WORKAROUND_SOURCE_EPISODES="${OFFICIAL_WORKAROUND_SOURCE_EPISODES:-25}"
# ____
REF_ROOT="${REF_ROOT:-/data2/local_userdata/gaobowen/VLA/bridgevla}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$REF_ROOT/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
BRIDGEVLA_PALIGEMMA_PATH="${BRIDGEVLA_PALIGEMMA_PATH:-$PROJECT_ROOT/data/ckpt/paligemma}"

# gbw____
# launcher 后面会进入 finetune/Colosseum；将用户提供的相对输出路径先
# 固定到仓库根目录，避免 MANIFEST 在 cd 后失效。
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
fi
if [[ "$MANIFEST" != /* ]]; then
    MANIFEST="$PROJECT_ROOT/$MANIFEST"
fi
# ____

test -x "$PYTHON_BIN"
test -f "$MODEL_FOLDER/$MODEL_NAME"
test -d "$COPPELIASIM_ROOT"
test -d "$BRIDGEVLA_PALIGEMMA_PATH"
test -f "$MANIFEST"
command -v xvfb-run >/dev/null
command -v jq >/dev/null

# gbw____
# Expand the user-facing variation selector once, before reading the manifest.
# This prevents the old comma-only case expression from silently selecting no
# jobs when the natural official spelling ``1-14`` is used.
declare -A SELECTED_VARIATIONS=()
if [[ "$VARIATION_FILTER" != "all" ]]; then
    for token in ${VARIATION_FILTER//,/ }; do
        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo="${BASH_REMATCH[1]}"
            hi="${BASH_REMATCH[2]}"
            if (( lo > hi )); then
                echo "[Error] invalid variation range: $token" >&2
                exit 1
            fi
            for ((variation_id = lo; variation_id <= hi; variation_id++)); do
                SELECTED_VARIATIONS["$variation_id"]=1
            done
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            SELECTED_VARIATIONS["$token"]=1
        else
            echo "[Error] invalid VARIATION_FILTER token: $token" >&2
            exit 1
        fi
    done
if (( ${#SELECTED_VARIATIONS[@]} == 0 )); then
        echo "[Error] VARIATION_FILTER selected no variations" >&2
        exit 1
    fi
fi
# ____

# gbw____
# gbw____
if [[ "$FILTER_MODE" != "none" && "$FILTER_MODE" != "filt3r_akf" ]]; then
    echo "[Error] FILTER_MODE must be none or filt3r_akf" >&2
    exit 1
fi
if [[ "$FILTER_DIAGNOSTICS" != "0" && "$FILTER_DIAGNOSTICS" != "1" ]]; then
    echo "[Error] FILTER_DIAGNOSTICS must be 0 or 1" >&2
    exit 1
fi
if [[ "$FILTER_DIAGNOSTICS_SHADOW_RAW" != "0" && "$FILTER_DIAGNOSTICS_SHADOW_RAW" != "1" ]]; then
    echo "[Error] FILTER_DIAGNOSTICS_SHADOW_RAW must be 0 or 1" >&2
    exit 1
fi
# ____
if [[ "$OFFICIAL_WORKAROUND" != "0" && "$OFFICIAL_WORKAROUND" != "1" ]]; then
    echo "[Error] OFFICIAL_WORKAROUND must be 0 or 1" >&2
    exit 1
fi
if ! [[ "$OFFICIAL_WORKAROUND_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] OFFICIAL_WORKAROUND_MAX_ATTEMPTS must be a positive integer" >&2
    exit 1
fi
if ! [[ "$OFFICIAL_WORKAROUND_SOURCE_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] OFFICIAL_WORKAROUND_SOURCE_EPISODES must be a positive integer" >&2
    exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] REPEATS must be a positive integer" >&2
    exit 1
fi
if ! [[ "$REPEAT_START" =~ ^[0-9]+$ ]]; then
    echo "[Error] REPEAT_START must be a non-negative integer" >&2
    exit 1
fi
# ____

# gbw____
export FILTER_MODE FILTER_DIAGNOSTICS FILTER_DIAGNOSTICS_SHADOW_RAW FILT3R_STAGES
# ____
export DATASET_ROOT MODEL_FOLDER MODEL_NAME OUTPUT_ROOT DEVICE
export COPPELIASIM_ROOT
export BRIDGEVLA_PALIGEMMA_PATH
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/finetune:$REF_ROOT/finetune/bridgevla/libs/RLBench_peract587:$REF_ROOT/finetune/bridgevla/libs/PyRep_stepjam231:$REF_ROOT/finetune/bridgevla/libs/YARR:$REF_ROOT/finetune/bridgevla/libs/peract_colab:$REF_ROOT/finetune/bridgevla/libs/point-renderer:$PROJECT_ROOT/finetune/Colosseum/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT/finetune/Colosseum"

# gbw____
matched_jobs=0
workaround_failed=0
# ____

run_one() {
    local repeat="$1"
    local base_task="$2"
    local variation="$3"
    local task_name="$4"
    local eval_datafolder="$5"
    local log_dir="$OUTPUT_ROOT/repeat_${repeat}/${variation}/${base_task}"
    local result_csv="$log_dir/$MODEL_NAME"
    result_csv="${result_csv%.pth}/eval_results.csv"

    if [[ "$FORCE" != "1" && -s "$result_csv" && "$(wc -l < "$result_csv")" -ge 2 ]]; then
        echo "[skip] repeat=$repeat task=$task_name existing=$result_csv"
        return 0
    fi
    mkdir -p "$log_dir"
    echo "[run] repeat=$repeat task=$task_name episodes=$EVAL_EPISODES max_steps=$EPISODE_LENGTH"
    if [[ "$DRY_RUN" == "1" ]]; then
        # gbw____
        echo "[dry-run] mode=$FILTER_MODE stages=$FILT3R_STAGES q_mode=${FILT3R_Q_MODE:-default} reset_mode=${FILT3R_RESET_MODE:-default} xvfb-run ... python eval.py --eval-datafolder $eval_datafolder --tasks $task_name --log-name $log_dir"
        # ____
        return 0
    fi
    local video_args=()
    if [[ "$SAVE_VIDEO" == "1" ]]; then
        video_args+=(--save-video)
    fi
    xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24 -ac" \
        "$PYTHON_BIN" eval.py \
        --model-folder "$MODEL_FOLDER" \
        --eval-datafolder "$eval_datafolder" \
        --tasks "$task_name" \
        --eval-episodes "$EVAL_EPISODES" \
        --start-episode "$START_EPISODE" \
        --episode-length "$EPISODE_LENGTH" \
        --log-name "$log_dir" \
        --device "$DEVICE" \
        --headless \
        --model-name "$MODEL_NAME" \
        "${video_args[@]}"
}

# gbw____
# 这六个单元是 Colosseum 数据中已知会在普通 reset 阶段触发
# ``Could not place the task ... in the scene`` 的异常单元。它们必须跳过
# 普通 eval.py，统一交给下方的官方成功 trial workaround；否则 set -e 会
# 在 workaround 尚未执行前结束整个 repeat。
is_official_workaround_cell() {
    local cell_key="$1:$2"
    case "$cell_key" in
        close_laptop_lid:1|close_laptop_lid:6|\
        wipe_desk:1|wipe_desk:6|\
        insert_onto_square_peg:1|insert_onto_square_peg:6)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}
# ____

for ((repeat = REPEAT_START; repeat < REPEAT_START + REPEATS; repeat++)); do
    # gbw____
    # 先把 manifest 读入 Bash 数组，再对每一行使用 here-string 解析。评测
    # 子进程运行时间很长，不能让一个持续的 process-substitution/IFS 读取
    # 状态和子进程交互；显式保存每行也能避免字段错位后继续执行错误任务。
    # ____
    mapfile -t manifest_jobs < <(jq -r '.valid_jobs[] | [.base_task, (.variation|tostring), .task_name, .eval_datafolder] | @tsv' "$MANIFEST")
    for job_row in "${manifest_jobs[@]}"; do
        IFS=$'\t' read -r base_task variation_id task_name eval_datafolder <<< "$job_row"
        [[ -n "$base_task" ]] || continue
        if [[ -n "$TASK_FILTER" && ! "$base_task" =~ $TASK_FILTER ]]; then
            continue
        fi
        # gbw____
        if [[ -z "${variation_id:-}" || ! "$variation_id" =~ ^[0-9]+$ ]]; then
            echo "[Error] invalid manifest variation for task=$base_task: <$variation_id>" >&2
            exit 1
        fi
        if [[ "$VARIATION_FILTER" != "all" && -z "${SELECTED_VARIATIONS[$variation_id]+selected}" ]]; then
            continue
        fi
        # gbw____
        if [[ "$OFFICIAL_WORKAROUND" == "1" ]] && is_official_workaround_cell "$base_task" "$variation_id"; then
            echo "[skip-workaround-cell] repeat=$repeat task=${base_task}_${variation_id}"
            continue
        fi
        # ____
        # ____
        matched_jobs=$((matched_jobs + 1))
        run_one "$repeat" "$base_task" "$variation_id" "$task_name" "$eval_datafolder"
    done
    unset manifest_jobs
done

# gbw____
# 官方 README 指定的 workaround 只作用于六个 task/variation 单元；不改变
# 其它 cell 的 episode0--24 选择，也不改变最终汇总目录结构。每个 repeat
# 都单独执行，helper 会把成功 trial 的审计 manifest 保存在对应目录。
if [[ "$OFFICIAL_WORKAROUND" == "1" ]]; then
    workaround_video_args=()
    if [[ "$SAVE_VIDEO" == "1" ]]; then
        workaround_video_args+=(--save-video)
    fi
    for ((repeat = REPEAT_START; repeat < REPEAT_START + REPEATS; repeat++)); do
        for cell in \
            "close_laptop_lid:1" "close_laptop_lid:6" \
            "wipe_desk:1" "wipe_desk:6" \
            "insert_onto_square_peg:1" "insert_onto_square_peg:6"; do
            base_task="${cell%%:*}"
            variation_id="${cell##*:}"
            if [[ "$VARIATION_FILTER" != "all" && -z "${SELECTED_VARIATIONS[$variation_id]+selected}" ]]; then
                continue
            fi
            if [[ -n "$TASK_FILTER" && ! "$base_task" =~ $TASK_FILTER ]]; then
                continue
            fi
            result_csv="$OUTPUT_ROOT/repeat_${repeat}/${variation_id}/${base_task}/$(basename "${MODEL_NAME%.pth}")/eval_results.csv"
            # gbw____
            # 与普通任务相同，workaround 也支持断点续跑；已有完整结果时不
            # 重复采集成功 trial，避免无意中改变已记录的评测样本。
            if [[ "$FORCE" != "1" && -s "$result_csv" && "$(wc -l < "$result_csv")" -ge 2 ]]; then
                echo "[skip-workaround] repeat=$repeat task=${base_task}_${variation_id} existing=$result_csv"
                continue
            fi
            # ____
            echo "[workaround] repeat=$repeat task=${base_task}_${variation_id} target_successes=$EVAL_EPISODES max_attempts=$OFFICIAL_WORKAROUND_MAX_ATTEMPTS"
            if [[ "$DRY_RUN" == "1" ]]; then
                # gbw____
                echo "[dry-run] mode=$FILTER_MODE stages=$FILT3R_STAGES q_mode=${FILT3R_Q_MODE:-default} reset_mode=${FILT3R_RESET_MODE:-default} $PYTHON_BIN run_official_workaround.py --task $base_task --variation $variation_id --repeat $repeat --filter-mode $FILTER_MODE --successes $EVAL_EPISODES --max-attempts $OFFICIAL_WORKAROUND_MAX_ATTEMPTS"
                # ____
                continue
            fi
            if ! xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24 -ac" \
                "$PYTHON_BIN" run_official_workaround.py \
                --task "$base_task" \
                --variation "$variation_id" \
                --repeat "$repeat" \
                --data-root "$DATASET_ROOT" \
                --output-root "$OUTPUT_ROOT" \
                --model-folder "$MODEL_FOLDER" \
                --model-name "$MODEL_NAME" \
                --python-bin "$PYTHON_BIN" \
                --eval-script "$PROJECT_ROOT/finetune/Colosseum/eval.py" \
                --device "$DEVICE" \
                --filter-mode "$FILTER_MODE" \
                --filter-diagnostics "$FILTER_DIAGNOSTICS" \
                --filter-diagnostics-shadow-raw "$FILTER_DIAGNOSTICS_SHADOW_RAW" \
                --episode-length "$EPISODE_LENGTH" \
                --source-episodes "$OFFICIAL_WORKAROUND_SOURCE_EPISODES" \
                --successes "$EVAL_EPISODES" \
                --max-attempts "$OFFICIAL_WORKAROUND_MAX_ATTEMPTS" \
                "${workaround_video_args[@]}"
            then
                workaround_failed=1
                echo "[Error] official workaround failed: ${base_task}_${variation_id}" >&2
            fi
        done
    done
fi

if (( matched_jobs == 0 )); then
    echo "[Error] no manifest job matched VARIATION_FILTER=$VARIATION_FILTER TASK_FILTER=$TASK_FILTER" >&2
    exit 1
fi
if (( workaround_failed != 0 )); then
    exit 1
fi
# ____
