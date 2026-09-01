#!/usr/bin/env bash
# gbw___
set -euo pipefail

# gbw____
PROJECT_ROOT="${PROJECT_ROOT:-/remote_userdata/gaobowen/BridgeVLA-Sequence}"
CONDA_BASE="${CONDA_BASE:-/home/gaobowen/anaconda3}"
# ____
CONDA_ENV="${CONDA_ENV:-bridgevla_plus_rlbench}"
TASKS="${TASKS:-close_jar}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
START_EPISODE="${START_EPISODE:-0}"
# gbw____
EVAL_SEED="${EVAL_SEED:-}"
EPISODE_SELECTION_INPUT="${EPISODE_SELECTION_INPUT:-}"
# ____
DEVICE="${DEVICE:-0}"
HEADLESS=1
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VISUALIZE="${VISUALIZE:-0}"
SAVE_MULTIVIEW="${SAVE_MULTIVIEW:-1}"
MULTIVIEW_SIZE="${MULTIVIEW_SIZE:-256}"
# gbw____
FILTER_MODE="${FILTER_MODE:-none}"
# gbw____
# 评估 A1 时显式设置 FILTER_MODE=filt3r_akf；none 是官方 A0 旁路。
# A1 的 baseline 参数是 constant Q、current reset、只过滤 Stage-1。
# ____
FILTER_DIAGNOSTICS="${FILTER_DIAGNOSTICS:-0}"
FILTER_DIAGNOSTICS_SHADOW_RAW="${FILTER_DIAGNOSTICS_SHADOW_RAW:-0}"
# gbw____
FILT3R_STAGES="${FILT3R_STAGES:-1}"
# gbw____
# 下面三项共同定义当前 A1 baseline 的默认路径：
# q_mode=constant、max token shift=0（无 trust-region）、reset_mode=current。
# ____
FILT3R_Q_MODE="${FILT3R_Q_MODE:-constant}"
FILT3R_JITTER_WEIGHT="${FILT3R_JITTER_WEIGHT:-1.0}"
FILT3R_MAX_TOKEN_SHIFT_RATIO="${FILT3R_MAX_TOKEN_SHIFT_RATIO:-0.0}"
FILT3R_RESET_MODE="${FILT3R_RESET_MODE:-current}"
# gbw____
FILT3R_DRIFT_SCOPE="${FILT3R_DRIFT_SCOPE:-global_token_median}"
# ____
# gbw____
# R0 calibration can set FILT3R_R and FILT3R_P_INIT_RATIO together;
# the Python adapter derives P_init=R*ratio when the ratio is non-empty.
FILT3R_R="${FILT3R_R:-1.0}"
FILT3R_P_INIT_RATIO="${FILT3R_P_INIT_RATIO:-}"
# gbw____
# A3: jitter-aware full-channel measurement covariance. fixed 是 A1/A2
# 默认值；jitter_aware 只在 A3/A4 候选中显式打开。
FILT3R_R_MODE="${FILT3R_R_MODE:-fixed}"
FILT3R_JITTER_R_LAMBDA="${FILT3R_JITTER_R_LAMBDA:-0.0}"
FILT3R_JITTER_R_MAX="${FILT3R_JITTER_R_MAX:-4.0}"
FILT3R_JITTER_R_TAU="${FILT3R_JITTER_R_TAU:-0.0}"
FILT3R_JITTER_R_COHERENCE_WEIGHT="${FILT3R_JITTER_R_COHERENCE_WEIGHT:-0.0}"
FILT3R_Q_USE_ADAPTIVE_R="${FILT3R_Q_USE_ADAPTIVE_R:-1}"
# ____
# ____
# gbw____
# gbw____
# A2 APNE 参数：Q/R 在 [Q_MIN,Q_MAX] 内由 drift sigmoid 因果调度。
# 0.5/35.6 让 g=1 的 sigmoid 中点保持 A1-clean 的 Q/R=18.05。
# ____
FILT3R_Q_MIN="${FILT3R_Q_MIN:-0.5}"
FILT3R_Q_MAX="${FILT3R_Q_MAX:-35.6}"
FILT3R_Q_SIGMOID_ALPHA="${FILT3R_Q_SIGMOID_ALPHA:-4.0}"
FILT3R_Q_SIGMOID_TAU="${FILT3R_Q_SIGMOID_TAU:-1.0}"
# gbw____
# Gain-space A2：目标 gain 的 bounded-sigmoid 与由 gain 反解 Q/R 的安全范围。
# 这些变量只在 FILT3R_Q_MODE=gain_space_adaptive 时生效。
# ____
FILT3R_GAIN_K_DELTA="${FILT3R_GAIN_K_DELTA:-0.015}"
FILT3R_GAIN_SIGMOID_ALPHA="${FILT3R_GAIN_SIGMOID_ALPHA:-4.0}"
FILT3R_GAIN_SIGMOID_TAU="${FILT3R_GAIN_SIGMOID_TAU:-1.0}"
FILT3R_GAIN_K_MIN="${FILT3R_GAIN_K_MIN:-0.90}"
FILT3R_GAIN_K_MAX="${FILT3R_GAIN_K_MAX:-0.98}"
FILT3R_GAIN_Q_MIN="${FILT3R_GAIN_Q_MIN:-4.0}"
FILT3R_GAIN_Q_MAX="${FILT3R_GAIN_Q_MAX:-40.0}"
# ____
# gbw____
# 新候选可显式把 g=1 标定到 nominal Q/R；默认关闭以保持历史结果。
FILT3R_Q_CALIBRATE_NOMINAL="${FILT3R_Q_CALIBRATE_NOMINAL:-0}"
# ____
# gbw____
# A2 coherence-gated Q：默认 0 保持已有 A2；screen 时显式提高该值，
# 让低 temporal coherence 的瞬时 drift 产生更低的 Q。
FILT3R_Q_COHERENCE_WEIGHT="${FILT3R_Q_COHERENCE_WEIGHT:-0.0}"
# gbw____
# A2 保守变体：保留 nominal Q/R 作为下界，只让持续 drift 提高 Q。
FILT3R_Q_FLOOR_NOMINAL="${FILT3R_Q_FLOOR_NOMINAL:-0}"
# gbw____
FILT3R_Q_POSITIVE_BOOST="${FILT3R_Q_POSITIVE_BOOST:-1.0}"
# gbw____
FILT3R_Q_POSITIVE_DEADBAND="${FILT3R_Q_POSITIVE_DEADBAND:-0.0}"
# ____
# gbw____
FILT3R_Q_INNOVATION_WEIGHT="${FILT3R_Q_INNOVATION_WEIGHT:-0.0}"
FILT3R_Q_INNOVATION_TAU="${FILT3R_Q_INNOVATION_TAU:-1.0}"
# gbw____
# A2-only jitter-gated Q；默认关闭，避免改变既有 A1/A2 结果。
FILT3R_Q_JITTER_PENALTY="${FILT3R_Q_JITTER_PENALTY:-0.0}"
FILT3R_Q_JITTER_TAU="${FILT3R_Q_JITTER_TAU:-1.0}"
# ____
# ____
# gbw____
FILT3R_Q_INNOVATION_GUARD_TAU="${FILT3R_Q_INNOVATION_GUARD_TAU:-0.0}"
# ____
# gbw____
# innovation-aligned Q 的慢速 normalized-innovation reference。
FILT3R_Q_INNOVATION_REFERENCE_BETA="${FILT3R_Q_INNOVATION_REFERENCE_BETA:-0.05}"
FILT3R_Q_INNOVATION_POWER="${FILT3R_Q_INNOVATION_POWER:-1.0}"
# ____
# ____
# ____
# ____
FILT3R_DRIFT_EMA_BETA="${FILT3R_DRIFT_EMA_BETA:-0.2}"
# gbw____
# token_ema_double 的慢速 drift reference；其他 drift scope 仅透传不使用。
FILT3R_DRIFT_REFERENCE_BETA="${FILT3R_DRIFT_REFERENCE_BETA:-0.05}"
# ____
# gbw____
# APNE 分支：innovation 能量 EMA 与 log(Q_hat/Q_nominal) 的 sigmoid 参数。
FILT3R_APNE_EMA_BETA="${FILT3R_APNE_EMA_BETA:-0.2}"
FILT3R_APNE_LOG_ALPHA="${FILT3R_APNE_LOG_ALPHA:-2.0}"
FILT3R_APNE_INNOVATION_CLIP="${FILT3R_APNE_INNOVATION_CLIP:-16.0}"
# gbw____
FILT3R_APNE_INNOVATION_REFERENCE="${FILT3R_APNE_INNOVATION_REFERENCE:-1.0}"
# ____
# gbw____
# APNE-clean 的可选保护：低 innovation evidence 不得降低 nominal Q。
FILT3R_APNE_Q_FLOOR_NOMINAL="${FILT3R_APNE_Q_FLOOR_NOMINAL:-0}"
# ____
# ____
# ____
# ____
# ____

# gbw____
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/datasets/rl_eval}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/data/ckpt/rl/model_80.pth}"
EXP_CFG_PATH="${EXP_CFG_PATH:-$PROJECT_ROOT/data/ckpt/rl/exp_cfg.yaml}"
MVT_CFG_PATH="${MVT_CFG_PATH:-$PROJECT_ROOT/data/ckpt/rl/mvt_cfg.yaml}"
PALIGEMMA_PATH="${PALIGEMMA_PATH:-$PROJECT_ROOT/data/ckpt/paligemma}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/rlbench_eval_close_jar}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$PROJECT_ROOT/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
# ____
# gbw____
# Normalize launcher paths before changing directory below; this keeps relative
# OUTPUT_DIR and diagnostics paths valid inside finetune/RLBench as well.
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
if [[ -n "${FILTER_DIAGNOSTICS_PATH:-}" ]]; then
    FILTER_DIAGNOSTICS_PATH="$(realpath -m "$FILTER_DIAGNOSTICS_PATH")"
fi
export OUTPUT_DIR
# ____
# gbw____
if [[ "$FILTER_DIAGNOSTICS" == "1" ]]; then
    FILTER_DIAGNOSTICS_PATH="${FILTER_DIAGNOSTICS_PATH:-$OUTPUT_DIR/filt3r_diagnostics.jsonl}"
    export FILTER_DIAGNOSTICS_PATH
fi
# ____

export DATASET_ROOT MODEL_PATH EXP_CFG_PATH MVT_CFG_PATH PALIGEMMA_PATH OUTPUT_DIR
export TASKS EVAL_EPISODES EPISODE_LENGTH START_EPISODE DEVICE HEADLESS
# gbw____
export EVAL_SEED
export EPISODE_SELECTION_INPUT
# ____
export SAVE_VIDEO VISUALIZE SAVE_MULTIVIEW MULTIVIEW_SIZE
# gbw____
export FILTER_MODE FILTER_DIAGNOSTICS
export FILTER_DIAGNOSTICS_SHADOW_RAW
# gbw____
export FILT3R_STAGES FILT3R_Q_MODE FILT3R_JITTER_WEIGHT
export FILT3R_MAX_TOKEN_SHIFT_RATIO FILT3R_RESET_MODE
# gbw____
export FILT3R_DRIFT_SCOPE
# ____
# gbw____
export FILT3R_R FILT3R_P_INIT_RATIO
# ____
# gbw____
export FILT3R_R_MODE FILT3R_JITTER_R_LAMBDA FILT3R_JITTER_R_MAX
export FILT3R_JITTER_R_TAU FILT3R_JITTER_R_COHERENCE_WEIGHT
export FILT3R_Q_USE_ADAPTIVE_R
# ____
# gbw____
export FILT3R_Q_MIN FILT3R_Q_MAX FILT3R_Q_SIGMOID_ALPHA
export FILT3R_Q_SIGMOID_TAU FILT3R_DRIFT_EMA_BETA
# gbw____
export FILT3R_GAIN_K_DELTA FILT3R_GAIN_SIGMOID_ALPHA
export FILT3R_GAIN_SIGMOID_TAU FILT3R_GAIN_K_MIN FILT3R_GAIN_K_MAX
export FILT3R_GAIN_Q_MIN FILT3R_GAIN_Q_MAX
# ____
# gbw____
export FILT3R_Q_CALIBRATE_NOMINAL
# ____
# gbw____
export FILT3R_DRIFT_REFERENCE_BETA
# ____
# gbw____
export FILT3R_Q_COHERENCE_WEIGHT
# gbw____
export FILT3R_Q_FLOOR_NOMINAL
# ____
# gbw____
export FILT3R_Q_POSITIVE_BOOST
# gbw____
export FILT3R_Q_POSITIVE_DEADBAND
# ____
# gbw____
export FILT3R_Q_INNOVATION_WEIGHT FILT3R_Q_INNOVATION_TAU
# gbw____
export FILT3R_Q_JITTER_PENALTY FILT3R_Q_JITTER_TAU
# ____
# ____
# gbw____
export FILT3R_Q_INNOVATION_GUARD_TAU
# ____
# gbw____
export FILT3R_Q_INNOVATION_REFERENCE_BETA FILT3R_Q_INNOVATION_POWER
# ____
# ____
# ____
# gbw____
export FILT3R_APNE_EMA_BETA FILT3R_APNE_LOG_ALPHA
export FILT3R_APNE_INNOVATION_CLIP
# gbw____
export FILT3R_APNE_INNOVATION_REFERENCE
# ____
# gbw____
export FILT3R_APNE_Q_FLOOR_NOMINAL
# ____
# ____
# ____
# ____
# ____

for task in $TASKS; do
    test -d "$DATASET_ROOT/$task"
done
test -f "$MODEL_PATH"
test -f "$EXP_CFG_PATH"
test -f "$MVT_CFG_PATH"
test -d "$PALIGEMMA_PATH"
test -d "$COPPELIASIM_ROOT"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONDONTWRITEBYTECODE=1
# gbw____
export PYTHONPATH="$PROJECT_ROOT/finetune:$PROJECT_ROOT/finetune/bridgevla/libs/PyRep:$PROJECT_ROOT/finetune/bridgevla/libs/RLBench:$PROJECT_ROOT/finetune/bridgevla/libs/YARR:$PROJECT_ROOT/finetune/bridgevla/libs/peract:$PROJECT_ROOT/finetune/bridgevla/libs/peract_colab:$PROJECT_ROOT/finetune/bridgevla/libs/point-renderer:$PROJECT_ROOT/finetune/Colosseum/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}"
# ____
export COPPELIASIM_ROOT
export BRIDGEVLA_PALIGEMMA_PATH="$PALIGEMMA_PATH"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

runner=(xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24 -ac" python)

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT/finetune/RLBench"

"${runner[@]}" - <<'PY'
import json
import os
# gbw____
import sys

import torch
# ____

from eval import eval as run_eval
from eval import load_agent

output_dir = os.environ["OUTPUT_DIR"]
tasks = os.environ["TASKS"].split()
device = int(os.environ["DEVICE"])
headless = os.environ["HEADLESS"] == "1"
save_video = os.environ["SAVE_VIDEO"] == "1"
visualize = os.environ["VISUALIZE"] == "1"
save_multiview = os.environ["SAVE_MULTIVIEW"] == "1"
multiview_size = int(os.environ["MULTIVIEW_SIZE"])
# gbw____
eval_seed_text = os.environ.get("EVAL_SEED", "").strip()
eval_seed = int(eval_seed_text) if eval_seed_text else None
if eval_seed is not None:
    import numpy as np

    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
# ____

agent = load_agent(
    model_path=os.environ["MODEL_PATH"],
    exp_cfg_path=os.environ["EXP_CFG_PATH"],
    mvt_cfg_path=os.environ["MVT_CFG_PATH"],
    eval_log_dir=output_dir,
    device=device,
)

scores = run_eval(
    agent=agent,
    tasks=tasks,
    eval_datafolder=os.environ["DATASET_ROOT"],
    start_episode=int(os.environ["START_EPISODE"]),
    eval_episodes=int(os.environ["EVAL_EPISODES"]),
    episode_length=int(os.environ["EPISODE_LENGTH"]),
    replay_ground_truth=False,
    device=device,
    headless=headless,
    logging=True,
    log_dir=output_dir,
    verbose=True,
    save_video=save_video,
    model_name=os.path.basename(os.environ["MODEL_PATH"]),
    visualize=visualize,
    visualize_root_dir=os.path.join(output_dir, "visualize"),
    save_multiview=save_multiview,
    multiview_dir=os.path.join(output_dir, "multiview_videos"),
    multiview_resolution=multiview_size,
    # gbw____
    eval_seed=eval_seed,
    episode_selection_path=os.path.join(output_dir, "episode_selection.json"),
    episode_selection_input_path=(
        os.environ.get("EPISODE_SELECTION_INPUT") or None
    ),
    # ____
)

# gbw____
filter_config = None
if os.environ["FILTER_MODE"] == "filt3r_akf":
    from dataclasses import asdict
    from bridgevla.mvt.filt3r_akf import Filt3rAKFConfig

    filter_config = asdict(Filt3rAKFConfig.from_environment())
# ____

summary = {
    "tasks": tasks,
    "eval_episodes": int(os.environ["EVAL_EPISODES"]),
    "episode_length": int(os.environ["EPISODE_LENGTH"]),
    "headless": headless,
    "scores": scores,
    "dataset_root": os.environ["DATASET_ROOT"],
    "model_path": os.environ["MODEL_PATH"],
    # gbw____
    "eval_seed": eval_seed,
    # ____
    # gbw____
    "filter_mode": os.environ["FILTER_MODE"],
    "filter_diagnostics_path": os.environ.get("FILTER_DIAGNOSTICS_PATH"),
    "filter_config": filter_config,
    # gbw____
    "seed_policy": "EVAL_SEED selects a deterministic 25-episode subset; eval_demo_seed is the sorted dataset position",
    "episode_selection_input": os.environ.get("EPISODE_SELECTION_INPUT") or None,
    # ____
    "runtime": {
        "device": f"cuda:{device}",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    },
    # ____
}
with open(os.path.join(output_dir, "summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)
print(json.dumps(summary, indent=2))
PY
#____
