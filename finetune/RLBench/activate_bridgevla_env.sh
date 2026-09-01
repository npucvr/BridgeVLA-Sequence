#!/usr/bin/env bash
# gbw____
# 共享代码路径下的 RLBench 运行环境激活脚本。
# 必须使用 source finetune/RLBench/activate_bridgevla_env.sh 调用。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 source 调用此脚本，而不是直接执行。" >&2
    exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/remote_userdata/gaobowen/BridgeVLA-Sequence}"
CONDA_ENV="${CONDA_ENV:-bridgevla_plus_rlbench}"

if [[ -f "$HOME/python/conda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${CONDA_BASE:-$HOME/python/conda3}"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
else
    echo "找不到 Conda，请先定位该节点已有的 conda.sh。" >&2
    return 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

case "$(hostname -s)" in
    server08|gpu8)
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
        ;;
    server11|gpu11)
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
        ;;
    server13|gpu13)
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
        ;;
    # gbw____
    server16|gpu16)
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
        ;;
    server17|gpu17)
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
        ;;
    # ____
    *)
        CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda 2>/dev/null || true)}"
        ;;
esac

if [[ ! -d "$CUDA_HOME" ]]; then
    echo "CUDA_HOME 不存在：$CUDA_HOME" >&2
    return 1
fi

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/datasets/rl_eval}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/data/ckpt/rl/model_80.pth}"
EXP_CFG_PATH="${EXP_CFG_PATH:-$PROJECT_ROOT/data/ckpt/rl/exp_cfg.yaml}"
MVT_CFG_PATH="${MVT_CFG_PATH:-$PROJECT_ROOT/data/ckpt/rl/mvt_cfg.yaml}"
PALIGEMMA_PATH="${PALIGEMMA_PATH:-$PROJECT_ROOT/data/ckpt/paligemma}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$PROJECT_ROOT/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
USER_XVFB_ROOT="$PROJECT_ROOT/finetune/user_xvfb/root"

export PROJECT_ROOT CONDA_BASE CONDA_ENV CUDA_HOME
export DATASET_ROOT MODEL_PATH EXP_CFG_PATH MVT_CFG_PATH PALIGEMMA_PATH
export COPPELIASIM_ROOT
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export BRIDGEVLA_PALIGEMMA_PATH="$PALIGEMMA_PATH"
export PYTHONDONTWRITEBYTECODE=1

if [[ -d "$USER_XVFB_ROOT/usr/bin" ]]; then
    export PATH="$USER_XVFB_ROOT/usr/bin:$PATH"
fi

export PYTHONPATH="$PROJECT_ROOT/finetune:$PROJECT_ROOT/finetune/bridgevla/libs/PyRep:$PROJECT_ROOT/finetune/bridgevla/libs/RLBench:$PROJECT_ROOT/finetune/bridgevla/libs/YARR:$PROJECT_ROOT/finetune/bridgevla/libs/peract:$PROJECT_ROOT/finetune/bridgevla/libs/peract_colab:$PROJECT_ROOT/finetune/bridgevla/libs/point-renderer:$PROJECT_ROOT/finetune/Colosseum/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}"

echo "BridgeVLA RLBench 环境已激活："
echo "  host=$(hostname -s)"
echo "  conda=$CONDA_ENV ($CONDA_BASE)"
echo "  python=$(command -v python)"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  COPPELIASIM_ROOT=$COPPELIASIM_ROOT"
# ____
