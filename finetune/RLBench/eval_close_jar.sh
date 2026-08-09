#!/bin/bash
set -e

export COPPELIASIM_ROOT="/remote_userdata/sunguodong/repos/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$LD_LIBRARY_PATH"
export LD_PRELOAD="$COPPELIASIM_ROOT/libssl.so.1.1:$COPPELIASIM_ROOT/libcrypto.so.1.1"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export DISPLAY=:1.0
export PALIGEMMA_PATH="/remote_userdata/sunguodong/repos/BridgeVLA/data/bridgevla_ckpt/paligemma-3b-pt-224"
export CUDA_VISIBLE_DEVICES=7
export TF_CPP_MIN_LOG_LEVEL=3
export BITSANDBYTES_NOWELCOME=1

CKPT_DIR="/data2/local_userdata/lizhe/VLA/BridgeVLA/checkpoints/bridgevla/rlbench"
EVAL_DATA="/remote_userdata/sunguodong/repos/BridgeVLA/data/RLBench_TRAIN_DATA"
LOG_DIR="/home/denghui/BridgeVLA-Sequence/logs/rlbench"

cd /home/denghui/BridgeVLA-Sequence/finetune/RLBench

conda run --no-capture-output -n bridgevla_rlbench \
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac' \
    python3 eval.py \
    --tasks close_jar \
    --model-folder "$CKPT_DIR" \
    --eval-datafolder "$EVAL_DATA" \
    --eval-episodes 5 \
    --log-name close_jar_eval \
    --log-dir "$LOG_DIR" \
    --device 0 \
    --headless \
    --save-video \
    --model-name model_80.pth
