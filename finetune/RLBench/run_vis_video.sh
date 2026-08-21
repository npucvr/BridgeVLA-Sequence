#!/usr/bin/env bash

set -euo pipefail

cd "$BRIDGEVLA_ROOT/finetune/RLBench"

export GPU=6
export BRIDGEVLA_RECORD_CAM_YAW_DEG=0
export BRIDGEVLA_RECORD_CAM_WIDTH=640
export BRIDGEVLA_RECORD_CAM_HEIGHT=360
export USE_TF=0
export USE_TORCH=1
export CUDA_VISIBLE_DEVICES="$GPU"

xvfb-run --auto-servernum \
  --server-args='-screen 0 1024x768x24 -ac' \
  python -u eval.py \
  --model-folder /data2/local_userdata/lizhe/VLA/BridgeVLA/checkpoints/bridgevla/rlbench \
  --model-name model_80.pth \
  --eval-datafolder /data2/local_userdata/lizhe/VLA/BridgeVLA/datasets/rlbench/eval \
  --tasks close_jar \
  --start-episode 0 \
  --eval-episodes 1 \
  --episode-length 25 \
  --log-name video_close_jar_yaw0_ep0 \
  --device 0 \
  --headless \
  --save-video