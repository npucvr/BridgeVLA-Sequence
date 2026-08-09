# AGENTS.md — BridgeVLA

## Architecture

- **`pretrain/`** — Pre-trains Paligemma-3B on object detection (2D heatmap prediction) using RoboPoint data.
- **`finetune/bridgevla/`** — Core VLA package (`bridgevla`) installed via `pip install -e finetune/`.
  - `models/bridgevla_agent.py` — Main agent (~1200 lines), the heart of the system.
  - `mvt/` — Multi-View Transformer (adapted from RVT/NVlabs), renders point clouds into 2D views.
  - `libs/` — Vendored dependencies: PerAct, YARR, point-renderer. Do not modify lightly.
  - Config is yacs-based (`config.py`, `configs/`).
- **`finetune/RLBench/`**, **`finetune/Colosseum/`**, **`finetune/GemBench/`** — Per-benchmark training/eval scripts. Each benchmark creates its own conda environment.

## Key constraints

- **Backbone is Paligemma** (`google/paligemma-3b-pt-224`) — a gated HuggingFace model. Auth is required before any training.
- **Each benchmark needs its own conda env** — RLBench, Colosseum, and GemBench have conflicting simulator dependencies.
- **No tests, no linting, no CI** in this repo. It is pure research code.
- **Checkpoints and data are on HuggingFace** (`LPY/BridgeVLA`), not in this repo. See README for links.

## Development commands

### Pretrain
```bash
cd pretrain
bash pretrain.sh --branches pre-training --config_path pretrain_config.yaml \
  --json_detection_path <path> --image_folder <path>
```

### Finetune (RLBench example; same pattern for Colosseum/GemBench)
```bash
cd finetune
pip install -e .                          # install bridgevla package
cd RLBench
bash train.sh --exp_cfg_path configs/rlbench_config.yaml \
  --exp_note debug --freeze_vision_tower --log_dir <path> \
  --load_pretrain --pretrain_path <path>
```

All training uses `torchrun` (distributed, multi-GPU). Config params are in per-benchmark YAML files under `configs/`.
