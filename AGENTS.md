# AGENTS.md — BridgeVLA

## 架构

- **`pretrain/`** — 用 RoboPoint 数据预训练 Paligemma-3B 做目标检测(2D 热图预测)。
- **`finetune/bridgevla/`** — 核心 VLA 包(`bridgevla`,`pip install -e finetune/` 安装)。
  - `models/bridgevla_agent.py` — 主 Agent(~1200 行),系统核心。
  - `mvt/` — 多视角 Transformer(改编自 RVT/NVlabs),把点云渲染成 2D 视图。
  - `libs/` — 内置依赖:PerAct、YARR、point-renderer,非必要勿改。
  - 配置基于 yacs(`config.py`、`configs/`)。
- **`finetune/RLBench/`**、**`finetune/Colosseum/`**、**`finetune/GemBench/`** — 各基准的训练/评估脚本,每个基准有独立 conda 环境。

## 关键约束

- **主干是 Paligemma**(`google/paligemma-3b-pt-224`)— HuggingFace 门控模型,训练前必须认证。
- **每个基准需要独立的 conda 环境** — RLBench、Colosseum、GemBench 的模拟器依赖互相冲突。
- **仓库无测试、无 lint、无 CI** — 纯研究代码。
- **权重和数据在 HuggingFace**(`LPY/BridgeVLA`),不在仓库里,见 README。

## 开发命令

### 预训练
```bash
cd pretrain
bash pretrain.sh --branches pre-training --config_path pretrain_config.yaml \
  --json_detection_path <path> --image_folder <path>
```

### 微调(以 RLBench 为例,Colosseum/GemBench 同理)
```bash
cd finetune
pip install -e .                          # 安装 bridgevla 包
cd RLBench
bash train.sh --exp_cfg_path configs/rlbench_config.yaml \
  --exp_note debug --freeze_vision_tower --log_dir <path> \
  --load_pretrain --pretrain_path <path>
```

所有训练用 `torchrun`(分布式、多 GPU)。配置参数在各基准的 `configs/` YAML 文件中。
