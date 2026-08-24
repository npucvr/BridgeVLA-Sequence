# AGENTS.md — BridgeVLA

## 架构

- **`pretrain/`** — 用 RoboPoint 数据预训练 Paligemma-3B 做目标检测(2D 热图预测)。
- **`finetune/bridgevla/`** — 核心 VLA 包(`bridgevla`,`pip install -e finetune/` 安装)。
  - `models/bridgevla_agent.py` — 主 Agent(~1200 行),系统核心。
  - `mvt/` — 多视角 Transformer(改编自 RVT/NVlabs),把点云渲染成 2D 视图。
  - `libs/` — 内置依赖:PerAct、YARR、point-renderer,非必要勿改。
  - 配置基于 yacs(`config.py`、`configs/`)。
- **`finetune/RLBench/`**、**`finetune/Colosseum/`**、**`finetune/GemBench/`** — 各基准的训练/评估脚本,每个基准有独立 conda 环境。

## Agent 临时工作区

- **`agent_playground/`** 是 Agent 的自由工作目录。一次性验证脚本、数据审计脚本、临时配置、日志、中间文件和可视化预览默认放在这里，不要散落到仓库根目录或核心源码目录。
- `agent_playground/` 中的内容默认不构成正式接口，任务结束后应删除无用产物；如果某个脚本需要长期复用，再提升到 `scripts/` 并补充说明。
- 大型数据、模型权重和正式实验结果仍按既有约定放在 `data/` 或实验输出目录，不要把它们复制到 `agent_playground/`。

## 关键约束

- **主干是 Paligemma**(`google/paligemma-3b-pt-224`)— HuggingFace 门控模型,训练前必须认证。
- **每个基准需要独立的 conda 环境** — RLBench、Colosseum、GemBench 的模拟器依赖互相冲突。
- **仓库无测试、无 lint、无 CI** — 纯研究代码。
- **权重和原始数据来源于 HuggingFace**(`LPY/BridgeVLA` 及相关数据集); 当前工作区的 `data/` 已包含 BridgeVLA checkpoint、RLBench 训练数据和评估数据。

## 运行机器与共享仓库

- **规范仓库路径：** `/remote_userdata/sunguodong/repos/BridgeVLA`。该路径位于共享 NFS（核验显示为 `192.168.1.202:/remote_userdata`），本地工作区和远程 GPU 机器应使用同一个路径；不要假设 `/home/sunguodong/repos/BridgeVLA` 在所有机器上存在。
- **候选 GPU 节点：** `server16`、`server17`、`server19`、`server20` 都可以作为运行节点；硬件和空闲状态仅作参考，不能把当前会话所在节点固化成默认节点。
  - `server16`：8 张 NVIDIA GeForce RTX 4090；已核验可 SSH 和访问 NFS。
  - `server17`：8 张 NVIDIA GeForce RTX 4090；已核验可 SSH 和访问 NFS。
  - `server19`：8 张 NVIDIA RTX 5880 Ada Generation（约 48 GiB/卡）；本次会话核验为当前主机，可直接本地执行命令。
  - `server20`：8 张 NVIDIA RTX 6000D；已核验可 SSH 和访问 NFS。
- **运行前必须核验实际节点：** 每次真正启动训练、评估或长时间实验前，必须在实际执行 shell 中先运行 `hostname`、进入规范仓库路径后运行 `pwd`，再运行 `nvidia-smi`。确认 hostname、仓库路径和 GPU 都符合预期后才能启动任务。
- **不要 SSH 到当前主机：** 如果本地 `hostname` 已经显示目标节点（例如当前显示 `server19`），直接在本地运行；不要再执行 `ssh server19`。SSH 只用于访问另一台节点，并且远程命令中也必须再次打印 `hostname` 确认目标机器。
- **GPU 状态是动态的：** 核验时检查显存和已有进程；不要因为机器能 SSH 或曾经空闲就直接占用 GPU。
- **工作目录：** 所有节点使用 `/remote_userdata/sunguodong/repos/BridgeVLA`。数据、checkpoint、日志和实验产物位于共享 NFS，GPU 计算在实际 hostname 对应的机器上完成。
- **公共运行时入口：** 使用仓库内 `scripts/bridgevla_runtime.sh`，不要依赖被 `.gitignore` 忽略的根目录 `env.sh`。该入口自动根据当前 hostname 查找 conda，并设置 CoppeliaSim、`PYTHONPATH`、GPU 检查和可选 Xvfb。
- **机器私有配置：** 如自动查找不到 conda，可在 `~/.config/bridgevla/$(hostname -s).env` 中设置 `BRIDGEVLA_CONDA_SH`、`BRIDGEVLA_CONDA_ENV` 和 `BRIDGEVLA_DISPLAY`；这些路径不进入 Git。当前 server16 的 conda 入口是 `/home/sunguodong/miniconda3/etc/profile.d/conda.sh`。
- **仿真显示依赖：** 需要启动 CoppeliaSim 的评估必须设置 `BRIDGEVLA_START_XVFB=1`；本次 server16 核验发现未安装 `Xvfb`，因此 Python/CUDA/RLBench 导入通过，但实际图形仿真运行前仍需提供 Xvfb 或可用的 `DISPLAY`。
- **基础核验命令：**
  ```bash
  # 在实际运行 shell 中执行；不要假设当前节点
  hostname
  cd /remote_userdata/sunguodong/repos/BridgeVLA
  pwd
  nvidia-smi

  # 从另一台机器 SSH 到目标节点时，远程命令中也要核验 hostname
  ssh server16 'hostname; cd /remote_userdata/sunguodong/repos/BridgeVLA; pwd; nvidia-smi'
  ssh server17 'hostname; cd /remote_userdata/sunguodong/repos/BridgeVLA; pwd; nvidia-smi'
  ssh server20 'hostname; cd /remote_userdata/sunguodong/repos/BridgeVLA; pwd; nvidia-smi'
  # 若目标是 server19，先确认当前 shell 不是 server19；在 server19 本机直接执行上面的本地核验命令
  ```

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
