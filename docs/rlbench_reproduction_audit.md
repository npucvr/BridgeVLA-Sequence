# BridgeVLA RLBench 官方复现记录

## 复现目标

使用官方发布的 `model_80.pth` checkpoint 和官方 EVAL 数据集，按照论文中的
RLBench 评估协议复现 BridgeVLA baseline 结果。

论文报告的评估设置为：18 个 RLBench 任务、每个任务 25 个 episodes、每个 episode
最多执行 25 个动作步骤，并进行 5 次重复评估。

## 官方评估数据

评估数据集：

- [LPY/BridgeVLA_RLBench_EVAL_DATA](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_EVAL_DATA/tree/main)

使用每个任务的 episodes `0..24`，共 18 个任务 × 25 个 episodes。运行前需将数据准备为
`task/all_variations/episodes/episode0..episode24` 的目录布局，并将
`EVAL_DATAFOLDER` 指向该目录。

## 评估协议

- checkpoint：`model_80.pth`（官方发布的第 80 个 epoch）。
- 任务：18 个 RLBench 任务，任务列表和顺序与论文一致。
- 每个任务：25 个 episodes。
- 每个 episode：最多 25 个动作步骤。
- 重复次数：5 次。
- 评估相机：`front`、`left_shoulder`、`right_shoulder`、`wrist`。
- 汇总方式：18 个任务成功率的非加权平均。

## 复现结果

在官方 EVAL 数据集上运行 5 次后的 18-task 平均成功率为：

```text
本次复现：87.42%
论文报告：88.20%
差值：    -0.78
```

各次运行之间的差异处于正常波动范围内，结果可以视为对论文 RLBench baseline 的
有效复现。

## 复现命令

下载并按上述目录布局准备官方 EVAL 数据后执行统一的五次评估入口：

```bash
EVAL_DATAFOLDER=/PATH/TO/RLBench_EVAL_DATA \
RESULT_LOG_DIR=rlbench_eval_split \
  bash scripts/rlbench_repro/run_repeated_eval.sh
```

默认使用一张 GPU 顺序运行。并行运行时，每个 CoppeliaSim 实例需要独立的
GPU 和 display，例如：

```bash
GPU_IDS=0,2 DISPLAY_IDS=:1.0,:2.0 \
  bash scripts/rlbench_repro/run_repeated_eval.sh
```

## 环境说明

上述结果是在当前服务器环境中得到的复现结果。若 PyTorch、TorchVision、
Transformers、NumPy、CoppeliaSim 或 RLBench 版本发生变化，应重新进行环境核验，
不能直接认为结果完全等同于论文原始环境。
