# Stage-1 Temporal Token Adapter 实验报告

## 1. 实验目的

验证在冻结 PaliGemma 和 action heads 的前提下，加入短期历史 token 是否有效。

需要区分三个模型：

| 模型 | 输入 | 可训练参数 | 作用 |
|---|---|---|---|
| 官方 baseline `model_80.pth` | 当前 token | 无 | 官方基线 |
| K=1 adapter-only | 当前 token | Stage-1 adapter | 控制组，隔离 adapter finetune 的影响 |
| K=4 adapter-only | 当前 token + 最近 3 个历史 token | Stage-1 adapter | 历史 token 实验组 |

K=1 adapter-only **不是**官方 baseline。adapter 的 `up` 层零初始化，因此训练前功能上等价于官方模型；训练后则可能改变原模型行为。

## 2. 训练设置

- 初始化 checkpoint：官方 `model_80.pth`
- 冻结：PaliGemma、action heads 和其他 backbone 参数
- 训练：只更新 `stage1_token_adapter`
- `train_iter=1000`
- `bs=4`
- 约 250 个 optimizer updates
- K=1：`stage1_history_len=1`
- K=4：`stage1_history_len=4`
- 原有 action loss 不变
- K=4 训练使用 replay keyframe token cache；在线 EVAL 使用因果历史 observation token

本实验是短程方向性实验，没有执行完整 100 epoch finetune。

## 3. EVAL 设置

- RLBench 18 tasks
- 每个 task 25 episodes
- 每次运行 450 episodes
- 每个模型重复 5 次
- 使用同一 checkpoint，不重新训练 5 个模型
- 官方论文 Table 1 报告的 BridgeVLA 平均成功率为 88.2%

论文：[BridgeVLA NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5c1a8aa04c1a2cf5013f28831870dafa-Paper-Conference.pdf)

## 4. 结果

下表中的 ± 是 5 次运行之间的 sample standard deviation，统计对象是每次运行的 18-task 平均成功率。

| 模型 | 平均成功率 | 运行间 std | 相对官方 baseline |
|---|---:|---:|---:|
| 官方 `model_80.pth` | **88.40%** | 0.84 | — |
| K=1 adapter-only | **86.40%** | 1.27 | -2.00 pp |
| K=4 adapter-only | **88.04%** | 0.62 | -0.36 pp |
| 论文 BridgeVLA | **88.20%** | — | — |

关键差异：

- K=4 相比 K=1：**+1.64 pp**
- K=4 相比官方 baseline：**-0.36 pp**
- K=4 相比论文结果：**-0.16 pp**
- K=1 相比论文结果：**-1.80 pp**

## 5. 结论

1. 官方 checkpoint 的复现结果为 88.40%，与论文 88.2% 基本一致，说明当前 EVAL 流程没有明显性能鸿沟。
2. K=1 adapter-only 低于官方 baseline，说明短程 adapter finetune 本身可能改变并损害原模型行为。
3. K=4 恢复到官方/论文水平，但目前没有证据表明它显著超过官方 baseline。
4. K=4 相对 K=1 的提升支持历史 token 有帮助，但提升幅度仍属于方向性结果，不能宣称新的 state of the art。

## 6. 结果文件

- 汇总：`data/stage1_runs_1000/five_run_comparison.json`
- K=1/K=4 任务级均值和方差：`data/stage1_runs_1000/stage1_five_run_summary.json`
- 官方五次结果：`data/bridgevla_ckpt/bridgevla/rlbench/eval/rlbench_5run_official/`
- K=1 五次结果：checkpoint 目录下的 `eval/rlbench_5run_k1/`
- K=4 五次结果：checkpoint 目录下的 `eval/rlbench_5run_k4/`

上述结果目录和 JSON 汇总均为本地实验产物，不纳入 Git；本报告保留可复核的
实验设置、统计口径和汇总数值。
