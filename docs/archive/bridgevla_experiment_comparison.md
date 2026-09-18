# BridgeVLA 当前实验、论文指标与训练成本对照

本文汇总仓库当前已经完成的 RLBench 实验，并与原始 BridgeVLA 论文及 BridgeVLA++ 的公开结果对照。性能表只保留 18-task 平均成功率以及 `place_cups`、`place_shape_in_shape_sorter`、`stack_blocks` 三个任务。

## 1. 结论摘要

- 当前本地 **token-only H-token** 路线在 `1000 optimizer updates` 的 18-task 平均成功率为 **88.49±0.88%**；相对本地复测的官方 `model_80.pth` baseline 为 **+1.11 pp**，相对原始 BridgeVLA 论文的 88.2% 为 **+0.29 pp**。
- 当前本地 **prior-observation** 路线（`hidden_state_observation_loss_weight=0.05`，2000 updates）达到 **89.24±0.87%**；相对本地 baseline 为 **+1.87 pp**，相对原始论文为 **+1.04 pp**。这仍是本仓库的 BridgeVLA hidden-state 实验，**不是 BridgeVLA++ 官方结果**。
- 官方 **BridgeVLA++** RLBench 平均成功率为 **93.7±0.6%**，三个选定任务分别为 **76.8±11.5% / 72.0±6.3% / 85.6±4.6%**。
- 本地路线只更新少量新增模块：token-only 为 **1,324,800（约 1.325M）** 个可训练参数；加入 prior-observation decoder 后为 **3,471,360（约 3.471M）**。PaliGemma/BridgeVLA 主体在本地 route-only 实验中冻结。

## 2. 性能对比

### 2.1 结果表

除特别注明外，数值为成功率（%），格式为 `5 次评测 run 的均值 ± sample std`。`Avg. Success` 是 18 个 task 的平均成功率；因为每个 task 每次都是 25 个 episode，task-level 平均与 pooled episode success rate 等价。

| 方法 / checkpoint | Avg. Success（18 tasks） | `place_cups` | `place_shape_in_shape_sorter` | `stack_blocks` | 相对原论文 Avg. Δ（pp） |
|---|---:|---:|---:|---:|---:|
| BridgeVLA 原论文 | 88.2 | 58.4±10.0 | 60.8±7.7 | 76.8±8.7 | — |
| 本地复测官方 BridgeVLA `model_80.pth` | 87.38±1.59 | 52.0±6.32* | 56.8±3.35* | 77.6±8.29* | -0.82 |
| 本地 token-only，1000 updates | 88.49±0.88 | 57.6±6.69 | 60.0±6.32 | 79.2±7.16 | +0.29 |
| 本地 prior-observation，2000 updates，weight=0.05 | 89.24±0.87 | 62.4±4.56 | 61.6±4.56 | 79.2±4.38 | +1.04 |
| BridgeVLA++ 官方论文 | 93.7±0.6 | 76.8±11.5 | 72.0±6.3 | 85.6±4.6 | +5.50 |

\* 本地官方 baseline 的三个 task-level std 是根据五个原始评测 CSV 重新计算的；baseline 汇总 JSON 只直接保存了 18-task overall std。

## 3. 训练成本对比

下表只列与当前 RLBench 结果直接相关的训练。参数量分为“总模型规模”和“本次实际更新参数”：本地两条路线都从已有 `model_80.pth` 初始化，主 PaliGemma/原有动作模块冻结。

| 方法 | 参数量（总模型 / 本次实际更新） | 训练量 | 训练资源 | 训练时间 |
|---|---|---|---|---|
| 本地 token-only，1000 updates | 约 3B 级主干（未精确 dump） / 1.325M | 1000 optimizer updates；`train_iter=4000`；batch=4 | 1× server17 GPU0 | 49m06s |
| 本地 prior-observation，2000 updates | 约 3B 级主干（冻结） / 3.471M | 2000 optimizer updates；`train_iter=8000`；batch=4 | 1× server17 GPU2 | 主训练循环 1h30m22s；完整墙钟时间未记录 |
| BridgeVLA 原论文 | PaliGemma 约 3B；完整可训练参数量未报告 | 2D pretrain：3,800 steps；RLBench：83,000 steps | 2D：8×A100；RLBench：48×H100 | 2D 约 2h；RLBench 约 20h（合计约 22h） |
| BridgeVLA++ 官方论文 | 总计约 3.19B（2.92B + 269.77M）；实际更新参数量未报告 | 2D pretrain：3,800 steps；RLBench：130 epochs（前 4 个 freeze epochs；batch=4/GPU） | 2D：8×A100；RLBench：32×H20 | 2D 约 2h；RLBench 墙钟时间未报告 |

说明：本地时间包含日志记录的初始化/加载过程；论文使用多卡和不同 GPU 型号，表中时间不能直接按墙钟做一一对比。prior-observation 的 `1h30m22s` 是日志进度条主循环时间，不代表完整墙钟时间。
