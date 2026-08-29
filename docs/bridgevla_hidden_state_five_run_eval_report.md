# BridgeVLA H-token sequence 路线五轮完整 RLBench 实验报告

## 1. 实验目的

本实验评估新的 H-token hidden-state sequence 路线是否能在 BridgeVLA 上带来性能或稳定性提升，并使用与官方 baseline 相同的完整 RLBench 评测协议进行比较。

H-token 路线的核心形式为：

$$
H_t = \operatorname{PaliGemma}(z_t, l),
\qquad
y_t^- = F_\phi(y_{t-1}, u_{t-1}),
$$

$$
y_t = U_\omega(y_t^-, H_t),
\qquad
\widetilde{H}_t = H_t + A_\psi(H_t, y_t).
$$

## 2. 训练与评测协议

### 2.1 训练

- Benchmark：RLBench
- Tasks：18
- Demonstrations：每个 task 100 条
- 初始化 checkpoint：`data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth`
- 训练方式：route-only，冻结 BridgeVLA 主体，仅训练新的 hidden-state route 模块
- Sequence window：4
- Batch size：4
- Epoch：1
- Optimizer updates：100、250、500、1000、2000
- 对应 `train_iter`：400、1000、2000、4000、8000
- 所有训练运行退出码：0

### 2.2 官方完整评测

- Tasks：18
- 每个 task 每轮 episode 数：25
- 每轮 episode 总数：450
- 每个 episode 最大步数：25
- 每个 checkpoint 的独立运行数：5
- 每个 checkpoint 总 episode 数：2250
- `start_episode=0`
- 每轮先计算 18 个 task success rate 的算术平均，再对 5 个 run-level 平均值取均值

记第 $i$ 次运行的 18-task 平均成功率为 $s_i$，报告指标为：

$$
\bar{s}=\frac{1}{5}\sum_{i=1}^{5}s_i,
\qquad
\operatorname{std}_{\mathrm{sample}}
=\sqrt{\frac{1}{4}\sum_{i=1}^{5}(s_i-\bar{s})^2}.
$$

因此标准差为五次独立运行的 sample standard deviation（`ddof=1`）。由于每个 task 均为 25 episodes，task mean 与所有 episode 的 pooled success rate 等价。

H-token checkpoint 的第 1 次运行复用此前已经完成的完整评测，第 2–5 次为追加评测；官方 baseline 则重新完成了 5 次独立完整评测。

## 3. 训练 checkpoint

| Optimizer updates | `train_iter` | Checkpoint |
|---:|---:|---|
| 100 | 400 | [`model_last.pth`](../data/hidden_state_sequence_full_v2/train/debug_train_iter%20400%20num_workers%201_hidden_state_enabled%20True/sequence_full_100/debug/08_28_18_10/model_last.pth) |
| 250 | 1000 | [`model_last.pth`](../data/hidden_state_sequence_full_v2/train/debug_train_iter%201000%20num_workers%201_hidden_state_enabled%20True/sequence_full_250/debug/08_28_18_10/model_last.pth) |
| 500 | 2000 | [`model_last.pth`](../data/hidden_state_sequence_full_v2/train/debug_train_iter%202000%20num_workers%201_hidden_state_enabled%20True/sequence_full_500/debug/08_28_18_10/model_last.pth) |
| 1000 | 4000 | [`model_last.pth`](../data/hidden_state_sequence_full_v2/train/debug_train_iter%204000%20num_workers%201_hidden_state_enabled%20True/sequence_full_1000/debug/08_28_18_10/model_last.pth) |
| 2000 | 8000 | [`model_last.pth`](../data/hidden_state_sequence_full_v2/train/debug_train_iter%208000%20num_workers%201_hidden_state_enabled%20True/sequence_full_2000/debug/08_28_18_36/model_last.pth) |

## 4. 五轮整体结果

`Success` 为五次 run-level task mean 的均值；括号内为运行间 sample standard deviation。`Δ` 相对同一官方 baseline 五轮均值，单位为 percentage points（pp）。

| 配置 | run_1 | run_2 | run_3 | run_4 | run_5 | Success（mean ± std） | 成功 episodes | Δ mean | Δ std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official baseline | 87.33% | 85.11% | 89.56% | 87.78% | 87.11% | **87.38 ± 1.59%** | 1966/2250 | — | — |
| H-token，100 updates | 87.11% | 87.78% | 85.11% | 88.44% | 86.22% | **86.93 ± 1.31%** | 1956/2250 | -0.44 | -0.28 |
| H-token，250 updates | 89.11% | 87.56% | 87.56% | 88.44% | 86.89% | **87.91 ± 0.87%** | 1978/2250 | +0.53 | -0.72 |
| H-token，500 updates | 88.22% | 87.33% | 87.33% | 88.00% | 86.22% | **87.42 ± 0.78%** | 1967/2250 | +0.04 | -0.81 |
| H-token，1000 updates | 89.11% | 90.00% | 89.56% | 86.67% | 85.56% | **88.18 ± 1.95%** | 1984/2250 | +0.80 | +0.36 |
| H-token，2000 updates | 85.33% | 87.78% | 86.67% | 87.78% | 87.33% | **86.98 ± 1.03%** | 1957/2250 | -0.40 | -0.56 |

## 5. 逐任务五轮统计

下表为五次运行的逐 task success rate，格式为 `mean ± sample std`，单位为百分比。

| Task | Official baseline | H-token 100 | H-token 250 | H-token 500 | H-token 1000 | H-token 2000 |
|---|---:|---:|---:|---:|---:|---:|
| `close_jar` | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `reach_and_drag` | 100.00 ± 0.00 | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.20 ± 1.79 | 98.40 ± 2.19 |
| `insert_onto_square_peg` | 88.80 ± 3.35 | 88.00 ± 4.90 | 92.00 ± 5.66 | 90.40 ± 6.07 | 91.20 ± 3.35 | 88.00 ± 4.90 |
| `meat_off_grill` | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `open_drawer` | 100.00 ± 0.00 | 97.60 ± 3.58 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.20 ± 1.79 |
| `place_cups` | 52.00 ± 6.32 | 49.60 ± 14.03 | 52.80 ± 5.22 | 49.60 ± 8.29 | 60.80 ± 15.07 | 50.40 ± 14.59 |
| `place_wine_at_rack_location` | 88.00 ± 8.49 | 88.00 ± 8.94 | 88.00 ± 7.48 | 90.40 ± 6.69 | 85.60 ± 10.43 | 88.80 ± 7.69 |
| `push_buttons` | 100.00 ± 0.00 | 97.60 ± 2.19 | 100.00 ± 0.00 | 97.60 ± 2.19 | 99.20 ± 1.79 | 100.00 ± 0.00 |
| `put_groceries_in_cupboard` | 77.60 ± 3.58 | 77.60 ± 3.58 | 79.20 ± 5.22 | 75.20 ± 3.35 | 78.40 ± 6.69 | 76.80 ± 4.38 |
| `put_item_in_drawer` | 98.40 ± 3.58 | 92.80 ± 6.57 | 95.20 ± 1.79 | 93.60 ± 2.19 | 97.60 ± 2.19 | 97.60 ± 2.19 |
| `put_money_in_safe` | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `light_bulb_in` | 81.60 ± 7.27 | 90.40 ± 3.58 | 91.20 ± 5.93 | 90.40 ± 3.58 | 86.40 ± 3.58 | 89.60 ± 5.37 |
| `slide_block_to_color_target` | 97.60 ± 5.37 | 97.60 ± 2.19 | 96.80 ± 3.35 | 94.40 ± 6.07 | 96.80 ± 3.35 | 97.60 ± 2.19 |
| `place_shape_in_shape_sorter` | 56.80 ± 3.35 | 62.40 ± 6.07 | 60.80 ± 7.69 | 61.60 ± 6.07 | 60.80 ± 7.16 | 52.80 ± 5.22 |
| `stack_blocks` | 77.60 ± 8.29 | 76.00 ± 8.49 | 75.20 ± 6.57 | 71.20 ± 7.16 | 75.20 ± 7.16 | 75.20 ± 6.57 |
| `stack_cups` | 78.40 ± 6.07 | 80.80 ± 1.79 | 80.00 ± 4.00 | 85.60 ± 2.19 | 81.60 ± 5.37 | 75.20 ± 8.67 |
| `sweep_to_dustpan_of_size` | 85.60 ± 2.19 | 76.80 ± 4.38 | 76.00 ± 4.00 | 81.60 ± 4.56 | 82.40 ± 3.58 | 84.00 ± 2.83 |
| `turn_tap` | 90.40 ± 7.27 | 90.40 ± 2.19 | 95.20 ± 3.35 | 92.80 ± 3.35 | 92.00 ± 4.90 | 92.00 ± 2.83 |

## 6. 结果分析

1. 官方 baseline 五轮平均为 **87.38 ± 1.59%**。
2. H-token 的最高平均值是 1000 updates 的 **88.18%**，相对 baseline 仅 **+0.80 pp**，但标准差升至 **1.95 pp**。
3. 100、250、500 和 2000 updates 的运行间标准差分别为 1.31、0.87、0.78 和 1.03 pp；只有在部分训练长度下低于 baseline，不能说明整体稳定性提升。
4. 训练长度与性能没有单调关系；2000 updates 的平均值降至 86.98%。
5. H-token 在 `light_bulb_in`、`place_shape_in_shape_sorter` 和部分 `turn_tap` 任务上有一定改善，但 `place_cups`、`stack_blocks` 和 `sweep_to_dustpan_of_size` 仍是主要瓶颈，且部分任务运行间波动较大。
6. 因此，在当前训练数据、sequence window 和 route-only 配置下，不能认定 H-token sequence 路线带来了可靠的性能或稳定性增益。

## 7. 结论与限定

当前最准确的实验结论是：**新的 H-token sequence 方案没有显示出稳定、可确认的性能提升，也没有整体降低运行间标准差。** 1000 updates 的小幅均值提升伴随更大的波动，应视为实验方差范围内的结果，而不是确定性增益。

这里的标准差反映同一个训练 checkpoint 的五次重复评估波动，不是不同训练 seed 之间的方差。当前 sequence sampler 使用长度为 4 的时间窗口；hidden-state prior 与真实环境状态也不是同一概念。

## 8. 数据、日志与复核

- H-token 五轮正式汇总：[`data/hidden_state_sequence_eval_repeats_v1/summary_official_5run.json`](../data/hidden_state_sequence_eval_repeats_v1/summary_official_5run.json)
- 官方 baseline 五轮正式汇总：[`data/official_baseline_eval_5runs_v1/summary_official_5run.json`](../data/official_baseline_eval_5runs_v1/summary_official_5run.json)
- H-token 评测日志：[`data/hidden_state_sequence_eval_repeats_v1/`](../data/hidden_state_sequence_eval_repeats_v1/)
- baseline 评测日志：[`data/official_baseline_eval_5runs_v1/`](../data/official_baseline_eval_5runs_v1/)
- H-token 训练日志：[`data/hidden_state_sequence_full_v2/`](../data/hidden_state_sequence_full_v2/)
- 旧的 H-token 单次结果汇总（仅作历史记录）：[`data/hidden_state_sequence_eval_v3/summary.json`](../data/hidden_state_sequence_eval_v3/summary.json)

共核验 30 个完整评测 CSV（5 个 H-token checkpoint × 5 次运行 + baseline × 5 次运行），每个 CSV 均包含 18 个 task 行，全部评测退出码为 0。聚焦路线测试 [`tests/test_stage2_hidden_state_route.py`](../tests/test_stage2_hidden_state_route.py) 和相关 Python 语法检查均通过。
