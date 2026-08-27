# BridgeVLA hidden-state 五轮完整评估报告

## 1. 实验目的

本实验评估 BridgeVLA 在 `current_correction` hidden-state 路由下，不同 optimizer updates checkpoint 的 RLBench 性能，并按照论文常用形式报告五轮独立完整评估的均值和样本标准差。

评估 checkpoint 为：100、250、500、1000 和 2000 optimizer updates。每个 checkpoint 均使用同一个训练 checkpoint，进行五轮独立完整 EVAL。

## 2. 评估协议

- Benchmark：RLBench
- Tasks：18
- 每个 task 每轮 episode 数：25
- 每轮 episode 总数：450
- 每个 episode 最大步数：25
- 每个 checkpoint 五轮总 episode 数：2250
- `start_episode=0`
- 评估路由：`stage1_hidden_state_enabled=True`、`stage1_adapter_mode=current_correction`
- hidden-state checkpoint 的 `run_1` 使用此前已完成的 full EVAL，`run_2`–`run_5` 为本实验追加的四轮评估

对每轮先计算 18 个 task success rate 的平均值，再在五个 run-level 平均值上计算：

$$
\bar{s}=\frac{1}{5}\sum_{i=1}^{5}s_i,
\qquad
\operatorname{std}_{\mathrm{sample}}
=\sqrt{\frac{1}{4}\sum_{i=1}^{5}(s_i-\bar{s})^2}.
$$

因此，报告中的标准差是五轮重复评估的 **sample standard deviation（`ddof=1`）**。

## 3. 整体结果

| checkpoint | run_1 | run_2 | run_3 | run_4 | run_5 | Success（mean ± std） | 相对官方 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Official baseline | 88.22% | 87.11% | 89.33% | 88.89% | 88.44% | **88.40 ± 0.84%** | — |
| 100 updates | 89.11% | 86.67% | 88.00% | 87.56% | 87.11% | **87.69 ± 0.94%** | -0.71 pp |
| 250 updates | 85.78% | 86.22% | 87.78% | 86.44% | 86.89% | **86.62 ± 0.76%** | -1.78 pp |
| 500 updates | 85.11% | 86.67% | 86.44% | 86.67% | 85.78% | **86.13 ± 0.68%** | -2.27 pp |
| 1000 updates | 86.22% | 86.00% | 87.56% | 86.22% | 88.00% | **86.80 ± 0.91%** | -1.60 pp |
| 2000 updates | 86.67% | 87.78% | 89.33% | 88.00% | 85.78% | **87.51 ± 1.36%** | -0.89 pp |

其中 `pp` 表示 percentage points。

## 4. 逐任务五轮统计

下表为每个 hidden-state checkpoint 在五轮评估上的逐 task success rate，格式为 `mean ± sample std`。

| Task | 100 updates | 250 updates | 500 updates | 1000 updates | 2000 updates |
|---|---:|---:|---:|---:|---:|
| `close_jar` | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| `reach_and_drag` | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 98.40 ± 3.58% |
| `insert_onto_square_peg` | 92.00 ± 4.00% | 88.80 ± 5.22% | 89.60 ± 2.19% | 88.80 ± 5.22% | 91.20 ± 3.35% |
| `meat_off_grill` | 100.00 ± 0.00% | 99.20 ± 1.79% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| `open_drawer` | 99.20 ± 1.79% | 99.20 ± 1.79% | 100.00 ± 0.00% | 97.60 ± 3.58% | 99.20 ± 1.79% |
| `place_cups` | 58.40 ± 5.37% | 50.40 ± 8.29% | 50.40 ± 4.56% | 52.00 ± 7.48% | 56.80 ± 7.16% |
| `place_wine_at_rack_location` | 90.40 ± 7.80% | 92.80 ± 6.57% | 87.20 ± 5.22% | 88.00 ± 4.90% | 93.60 ± 2.19% |
| `push_buttons` | 99.20 ± 1.79% | 98.40 ± 2.19% | 98.40 ± 2.19% | 100.00 ± 0.00% | 99.20 ± 1.79% |
| `put_groceries_in_cupboard` | 75.20 ± 3.35% | 76.80 ± 3.35% | 76.80 ± 3.35% | 75.20 ± 3.35% | 77.60 ± 4.56% |
| `put_item_in_drawer` | 98.40 ± 2.19% | 96.00 ± 2.83% | 99.20 ± 1.79% | 92.00 ± 9.38% | 94.40 ± 2.19% |
| `put_money_in_safe` | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| `light_bulb_in` | 87.20 ± 5.93% | 86.40 ± 4.56% | 84.00 ± 4.00% | 85.60 ± 4.56% | 90.40 ± 2.19% |
| `slide_block_to_color_target` | 96.80 ± 3.35% | 92.80 ± 4.38% | 96.00 ± 4.00% | 96.00 ± 4.00% | 93.60 ± 2.19% |
| `place_shape_in_shape_sorter` | 53.60 ± 4.56% | 59.20 ± 3.35% | 52.80 ± 1.79% | 56.00 ± 4.00% | 59.20 ± 9.55% |
| `stack_blocks` | 76.80 ± 5.22% | 76.80 ± 3.35% | 75.20 ± 5.22% | 75.20 ± 5.22% | 72.00 ± 11.31% |
| `stack_cups` | 77.60 ± 6.69% | 76.80 ± 5.93% | 80.80 ± 7.16% | 82.40 ± 6.69% | 80.80 ± 5.22% |
| `sweep_to_dustpan_of_size` | 85.60 ± 2.19% | 79.20 ± 3.35% | 74.40 ± 2.19% | 86.40 ± 2.19% | 80.00 ± 4.00% |
| `turn_tap` | 88.00 ± 4.90% | 86.40 ± 6.07% | 85.60 ± 6.69% | 87.20 ± 5.22% | 88.80 ± 3.35% |

## 5. 结果分析

1. 在本实验配置下，训练长度没有带来单调的性能提升。
2. 100 updates 的整体均值最高，为 `87.69 ± 0.94%`，但仍低于官方 baseline 的 `88.40 ± 0.84%`。
3. 500 updates 的均值最低，为 `86.13 ± 0.68%`。
4. 2000 updates 相比 250 和 500 updates 有所恢复，但仍低于官方 baseline，且重复评估波动最大（`±1.36%`）。
5. `place_cups`、`place_shape_in_shape_sorter` 和 `stack_blocks` 是整体性能较低或波动较大的任务，应作为后续诊断重点。

## 6. 限定与解释

- 这里的标准差反映同一个训练 checkpoint 的重复评估波动，不是不同训练 seed 之间的方差。
- 当前 replay sampler 提供独立 transition，没有按 episode 形成真实时序样本，因此不能据此证明 `F_phi` 获得了真实 sequence-training 梯度。
- hidden state 是任务相关的内部表示，不等同于真实环境状态。

## 7. 数据与复核

- 完整结构化结果（包括每轮结果、逐 task mean/std、variance 和原始 CSV 路径）：[`data/stage1_hidden_state_runs/five_run_summary.json`](../data/stage1_hidden_state_runs/five_run_summary.json)
- 简版论文式结果表：[`data/stage1_hidden_state_runs/five_run_paper_report.md`](../data/stage1_hidden_state_runs/five_run_paper_report.md)
- 共验证 25 个 CSV，每个包含 18 个 task 行；20 个新增评估日志均包含 18 个 `[Evaluation] Finished` 标记并以 `exit=0` 结束。
