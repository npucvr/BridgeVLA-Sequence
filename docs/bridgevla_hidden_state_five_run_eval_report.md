# BridgeVLA 纯 token-only H-token 路线五轮完整 RLBench 实验报告

## 1. 实验目的

本实验在不改动 action features、rotation/gripper/collision logits 或 translation heatmap 的前提下，评估纯 token-only H-token hidden-state sequence 路线在 BridgeVLA 上的效果。隐藏状态只能通过修正视觉 token 影响动作策略。

路线形式为：

$$
H_t=\operatorname{PaliGemma}(o_t,l),
\qquad y_t^-=F_\phi(y_{t-1},u_{t-1}),
$$

$$
y_t=U_\omega(y_t^-,H_t),
\qquad \widetilde H_t=H_t+A_\psi(H_t,y_t),
$$

$$
u_t=\pi_\theta(\widetilde H_t).
$$

因此，hidden state 不直接注入 action features、rotation/gripper/collision logits 或 translation heatmap；动作解码只接收修正后的 visual tokens `H_tilde`。

## 2. 训练协议

- Benchmark：RLBench
- Demonstrations：每个 task 100 条
- 初始化 checkpoint：`data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth`
- 训练方式：`hidden_state_sequence_training`，完整 episode，route-only
- Sequence window：4（保留 legacy 配置；采样实际覆盖完整 episode）
- Batch size：4
- Epoch：1
- Vision tower：冻结
- 训练模块：仅 `A_psi`、`F_phi`、`U_omega`
- `bptt_length=1`：每个时间步之间截断梯度，但不截断 hidden-state 的数值传递；hidden state 只在 episode 起点重置
- 训练 checkpoint：100、250、500、1000、2000 optimizer updates
- 对应 `train_iter`：400、1000、2000、4000、8000
- 五个训练任务均完成，manifest 状态为 `complete`

| Updates | `train_iter` | Checkpoint |
|---:|---:|---|
| 100 | 400 | `outputs/token_only_full_episode_v1/train/debug_train_iter 400 num_workers 1_hidden_state_enabled True/token_only_100/debug/08_31_15_11/model_last.pth` |
| 250 | 1000 | `outputs/token_only_full_episode_v1/train/debug_train_iter 1000 num_workers 1_hidden_state_enabled True/token_only_250/debug/08_31_15_11/model_last.pth` |
| 500 | 2000 | `outputs/token_only_full_episode_v1/train/debug_train_iter 2000 num_workers 1_hidden_state_enabled True/token_only_500/debug/08_31_15_11/model_last.pth` |
| 1000 | 4000 | `outputs/token_only_full_episode_v1/train/debug_train_iter 4000 num_workers 1_hidden_state_enabled True/token_only_1000/debug/08_31_15_25/model_last.pth` |
| 2000 | 8000 | `outputs/token_only_full_episode_v1/train/debug_train_iter 8000 num_workers 1_hidden_state_enabled True/token_only_2000/debug/08_31_15_28/model_last.pth` |

## 3. 官方完整评测协议

- Tasks：18
- 每个 task 每次运行 25 episodes
- 每次运行共 450 episodes
- 每个 episode 最大步数：25
- 每个 checkpoint：5 次独立运行
- 每个 checkpoint 共 2250 episodes
- `start_episode=0`
- 评测入口：`finetune/RLBench/eval.py --tasks all`

第 $i$ 次运行的 18-task 平均成功率记为 $s_i$，报告均值和运行间 sample standard deviation：

$$
\bar{s}=\frac{1}{5}\sum_{i=1}^{5}s_i,
\qquad
\operatorname{std}_{\mathrm{sample}}
=\sqrt{\frac{1}{4}\sum_{i=1}^{5}(s_i-\bar{s})^2}.
$$

由于每个 task 都有 25 episodes，task success rate 的算术平均与 pooled episode success rate 等价。对照 baseline 使用 `outputs/official_baseline_eval_5runs_v1/summary_official_5run.json`：

- Official baseline：**87.377778 ± 1.590093%**
- 成功 episodes：1966/2250

## 4. 五次运行总体结果

`run_i` 为该次运行 18 个 task success rate 的算术平均；标准差为五次 run-level mean 的 sample standard deviation。`Δ mean` 和 `Δ std` 相对官方 baseline，单位为 percentage points（pp）。

| 配置 | run_1 | run_2 | run_3 | run_4 | run_5 | mean ± sample std | 成功 episodes | Δ mean | Δ std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official baseline | 87.333333 | 85.111111 | 89.555556 | 87.777778 | 87.111111 | **87.377778 ± 1.590093%** | 1966/2250 | — | — |
| Token-only，100 updates | 87.111111 | 88.444444 | 89.111111 | 88.444444 | 88.222222 | **88.266667 ± 0.726908%** | 1986/2250 | +0.888889 | -0.863185 |
| Token-only，250 updates | 88.000000 | 88.888889 | 86.222222 | 86.444444 | 87.333333 | **87.377778 ± 1.104424%** | 1966/2250 | +0.000000 | -0.485669 |
| Token-only，500 updates | 88.888889 | 87.777778 | 86.888889 | 86.666667 | 87.333333 | **87.511111 ± 0.880516%** | 1969/2250 | +0.133333 | -0.709577 |
| Token-only，1000 updates | 89.111111 | 88.888889 | 89.333333 | 87.777778 | 87.333333 | **88.488889 ± 0.880516%** | 1991/2250 | +1.111111 | -0.709577 |
| Token-only，2000 updates | 88.000000 | 87.333333 | 89.111111 | 88.444444 | 87.555556 | **88.088889 ± 0.713191%** | 1982/2250 | +0.711111 | -0.876902 |

## 5. 逐任务五轮统计

以下为 token-only checkpoint 的五次运行逐 task success rate，格式为 `mean ± sample std`，单位为百分比。

| Task | 100 updates | 250 updates | 500 updates | 1000 updates | 2000 updates |
|---|---:|---:|---:|---:|---:|
| `close_jar` | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `reach_and_drag` | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `insert_onto_square_peg` | 92.00 ± 2.83 | 90.40 ± 6.69 | 88.80 ± 3.35 | 92.80 ± 1.79 | 88.00 ± 5.66 |
| `meat_off_grill` | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `open_drawer` | 100.00 ± 0.00 | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.20 ± 1.79 |
| `place_cups` | 60.80 ± 8.67 | 52.80 ± 11.10 | 57.60 ± 7.27 | 57.60 ± 6.69 | 64.00 ± 11.31 |
| `place_wine_at_rack_location` | 91.20 ± 5.22 | 90.40 ± 8.29 | 88.00 ± 2.83 | 85.60 ± 6.69 | 88.80 ± 7.69 |
| `push_buttons` | 99.20 ± 1.79 | 98.40 ± 2.19 | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `put_groceries_in_cupboard` | 76.80 ± 3.35 | 77.60 ± 4.56 | 74.40 ± 6.07 | 79.20 ± 4.38 | 76.80 ± 6.57 |
| `put_item_in_drawer` | 95.20 ± 1.79 | 96.00 ± 0.00 | 93.60 ± 7.80 | 99.20 ± 1.79 | 98.40 ± 2.19 |
| `put_money_in_safe` | 99.20 ± 1.79 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| `light_bulb_in` | 89.60 ± 3.58 | 92.00 ± 6.93 | 88.80 ± 4.38 | 88.00 ± 8.49 | 88.80 ± 3.35 |
| `slide_block_to_color_target` | 97.60 ± 3.58 | 96.80 ± 1.79 | 96.80 ± 3.35 | 98.40 ± 2.19 | 96.00 ± 5.66 |
| `place_shape_in_shape_sorter` | 57.60 ± 6.69 | 57.60 ± 4.56 | 58.40 ± 9.21 | 60.00 ± 6.32 | 60.00 ± 4.90 |
| `stack_blocks` | 72.80 ± 5.93 | 66.40 ± 4.56 | 76.00 ± 5.66 | 79.20 ± 7.16 | 65.60 ± 9.21 |
| `stack_cups` | 82.40 ± 6.07 | 81.60 ± 3.58 | 79.20 ± 4.38 | 81.60 ± 3.58 | 81.60 ± 2.19 |
| `sweep_to_dustpan_of_size` | 84.00 ± 2.83 | 82.40 ± 4.56 | 84.80 ± 4.38 | 80.80 ± 5.22 | 87.20 ± 1.79 |
| `turn_tap` | 90.40 ± 6.07 | 91.20 ± 3.35 | 90.40 ± 4.56 | 90.40 ± 3.58 | 91.20 ± 3.35 |

## 6. 结果分析

1. 1000 updates 的平均成功率最高，为 **88.488889%**，相对 baseline 为 **+1.111111 pp**；其运行间 sample std 为 **0.880516 pp**，低于 baseline 的 1.590093 pp。
2. 100 updates 为 **+0.888889 pp**，250 updates 与 baseline 相同，500 updates 为 **+0.133333 pp**，2000 updates 为 **+0.711111 pp**。
3. 五个 token-only checkpoint 的运行间标准差都低于本次 baseline，但这只反映同一 checkpoint 的五次官方评估波动，不能替代训练 seed 方差。
4. 性能不随训练长度单调增加：1000 updates 达到峰值后，2000 updates 回落至 88.088889%。因此，1000 updates 的提升应视为当前实验配置下的结果，不宜外推为确定性的普遍增益。
5. `place_cups`、`place_shape_in_shape_sorter` 和 `stack_blocks` 仍是主要瓶颈；其中 `place_cups` 在 2000 updates 达到 64.00%，`sweep_to_dustpan_of_size` 在 2000 updates 达到 87.20%，但任务间差异和运行间波动仍然存在。

### 6.1 指标提升集中在哪里

相对 baseline 的提升并不是所有任务均匀增加，而是集中在若干需要持续记忆或多阶段动作组织的任务：

| Task | baseline | 100 updates | 250 updates | 500 updates | 1000 updates | 2000 updates |
|---|---:|---:|---:|---:|---:|---:|
| `light_bulb_in` | 81.60 | +8.00 | +10.40 | +7.20 | +6.40 | +7.20 |
| `place_cups` | 52.00 | +8.80 | +0.80 | +5.60 | +5.60 | +12.00 |
| `stack_cups` | 78.40 | +4.00 | +3.20 | +0.80 | +3.20 | +3.20 |
| `place_shape_in_shape_sorter` | 56.80 | +0.80 | +0.80 | +1.60 | +3.20 | +3.20 |
| `insert_onto_square_peg` | 88.80 | +3.20 | +1.60 | +0.00 | +4.00 | -0.80 |
| `stack_blocks` | 77.60 | -4.80 | -11.20 | -1.60 | +1.60 | -12.00 |
| `sweep_to_dustpan_of_size` | 85.60 | -1.60 | -3.20 | -0.80 | -4.80 | +1.60 |

1000 updates 的总增益主要由 `light_bulb_in`（+6.40 pp）、`place_cups`（+5.60 pp）、`insert_onto_square_peg`（+4.00 pp）、`place_shape_in_shape_sorter`（+3.20 pp）和 `stack_cups`（+3.20 pp）贡献；2000 updates 中 `place_cups` 和 `light_bulb_in` 的提升仍在，但被 `stack_blocks` 的 -12.00 pp 退化部分抵消。这说明当前方案改善了部分时序决策，却没有解决所有长任务。

### 6.2 指标提升的可能原因

以下是与实现和结果模式一致的机制解释，但不是由本次单组实验单独证明的因果结论：

1. **完整 episode 的状态连续性减少了随机窗口断点。** `SequenceReplayBuffer` 从 episode 首 transition 开始按 terminal/timeout 顺序展开，padding 通过 `valid_mask=0` 排除；训练状态只在 episode 起点初始化。因此，`U_omega`/`F_phi` 能在连续观测和动作上下文中工作，而不是每个随机窗口都从零状态开始。这与 `light_bulb_in`、`place_cups`、`stack_cups` 等多阶段任务的提升相吻合。
2. **route-only 限制了干预范围。** 当前路径为 `H_t -> U_omega -> y_t -> A_psi -> H_tilde -> policy`，没有直接修改 action features 或其他动作 head。`A_psi` 的 output projection 和 hidden branch 零初始化，训练初始点接近原始 BridgeVLA；因此模型可在保留原有高成功率任务的同时，对困难任务提供小的 token residual 修正。stage-two 的第二次视觉处理还复用第一次 `U_omega` 得到的 posterior，避免同一决策内出现两个不一致的 hidden-state 更新。
3. **冻结视觉塔和低参数适配降低了遗忘与训练噪声。** 视觉表征以及主策略保持固定，只训练 `A_psi`、`F_phi`、`U_omega`。结果中 `close_jar`、`meat_off_grill` 等任务保持 100%，且五个 checkpoint 的 run-level 标准差均低于 baseline，这与“在原策略附近做受限适配”的解释一致，但也可能只是小样本评估波动。需要区分：单独 `--freeze_vision_tower` 并不等价于只训练这三个模块，本次实验依靠显式 route-only 配置完成严格冻结。
4. **`bptt_length=1` 提供了稳定的局部训练目标。** 它在每步切断旧 `U_omega` 图，同时保留 hidden state 的数值传递，并让当前一步的 `F_phi` 仍获得下一步 action loss 的梯度。这样避免了长 episode 反向图和跨长链错误 credit assignment，可能有利于稳定训练；但它并没有学习跨多步的长期 credit assignment，提升不能简单归因于“长 BPTT”。
5. **采样分布和有效监督量发生了改变。** full-episode 模式按 episode anchor 采样，但每个 episode 的所有 valid transition 都参与监督，长 episode 因而贡献更多有效步；这与 legacy 随机 transition/window 训练并不严格等价。指标提升可能部分来自更合适的时序数据分布，而不只是 hidden-state 模块本身。
6. **padding 会带来计算开销而非额外监督。** batch 的 padded length 取所有保留 episode 的最大长度，短 episode 的重复帧虽被 `valid_mask=0` 屏蔽、不推进状态且不计 loss，仍可能经过 forward。因此，full-episode 的收益不能解释为“增加了 padding 数据”，且长度差异较大时会降低吞吐。

### 6.3 不能从本次结果推出的结论

- 序列训练使用 demonstration action 进行 teacher forcing：`F_phi` 推进时读的是 replay 中的 expert action，而部署时使用模型自己的 action；因此真实执行中的 action distribution shift 可能削弱收益。
- 五次评估只覆盖一个训练 checkpoint/数据配置的评估随机性，不能代表训练 seed 方差。1000 updates 的 +1.11 pp 尚未经过显著性检验；以 token std=0.881、baseline std=1.590、各 n=5 粗略估计，独立比较的标准误约为 0.813 pp，差异相对运行噪声并不强。
- 每个 task 每次只有 25 episodes，task success rate 的量化步长为 4 pp；18 个 task 在汇总中等权。五个 checkpoint 和 18 个 task 的多重比较，以及事后选择最高的 1000 updates，会放大“最佳点”被偶然选中的可能。
- 当前 token 与 baseline 不是共享训练 seed/同一 checkpoint 的严格配对观测，不能将 run-level 差异当作配对因果证据。
- 不能据此证明模型学到了真实、可泛化的长期环境状态，也不能区分 hidden state、采样分布、参数冻结或低秩 residual 正则化各自的贡献。
- 仓库默认配置中的 `sequence_training` 和 `hidden_state_enabled` 均为关闭状态；本实验依靠训练命令显式开启 sequence/route/hidden-state 选项。仅设置 `full_episode` 或 `bptt_length` 不会自动启用整条路径。

### 6.4 建议的后续验证

为把“可能原因”变成可验证结论，下一轮应在相同 checkpoint、数据、updates 和评估 seeds 下做：

1. full-episode 与随机 window 的对照；
2. `bptt_length=1` 与更长 BPTT 的对照；
3. route-only、直接 action-feature 注入、解冻视觉塔的消融；
4. teacher-forced hidden-state 与模型 rollout hidden-state 的对照；
5. 多训练 seed，并记录有效 transition 数、BC loss、hidden-state norm/漂移和每任务 episode 长度分层统计。

## 7. 结论与限定

在严格纯 token-only H-token 路线下，当前完整五轮官方 RLBench 结果显示：**1000 updates 获得了相对 baseline 的 +1.11 pp 平均提升，同时运行间标准差较低；但不同训练长度并不呈单调趋势，不能仅凭本实验认定该路线带来了稳定、普遍的性能增益。**

本实验没有将 hidden state 直接加入 action features 或 translation heatmap，所有结果均来自修正后的 visual-token route。`bptt_length=1` 只截断跨时间步的梯度，不会截断 episode 内 hidden state 的数值连续性；标准差是同一训练 checkpoint 的五次独立官方评估之间的 sample standard deviation，不是不同训练 seed 之间的方差。

## 8. 数据、日志与复核

- 训练 manifest：`outputs/token_only_full_episode_v1/training_manifest.json`
- 当前五轮汇总：`outputs/token_only_full_episode_eval_v1/summary_token_only_full_episode_5run.json`
- 当前评测日志与 CSV：`outputs/token_only_full_episode_eval_v1/`
- 官方 baseline 汇总：`outputs/official_baseline_eval_5runs_v1/summary_official_5run.json`
- 基础 checkpoint：`data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth`

最终复核结果：

- 5 个 checkpoint 文件均存在；
- 25 次评测进程均以状态 0 完成；
- 25 个 `eval_results.csv` 均存在且各含 18 个 task；
- 25 次评测的任务顺序和统计口径一致；
- 当前正式评测日志未发现 `Traceback`、`RuntimeError`、CUDA OOM、Xvfb 启动失败或非零结束状态。
