# BridgeVLA Prior-Observation Hidden-State 实验结果汇报

## 1. 结论摘要

本实验完成了从历史状态先验 `y_t^-` 预测当前观测表征的辅助训练，并按完整论文 RLBench 协议完成评估。

| 项目 | 结果 |
|---|---:|
| 训练协议 | token-only hidden-state route，2000 optimizer updates |
| 辅助损失 | `hidden_state_observation_loss_weight=0.05` |
| 评估任务 | 18 个 RLBench tasks |
| 每次运行 | 25 episodes/task，450 episodes/run |
| 独立重复 | 5 次，共 2250 episodes |
| 成功 episodes | **2008/2250** |
| 总体成功率 | **89.2444%** |
| 5-run sample std | **0.8692%** |
| 相对论文表格均值 | **+1.04 percentage points** |

正式训练与评估均已正常结束，无 OOM、DDP、CoppeliaSim 或 traceback 错误。

## 2. 实验目标与因果时序

辅助 decoder 的输入严格为当前观测吸收之前的先验状态：

$$
 y_t^- = F_\phi(y_{t-1},u_{t-1}),
$$

$$
 \hat h_t = D_\eta(y_t^-),
 \qquad
 y_t = U_\omega(y_t^-,H_t).
$$

当前 visual tokens `H_t` 只作为 detached target：

$$
 \mathcal L_{\mathrm{obs}}
 = -\log p_\eta(\phi(H_t)\mid y_t^-).
$$

因此 decoder 不接收当前 `H_t`，也不接收 posterior `y_t`。当前实现使用每个 rendered view 的 visual tokens 做平均池化并独立归一化，作为紧凑的 `H_t` 目标，而不是重建原始像素 `z_t`。

## 3. 训练配置

| 配置项 | 值 |
|---|---|
| 初始化 checkpoint | `data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth` |
| hidden-state | enabled |
| prior observation prediction | enabled |
| sequence trainer | enabled，full episode |
| sequence length | 4（保留 legacy 配置） |
| `bptt_length` | 1 |
| batch size | 4 |
| `train_iter` | 8000 |
| optimizer updates | 2000（单卡，`8000 / 4`） |
| epoch | 1 |
| route mode | hidden-state route-only |
| vision tower | frozen |
| trainable modules | `A_psi`、`F_phi`、`U_omega`、`ObservationDecoder` |
| observation loss weight | `0.05` |
| PaliGemma | `/remote_userdata/sunguodong/repos/BridgeVLA/data/bridgevla_ckpt/paligemma-3b-pt-224` |
| 训练节点/GPU | `server17` / physical GPU2 |
| 训练数据 | `/data3/local_userdata/sunguodong/BridgeVLA/data/RLBench_TRAIN_DATA` |
| replay data | `/data3/local_userdata/sunguodong/BridgeVLA/data/replay_train_k4` |

训练日志最终显示：

```text
100%|██████████| 2000/2000 ...
[Finish]
```

正式 checkpoint：

```text
outputs/prior_observation_prediction_v1/train/
  debug_train_iter 8000 num_workers 1 rvt.hidden_state_observation_loss_weight 0.05_*/
  prior_obs_w005_2000updates/debug/09_01_21_38/model_last.pth
```

该 checkpoint 大小约 `7.5 GiB`，包含 5 组 `mvt1.observation_decoder.*` 参数，参数值均为有限值。

## 4. Smoke 验证

正式训练前完成了单卡 one-update smoke run，验证内容包括：

- local PaliGemma checkpoint 离线加载；
- prior decoder 在 `U_omega` 前执行；
- target 不向 visual encoder 反向传播；
- stage-two 不重复计算 observation prediction；
- checkpoint 能保存并包含 observation decoder 参数。

Smoke checkpoint：

```text
outputs/prior_observation_prediction_v1/train/
  debug_train_iter 4 num_workers 0 rvt.hidden_state_observation_loss_weight 0.05_*/
  prior_obs_smoke_gpu2_v2/debug/09_01_21_36/model_last.pth
```

## 5. 完整论文评估协议

评估使用 held-out 数据集，而非训练 demonstrations：

- 数据：`/data3/local_userdata/sunguodong/BridgeVLA/data/RLBench_EVAL_DATA`；
- tasks：18；
- 每个 task：25 episodes；
- 每次运行：450 episodes；
- 重复次数：5；
- 总 episodes：2250；
- 每个 episode 最大控制步数：25；
- `start_episode=0`；
- GPU：server17 GPU1、GPU2；
- displays：`:4.0`、`:5.0`；
- CoppeliaSim 实例使用独立 display；
- 评估结果写入 NFS，数据从 server17 本地高速目录读取。

为复用现有五重复 launcher，评估 staging 目录中的 `model_80.pth` 是指向上述训练后 `model_last.pth` 的符号链接；同时 staging 目录链接了训练生成的 `exp_cfg.yaml` 和 `mvt_cfg.yaml`。

## 6. 五次运行结果

| Run | 成功 episodes | 成功率 |
|---:|---:|---:|
| run 1 | 403/450 | 89.555556% |
| run 2 | 398/450 | 88.444444% |
| run 3 | 397/450 | 88.222222% |
| run 4 | 404/450 | 89.777778% |
| run 5 | 406/450 | 90.222222% |
| **均值** | **2008/2250** | **89.244444%** |
| **sample std** | — | **0.869227%** |

## 7. 逐任务五轮统计

下表为五次 run 的 task-level success rate 均值与 sample standard deviation，单位为百分比。

| Task | Mean ± sample std |
|---|---:|
| `close_jar` | 100.00 ± 0.00 |
| `reach_and_drag` | 99.20 ± 1.79 |
| `insert_onto_square_peg` | 87.20 ± 1.79 |
| `meat_off_grill` | 100.00 ± 0.00 |
| `open_drawer` | 100.00 ± 0.00 |
| `place_cups` | 62.40 ± 4.56 |
| `place_wine_at_rack_location` | 88.80 ± 5.22 |
| `push_buttons` | 99.20 ± 1.79 |
| `put_groceries_in_cupboard` | 79.20 ± 5.22 |
| `put_item_in_drawer` | 99.20 ± 1.79 |
| `put_money_in_safe` | 100.00 ± 0.00 |
| `light_bulb_in` | 89.60 ± 5.37 |
| `slide_block_to_color_target` | 95.20 ± 3.35 |
| `place_shape_in_shape_sorter` | 61.60 ± 4.56 |
| `stack_blocks` | 79.20 ± 4.38 |
| `stack_cups` | 82.40 ± 4.56 |
| `sweep_to_dustpan_of_size` | 87.20 ± 1.79 |
| `turn_tap` | 96.00 ± 2.83 |

## 8. 对照结果

### 8.1 与论文表格 task-average 对照

现有 `compare_to_paper.py` 对论文表格中的 task-average `88.20%` 进行比较：

- prior-observation hidden-state：`89.24%`；
- 论文表格均值：`88.20%`；
- 差值：`+1.04 pp`；
- mean absolute per-task delta：`1.33 pp`。

### 8.2 与已有 official baseline 五轮结果对照

已有审计 baseline 为 `87.377778 ± 1.590093%`，成功 `1966/2250`。本实验为：

- 成功率：`89.244444%`；
- 成功 episodes：`2008/2250`；
- 相对 baseline：`+1.866667 pp`；
- 成功 episodes：`+42`。

两种对照的统计口径不同：论文表格对 task-level 均值进行比较，official baseline 使用已有五轮 episode 汇总结果；原始五轮 CSV 和脚本输出应作为最终审计依据。

## 9. 产物与复现入口

完整评估目录：

```text
outputs/prior_observation_prediction_v1/eval/prior_obs_w005_full_5run/
```

主要文件：

- 汇总：`outputs/prior_observation_prediction_v1/eval/prior_obs_w005_full_5run_launcher/aggregate.txt`
- 完整 launcher 状态：`outputs/prior_observation_prediction_v1/eval/prior_obs_w005_full_5run_launcher/status.txt`
- launcher 日志：`outputs/prior_observation_prediction_v1/eval/prior_obs_w005_full_5run_launcher/launcher_stdout.log`
- 每次运行的 episode 日志：`.../prior_obs_w005_full_5run_launcher/run_1.log` 至 `run_5.log`
- 每次运行的 task CSV：`.../prior_obs_w005_full_5run/run_1/model_80/eval_results.csv` 至 `run_5/model_80/eval_results.csv`
- 单次评估配置：`.../prior_obs_w005_full_5run/run_1/eval_config.yaml`

评估 launcher 使用的协议命令等价于：

```bash
export MODEL_FOLDER=outputs/prior_observation_prediction_v1/eval_model_last
export EVAL_DATAFOLDER=/data3/local_userdata/sunguodong/BridgeVLA/data/RLBench_EVAL_DATA
export EVAL_OUTPUT_ROOT=outputs/prior_observation_prediction_v1/eval
export RESULT_LOG_DIR=prior_obs_w005_full_5run
export LOG_DIR=outputs/prior_observation_prediction_v1/eval/prior_obs_w005_full_5run_launcher
export GPU_IDS=1,2
export DISPLAY_IDS=:4.0,:5.0
export EVAL_EPISODES=25
export EPISODE_LENGTH=25

bash scripts/rlbench_repro/run_repeated_eval.sh
```

## 10. 结论与限制

在当前训练预算和单个辅助损失权重 `0.05` 下，prior-observation hidden-state 路线在完整 RLBench 五重复评估中达到 `89.24%`，没有观察到相对已有 baseline 的性能下降。

该结果支持以下较弱结论：历史状态先验预测当前视觉表征的辅助目标可以与当前观测吸收共存，并且在本次实验中未破坏 token-only action route。

仍不应据此声称 hidden state 已恢复 simulator true state。当前实验还有以下限制：

1. 观测目标是 pooled visual-token representation，而不是原始图像或完整 `z_t`；
2. 只测试了一个训练 checkpoint 和一个辅助权重；
3. 尚未完成 `weight=0` 的同预算 matched ablation；
4. 尚未实现 simulator-state reconstruction loss `\mathcal L_x`；
5. 当前训练仍未实现跨 batch 的 episode hidden-state carry。
