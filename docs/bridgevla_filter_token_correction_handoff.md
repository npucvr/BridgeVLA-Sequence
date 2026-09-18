# BridgeVLA 滤波器式 token correction — 交接文档

更新日期：2026-09-16　适用的方法设计：`docs/bridgevla_filter_token_correction_design.md`

## 0. 一句话状态

方法设计已完成、代码已接入并通过端到端训练验证。首次 400 步 smoke 查出两处尺度设定缺陷，导致 token 修正通路在数值上等于关闭（有效修正仅 $5\times10^{-6}$）；本轮已完成 $B$ 的标定与 $R$ 的初始化修复，CPU/真实维度前向及 `server17` GPU7 的同配置 400-update smoke 均已通过数值健康检查。

**已完成一次正式 2000-update 训练及五次 held-out RLBench 评估：18-task 平均 87.87%，与论文表中 88.20% 的差值为 −0.33pp。** 该结果仍不是与 $λ_{\mathrm{innov}}=0$ 或原模型的严格匹配因果对照。

## 1. 目标与硬约束

目标：在**冻结** BridgeVLA（PaliGemma + 全部动作模块）的前提下，用一个约 20 万参数的可训练模块，把视觉 token 修正为

$$
\widetilde H_t=H_t+\alpha\,B\,C_\theta K_te_t,
\qquad
e_t=z_t-C_\theta\mu_t^-,
\qquad
K_t=\Sigma_t^-C_\theta^\top\left(C_\theta\Sigma_t^-C_\theta^\top+R_\theta\right)^{-1}
$$

约束：不加重视觉主干、不生成视频、不做候选动作规划、不改原 action decoder；$\alpha$ 零初始化，初始时严格等于原模型。

与 LoRA 类方法的区别是**表达能力**：LoRA 没有状态、没有 $u_{t-1}$、没有新息，表达不了"只在观测偏离动作预测时才修正"。

## 2. 代码改动清单

新增：

| 文件 | 内容 |
|---|---|
| `finetune/bridgevla/hidden_state/filter_correction.py` | `MeasurementProjection`（冻结列正交 $P$ 及精确伴随）、`FilterCorrection`（滤波器模型 + 信息形式卡尔曼更新 + token 修正 + §10.3 诊断量）、`pack_state`/`unpack_state`；$R$ 默认初始化为 48 |

修改（当前 hidden-state 路线收敛为 filter 实现；默认关闭时原始
BridgeVLA 路径仍保持不变）：

| 文件 | 改动 |
|---|---|
| `finetune/bridgevla/hidden_state/__init__.py` | 只导出 `F_phi`、`FilterCorrection` 与 `MeasurementProjection` |
| `finetune/bridgevla/mvt/mvt_single.py` | 只保留 filter 构造、forward、打包信念初始化与转移 |
| `finetune/bridgevla/mvt/mvt.py` | **补声明 filter 参数**（见 §7.1） |
| `finetune/bridgevla/mvt/config.py` | 6 个 filter 侧开关，含 `hidden_state_filter_init_log_measure_noise` |
| `finetune/bridgevla/config.py` | `rvt.hidden_state_filter_innovation_loss_weight` |
| `finetune/bridgevla/models/bridgevla_agent.py` | 新息损失项、校验、日志键、§10.3 诊断量记录；`train()` 只保留 filter 模块 |
| `finetune/RLBench/train.py` | checkpoint 兼容过滤、filter 冻结模块列表、参数校验 |

删除：`hidden_state/observation_decoder.py`、`hidden_state/observation_update.py`、
`hidden_state/token_correction.py` 及其专用配置和损失分支。旧实验文档、checkpoint
与评估结果不删除；旧路线代码不再作为运行时实现维护。

配置开关：`hidden_state_filter_correction`（默认 False）、`hidden_state_filter_measure_dim`（64）、`hidden_state_filter_grid`（4）、`hidden_state_filter_full_covariance`（True）、`hidden_state_filter_seed`（0）、`hidden_state_filter_init_log_measure_noise`（48.0）、`rvt.hidden_state_filter_innovation_loss_weight`（0.0）。

开启 hidden-state 后只构建 `F_phi` 与 `FilterCorrection`；旧的
`U_omega`、`A_psi`、`observation_decoder` 已从运行时代码删除。

## 3. 文档与产物位置

正式文档：

- `docs/bridgevla_filter_token_correction_design.md` —— 方法设计（§3 形式化、§5 流程图、§10.2 离线验证结果、§11.5/§11.6 smoke 诊断）
- `docs/archive/` —— 已归档的旧路线文档（hidden-state / prior-observation）

工作产物（`agent_playground/` 已被 `.gitignore` 排除，**长期需要请提升到 `scripts/` 或 `docs/`**）：

| 路径 | 内容 |
|---|---|
| `agent_playground/filter_tests/test_filter_correction.py` | CPU 单元测试，16 项 / 65 断言 |
| `agent_playground/filter_tests/smoke_real_geometry.py` | 真实几何冒烟 |
| `agent_playground/filter_stats/dump_measurements.py` | 测量采集（复用训练管线，含渲染器 dtype shim） |
| `agent_playground/filter_stats/innovation_statistics.py` | 离线新息统计（CPU） |
| `agent_playground/filter_stats/make_synthetic.py` | 统计脚本的合成数据校验 |
| `agent_playground/filter_stats/RESULTS.md` | §10.2 完整结果与限制 |
| `agent_playground/filter_stats/SMOKE_RESULTS.md` | smoke 结果与根因对账 |
| `agent_playground/filter_stats/out/` | 采集数据（`.npz`）与统计报告（`.json`） |
| `outputs/filter_route_smoke/` | 初版训练日志与 checkpoint（**两个 8 GB 文件，如不需要可删**） |
| `outputs/filter_route_smoke_calibrated/` | 本轮标定修复 smoke 日志与 checkpoint |
| `agent_playground/filter_stats/out/calibrated_probe_4ep_dim64.*` | 4 个 `close_jar` demonstration 的 route checkpoint 诊断（4 episodes / 12 steps） |

## 4. 已验证的事实

1. **数学正确**：伴随性质、与显式构造 $S$ 求逆的暴力实现逐项一致、零初始化严格恒等、打包/解包往返、协方差传播保持对称正定、梯度可达而 $P$ 无梯度。
2. **接入完整**：400 optimizer updates 端到端跑通（序列状态递推、反向传播、优化器、损失接线、checkpoint 存取），11 分 05 秒，无报错。
3. **状态传递无需改 agent 序列循环**：信念打包成 $[B,d+d^2]$ 张量即可复用现有的行索引 / `index_copy` / `torch.where` / `detach`。
4. **承重假设未被推翻**（§10.2）：$z_t$ 可由 $(z_{t-1},u_{t-1})$ 预测（held-out 解释方差 **71%**），动作携带特定信息（打乱动作后增益完全消失），新息 lag-1 自相关 −0.026、能量占比 1.6%。
5. **参数规模**：2 视角时 `FilterCorrection` 本体 137,345；实际 3 视角、$m=3072$ 时约 20.4 万。

## 5. 已知缺陷与必须的修法

### 5.1 修正通路被三重衰减压掉（最关键）

实测有效修正 $\lvert\alpha\rvert\|\Delta H\|/\|H\|=4.93\times10^{-6}$，而

$$
\lvert\alpha\rvert\times\frac{1}{\sqrt{D}}\times\frac{1}{\text{cell\_tokens}}
=3.36\times10^{-3}\times2.21\times10^{-2}\times6.25\times10^{-2}
=4.64\times10^{-6}
$$

与实测吻合。伴随算子数学上正确，但 $P$ 是收缩算子，$P^\top$ 同样收缩。

**修法状态**：已改为列正交的半正交投影、逐格求和及精确伴随复制（不再除 `cell_tokens`），并由 CPU 伴随/尺度测试、真实维度前向和端到端 smoke 验证。4 回合真实前向的 `residual_ratio` 为 $3.162425\times10^{-3}$，旧版为 $4.93\times10^{-6}$；正式性能实验仍然禁止启动。

### 5.2 结构性增益塌缩

$m=3072$ 个测量对 $d=64$ 个状态维度、$R\approx0.69$，$C^\top R^{-1}C$ 对角约 70，一次更新把 $\operatorname{tr}(\Sigma)$ 从 64 压到 0.473，此后 $K$ 退化为近似常数且接近 0。

**修法状态**：已新增 `hidden_state_filter_init_log_measure_noise` 配置，默认值并在 smoke 命令中显式设为 **48.0**。量级估算：$C_\theta$ 按 $\mathcal N(0,1/d)$ 初始化时 $\left(C^\top R^{-1}C\right)_{ii}\approx m/(dR)$，代入 $m=3072$、$d=64$ 得信息项约 $48/R$。旧值 $R=0.69$ 给出约 70，$\operatorname{tr}(\Sigma^+)$ 从 64 塌到 0.473；$R\approx48$ 使真实维度未训练首步后验 trace 为 32.15，符合只减半的目标。标定后 4 回合真实前向的 posterior trace 为 19.57753，未再出现首步塌缩；`measure_noise_trace=147474.3` 与 $3072\times48$ 同量级。这里 `init_log_measure_noise` 是 softplus 的逆参数，在 48 处近似等于实际噪声值。

### 5.3 其他

- **滤波模块共用 `peract.lr=8e-5`**：$\alpha$ 400 步只走了 $3.4\times10^{-3}$，建议单独 lr。
- **$\alpha$ 变负**（$-3.36\times10^{-3}$）：说明当前修正对动作损失有害。需一次 $\lambda_{\mathrm{innov}}=0$ 的对照来区分"动作损失在压它"还是"新息损失在推它"。方向与旧路线在 matched 对照里的 −0.93pp 一致，值得警惕。
- **没有周期性文本日志**：`print_loss_log` 在仓库里**没有任何调用者**，损失只进 wandb。判断"更新数是否足够"前必须补这个。

## 6. 复现命令

### 6.1 环境

节点：训练用 `server17`（GPU 6/7）；`server19` 的 8 张卡常被他人占满，**每次运行前必须重新核验**。

server17 上的环境与本地数据（NFS 不是热数据读取路径，replay 已在节点本地）：

```bash
export BRIDGEVLA_RLBENCH_DATA_FOLDER=/data3/local_userdata/sunguodong/BridgeVLA/data/RLBench_TRAIN_DATA
export BRIDGEVLA_RLBENCH_TRAIN_REPLAY_DIR=/data3/local_userdata/sunguodong/BridgeVLA/data/replay_train_k4
source scripts/bridgevla_runtime.sh      # 解析出 python=/home/sunguodong/miniconda3/envs/bridgevla/bin/python
```

### 6.2 单元测试（任意节点，CPU）

```bash
/home/sunguodong/miniforge3/envs/bridgevla/bin/python \
  agent_playground/filter_tests/test_filter_correction.py
```

### 6.3 训练（server17）

```bash
cd finetune/RLBench
CUDA_VISIBLE_DEVICES=7 GPUS_PER_NODE=1 setsid nohup bash train.sh \
  --log_dir /remote_userdata/sunguodong/repos/BridgeVLA/outputs/filter_route_smoke_calibrated \
  --exp_note filter_calibrated_400updates --debug --epochs 1 --num_train 100 \
  --data_folder "$BRIDGEVLA_RLBENCH_DATA_FOLDER" \
  --train_replay_storage_dir "$BRIDGEVLA_RLBENCH_TRAIN_REPLAY_DIR" \
  --init_checkpoint /remote_userdata/sunguodong/repos/BridgeVLA/data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth \
  --hidden_state_route_only --hidden_state_sequence_training \
  --hidden_state_sequence_full_episode --hidden_state_sequence_bptt_length 1 \
  --hidden_state_sequence_length 4 \
  --exp_cfg_opts "bs 2 train_iter 800 num_workers 1 rvt.hidden_state_filter_innovation_loss_weight 0.05" \
  --mvt_cfg_opts "hidden_state_enabled True hidden_state_filter_correction True hidden_state_filter_init_log_measure_noise 48.0 hidden_state_dim 64 paligemma_path /remote_userdata/sunguodong/repos/BridgeVLA/data/bridgevla_ckpt/paligemma-3b-pt-224" \
  > <out>/train.log 2>&1 < /dev/null &
```

`train_iter=800`、`bs=2`、单卡 ⇒ `800/(2×1)=400` optimizer updates。

### 6.4 采集测量（离线统计用）

见 `agent_playground/filter_stats/dump_measurements.py` 的模块 docstring；跑完会额外打印 §10.3 的诊断量。

### 6.5 离线统计（CPU）

```bash
python agent_playground/filter_stats/innovation_statistics.py --input out/keypoint_probe.npz
```

## 7. 踩过的坑（务必先读，避免重复）

### 7.1 `mvt.py` 用 `locals()` 构造子模块参数

`mvt.py` 里是 `args = copy.deepcopy(locals())` 再 `MVTSingle(**args)`。**新增配置项必须同时在 `mvt.py` 的显式签名里声明**，否则会变成未知关键字直接 TypeError。建议每次加配置项后跑一次静态检查：AST 解析 `MVT.__init__` / `RVTAgent.__init__` 的签名，比对全部配置键是否被接受。

### 7.2 `stage_two` 必须保持 True

`rvt2.yaml` 设 `stage_two: True`，且 `RVTAgent.get_q` 从第二遍读取 `feat_x`。设成 False 会直接崩。因此滤波模块每步被调用两次（第一遍编码 + 第二遍 view-space 精修）；约定为**每步只推进一次信念，但每一遍都修正自己的 token**。

### 7.3 渲染器 dtype 冲突

`trans_pc` 会把点云降为 float32，而渲染器相机位姿是 float64，且颜色通道**必须**是 float32——TorchScript 不做类型提升，会报 `expected scalar type Double but found Float`。采集脚本里有一个 dtype shim（两种精度都试、颜色固定 float32），**只作用于采集脚本，没有改 vendored 包**。若正式训练也触发，需要单独处理。

### 7.4 replay 缓存是关键点级，不是逐帧

`replay_train_k4` 每段回合只有 1–6 步（实测 `{1:8, 3:1, 4:4, 5:7, 6:10}`）。要在设计分辨率 $m=3072$ 下做统计，必须改用 `RLBench_TRAIN_DATA/<task>/all_variations/episodes/episodeN/` 的**逐帧**观测自建输入。

### 7.5 远端任务会被本地 ssh 超时杀掉

ssh 客户端被中断时远端进程收到 SIGHUP 而终止（曾白跑 10 分钟）。**必须用 `setsid nohup ... &` 完全脱离会话**，再轮询产物文件。

### 7.6 卸载 / 关闭 wandb 会丢掉全部损失可见性

`print_loss_log` 无调用者，损失只进 wandb。`WANDB_MODE=disabled` 后日志里只有 tqdm 进度条。要么让 `--debug` 走 wandb offline 再解析，要么先补文本日志。

### 7.7 已修的 bug

`RVTAgent.train()` 原本硬编码旧路线模块，滤波模式下这些模块不存在 →
真实训练会直接崩。现在训练模式只保留 `F_phi` 与 `filter_correction`。

## 8. 未验证项

- $λ_{\mathrm{innov}}=0$ 的严格匹配对照及原模型对照尚未完成，当前结果不能单独归因于滤波修正；
- 逐迭代损失曲线未采集；
- 正式训练仅验证了一个 2000-update seed，跨 seed 稳定性未知；
- 显存与吞吐未系统记录；
- §10.1 的机制主张（收益是否集中在高歧义帧）**完全未检验**，需闭环 rollout；
- `stage_two` 约定只有接口与单测覆盖，无端到端对照。

## 9. 开放决策

1. $P$ 的尺度修复已实现并通过端到端 smoke：列正交投影 + 求和/复制伴随；长期稳定性仍待更长训练验证；
2. $R_\theta$ 的初始化已实现并通过端到端 smoke：默认 48.0；正式训练中的长期校准仍待验证；
3. 滤波模块是否单独设 lr；
4. 是否需要 $\lambda_{\mathrm{innov}}=0$ 的对照；
5. 手设卡尔曼基线是否复现；
6. 周期性文本日志的实现方式。

## 10. 建议的下一步顺序

1. **已完成** §5.1（$B$ 标定）与 §5.2（$R$ 初始化），并加单元测试断言投影/提升尺度及后验不塌缩；
2. 补周期性文本日志；
3. **已完成**：同配置 400 步 smoke；4 回合真实前向的判据为 `residual_ratio=3.162425e-3`、`posterior_trace=19.57753`、`measure_noise_trace=147474.3`、`process_noise_trace=0.65058`，均未触发数值退化；
4. 下一步跑 $\lambda_{\mathrm{innov}}=0$ 对照，判断新旧 smoke 中 $\alpha$ 方向变化的来源；
5. 正式滤波路线实验及五次评估已完成；后续对照仍须匹配 bs / 更新数 / 数据 / seed（历史上 bs 不同就造成过 1.77pp 的差异）；
6. 机制主张需要闭环 rollout 数据，优先级排在性能对照之后。

## 11. 本轮续接结果（2026-09-15）

### 11.1 验证结果

- CPU 单元测试：16 项 / 65 断言全部通过；
- 真实维度无训练前向：$D=2048$、3 视角、$m=3072$、$d=64$、可训练参数 203,905；$R=48$ 时首步 posterior trace 为 32.15，$|\alpha|=3.36\times10^{-3}$ 时 residual ratio 为 $1.762560\times10^{-4}$；
- 端到端 smoke：`server17` GPU7，400/400 updates，约 10 分 24 秒，`[Finish]`，无 traceback/OOM/NaN，`model_0.pth` 与 `model_last.pth` 均写出；
- route checkpoint 诊断：`close_jar` 4 episodes / 12 steps，`hidden_state_dim=64`，residual ratio $3.162425\times10^{-3}$、posterior trace 19.57753、$\operatorname{tr}(R)=147474.3$、$\operatorname{tr}(Q)=0.65058$、mean correction norm 19.56172、$|\alpha|=6.665071\times10^{-3}$。

### 11.2 边界与下一步

这轮先证明尺度修复后的通路在 400 steps 和 4 回合真实前向中数值可用；随后完成正式 2000-update 训练和五次评估，结果见 §12。**当前仍没有严格匹配对照和因果结论**。另一次诊断初跑因脚本默认 `hidden_state_dim=128` 而 checkpoint 为 64 失败，补 `--hidden-dim 64` 后加载和诊断成功；以后复现必须显式保持该维度一致。

## 12. 正式训练与五次评估结果（2026-09-16）

### 12.1 协议与完成性

- 训练：`server17` GPU7，`bs=2`、单卡、`train_iter=4000`（即 2000 optimizer updates）、`hidden_state_filter_correction=True`、`hidden_state_dim=64`、`hidden_state_filter_init_log_measure_noise=48.0`、$\lambda_{\mathrm{innov}}=0.05$，从 `model_80.pth` 初始化；日志报告 `2000/2000` 和 `[Finish]`。
- 评估：使用 server17 节点本地 held-out `RLBench_EVAL_DATA`，18 个任务、每任务 25 episodes、最大 25 control steps；五个 run 均写出 18 个唯一任务行。
- 结果目录：`outputs/filter_route_formal_2000_20260915/eval_five_repeats_final/`；run1 复用首次完整结果，run2--5 在恢复流程中重新完成，旧的中断部分结果未纳入汇总。
- 完成性：`COMPLETE`、`aggregate.txt` 均存在；评估进程和本轮自建 Xvfb 均已退出。

### 12.2 结果

五次 run 的 18-task 平均分别为 `86.44`、`88.67`、`87.78`、`88.00`、`88.44`，均值为 **87.87%**（run-level sample std 约 `0.87pp`）。逐任务均值 ± sample std 如下：

| task | success rate |
|---|---:|
| `close_jar` | 100.00 ± 0.00 |
| `reach_and_drag` | 100.00 ± 0.00 |
| `insert_onto_square_peg` | 89.60 ± 2.19 |
| `meat_off_grill` | 100.00 ± 0.00 |
| `open_drawer` | 100.00 ± 0.00 |
| `place_cups` | 52.80 ± 10.35 |
| `place_wine_at_rack_location` | 88.00 ± 10.58 |
| `push_buttons` | 99.20 ± 1.79 |
| `put_groceries_in_cupboard` | 77.60 ± 2.19 |
| `put_item_in_drawer` | 94.40 ± 4.56 |
| `put_money_in_safe` | 100.00 ± 0.00 |
| `light_bulb_in` | 88.80 ± 8.67 |
| `slide_block_to_color_target` | 97.60 ± 2.19 |
| `place_shape_in_shape_sorter` | 57.60 ± 8.29 |
| `stack_blocks` | 78.40 ± 8.29 |
| `stack_cups` | 79.20 ± 3.35 |
| `sweep_to_dustpan_of_size` | 86.40 ± 2.19 |
| `turn_tap` | 92.00 ± 2.83 |

论文表中 18-task 平均为 `88.20%`，本次差值为 `−0.33pp`；`place_cups`、`place_wine_at_rack_location` 和 `put_item_in_drawer` 的方差或均值差距较明显。该结果证明当前路线可完成正式训练和评估，但**不能证明相对原模型或其他路线的因果增益**。

## 13. 弱任务 episode/variation 诊断（2026-09-16）

### 13.1 分析口径

- 对正式结果的五个最终日志逐行解析，得到 `5 × 18 × 25 = 2250` 条 episode 记录；90 个 task-run 组合均为 25 条，没有把中断 run 的残留结果混入汇总。
- `finetune/RLBench/eval.py` 将 `ep` 传入 `reset_to_demo(eval_demo_seed)`，环境再按 `from_episode_number=ep` 读取 held-out demo。因此，同一 task 的 episode 编号对应固定的评估初始场景/variation；五次 run 的差异至少包含策略和仿真执行的重复波动。
- 下文的 route 统计均为当前正式 filter checkpoint 的五次 run 合并结果；baseline 对照来自 `outputs/official_baseline_eval_5runs_v1/console_run{1..5}.log`，使用相同 18-task、25-episode held-out 协议。该对照用于定位瓶颈，不替代严格的 matched causal ablation。

### 13.2 主要瓶颈

`success=0/5` 表示该固定 episode 在五次 route run 中均失败；`fail@25` 是失败 episode 中跑满 25 steps 的数量。

| task | route 成功率 | 固定失败 episode（0/5） | 失败数 | fail@25 | 初步信号 |
|---|---:|---|---:|---:|---|
| `place_cups` | 52.8% | 5, 14, 19, 20 | 59 | 57 | 同一 cup 数量内也有固定难场景，非单纯长程数量效应 |
| `place_shape_in_shape_sorter` | 57.6% | 0, 3, 6, 15, 21, 22, 23 | 53 | 53 | 目标形状/几何定位是首要嫌疑 |
| `put_groceries_in_cupboard` | 77.6% | 5, 15, 17 | 28 | 20 | tuna 固定失败；另有 8 次提前终止，需补抓异常类型 |
| `stack_cups` | 79.2% | 5, 8, 15 | 26 | 24 | base cup 颜色/位置与遮挡相关的嫌疑较强 |
| `stack_blocks` | 78.4% | 13, 18 | 27 | 18 | 2 yellow、3 navy、4 teal 等变体偏弱，但每个变体样本较少 |
| `light_bulb_in` | 88.8% | 无 | 14 | 13 | 没有固定全失败场景，主要是重复波动；适合作为 route 的正向对照 |

按 `variation_number` 聚合后，最明显的选择性失败如下（分子/分母为五次 route run 合并后的该 variation trial 数）：

| task | 较弱 variation | 较强 variation | 解释边界 |
|---|---|---|---|
| `place_shape_in_shape_sorter` | cube `0/10`、triangular prism `3/15`、star `11/30` | cylinder `26/35`、moon `32/35` | 同一形状内部仍有 episode 差异，不能把全部差异归因于颜色/语言 |
| `put_groceries_in_cupboard` | tuna `0/10`、chocolate jello `5/10`、coffee `8/15` | crackers/mustard/spam/sugar `35/35` | 目标物识别、抓取点或放置路径均可能参与，当前日志不能区分 |
| `stack_cups` | black `2/10`、silver `5/10`、orange `10/15` | red/azure/blue/lime/cyan/magenta/purple `均为 5/5 或 10/10` | 目标 base cup 的选择和后续堆叠几何需要逐步轨迹证据 |
| `place_cups` | 1 cup `19/35`、2 cups `34/65`、3 cups `13/25` | — | 三种数量几乎同样弱；更像空间布局/多次抓放，而不是语言中的数量词本身 |

固定难 episode 与官方 baseline 有较大重合：`place_shape_in_shape_sorter` 有 7 个共同固定失败 episode（0、3、6、15、21、22、23），`put_groceries_in_cupboard` 的 5、15、17 和 `stack_cups` 的 5、8、15 也完全重合；`place_cups` 的 19、20 以及 `stack_blocks` 的 13 同样重合。这说明这些点首先应按数据场景/目标物诊断，而不是直接扩大 correction 强度。

### 13.3 与 baseline 的定位性对照

同一 held-out split 的五次 run 合并结果如下：

| task | official baseline | filter route | delta |
|---|---:|---:|---:|
| `place_cups` | 52.0% | 52.8% | +0.8pp |
| `place_shape_in_shape_sorter` | 56.8% | 57.6% | +0.8pp |
| `put_groceries_in_cupboard` | 77.6% | 77.6% | +0.0pp |
| `stack_cups` | 78.4% | 79.2% | +0.8pp |
| `stack_blocks` | 77.6% | 78.4% | +0.8pp |
| `light_bulb_in` | 81.6% | 88.8% | +7.2pp |

更细的 variation 对照也支持同一结论：shape sorter 的 cube 为 baseline/filter `0/10` vs `0/10`，groceries 的 tuna 为 `0/10` vs `0/10`、coffee 为 `8/15` vs `8/15`，stack cups 的 silver 与 orange 分别为 `5/10` vs `5/10`、`10/15` vs `10/15`。`light_bulb_in` 是例外：filter 在 yellow、magenta、teal variation 上分别为 `5/5`、`4/10`、`8/10`，baseline 为 `2/5`、`1/10`、`6/10`；该信号值得作为正向机制案例继续检查，但样本量仍小，不能单独形成因果结论。

失败终止形态也不同：shape sorter 的 53 次失败全部跑满 25 steps，说明是“没有达到成功条件”的超时型失败；groceries 有 8 次短于 25 steps，stack blocks 有 9 次短于 25 steps。当前正式评估关闭了逐 episode error summary，短 episode 只能确认提前终止，尚不能区分 `IKError`、`ConfigurationPathError` 和 `InvalidActionError`。

### 13.4 下一步实验优先级

1. 先做小规模 `save_video + visualize`，只覆盖固定难点和同 variation 的成功控制：shape sorter 的 cube/triangular prism/star，groceries 的 tuna 与 coffee，stack cups 的 black/silver/orange，以及 place cups 的 5、14、19、20；同时记录每步 action、terminal/timeout 和异常类型。
2. 在相同 episode 集上保存 filter 诊断量（residual ratio、$|\alpha|$、posterior trace、correction norm），检查修正是否在“抓错目标/放置失败”之前已经介入，还是只在正常轨迹上改变了动作细节。
3. 用 `light_bulb_in` 作为正向对照，分析 route 为什么减少 timeout；用 `put_item_in_drawer` 作为回归保护项（当前 route 为 94.4%，baseline 为 98.4%）。
4. 只有当轨迹证据确认 correction 作用点后，再做严格 matched 的 $λ_{\mathrm{innov}}=0$ vs `0.05` 对照；目前不建议仅凭全任务平均值调大 loss 权重。

### 13.5 BridgeVLA++ 的 per-task action-step 对齐（2026-09-16）

官方 [BridgeVLA-Seq](https://github.com/npucvr/BridgeVLA-Seq) 的 RLBench 评估代码包含 `configs/eval_step_limit.yml`：`place_cups` 和 `stack_blocks` 的最大 keyframe action steps 为 **35**，其余 16 个 task 仍为 **25**。`eval.py` 默认加载这张表，并在每个 task 开始时把该值同时写入 rollout loop 和环境的 `time_in_state`；代码注释定义“一步”为一次 next-keyframe prediction 后由 planner 执行。因此这改变的是**评估 horizon**，不是 action 向量维度或 action chunk。论文附录也明确写成“25-keyframe-step budget，Place Cups 和 Stack Blocks 为 35”。

当前 checkout 仍是原始 BridgeVLA 评估入口：`finetune/RLBench/eval.py` 和 `scripts/rlbench_repro/run_eval.sh` 对所有 task 使用全局 `episode_length=25`。所以本节正式 filter route 结果在 `place_cups`、`stack_blocks` 上没有与 BridgeVLA++ 的公开 RLBench protocol 完全对齐。

| task | BridgeVLA++ step limit | 本轮 route 的 25-step 结果 | 25-step 失败中跑满上限 | 成功 episode 平均长度 |
|---|---:|---:|---:|---:|
| `place_cups` | **35** | 66/125（52.8%） | 57/59 | 17.68；按 1/2/3 cups 为 15.63/17.79/20.38 |
| `stack_blocks` | **35** | 98/125（78.4%） | 18/27 | 17.31；按 2/3/4 blocks 为 11.47/18.22/23.13 |
| `place_shape_in_shape_sorter` | 25 | 72/125（57.6%） | 53/53 | 6.21 |
| `put_groceries_in_cupboard` | 25 | 97/125（77.6%） | 20/28 | 5.54 |
| `stack_cups` | 25 | 99/125（79.2%） | 24/26 | 10.25 |
| `light_bulb_in` | 25 | 111/125（88.8%） | 13/14 | 6.18 |

**判断**：

- `stack_blocks` 是最强的 horizon 嫌疑：4-block 成功轨迹平均已到 23.13 步，接近当前 25 步上限；增加到 35 步有明确的可检验理由。
- `place_cups` 也确实随 cup 数量变长，且失败大多跑满 25 步；但 1/2/3 cups 的成功率几乎相同（54.3%/52.3%/52.0%），并且存在同一数量内固定失败 episode，因此不能把问题归结为动作预算。
- `shape sorter`、`groceries`、`stack cups` 的官方 step limit 没有放宽；它们的固定失败 variation/scene 和目标几何信号仍然更强。`fail@25` 只能说明没有在上限内成功，不能单独证明“再给步数就能成功”。

因此下一项应先做同 checkpoint、同 25 个 episode、同 eval seed 的 `episode_length=25` vs `35` horizon-only 对照，优先 `place_cups` 和 `stack_blocks`，并统计新增的 26--35 步是否真正带来成功。如果只是在 35 步继续失败，应回到目标物/遮挡/放置几何诊断。另需注意，官方 BridgeVLA++ 代码还包含 RLBench 缺失 mesh 的 renderable 修复；它与 step limit 是两个独立的 protocol 差异，不能在一次对照中混为步数效应。

### 13.6 首轮 25 vs 35 step horizon-only 对照（2026-09-16）

已在同一 filter checkpoint、同一 node-local held-out 数据、同一 25 个 episode 和同一任务顺序下完成首轮对照；仅改变 `episode_length`，未启用 `visualize` 或 `save_video`。结果文件位于 `outputs/step_validation/filter_25/` 和 `outputs/step_validation/filter_35/`，原始 episode 日志位于 `agent_playground/step_validation/stdout_25.log` 与 `stdout_35.log`。

| episode length | `place_cups` | `stack_blocks` | 平均 episode length |
|---:|---:|---:|---:|
| 25 | 12/25 = 48.0% | 20/25 = 80.0% | 21.36 / 17.88 |
| 35 | 14/25 = 56.0% | 20/25 = 80.0% | 25.96 / 20.48 |
| delta | **+8.0pp** | **+0.0pp** | +4.60 / +2.60 |

按 `(task, episode)` 对齐后，`place_cups` 有 5 个 25→35 rescue、3 个反向退化；`stack_blocks` 有 3 个 rescue、3 个反向退化。若严格要求 35-step 轨迹实际使用第 26--35 步且 25-step 同 episode 失败，则两个 task 各只有 1 个样本：`place_cups/episode21` 在 26 步成功，`stack_blocks/episode2` 在 31 步成功。其余 rescue 在 25 步以内就成功，不能归因于新增 horizon；反向退化也说明单轮评估存在明显 episode-level variance 或 simulator nondeterminism。

**阶段性结论**：首轮支持 `place_cups` 可能有小幅 horizon 效应，但没有支持 `stack_blocks` 的净收益；不能据此把低成功率主要归因于 action-step 上限，也不能直接把 35-step 结果当作最终提升。当前应再做至少 4 轮相同的 paired 对照（共 5 轮），再报告均值、离散度和逐 episode rescue 率。无论如何，官方 renderable-mesh 修复仍是独立 protocol 差异，不能与本轮步数效应合并解释。
