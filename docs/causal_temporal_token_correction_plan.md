# BridgeVLA Stage-1 历史 Token Adapter 实验方案

## 目标

验证最近的历史视觉 tokens 是否能改善 BridgeVLA 的动作预测。第一版只处理视觉历史，
不修改 PaliGemma、不修改 action heads，也不加入实际 EEF motion。

## 固定实验定义

```text
初始化：官方 BridgeVLA model_80.pth
PaliGemma：冻结
action heads：冻结
可训练参数：Stage-1 Temporal Token Adapter
历史窗口：K=4（当前关键帧 + 最近 3 个关键帧）
训练 loss：沿用原有 action loss
评估：官方 RLBench EVAL split
```

K=4 是有意的短窗口。关键帧不足 4 个时使用 padding mask；超过 4 个时只保留最近
4 个，不要求覆盖完整任务。

## 当前数据流

```text
当前多视角观测 + language_goal
    -> MVT / PaliGemma
    -> Stage-1 visual tokens
    -> Temporal Token Adapter
    -> 原有 up0 / waypoint / action heads
    -> 当前关键帧动作
```

在 [`mvt_single.py`](../finetune/bridgevla/mvt/mvt_single.py) 中，PaliGemma 输出的
视觉 tokens 形状约为：

```text
h_t: [N_img * 256, 2048]
```

Adapter 输出保持相同形状，之后继续使用原来的 Stage-1 数据管线。

## 历史窗口

对当前关键帧 `t`，构造：

```text
H_t = [h_{t-3}, h_{t-2}, h_{t-1}, h_t]
H_t: [B, 4, N_img * 256, 2048]
```

时间融合只在相同 view 和 patch 位置上进行。一个最小实现是低维时间注意力：

```text
z_i = W_down(LayerNorm(h_i))
a_i = softmax(score(z_t, z_i))
c_t = sum_i(a_i * z_i)
h'_t = h_t + W_up(GELU(c_t))
```

其中 `W_down` 将 2048 维压缩到较小的 bottleneck，`W_up` 再恢复到 2048 维。最后一层
建议零初始化，使训练开始时接近原始 `model_80.pth` 的行为。

任务开始时清空历史 buffer：

```text
t=0: [PAD, PAD, PAD, h0]
t=1: [PAD, PAD, h0,  h1]
```

任务结束时只使用已有的最近窗口；不能跨 episode 复用 tokens。

Replay 的当前输入由 `sample_frame` 标识，因此历史 cache 只选择
`keypoint_frame < sample_frame` 的关键帧；当当前输入本身就是一个关键帧时，不会把同一
帧重复放进 history。`keypoint_idx` 仅用于校验目标动作的 episode 位置。Stage-2 crop
会重新走当前 PaliGemma token 路径，不复用对应原始视图的 Stage-1 history。
在线 EVAL 时由 `RVTAgent.reset()` 清空窗口，并将前序 control observation 的 Stage-1 token
按同样的因果顺序缓存；训练仍使用 replay 的 keyframe cache。

## 数据实现

当前 replay 使用 `timesteps=1`，不能直接提供历史窗口。第一版应从原始 RLBench
episode 的 `low_dim_obs.pkl` 和图像中生成关键帧窗口：

```text
原始 episode
    -> 当前 keypoint_discovery
    -> 每个关键帧提取 PaliGemma Stage-1 tokens
    -> 构造长度为 4 的 token window
    -> 使用当前关键帧动作作为监督
```

由于 PaliGemma 冻结，先离线缓存每个关键帧的 Stage-1 tokens，训练时只读取缓存，
避免重复运行 PaliGemma。缓存脚本为
[`precompute_stage1_tokens.py`](../scripts/rlbench_train/precompute_stage1_tokens.py)，
每个文件保存一个 episode 的 `keypoint_frames` 和 `[N_keypoint, 768, 2048]`
tokens。`--data_root` 同时支持 `task/all_variations/episodes` 和官方的
`train/task/all_variations/episodes` 两种布局：

```bash
source scripts/bridgevla_runtime.sh
python scripts/rlbench_train/precompute_stage1_tokens.py \
  --data_root data/RLBench_TRAIN_DATA \
  --output_root data/stage1_token_cache_k4 \
  --checkpoint data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth \
  --paligemma_path data/bridgevla_ckpt/paligemma-3b-pt-224 \
  --device cuda:0
```

K=4 训练入口需要 `--stage1_adapter_only`、`--init_checkpoint`、
`--stage1_token_cache_dir`，并通过 `--mvt_cfg_opts
"stage1_history_len 4"` 设置窗口长度；K=1 对照将 `stage1_history_len` 设为 1
并省略 cache 参数。训练入口同时支持 `--data_folder` 指向
`task/all_variations/episodes` 布局和 `--clip_cache_dir`。正式训练可使用：

建议先按固定 update budget 做阶段训练，而不是直接跑完整的 100 epoch。当前入口将
`train_iter` 按 `bs` 换算为 optimizer updates，下面示例每组先处理 1,000 个 sample，
即 bs=4 时约 250 updates（`epochs=1`、`train_iter=1000`）：

```bash
cd finetune/RLBench
python train.py \
  --epochs 1 --num_train 25 \
  --data_folder ../../data/RLBench_TRAIN_DATA \
  --clip_cache_dir ../../data/clip_cache \
  --train_replay_storage_dir ../../data/replay_train_k4 \
  --stage1_token_cache_dir ../../data/stage1_token_cache_k4 \
  --init_checkpoint ../../data/bridgevla_ckpt/bridgevla/rlbench/model_80.pth \
  --stage1_adapter_only \
  --mvt_cfg_path ../bridgevla/mvt/configs/rvt2.yaml \
  --mvt_cfg_opts "paligemma_path ../../data/bridgevla_ckpt/paligemma-3b-pt-224 stage1_history_len 4" \
  --exp_cfg_opts "tasks all bs 4 train_iter 1000"
```

K=1 只需把 `stage1_history_len` 改为 `1` 并删除 cache 参数；两组应使用相同
replay、训练步数和日志/评估设置。短训后先用 `eval.py --eval-episodes 3` 做每任务
方向性比较，确认趋势后再扩展到每任务 25 episodes。

对照定义要区分两种 K=1：发布的 `model_80.pth` 是官方无 adapter baseline；从该 checkpoint
继续训练、但 `stage1_history_len=1` 的 K=1 是 adapter-only control，不能代替官方 baseline。
论文 RLBench 结果使用 18 tasks、每任务 25 trials；最终比较应统一采用该协议。

本次按五次 EVAL 协议完成统一比较；每次为 18 tasks × 25 episodes/task = 450
episodes。以每次 18-task 平均成功率计算，结果为：官方 `model_80.pth`
`88.40 ± 0.84%`，K=1 adapter-only control `86.40 ± 1.27%`，K=4
`88.04 ± 0.62%`（± 为五次运行间的 sample std）。论文 Table 1 的 BridgeVLA
平均成功率为 88.2%；因此官方 checkpoint 为 `+0.20` 个百分点，K=1 control 为
`-1.80` 个百分点，K=4 为 `-0.16` 个百分点；K=4 比 K=1 高 `+1.64` 个百分点。
论文正文明确写明 RLBench 的 BridgeVLA 结果总共评估五次；仓库的
`run_repeated_eval.sh` 是对同一个发布的 `model_80.pth` checkpoint 做五次 EVAL，再由
`aggregate_runs.py` 计算 mean/std，并不是五个训练 checkpoint。
当前 `eval.py` 将 `ep` 直接作为 `from_episode_number`；五次运行使用同一 held-out
episode 编号，但由独立评估进程收集重复结果，不涉及重新训练五个模型。

实际 EVAL 数据中关键帧数量中位数约为 5；K=4 是短期历史实验，不追求完整任务记忆。

## Loss 与参数更新

继续使用 [`bridgevla_agent.py`](../finetune/bridgevla/models/bridgevla_agent.py) 中的
原有动作损失：

```text
L_action = L_translation
         + L_rotation
         + L_gripper
         + L_collision
```

不增加文本生成 loss、token MSE 或 belief auxiliary loss。

PaliGemma 可以在 `no_grad()` 下提取缓存 tokens。action heads 虽然参数冻结，但其
forward 不能放入 `no_grad()`，否则梯度无法回传到 Adapter。

## 最小对照

```text
K=1：只使用当前 token
K=4：使用当前 token + 最近 3 个历史 token
```

两组实验使用相同 checkpoint、数据、loss、训练步数和官方 EVAL 协议。首轮只判断
K=4 是否优于当前观测 baseline；暂不加入 `u^ach`、在线更新或 TTT。
