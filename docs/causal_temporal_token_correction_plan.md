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

由于 PaliGemma 冻结，建议先离线缓存每个关键帧的 Stage-1 tokens，训练时只读取缓存，
避免重复运行 PaliGemma。

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
