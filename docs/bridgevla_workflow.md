# BridgeVLA 训练与推理验证流程（RLBench）

本文只描述当前仓库中的两条主线：离线监督训练，以及加载模型后的在线推理验证。BridgeVLA 的基本映射是：

```text
当前多视角观测 + 任务语言  ->  下一个 keypoint action
```

图使用 D2 编写；`.d2` 源文件和供 VS Code Markdown Preview 显示的 `.svg` 均位于 [`docs/diagrams/`](diagrams/)。

## 1. 监督训练

RLBench 专家 demo 会先被处理成 replay cards。每张 card 是一条独立监督样本：

| 内容 | 数据 |
| --- | --- |
| 输入 | 当前时刻四相机 RGB、点云和任务语言 |
| 标签 | 下一个 keypoint 的 xyz、quaternion、gripper open/close 和 ignore collision |

训练时从 replay 随机抽取 `timesteps = 1` 的 batch，因此网络不会看到观测历史、未来图像或 keypoint 时间戳。

![BridgeVLA 监督训练时序](diagrams/training_sequence.svg)

图源：[`training_sequence.d2`](diagrams/training_sequence.d2)

训练中的关键步骤：

1. 融合四相机点云，裁剪并归一化到场景空间。
2. MVT 将点云正交渲染成三个视图，PaliGemma 联合处理图像和语言。
3. Translation head 预测多视角热图；其他 heads 预测旋转、夹爪和碰撞状态。
4. 默认 two-stage MVT 在训练 Stage 2 时使用带扰动的真实 waypoint；这是 teacher forcing。
5. 汇总各 action 分量的交叉熵并更新模型：

```text
L = L_translation + L_rotation_x + L_rotation_y + L_rotation_z
  + L_gripper + L_collision
```

这是离线行为克隆：reward 会保存在 replay 结构中，但不参与该 action loss。

## 2. 推理验证

`eval.py` 加载 checkpoint，创建 BridgeVLA Agent 和 RLBench 环境。默认每个任务评估 25 个 episode，每个 episode 最多执行 25 次高层决策；正常评估不会读取未来专家 action。

模型每次只接收最新观测和固定的任务语言，并输出九维 action：

```text
[x, y, z, qx, qy, qz, qw, gripper, ignore_collision]
```

![BridgeVLA 在线推理验证时序](diagrams/online_rollout_sequence.svg)

图源：[`online_rollout_sequence.d2`](diagrams/online_rollout_sequence.d2)

一次在线循环为：

1. `RolloutGenerator` 将当前观测和语言交给 `agent.act()`。
2. `env.step(action)` 使用 `EndEffectorPoseViaPlanning2` 和 `MoveArmThenGripper` 执行目标位姿及夹爪动作。
3. 路径规划和底层仿真同步执行；`env.step()` 返回新观测后才会触发下一次推理。
4. episode 结束后汇总 return；在该任务设置中它被记录为 success rate。

因此它是在线闭环的高层 waypoint 控制，而不是逐相机帧的实时控制。模型不知道 episode 总长度，也不知道下一个 keypoint 的时间戳。

## 3. 训练与推理的核心差异

| 项目 | 训练 | 推理验证 |
| --- | --- | --- |
| 数据 | 随机单步 replay card | 环境返回的最新观测 |
| Stage 2 中心 | 带扰动的真实 waypoint | Stage 1 预测 waypoint |
| Action | 仅计算监督 loss | 由规划器和机器人执行 |
| 下一步触发 | sampler 抽取下一张 card | 上一个 `env.step()` 完成 |
| 未来信息 | 数据构建阶段可见，网络不可见 | 完全不可见 |

推理时 Stage 1 的定位误差会传给 Stage 2，执行误差还会改变下一次观测；这是训练与在线运行之间最重要的差异。

## 4. 主要源码入口

- Replay 构建：[`dataset.py`](../finetune/RLBench/utils/dataset.py)、[`get_dataset.py`](../finetune/RLBench/utils/get_dataset.py)
- 训练入口：[`train.py`](../finetune/RLBench/train.py)、[`bridgevla_agent.py`](../finetune/bridgevla/models/bridgevla_agent.py) 中的 `update()`
- Two-stage MVT：[`mvt.py`](../finetune/bridgevla/mvt/mvt.py)、[`mvt_single.py`](../finetune/bridgevla/mvt/mvt_single.py)
- 推理验证：[`eval.py`](../finetune/RLBench/eval.py)、[`rollout_generator.py`](../finetune/bridgevla/libs/YARR/yarr/utils/rollout_generator.py)
- Action 执行：[`rlbench_planning.py`](../finetune/RLBench/utils/rlbench_planning.py)、[`action_mode.py`](../finetune/bridgevla/libs/RLBench/rlbench/action_modes/action_mode.py)

## 5. 总结

训练阶段用单时刻 replay cards 监督“当前观测与语言到下一 keypoint action”的映射；验证阶段则反复执行 `观察 -> 预测 -> 规划执行 -> 新观察`，直到任务成功、失败或达到最大决策步数。
