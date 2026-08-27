# BridgeVLA 的概率机器人建模

当前 BridgeVLA 只考虑观测时刻 $t_o$ 和目标时刻 $t_g$，满足 $t_o<t_g$。在每次决策开始时，$y_{t_o}$ 是策略当前的 hidden state；动作执行结束后得到新的 hidden state $y_{t_g}$。

## 符号

| 符号 | 含义 |
|---|---|
| $t_o$ | 观测时刻 |
| $t_g$ | 目标时刻 |
| $x_{t_o},x_{t_g}$ | 两个时刻的环境隐状态 |
| $z_{t_o}$ | 观测时刻可获得的 RGB、深度和点云观测 |
| $l$ | 语言任务目标 |
| $H_{t_o}$ | PaliGemma 根据 $z_{t_o}$ 和 $l$ 产生的 tokens |
| $y_{t_o},y_{t_g}$ | BridgeVLA 在两个时刻的 hidden state，$y\in\mathbb{R}^{d_y}$ |
| $y_{\mathrm{init}}$ | episode 首次决策时的初始 hidden state，默认 $\mathbf{0}\in\mathbb{R}^{d_y}$ |
| $\widetilde H_{t_o}$ | 注入 $y_{t_o}$ 后的修正 tokens |
| $u_{t_o\to t_g}$ | 从观测时刻通向目标时刻的 waypoint action |
| $p(\cdot\mid\cdot)$ | 观测或环境转移的概率模型 |
| $\pi_\theta(\cdot\mid\cdot)$ | BridgeVLA 的 waypoint policy |
| $F_\phi$ | 更新 hidden state 的可训练模块 |
| $A_\psi$ | 修正 tokens 的可训练模块 |
| $\pi^*$ | 已知真实环境状态时的理想策略 |
| $\pi^*_{\mathrm{obs}},\pi^*_y$ | 分别表示只使用观测和同时使用观测、hidden state 时的理想策略 |

## 理论基础：观测条件策略的局限

如果真实环境状态 $x_{t_o}$ 完整且可获得，理想的状态反馈策略可以写成 $\pi^*(u_{t_o\to t_g}\mid x_{t_o},l)$。但实际只能通过观测模型 $p(z_{t_o}\mid x_{t_o})$ 获得 $z_{t_o}$。

观测噪声和遮挡会使不同的 $x_{t_o}$ 产生相似的 $z_{t_o}$。只使用当前观测时，理想策略可以写成：

$$
\pi^*_{\mathrm{obs}}(u_{t_o\to t_g}\mid z_{t_o},l)
=\int
\pi^*(u_{t_o\to t_g}\mid x_{t_o},l)\,
p(x_{t_o}\mid z_{t_o},l)\,\mathrm{d}x_{t_o}.
$$

如果 $y_{t_o}$ 能够保留与当前环境状态有关的时序信息，则策略可以使用更丰富的条件：

$$
\pi^*_{y}(u_{t_o\to t_g}\mid z_{t_o},y_{t_o},l)
=\int
\pi^*(u_{t_o\to t_g}\mid x_{t_o},l)\,
p(x_{t_o}\mid z_{t_o},y_{t_o},l)\,\mathrm{d}x_{t_o}.
$$

引入 $y$ 的理论目标，是让当前 hidden state 在观测之外提供额外的状态信息。其信息增益可以形式化为：

$$
I(x_{t_o};y_{t_o}\mid z_{t_o},l)
=\mathsf{H}(x_{t_o}\mid z_{t_o},l)
-\mathsf{H}(x_{t_o}\mid z_{t_o},y_{t_o},l)\ge 0.
$$

当该互信息大于零时，$y_{t_o}$ 确实弥补了部分观测造成的信息缺失；但这个性质不是结构自动保证的，需要通过时序训练和消融实验验证。$y$ 仍然是任务相关的 hidden state，不等同于真实环境状态 $x$。

## 当前流程

### 1. 由环境状态生成观测

$$
z_{t_o} \sim p(z_{t_o}\mid x_{t_o})
$$

观测时刻的环境状态 $x_{t_o}$ 通过观测模型产生当前观测 $z_{t_o}$。

### 2. 由观测和语言生成 tokens

$$
H_{t_o}=\operatorname{PaliGemma}(z_{t_o},l)
$$

PaliGemma 将当前观测和语言目标编码为 tokens $H_{t_o}$。

### 3. 使用当前 tokens 和 hidden state 进行修正

在每个 episode 的首次决策时，初始化 $y_{t_o}=y_{\mathrm{init}}$，其中 $y_{\mathrm{init}}=\mathbf{0}\in\mathbb{R}^{d_y}$；后续决策沿用上一轮执行结束后得到的 hidden state。

$$
\widetilde H_{t_o}=H_{t_o}+A_\psi(H_{t_o},y_{t_o})
$$

修正模块同时接收当前 tokens $H_{t_o}$ 和当前 hidden state $y_{t_o}$。因此，$y_{t_o}$ 不直接替代当前观测，而是作为时序状态参与当前 tokens 的修正。

### 4. 由修正 tokens 生成 waypoint

$$
\hat u_{t_o\to t_g}
\sim
\pi_\theta(u_{t_o\to t_g}\mid\widetilde H_{t_o})
$$

现有 MVT 处理和 action heads 统一包含在策略 $\pi_\theta$ 中。

### 5. 执行 waypoint 并到达目标状态

$$
x_{t_g}\sim p(x_{t_g}\mid x_{t_o},\hat u_{t_o\to t_g})
$$

运动规划器执行 waypoint，使环境从 $x_{t_o}$ 转移到目标状态 $x_{t_g}$。

### 6. 执行结束后更新 hidden state

$$
y_{t_g}=F_\phi(y_{t_o},\hat u_{t_o\to t_g})
$$

$F_\phi$ 使用 $y_{t_o}$ 和刚执行的 waypoint 得到目标时刻的 hidden state $y_{t_g}$，供下一次决策使用。后续可以进一步利用目标时刻的新观测：先计算 $H_{t_g}=\operatorname{PaliGemma}(z_{t_g},l)$，再将更新扩展为 $y_{t_g}=F_\phi(y_{t_o},\hat u_{t_o\to t_g},H_{t_g})$，使 hidden state 也直接吸收新观测；这属于后续尝试，当前最简方案仍使用前式。

## 训练目标

RLBench 的专家监督目标记为 $u^*_{t_o\to t_g}=\operatorname{Waypoint}(x_{t_g})$。训练时按时间顺序递推 $y$，行为克隆目标为：

$$
\mathcal L_{\mathrm{BC}}
= -\mathbb E_{(z_{t_o},l,u^*_{t_o\to t_g})}
\left[
\log \pi_\theta(u^*_{t_o\to t_g}\mid\widetilde H_{t_o})
\right].
$$

## 轻量验证实现

当前代码提供一个显式开关 `stage1_hidden_state_enabled`，默认关闭以保持旧 checkpoint 的行为；快速验证时打开该开关，并保持 `stage1_adapter_mode="current_correction"`。RLBench 训练入口可使用 `--mvt_cfg_opts "stage1_hidden_state_enabled True stage1_adapter_mode current_correction"`。实现只新增两个小模块：

1. `Stage1TokenCorrectionAdapter` 使用 token bottleneck，并把 $y_{t_o}$ 投影到同一个 bottleneck 后广播到各个 token；输出层零初始化，因此初始输出仍等于 $H_{t_o}$。
2. `Stage1HiddenStateTransition` 使用 action encoder 和紧凑的 `GRUCell` 实现 $F_\phi(y_{t_o},\hat u_{t_o\to t_g})$。

在线 rollout 中，episode 开始时通过 `reset()` 清空 $y$；第一次 `act()` 使用零初值；产生的 waypoint 在环境执行完成后，于下一次 `act()` 开始前更新 $y$。这样不会把尚未执行的动作反馈给当前决策，也不会在 `stage_two` 的两次 MVT forward 中重复更新。

当前 replay buffer 仍然独立采样单个 transition。因而普通单步行为克隆可以快速验证 token correction 的形状、初始化和 action loss 路径，但只有按时间顺序展开至少两个决策，第二个决策的行为克隆损失才会为 $F_\phi$ 提供梯度。现阶段的快速检查包含一个两步合成展开；真实序列训练应在该检查通过后再接入。
