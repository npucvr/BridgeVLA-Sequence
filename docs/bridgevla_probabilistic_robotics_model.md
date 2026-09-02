# BridgeVLA 的概率机器人建模

本文只保留与 BridgeVLA 数据管线直接对应的概率表达。$x_t$ 是环境真实状态，$z_t$ 是策略可见的观测，$H_t$ 是视觉编码结果，$u_t$ 是动作，$l$ 是语言目标。模型不直接访问 $x_t$。

## 1. 原始 BridgeVLA 数据管线

原始策略是一个最基本的 POMDP 闭环：

$$
 x_t
 \xrightarrow{\;p_{\mathrm{env}}(z_t\mid x_t)\;}
 z_t
 \xrightarrow{\;E_\rho(\cdot,l)\;}
 H_t
 \xrightarrow{\;\pi_\theta\;}
 u_t
 \xrightarrow{\;p_{\mathrm{env}}(x_{t+1}\mid x_t,u_t)\;}
 x_{t+1}.
$$

等价地，四个基本关系为

$$
 z_t\sim p_{\mathrm{env}}(z_t\mid x_t),
 \qquad H_t=E_\rho(z_t,l),
$$

$$
 u_t\sim\pi_\theta(u_t\mid H_t),
 \qquad x_{t+1}\sim p_{\mathrm{env}}(x_{t+1}\mid x_t,u_t).
$$

这里 $x_t$ 通过环境观测模型产生 $z_t$，而不是直接产生动作；动作由编码观测后的 VLA action decoder $\pi_\theta$ 产生。

## 2. Hidden-state 扩展

实验只在上述闭环中加入一个由历史维护的内部记忆。为避免符号混淆，本文约定：

- $y_t$：吸收当前观测前的 prior hidden state；
- $y_t^+$：吸收当前观测后的 updated hidden state。

episode 起点初始化为

$$
 y_1=y_{\mathrm{init}}.
$$

对后续时刻，prior 由上一个 updated state 和动作递推：

$$
 y_t=F_\phi(y_{t-1}^+,u_{t-1}),
 \qquad t\geq 2.
$$

当前观测编码为 $H_t=E_\rho(z_t,l)$ 后，进行 observation update：

$$
 y_t^+=U_\omega(y_t,H_t).
$$

再由 updated state 修正 visual tokens，并沿用原始策略产生动作：

$$
 \widetilde H_t=H_t+A_\psi(H_t,y_t^+),
 \qquad
 u_t\sim\pi_\theta(u_t\mid\widetilde H_t).
$$

动作执行后，进入下一时刻：

$$
 y_{t+1}=F_\phi(y_t^+,u_t).
$$

因此，隐状态内部的数据流为

```text
y_t -- observation prediction ----------------------> r_t target
 |
 +-- H_t --> U_omega --> y_t^+ --> A_psi --> H_tilde
                                                |
                                           pi_theta
                                                |
                                                u_t
                                                |
                                         F_phi(y_t^+, u_t)
                                                |
                                             y_{t+1}
```

$y_t$ 是模型的 predictive memory，不是环境状态 $x_t$ 的复制。若使用概率记号描述内部转移，当前确定性实现可写为

$$
 p_\phi(y_{t+1}\mid y_t^+,u_t)
 =\delta\!\left(y_{t+1}-F_\phi(y_t^+,u_t)\right).
$$

它与环境真实转移

$$
 p_{\mathrm{env}}(x_{t+1}\mid x_t,u_t)
$$

不是同一个分布。

## 3. Prior-observation prediction

prior hidden state $y_t$ 还用于预测当前观测的视觉表征。令

$$
 r_t=\operatorname{sg}\!\left(\Phi(H_t)\right),
 \qquad
 \widehat r_t=D_\eta(y_t),
$$

其中 $\operatorname{sg}$ 表示停止梯度，$r_t$ 是监督目标，$\widehat r_t$ 是 prior decoder 的预测。

如果第 $v$ 个视角的 visual tokens 为 $H_t^{(v)}\in\mathbb R^{S\times D}$，当前实现的目标为

$$
 r_t
 =\operatorname{concat}_{v=1}^{V}
 \operatorname{normalize}\!\left(
 \frac{1}{S}\sum_{s=1}^{S}H_t^{(v,s)}
 \right)
 \in\mathbb R^{V D}.
$$

也就是说，$r_t$ 是当前 $H_t$ 的 per-view pooled、normalized representation，而不是：

- 环境真实状态 $x_t$；
- 原始 RGB/depth 观测 $z_t$；
- 完整的 visual-token 序列 $H_t$；
- 动作 $u_t$。

decoder 建模 representation-space 条件分布

$$
 p_\eta(r_t\mid y_t)
 =\mathcal N\!\left(
 r_t;\mu_\eta(y_t),
 \operatorname{diag}\sigma_\eta^2
 \right),
$$

并使用负对数似然

$$
 \mathcal L_{\mathrm{obs}}
 =-\sum_t m_t\log p_\eta(r_t\mid y_t),
$$

其中 $m_t$ 是 sequence valid/loss mask。该辅助目标表示“历史 prior 对当前视觉表征的条件预测”。由于

$$
 y_t\in\sigma(z_{1:t-1},u_{1:t-1},l),
$$

而 $r_t$ 来自当前 $z_t$，当前观测只作为停止梯度 target，因而不构成信息泄露。当前 $H_t$ 仍由 $U_\omega$ 吸收到 $y_t^+$，并参与 token correction；预测结果 $\widehat r_t$ 不替换 $H_t$，也不进入主控制路径。

## 4. 训练目标

行为克隆目标为

$$
 \mathcal L_{\mathrm{BC}}
 =-\sum_t m_t\log\pi_\theta(u_t^*\mid\widetilde H_t).
$$

当前实验的总目标为

$$
 \mathcal L
 =\mathcal L_{\mathrm{BC}}
 +\lambda_{\mathrm{obs}}\mathcal L_{\mathrm{obs}}.
$$

当 $\lambda_{\mathrm{obs}}=0$ 时，模型退化为原始 BridgeVLA 路径。当前不使用 simulator-state reconstruction loss，因此不要求 $y_t$ 在数值上等于 $x_t$。

## 5. 时间展开与梯度语义

沿 episode 展开时：

$$
 y_1=y_{\mathrm{init}},
 \qquad
 y_t=F_\phi(y_{t-1}^+,u_{t-1}),
 \qquad
 y_t^+=U_\omega(y_t,H_t).
$$

每个 episode 只在起点初始化一次 hidden state；padding 由 sequence mask 排除。`hidden_state_sequence_bptt_length=1` 截断反向梯度，但不截断 hidden state 的数值传递。因此这里建模的是一个连续的 recurrent latent process，而不是相互独立的 transition。
