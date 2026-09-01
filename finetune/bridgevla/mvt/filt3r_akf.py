"""Causal, inference-only AKF between PaliGemma tokens and ConvexUpSample."""

# gbw____
from dataclasses import dataclass
import json
import os
from typing import Dict, Optional, Tuple

import torch
# gbw____
# A2 spatial-coherence Q 只需要无参数的局部池化，不引入可训练模块。
import torch.nn.functional as F
# ____
# ____


# gbw____
CANONICAL_TOKEN_TAIL = (3, 16, 16, 2048)
# gbw____
# 这些阈值只用于 diagnostics，不参与 Kalman 更新、reset 或 action 计算。
# jitter_score > 1 表示二阶变化超过当前历史 drift EMA；raw match tolerance
# 用于判断实际返回给 ConvexUpSample 的 token 是否等于输入 raw token。
DIAGNOSTIC_JITTER_THRESHOLD = 1.0
DIAGNOSTIC_RAW_MATCH_TOLERANCE = 1e-6
# ____


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_stages(name: str = "FILT3R_STAGES") -> Tuple[int, ...]:
    raw = os.environ.get(name, "1").strip().lower()
    if raw in {"both", "1,2", "2,1", "1+2", "2+1"}:
        return (1, 2)
    try:
        stages = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError(
            f"invalid {name}={raw!r}; expected 1, 2, or both"
        ) from exc
    if not stages or any(stage not in (1, 2) for stage in stages):
        raise ValueError(f"invalid {name}={raw!r}; expected 1, 2, or both")
    return stages


@dataclass(frozen=True)
class Filt3rAKFConfig:
    """One universal full-channel causal AKF configuration."""

    # gbw____
    # 当前实验版本：
    # A1-clean = constant Q + fixed full-channel R + reset_mode=none；
    # A2 = drift-adaptive bounded-sigmoid Q + 同一 fixed R + reset_mode=none。
    # 旧 B1/B2/B3 参数仍保留兼容性，但不属于本轮 A2 判定路径。
    # ____

    p_init: float = 1.5
    gamma_p: float = 1.0
    target_gain: float = 0.95
    r: float = 1.0
    # gbw____
    # A3 的最小实现：jitter 只改变 measurement covariance；Q 是否仍以
    # 固定 R0 为尺度由 q_use_adaptive_r 控制。默认 fixed 保持 A1/A2 结果。
    # ____
    r_mode: str = "fixed"
    jitter_r_lambda: float = 0.0
    jitter_r_max: float = 4.0
    # gbw____
    # tau 以上的二阶变化才进入 R；coherence weight=1 时只放大时间上
    # 不一致的 jitter，避免把持续真实运动全部当成 measurement noise。
    # ____
    jitter_r_tau: float = 0.0
    jitter_r_coherence_weight: float = 0.0
    # gbw____
    # A3 可选的时间一致性定义。direction 保留历史 hidden-channel
    # cosine；magnitude 使用相邻 raw displacement 的幅度连续性，避免
    # 特征方向旋转时把 jitter-aware R 的抑制误关掉。
    # ____
    jitter_r_coherence_mode: str = "direction"
    # gbw____
    # A3 的 R_t 作用粒度。token 保留已有“整 token 共享倍率”路径；
    # channel 则按当前 standardized innovation 在 2048 channel 上分配
    # R_t，仍是同一个因果 Kalman filter，不引入训练参数。
    # ____
    jitter_r_channel_mode: str = "token"
    q_use_adaptive_r: bool = True
    k_min: float = 0.01
    k_max: float = 0.99
    delta_floor: float = 1e-2
    eps: float = 1e-6
    stages: Tuple[int, ...] = (1,)
    reset_local_tau: float = 3.0
    reset_camera_quantile: float = 0.75
    reset_camera_tau: float = 2.0
    reset_camera_ema_beta: float = 0.2
    # gbw____
    # innovation_coherent reset 只允许“异常且与上一帧运动方向一致”的
    # local token 在连续多帧后 takeover；不把单帧、方向反复的 jitter
    # 直接替换成 raw token。默认阈值仅供新 A4 变体使用。
    # ____
    reset_coherence_tau: float = 0.5
    max_filtered_residual_ratio: float = 0.05
    # gbw____
    q_mode: str = "constant"
    drift_ema_beta: float = 0.2
    # gbw____
    # 默认 hold 完全复现 A1/A2；constant_velocity 只使用已经返回过的
    # filtered token 估计一阶运动，并在下一帧形成衰减预测。它不读取
    # task label，也不引入训练参数，仍是 training-free 的因果状态模型。
    # ____
    prediction_mode: str = "hold"
    velocity_ema_beta: float = 0.2
    velocity_decay: float = 0.5
    velocity_gate_tau: float = 0.5
    # gbw____
    # 仅 token_ema_double 使用；较小的 beta 形成长期 drift reference。
    drift_reference_beta: float = 0.05
    # ____
    # gbw____
    # gbw____
    # A2 可显式选择 drift 的历史尺度：旧实验保留全局 median；token_ema
    # 为每个 view/spatial token 维护自己的 EMA；token_ema_relative 进一步
    # 用当前帧 EMA 的 robust spatial median 标定长期运动强度；
    # global_token_mean_persistent 还维护一条更慢的全局 reference，
    # 只让持续的整帧变化提高 Q。
    # gbw____
    # token_ema_persistent 还要求 raw drift 的方向具有时间一致性，
    # 让孤立或反向变化回到中性 Q，避免单帧 spike 立即放大过程噪声。
    # token_ema_double 使用快、慢两条因果 EMA 的比值表达持续 drift，
    # 避免直接把单帧 raw_delta 当成 process-noise 证据。
    # ____
    # ____
    drift_scope: str = "global_token_median"
    q_power: float = 1.0
    q_scale_min: float = 0.25
    q_scale_max: float = 4.0
    # gbw____
    # A2 的 Q 是实际的 Q/R 比值，而不是旧 motion-jitter 的乘法尺度。
    # gbw____
    # q_sigmoid_tau 在归一化 drift g_t=1 处设定拐点；默认上下界取为
    # 0.5/35.6，使 sigmoid 中点的 Q/R=18.05 与 A1-clean 的中性
    # process-noise ratio 对齐，而不是在中性 drift 时无意中把 gain 降到约 0.87。
    # ____
    q_min: float = 0.5
    q_max: float = 35.6
    q_sigmoid_alpha: float = 4.0
    q_sigmoid_tau: float = 1.0
    # gbw____
    # gain-space A2 先直接调度目标 Kalman gain，再反解 Q/R。这样 Q 的
    # 有界范围只负责安全裁剪，不会因为 q_min/q_max 的线性坐标选择而
    # 改变中性 gain。所有参数都是 inference-only 的固定超参数。
    # ____
    gain_k_delta: float = 0.015
    gain_sigmoid_alpha: float = 4.0
    gain_sigmoid_tau: float = 1.0
    gain_k_min: float = 0.90
    gain_k_max: float = 0.98
    gain_q_min: float = 4.0
    gain_q_max: float = 40.0
    # gbw____
    # 新候选的可选标定：把 bounded-sigmoid 的 g=1 映射精确对齐到
    # nominal Q/R，而不是依赖 q_min/q_max 恰好关于 nominal 对称。
    # 默认关闭以保持已有 A2 结果可复现。
    # ____
    q_calibrate_nominal: bool = False
    # gbw____
    # coherence penalty 将“非持续 drift”映射为较低 Q：方向一致时 penalty
    # 为零，方向反复或孤立变化时 Q 下调，从而只对真正持续的状态变化
    # 提高 Kalman gain。默认 0 保持既有 A2 行为。
    # ____
    q_coherence_weight: float = 0.0
    # gbw____
    # A2-safe 变体：将 nominal Q/R 作为 drift-adaptive Q 的下界。
    # 低于历史 drift 的帧不再降低 Q，只有正向 drift 才能提高 Q。
    # ____
    q_floor_nominal: bool = False
    # gbw____
    # nominal floor 开启时只放大正向 drift 的超额部分；默认 1 保持
    # 原始 bounded-sigmoid 输入不变。
    q_positive_boost: float = 1.0
    # gbw____
    # 只在 nominal floor 分支生效：低于 1+deadband 的轻微正向 drift
    # 仍保持 nominal Q，避免把测量噪声误当成需要提高 process noise 的运动。
    # 超过 deadband 的部分才进入 bounded sigmoid；默认 0 保持旧行为。
    # ____
    q_positive_deadband: float = 0.0
    # gbw____
    # 在 nominal floor 之上，用 normalized innovation 的超额部分补充
    # drift 证据；再乘 temporal coherence，表示“当前偏差确实沿着历史
    # 运动方向增长”。默认 0，不改变既有 A2。
    # ____
    q_innovation_weight: float = 0.0
    q_innovation_tau: float = 1.0
    # gbw____
    # A2-only 的鲁棒 Q 惩罚：只对“瞬时加速度超过阈值且与上一帧方向
    # 不一致”的 token 降低 bounded-sigmoid 输入。它不改变 R、reset 或
    # token 的直接取值；默认 0 保持所有既有实验完全不变。
    # ____
    q_jitter_penalty: float = 0.0
    q_jitter_tau: float = 1.0
    # gbw____
    # 允许 jitter penalty 在 nominal floor 之下工作；默认关闭，保证
    # 历史 q_floor 候选仍保持“只升不降”的旧语义。
    # ____
    q_jitter_breaks_floor: bool = False
    # gbw____
    # innovation-gated Q penalty：只在当前观测相对历史状态的 normalized
    # innovation 超过阈值、且 raw drift 与上一帧不一致时降低 Q。它把
    # “观测异常”与“持续状态变化”分开，仍只作用于过程噪声调度。
    # ____
    q_innovation_penalty: float = 0.0
    q_innovation_penalty_tau: float = 1.0
    # gbw____
    # selective-jitter gate 是 A2 的稀疏 Q-only 保护：正常 token 固定回到
    # nominal Q/R，只有同时满足二阶突变、normalized innovation、时间
    # 不一致和局部 robust outlier 条件时，才把 Q/R 降到指定的低值。
    # 它不替换 raw token，也不引入训练参数或 task-specific 分支。
    q_selective_gate: bool = False
    q_selective_jitter_tau: float = 0.0
    q_selective_innovation_tau: float = 0.0
    q_selective_coherence_tau: float = 1.0
    q_selective_local_score_tau: float = 0.0
    q_selective_q_ratio: float = 1.0
    # gbw____
    # soft gate 不在阈值处突然切换 Q，而是把 jitter、innovation、局部
    # 异常和时间不一致性映射为 [0,1] 的连续异常权重。默认关闭，保证
    # S122-S140 的历史数值路径不变。
    # ____
    q_selective_soft_gate: bool = False
    q_selective_softness: float = 0.5
    # gbw____
    # selective gate 的正常分支可选地保留正向 drift：低于 nominal 的
    # 稳定帧仍回到 A1 gain，只有持续且幅度增大的运动才提高 Q；异常 gate
    # 仍使用 q_selective_q_ratio。默认关闭以保持 S122-S129 数值不变。
    # ____
    q_selective_positive_drift: bool = False
    # ____
    # gbw____
    # 只有当 normalized innovation 不超过该阈值时，才允许 drift-Q
    # 低于 nominal Q；设为 0 表示关闭。这样动态帧不会因为低 Q 被滞后，
    # 但稳定帧仍可以使用历史状态进行更强融合。
    # ____
    q_innovation_guard_tau: float = 0.0
    # ____
    # gbw____
    # innovation-aligned drift scope 的慢速归一化基线与响应强度。
    # 它只改变 bounded-sigmoid 的输入，不改变观测 R 或 reset。
    # ____
    q_innovation_reference_beta: float = 0.05
    q_innovation_power: float = 1.0
    # ____
    # gbw____
    # APNE 分支用 innovation 能量的 EMA 估计 Q；在 log(Q_hat/Q_nominal)
    # 上做 bounded sigmoid，使 Q_hat=Q_nominal 时仍回到 A1 的中性点。
    apne_ema_beta: float = 0.2
    apne_log_alpha: float = 2.0
    apne_innovation_clip: float = 16.0
    # gbw____
    # normalized APNE 的无量纲中性 evidence；通常为 1，避免把
    # hidden-feature 的绝对 energy 当作 R 的物理单位。
    # ____
    apne_innovation_reference: float = 1.0
    # gbw____
    # robust APNE 使用 channel-wise standardized innovation 的 MAD 尺度，
    # 而不是直接对 2048 个 channel 求均方。它仍然只维护逐 spatial token
    # 的 EMA；默认关闭以保持已有 APNE 数值路径完全不变。
    # ____
    apne_robust: bool = False
    # gbw____
    # robust APNE 的统计器：mad 是历史默认的 median absolute deviation；
    # winsorized_mean 对 standardized innovation^2 做固定上限 4.0 的
    # winsorization，保留少量重要 channel 的证据，同时避免极少数异常
    # channel 把整 token 的 Q 直接推到上界。默认 mad 不改变旧实验。
    # ____
    apne_robust_stat: str = "mad"
    # gbw____
    # 物理 APNE 将 normalized innovation energy 转成 Q/R：
    # (e-1)*(P+R)/R，而不是把 e 本身当作 Q evidence。默认关闭，
    # 保持当前 robust APNE screen 的数值路径不变。
    # ____
    apne_physical: bool = False
    # gbw____
    # APNE-drift 融合只在 log(Q/R) 证据空间中加入有界的长期 drift
    # 修正；blend=0 保持纯 APNE。coherence 会抑制单帧方向反转的
    # jitter，整个修正仍只依赖当前观测与历史 EMA。
    # ____
    apne_drift_blend: float = 0.0
    apne_drift_clip: float = 2.0
    # gbw____
    # APNE 长期 drift 的一致性证据。direction 使用 hidden channel 的
    # cosine；magnitude 使用相邻帧 token 位移幅度的一致性。后者避免
    # 视觉特征方向旋转导致 coherence≈0，从而把有效 drift 错误压掉。
    # 默认 direction 保持历史候选数值不变。
    # ____
    apne_drift_coherence: str = "direction"
    # gbw____
    # 可选的 APNE 正向 evidence gate：只对高于中性点的 log evidence
    # 使用 raw displacement coherence 做收缩；低 evidence 仍由
    # apne_q_floor_nominal 负责保护。权重为 0 时严格退化为旧路径。
    # ____
    apne_evidence_coherence_weight: float = 0.0
    # gbw____
    # nominal floor 的可选细分：只有 APNE process evidence 偏低且 raw
    # displacement 幅度不连续时，才允许 Q 低于 A1 nominal；高幅度连续的
    # 动态 token 仍保持 nominal 或更高，避免 place_cups 等任务被滞后。
    # 默认关闭，保持已有 APNE 候选严格可复现。
    # ____
    apne_floor_break_low_evidence: bool = False
    apne_floor_break_evidence_tau: float = 0.8
    apne_floor_break_coherence_tau: float = 0.72
    # gbw____
    # 可选的 jitter 上限证据。大于 0 时，只有低 evidence 且二阶
    # jitter 不高的 token 才能突破 floor；0 表示不增加该条件。
    # ____
    apne_floor_break_jitter_tau: float = 0.0
    # gbw____
    # APNE 的可选 Q-only 鲁棒项：当当前 raw innovation 的二阶变化
    # 超过历史 drift 尺度时，降低 APNE 估计出的过程噪声；默认关闭，
    # 因而不改变既有 A1/A2 配置。
    # ____
    apne_jitter_penalty: float = 0.0
    apne_jitter_tau: float = 1.0
    # gbw____
    # 只对 APNE 的 Q/R 比值设 nominal floor；默认关闭以保留历史行为。
    # 开启后仍允许高创新证据提高 Q，但禁止低证据造成比 A1 更低的 gain。
    # ____
    apne_q_floor_nominal: bool = False
    # ____
    # gbw____
    jitter_weight: float = 1.0
    max_token_shift_ratio: float = 0.0
    # ____
    reset_mode: str = "current"
    reset_patience: int = 2
    innovation_gate_tau: float = 6.0
    # ____

    def __post_init__(self) -> None:
        if self.p_init <= 0.0 or self.gamma_p < 0.0:
            raise ValueError("p_init must be positive and gamma_p non-negative")
        if not 0.0 < self.target_gain < 1.0:
            raise ValueError("target_gain must be in (0, 1)")
        if self.r <= 0.0 or self.delta_floor <= 0.0 or self.eps <= 0.0:
            raise ValueError("r, delta_floor, and eps must be positive")
        # gbw____
        if self.r_mode not in {"fixed", "jitter_aware"}:
            raise ValueError("r_mode must be fixed or jitter_aware")
        if self.jitter_r_lambda < 0.0:
            raise ValueError("jitter_r_lambda must be non-negative")
        if self.jitter_r_max <= 0.0:
            raise ValueError("jitter_r_max must be positive")
        # gbw____
        if self.jitter_r_tau < 0.0:
            raise ValueError("jitter_r_tau must be non-negative")
        if not 0.0 <= self.jitter_r_coherence_weight <= 1.0:
            raise ValueError(
                "jitter_r_coherence_weight must be in [0, 1]"
            )
        # ____
        # ____
        if not 0.0 <= self.k_min <= self.k_max <= 1.0:
            raise ValueError("gain clamps must satisfy 0 <= k_min <= k_max <= 1")
        if not self.stages or any(stage not in (1, 2) for stage in self.stages):
            raise ValueError("stages must contain only 1 and/or 2")
        if self.reset_local_tau < 0.0 or self.reset_camera_tau < 0.0:
            raise ValueError("reset thresholds must be non-negative")
        # gbw____
        if self.reset_coherence_tau < 0.0 or self.reset_coherence_tau > 1.0:
            raise ValueError("reset coherence threshold must be in [0, 1]")
        # ____
        if not 0.0 < self.reset_camera_quantile < 1.0:
            raise ValueError("reset_camera_quantile must be in (0, 1)")
        if not 0.0 < self.reset_camera_ema_beta <= 1.0:
            raise ValueError("reset_camera_ema_beta must be in (0, 1]")
        if self.max_filtered_residual_ratio < 0.0:
            raise ValueError("max_filtered_residual_ratio must be non-negative")
        # gbw____
        if self.q_mode not in {
            "constant",
            "motion_jitter_adaptive",
            "drift_adaptive_sigmoid",
            # gbw____
            "gain_space_adaptive",
            # ____
            "apne_bounded_sigmoid",
            "apne_normalized_sigmoid",
        }:
            raise ValueError(
                "q_mode must be constant, motion_jitter_adaptive, "
                "drift_adaptive_sigmoid, gain_space_adaptive, "
                "apne_bounded_sigmoid, or "
                "apne_normalized_sigmoid"
            )
        if not 0.0 < self.drift_ema_beta <= 1.0:
            raise ValueError("drift_ema_beta must be in (0, 1]")
        # gbw____
        if not 0.0 < self.drift_reference_beta <= 1.0:
            raise ValueError("drift_reference_beta must be in (0, 1]")
        # ____
        # gbw____
        # gbw____
        if self.drift_scope not in {
            "global_token_median",
            # gbw____
            # FILT3R 原始实现使用所有 latent token 的 mean delta；该 scope
            # 用于验证源码级 global process-noise 估计在 BridgeVLA 上的效果。
            "global_token_mean",
            # gbw____
            "global_token_mean_persistent",
            # ____
            # ____
            "token_ema",
            "token_ema_relative",
            # gbw____
            "token_ema_persistent",
            "token_ema_double",
            # gbw____
            # token_ema_coherent 用当前 drift 相对自身历史 EMA 的变化，
            # 再用相邻帧方向一致性确认是否为持续运动。
            "token_ema_coherent",
            # gbw____
            # magnitude coherence 不比较 2048 维方向，而比较相邻 raw
            # drift 的幅度是否连续，避免 hidden feature 方向旋转把真实
            # scene motion 错判成 incoherent。
            "token_ema_magnitude_coherent",
            # ____
            # gbw____
            # token_ema_spatial_coherent 在时间方向一致性之外，要求当前
            # drift 与 3x3 邻域的变化幅度相近；孤立 spatial spike 因而只
            # 获得较低 Q，仍不引入任何可训练参数。
            "token_ema_spatial_coherent",
            # ____
            # ____
            # gbw____
            # token_ema_innovation_aligned 还用 normalized innovation 的
            # 慢速 EMA 判断当前 drift 是应当更接近 raw 还是更接近 history。
            "token_ema_innovation_aligned",
            # gbw____
            # token_ema_apne 用 innovation 方差扣除上一时刻 P 与固定 R
            # 估计过程噪声证据，再通过已有 drift-adaptive sigmoid 生成 Q。
            "token_ema_apne",
            # token_ema_difference_apne 使用 raw 相邻帧差分；在随机游走
            # measurement model 下 Var(z_t-z_{t-1})≈Q_t+2R_t，避免把
            # 当前滤波器的滞后误差再次当成过程噪声。
            "token_ema_difference_apne",
            # ____
            # ____
        }:
            raise ValueError(
                "drift_scope must be global_token_median, global_token_mean, token_ema, "
                "token_ema_relative, token_ema_persistent, token_ema_double, "
                # gbw____
                "global_token_mean_persistent, token_ema_coherent, "
                "token_ema_magnitude_coherent, token_ema_spatial_coherent, "
                "token_ema_innovation_aligned, "
                "token_ema_apne, or token_ema_difference_apne"
                # ____
            )
        # ____
        if self.q_power <= 0.0:
            raise ValueError("q_power must be positive")
        # gbw____
        if self.prediction_mode not in {
            "hold",
            "constant_velocity",
            "gated_velocity",
        }:
            raise ValueError(
                "prediction_mode must be hold, constant_velocity, or "
                "gated_velocity"
            )
        if not 0.0 < self.velocity_ema_beta <= 1.0:
            raise ValueError("velocity_ema_beta must be in (0, 1]")
        if not 0.0 <= self.velocity_decay <= 1.0:
            raise ValueError("velocity_decay must be in [0, 1]")
        if not 0.0 <= self.velocity_gate_tau <= 1.0:
            raise ValueError("velocity_gate_tau must be in [0, 1]")
        # ____
        if not 0.0 < self.q_scale_min <= self.q_scale_max:
            raise ValueError("q scale bounds must be positive and ordered")
        # gbw____
        if not 0.0 < self.q_min <= self.q_max:
            raise ValueError("A2 Q bounds must be positive and ordered")
        if self.q_sigmoid_alpha <= 0.0:
            raise ValueError("A2 q_sigmoid_alpha must be positive")
        if self.q_sigmoid_tau < 0.0:
            raise ValueError("A2 q_sigmoid_tau must be non-negative")
        # gbw____
        if self.gain_k_delta < 0.0:
            raise ValueError("gain_k_delta must be non-negative")
        if self.gain_sigmoid_alpha <= 0.0:
            raise ValueError("gain_sigmoid_alpha must be positive")
        if self.gain_sigmoid_tau < 0.0:
            raise ValueError("gain_sigmoid_tau must be non-negative")
        if not 0.0 <= self.gain_k_min <= self.gain_k_max < 1.0:
            raise ValueError("gain-space clamps must satisfy 0 <= min <= max < 1")
        if self.gain_q_min <= 0.0 or self.gain_q_min > self.gain_q_max:
            raise ValueError("gain-space Q/R bounds must be positive and ordered")
        # ____
        # gbw____
        if self.q_calibrate_nominal:
            nominal_q_ratio = self.target_gain * (
                1.0 / (1.0 - self.target_gain) - self.gamma_p
            )
            if self.q_max <= nominal_q_ratio:
                raise ValueError(
                    "nominal-centered Q calibration requires q_max above "
                    "the nominal Q/R ratio"
                )
        # ____
        # gbw____
        if self.q_coherence_weight < 0.0:
            raise ValueError("q_coherence_weight must be non-negative")
        # gbw____
        if self.q_positive_boost <= 0.0:
            raise ValueError("q_positive_boost must be positive")
        # gbw____
        if self.q_positive_deadband < 0.0:
            raise ValueError("q_positive_deadband must be non-negative")
        # ____
        # gbw____
        if self.q_innovation_weight < 0.0:
            raise ValueError("q_innovation_weight must be non-negative")
        if self.q_innovation_tau < 0.0:
            raise ValueError("q_innovation_tau must be non-negative")
        # gbw____
        if self.q_jitter_penalty < 0.0:
            raise ValueError("q_jitter_penalty must be non-negative")
        if self.q_jitter_tau < 0.0:
            raise ValueError("q_jitter_tau must be non-negative")
        # gbw____
        if self.q_innovation_penalty < 0.0:
            raise ValueError("q innovation penalty must be non-negative")
        if self.q_innovation_penalty_tau < 0.0:
            raise ValueError("q innovation penalty tau must be non-negative")
        # gbw____
        if self.q_selective_jitter_tau < 0.0:
            raise ValueError("selective jitter tau must be non-negative")
        if self.q_selective_innovation_tau < 0.0:
            raise ValueError(
                "selective innovation tau must be non-negative"
            )
        if not 0.0 <= self.q_selective_coherence_tau <= 1.0:
            raise ValueError("selective coherence tau must be in [0, 1]")
        if self.q_selective_local_score_tau < 0.0:
            raise ValueError(
                "selective local-score tau must be non-negative"
            )
        if self.q_selective_q_ratio <= 0.0:
            raise ValueError("selective Q/R ratio must be positive")
        # gbw____
        if self.q_selective_softness <= 0.0:
            raise ValueError("selective soft gate width must be positive")
        # ____
        # ____
        # ____
        # ____
        # gbw____
        if self.q_innovation_guard_tau < 0.0:
            raise ValueError("q_innovation_guard_tau must be non-negative")
        # ____
        # gbw____
        if not 0.0 < self.q_innovation_reference_beta <= 1.0:
            raise ValueError("q_innovation_reference_beta must be in (0, 1]")
        if self.q_innovation_power < 0.0:
            raise ValueError("q_innovation_power must be non-negative")
        # ____
        # ____
        # ____
        # gbw____
        if not 0.0 < self.apne_ema_beta <= 1.0:
            raise ValueError("APNE EMA beta must be in (0, 1]")
        if self.apne_log_alpha <= 0.0:
            raise ValueError("APNE log sigmoid alpha must be positive")
        if self.apne_innovation_clip < 1.0:
            raise ValueError("APNE innovation clip must be at least one")
        # gbw____
        if self.apne_innovation_reference <= 0.0:
            raise ValueError("APNE innovation reference must be positive")
        # gbw____
        if self.apne_robust_stat not in {"mad", "winsorized_mean"}:
            raise ValueError(
                "APNE robust statistic must be mad or winsorized_mean"
            )
        # ____
        # gbw____
        if self.apne_drift_blend < 0.0:
            raise ValueError("APNE drift blend must be non-negative")
        if self.apne_drift_clip < 1.0:
            raise ValueError("APNE drift clip must be at least one")
        # gbw____
        if self.apne_drift_coherence not in {"direction", "magnitude"}:
            raise ValueError(
                "APNE drift coherence must be direction or magnitude"
            )
        # ____
        # gbw____
        if not 0.0 <= self.apne_evidence_coherence_weight <= 1.0:
            raise ValueError(
                "APNE evidence coherence weight must be in [0, 1]"
            )
        # ____
        # gbw____
        if self.apne_floor_break_evidence_tau < 0.0:
            raise ValueError(
                "APNE floor-break evidence threshold must be non-negative"
            )
        if not 0.0 <= self.apne_floor_break_coherence_tau <= 1.0:
            raise ValueError(
                "APNE floor-break coherence threshold must be in [0, 1]"
            )
        # ____
        # gbw____
        if self.apne_floor_break_jitter_tau < 0.0:
            raise ValueError(
                "APNE floor-break jitter threshold must be non-negative"
            )
        # ____
        # gbw____
        if self.jitter_r_coherence_mode not in {"direction", "magnitude"}:
            raise ValueError(
                "jitter R coherence mode must be direction or magnitude"
            )
        # ____
        # gbw____
        if self.jitter_r_channel_mode not in {"token", "channel"}:
            raise ValueError(
                "jitter R channel mode must be token or channel"
            )
        # ____
        # ____
        # ____
        # gbw____
        if self.apne_jitter_penalty < 0.0:
            raise ValueError("APNE jitter penalty must be non-negative")
        if self.apne_jitter_tau < 0.0:
            raise ValueError("APNE jitter tau must be non-negative")
        # ____
        # ____
        # ____
        # gbw____
        if self.jitter_weight <= 0.0:
            raise ValueError("jitter_weight must be positive")
        if self.max_token_shift_ratio < 0.0:
            raise ValueError("max_token_shift_ratio must be non-negative")
        # ____
        if self.reset_mode not in {
            "none",
            "current",
            "innovation_repartition_norm",
            # gbw____
            "innovation_coherent",
            # ____
        }:
            raise ValueError(
                "reset_mode must be none, current, innovation_repartition_norm, "
                "or innovation_coherent"
            )
        if self.reset_patience < 1:
            raise ValueError("reset_patience must be at least one")
        if self.innovation_gate_tau <= 0.0:
            raise ValueError("innovation_gate_tau must be positive")
        # ____

        q_ratio = self.target_gain * (
            1.0 / (1.0 - self.target_gain) - self.gamma_p
        )
        if q_ratio <= 0.0:
            raise ValueError("target_gain and gamma_p must produce a positive Q/R")

    @classmethod
    def from_environment(cls) -> "Filt3rAKFConfig":
        # gbw____
        # R0 校准时用同一比例同步设置 P_init，避免只降低 R 而把
        # 初始协方差留在旧尺度。未设置 ratio 时保持历史默认行为。
        r_value = _env_float("FILT3R_R", cls.r)
        p_init_value = _env_float("FILT3R_P_INIT", cls.p_init)
        p_init_ratio_text = os.environ.get("FILT3R_P_INIT_RATIO", "").strip()
        if p_init_ratio_text:
            p_init_value = r_value * float(p_init_ratio_text)
        # ____
        return cls(
            p_init=p_init_value,
            gamma_p=_env_float("FILT3R_GAMMA_P", cls.gamma_p),
            target_gain=_env_float("FILT3R_TARGET_GAIN", cls.target_gain),
            r=r_value,
            # gbw____
            r_mode=os.environ.get("FILT3R_R_MODE", cls.r_mode).strip().lower(),
            jitter_r_lambda=_env_float(
                "FILT3R_JITTER_R_LAMBDA", cls.jitter_r_lambda
            ),
            jitter_r_max=_env_float("FILT3R_JITTER_R_MAX", cls.jitter_r_max),
            # gbw____
            jitter_r_tau=_env_float("FILT3R_JITTER_R_TAU", cls.jitter_r_tau),
            jitter_r_coherence_weight=_env_float(
                "FILT3R_JITTER_R_COHERENCE_WEIGHT",
                cls.jitter_r_coherence_weight,
            ),
            # gbw____
            jitter_r_coherence_mode=os.environ.get(
                "FILT3R_JITTER_R_COHERENCE_MODE",
                cls.jitter_r_coherence_mode,
            ).strip().lower(),
            # ____
            # gbw____
            jitter_r_channel_mode=os.environ.get(
                "FILT3R_JITTER_R_CHANNEL_MODE",
                cls.jitter_r_channel_mode,
            ).strip().lower(),
            # ____
            # ____
            q_use_adaptive_r=_env_bool(
                "FILT3R_Q_USE_ADAPTIVE_R", cls.q_use_adaptive_r
            ),
            # ____
            k_min=_env_float("FILT3R_K_MIN", cls.k_min),
            k_max=_env_float("FILT3R_K_MAX", cls.k_max),
            delta_floor=_env_float("FILT3R_DELTA_FLOOR", cls.delta_floor),
            eps=_env_float("FILT3R_EPS", cls.eps),
            stages=_env_stages(),
            reset_local_tau=_env_float(
                "FILT3R_RESET_LOCAL_TAU", cls.reset_local_tau
            ),
            reset_camera_quantile=_env_float(
                "FILT3R_RESET_CAMERA_QUANTILE", cls.reset_camera_quantile
            ),
            reset_camera_tau=_env_float(
                "FILT3R_RESET_CAMERA_TAU", cls.reset_camera_tau
            ),
            reset_camera_ema_beta=_env_float(
                "FILT3R_RESET_CAMERA_EMA_BETA", cls.reset_camera_ema_beta
            ),
            # gbw____
            reset_coherence_tau=_env_float(
                "FILT3R_RESET_COHERENCE_TAU", cls.reset_coherence_tau
            ),
            # ____
            max_filtered_residual_ratio=_env_float(
                "FILT3R_MAX_FILTERED_RESIDUAL_RATIO",
                cls.max_filtered_residual_ratio,
            ),
            # gbw____
            q_mode=os.environ.get("FILT3R_Q_MODE", cls.q_mode).strip().lower(),
            drift_ema_beta=_env_float(
                "FILT3R_DRIFT_EMA_BETA", cls.drift_ema_beta
            ),
            # gbw____
            prediction_mode=os.environ.get(
                "FILT3R_PREDICTION_MODE", cls.prediction_mode
            ).strip().lower(),
            velocity_ema_beta=_env_float(
                "FILT3R_VELOCITY_EMA_BETA", cls.velocity_ema_beta
            ),
            velocity_decay=_env_float(
                "FILT3R_VELOCITY_DECAY", cls.velocity_decay
            ),
            velocity_gate_tau=_env_float(
                "FILT3R_VELOCITY_GATE_TAU", cls.velocity_gate_tau
            ),
            # ____
            # gbw____
            drift_reference_beta=_env_float(
                "FILT3R_DRIFT_REFERENCE_BETA", cls.drift_reference_beta
            ),
            # ____
            # gbw____
            drift_scope=os.environ.get(
                "FILT3R_DRIFT_SCOPE", cls.drift_scope
            ).strip().lower(),
            # ____
            q_power=_env_float("FILT3R_Q_POWER", cls.q_power),
            q_scale_min=_env_float("FILT3R_Q_SCALE_MIN", cls.q_scale_min),
            q_scale_max=_env_float("FILT3R_Q_SCALE_MAX", cls.q_scale_max),
            # gbw____
            q_min=_env_float("FILT3R_Q_MIN", cls.q_min),
            q_max=_env_float("FILT3R_Q_MAX", cls.q_max),
            q_sigmoid_alpha=_env_float(
                "FILT3R_Q_SIGMOID_ALPHA", cls.q_sigmoid_alpha
            ),
            q_sigmoid_tau=_env_float(
                "FILT3R_Q_SIGMOID_TAU", cls.q_sigmoid_tau
            ),
            # gbw____
            gain_k_delta=_env_float(
                "FILT3R_GAIN_K_DELTA", cls.gain_k_delta
            ),
            gain_sigmoid_alpha=_env_float(
                "FILT3R_GAIN_SIGMOID_ALPHA", cls.gain_sigmoid_alpha
            ),
            gain_sigmoid_tau=_env_float(
                "FILT3R_GAIN_SIGMOID_TAU", cls.gain_sigmoid_tau
            ),
            gain_k_min=_env_float("FILT3R_GAIN_K_MIN", cls.gain_k_min),
            gain_k_max=_env_float("FILT3R_GAIN_K_MAX", cls.gain_k_max),
            gain_q_min=_env_float("FILT3R_GAIN_Q_MIN", cls.gain_q_min),
            gain_q_max=_env_float("FILT3R_GAIN_Q_MAX", cls.gain_q_max),
            # ____
            # gbw____
            q_calibrate_nominal=_env_bool(
                "FILT3R_Q_CALIBRATE_NOMINAL", cls.q_calibrate_nominal
            ),
            # ____
            # gbw____
            q_coherence_weight=_env_float(
                "FILT3R_Q_COHERENCE_WEIGHT", cls.q_coherence_weight
            ),
            # ____
            # gbw____
            q_floor_nominal=_env_bool(
                "FILT3R_Q_FLOOR_NOMINAL", cls.q_floor_nominal
            ),
            # gbw____
            q_positive_boost=_env_float(
                "FILT3R_Q_POSITIVE_BOOST", cls.q_positive_boost
            ),
            # gbw____
            q_positive_deadband=_env_float(
                "FILT3R_Q_POSITIVE_DEADBAND", cls.q_positive_deadband
            ),
            # ____
            # gbw____
            q_innovation_weight=_env_float(
                "FILT3R_Q_INNOVATION_WEIGHT", cls.q_innovation_weight
            ),
            q_innovation_tau=_env_float(
                "FILT3R_Q_INNOVATION_TAU", cls.q_innovation_tau
            ),
            # gbw____
            q_jitter_penalty=_env_float(
                "FILT3R_Q_JITTER_PENALTY", cls.q_jitter_penalty
            ),
            q_jitter_tau=_env_float(
                "FILT3R_Q_JITTER_TAU", cls.q_jitter_tau
            ),
            # gbw____
            q_jitter_breaks_floor=_env_bool(
                "FILT3R_Q_JITTER_BREAKS_FLOOR",
                cls.q_jitter_breaks_floor,
            ),
            # ____
            # gbw____
            q_innovation_penalty=_env_float(
                "FILT3R_Q_INNOVATION_PENALTY",
                cls.q_innovation_penalty,
            ),
            q_innovation_penalty_tau=_env_float(
                "FILT3R_Q_INNOVATION_PENALTY_TAU",
                cls.q_innovation_penalty_tau,
            ),
            # gbw____
            q_selective_gate=_env_bool(
                "FILT3R_Q_SELECTIVE_GATE", cls.q_selective_gate
            ),
            q_selective_jitter_tau=_env_float(
                "FILT3R_Q_SELECTIVE_JITTER_TAU",
                cls.q_selective_jitter_tau,
            ),
            q_selective_innovation_tau=_env_float(
                "FILT3R_Q_SELECTIVE_INNOVATION_TAU",
                cls.q_selective_innovation_tau,
            ),
            q_selective_coherence_tau=_env_float(
                "FILT3R_Q_SELECTIVE_COHERENCE_TAU",
                cls.q_selective_coherence_tau,
            ),
            q_selective_local_score_tau=_env_float(
                "FILT3R_Q_SELECTIVE_LOCAL_SCORE_TAU",
                cls.q_selective_local_score_tau,
            ),
            q_selective_q_ratio=_env_float(
                "FILT3R_Q_SELECTIVE_Q_RATIO", cls.q_selective_q_ratio
            ),
            # gbw____
            q_selective_soft_gate=_env_bool(
                "FILT3R_Q_SELECTIVE_SOFT_GATE", cls.q_selective_soft_gate
            ),
            q_selective_softness=_env_float(
                "FILT3R_Q_SELECTIVE_SOFTNESS", cls.q_selective_softness
            ),
            # ____
            # gbw____
            q_selective_positive_drift=_env_bool(
                "FILT3R_Q_SELECTIVE_POSITIVE_DRIFT",
                cls.q_selective_positive_drift,
            ),
            # ____
            # ____
            # ____
            # ____
            # ____
            # gbw____
            q_innovation_guard_tau=_env_float(
                "FILT3R_Q_INNOVATION_GUARD_TAU", cls.q_innovation_guard_tau
            ),
            # ____
            # gbw____
            q_innovation_reference_beta=_env_float(
                "FILT3R_Q_INNOVATION_REFERENCE_BETA",
                cls.q_innovation_reference_beta,
            ),
            q_innovation_power=_env_float(
                "FILT3R_Q_INNOVATION_POWER", cls.q_innovation_power
            ),
            # ____
            # ____
            # ____
            # gbw____
            apne_ema_beta=_env_float(
                "FILT3R_APNE_EMA_BETA", cls.apne_ema_beta
            ),
            apne_log_alpha=_env_float(
                "FILT3R_APNE_LOG_ALPHA", cls.apne_log_alpha
            ),
            apne_innovation_clip=_env_float(
                "FILT3R_APNE_INNOVATION_CLIP", cls.apne_innovation_clip
            ),
            # gbw____
            apne_innovation_reference=_env_float(
                "FILT3R_APNE_INNOVATION_REFERENCE",
                cls.apne_innovation_reference,
            ),
            # gbw____
            apne_robust=_env_bool("FILT3R_APNE_ROBUST", cls.apne_robust),
            # gbw____
            apne_robust_stat=os.environ.get(
                "FILT3R_APNE_ROBUST_STAT", cls.apne_robust_stat
            ).strip().lower(),
            # ____
            # gbw____
            apne_physical=_env_bool(
                "FILT3R_APNE_PHYSICAL", cls.apne_physical
            ),
            # gbw____
            apne_drift_blend=_env_float(
                "FILT3R_APNE_DRIFT_BLEND", cls.apne_drift_blend
            ),
            apne_drift_clip=_env_float(
                "FILT3R_APNE_DRIFT_CLIP", cls.apne_drift_clip
            ),
            # gbw____
            apne_drift_coherence=os.environ.get(
                "FILT3R_APNE_DRIFT_COHERENCE",
                cls.apne_drift_coherence,
            ).strip().lower(),
            # gbw____
            apne_evidence_coherence_weight=_env_float(
                "FILT3R_APNE_EVIDENCE_COHERENCE_WEIGHT",
                cls.apne_evidence_coherence_weight,
            ),
            # ____
            # gbw____
            apne_floor_break_low_evidence=_env_bool(
                "FILT3R_APNE_FLOOR_BREAK_LOW_EVIDENCE",
                cls.apne_floor_break_low_evidence,
            ),
            apne_floor_break_evidence_tau=_env_float(
                "FILT3R_APNE_FLOOR_BREAK_EVIDENCE_TAU",
                cls.apne_floor_break_evidence_tau,
            ),
            apne_floor_break_coherence_tau=_env_float(
                "FILT3R_APNE_FLOOR_BREAK_COHERENCE_TAU",
                cls.apne_floor_break_coherence_tau,
            ),
            # gbw____
            apne_floor_break_jitter_tau=_env_float(
                "FILT3R_APNE_FLOOR_BREAK_JITTER_TAU",
                cls.apne_floor_break_jitter_tau,
            ),
            # ____
            # ____
            # ____
            # ____
            # ____
            # ____
            # ____
            # gbw____
            apne_jitter_penalty=_env_float(
                "FILT3R_APNE_JITTER_PENALTY", cls.apne_jitter_penalty
            ),
            apne_jitter_tau=_env_float(
                "FILT3R_APNE_JITTER_TAU", cls.apne_jitter_tau
            ),
            # ____
            # gbw____
            # APNE 的保守模式：低创新证据只能保持 nominal Q，不能把
            # bounded-sigmoid 的 Q/R 压到 A1 中性值以下。
            # ____
            apne_q_floor_nominal=_env_bool(
                "FILT3R_APNE_Q_FLOOR_NOMINAL", cls.apne_q_floor_nominal
            ),
            # ____
            # ____
            # gbw____
            jitter_weight=_env_float(
                "FILT3R_JITTER_WEIGHT", cls.jitter_weight
            ),
            max_token_shift_ratio=_env_float(
                "FILT3R_MAX_TOKEN_SHIFT_RATIO", cls.max_token_shift_ratio
            ),
            # ____
            reset_mode=os.environ.get(
                "FILT3R_RESET_MODE", cls.reset_mode
            ).strip().lower(),
            reset_patience=_env_int(
                "FILT3R_RESET_PATIENCE", cls.reset_patience
            ),
            innovation_gate_tau=_env_float(
                "FILT3R_INNOVATION_GATE_TAU", cls.innovation_gate_tau
            ),
            # ____
        )


@dataclass
class _TokenState:
    # gbw____
    # A1 每个 state key 保存四类核心历史：filtered token=S、协方差=P、上一帧
    # raw candidate=Z_{t-1}，以及用于 reset 的 camera delta EMA。其余字段是后续
    # B1/B2 扩展所需；A1 constant/current 路径不会依赖它们改变正常更新。
    # ____
    filtered: torch.Tensor
    covariance: torch.Tensor
    previous_candidate: torch.Tensor
    camera_delta_ema: Optional[torch.Tensor]
    # gbw____
    # constant_velocity 模式保存上一帧已滤波的 token 速度；hold 模式为
    # None，因此 A1/A2 的内存和数值路径保持原样。
    # ____
    velocity: Optional[torch.Tensor]
    # gbw____
    drift_ema: Optional[torch.Tensor]
    # gbw____
    # token_ema_double 的慢速历史尺度；它与 drift_ema 形状相同，仍按
    # view/spatial token 维护，不改变 full-channel token 接口。
    drift_reference_ema: Optional[torch.Tensor]
    # ____
    # gbw____
    # APNE 只保存每个 spatial token 的 innovation 能量 EMA；Q 最终沿 2048
    # 个 channel 广播，避免额外维护一个新的 channel-wise 网络或参数。
    innovation_energy_ema: Optional[torch.Tensor]
    # ____
    # gbw____
    # token_ema_apne 的逐 token 过程噪声证据 EMA；它只保存历史统计量，
    # 不增加可训练参数，也不改变 A1/A2 其他 scope 的状态接口。
    process_noise_ratio_ema: Optional[torch.Tensor]
    # ____
    # gbw____
    # innovation-aligned Q 维护逐 spatial token 的慢速 normalized-
    # innovation reference；第一帧有效更新从当前观测初始化。
    # ____
    innovation_reference_ema: Optional[torch.Tensor]
    # ____
    previous_delta: Optional[torch.Tensor]
    local_high_count: Optional[torch.Tensor]
    camera_high_count: Optional[torch.Tensor]
    # ____


class A1UnifiedTokenKalmanFilter:
    """A1 universal causal AKF for canonical PaliGemma image-token grids."""

    VALID_MODES = frozenset(("none", "filt3r_akf"))

    def __init__(
        self,
        mode: Optional[str] = None,
        config: Optional[Filt3rAKFConfig] = None,
        diagnostics_enabled: Optional[bool] = None,
    ) -> None:
        # gbw____
        # 这个类是 A1 的核心入口。它只保存时间状态，不改变 PaliGemma 或
        # ConvexUpSample 的模型参数；FILTER_MODE=none 时，apply() 会完全旁路。
        # ____
        self.mode = mode or os.environ.get("FILTER_MODE", "none")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"unsupported FILTER_MODE={self.mode!r}; expected none or filt3r_akf"
            )
        self.config = config or Filt3rAKFConfig.from_environment()
        self.diagnostics_enabled = (
            _env_bool("FILTER_DIAGNOSTICS")
            if diagnostics_enabled is None
            else diagnostics_enabled
        )
        # gbw____
        self.shadow_raw_enabled = self.diagnostics_enabled and _env_bool(
            "FILTER_DIAGNOSTICS_SHADOW_RAW"
        )
        # ____
        self.diagnostics_path = os.environ.get("FILTER_DIAGNOSTICS_PATH")
        self._episode_index = 0
        self._states: Dict[
            Tuple[int, int, Tuple[int, ...], str, str], _TokenState
        ] = {}
        self._last_diagnostics: Optional[dict] = None
        self._call_index = 0
        self._context: Dict[str, object] = {}

    @property
    def enabled(self) -> bool:
        return self.mode == "filt3r_akf"

    def enabled_for_stage(self, stage: int) -> bool:
        return self.enabled and stage in self.config.stages

    @property
    def last_diagnostics(self) -> Optional[dict]:
        return self._last_diagnostics

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def episode_index(self) -> int:
        return self._episode_index

    def set_context(
        self, *, task_label: Optional[str] = None, timestep: Optional[int] = None
    ) -> None:
        # gbw____
        # task_label 和 timestep 只用于 diagnostics，不能用来选择滤波参数。
        # 因此所有 RLBench task 实际共用同一套 A1 数学更新。
        # ____
        # Task label is diagnostics-only; it never selects a filter or parameter.
        self._context = {
            "task_label": None if task_label is None else str(task_label),
            "timestep": None if timestep is None else int(timestep),
            "eval_seed": os.environ.get("EVAL_SEED"),
        }

    def reset(self, reason: str = "agent_reset") -> int:
        # gbw____
        # episode/task/terminal 边界调用这里。清空后，下一帧会重新执行
        # S_0=Z_0、P_0=p_init，避免上一条 episode 的视觉状态泄漏到下一条。
        # ____
        count = len(self._states)
        self._states.clear()
        self._last_diagnostics = None
        if self.mode != "none":
            cleared_episode_index = self._episode_index
            self._episode_index += 1
            if self.diagnostics_enabled:
                self._append_diagnostics(
                    {
                        "event": "reset",
                        "algorithm": "A1_U_IGAKF",
                        "reason": reason,
                        "cleared_states": count,
                        "filter_mode": self.mode,
                        "call_index": self._call_index,
                        "episode_index": cleared_episode_index,
                        "next_episode_index": self._episode_index,
                    }
                )
        return count

    @staticmethod
    def _validate(tokens: torch.Tensor) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("U-IGAKF tokens must be a torch.Tensor")
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != CANONICAL_TOKEN_TAIL:
            raise ValueError(
                "the unified AKF requires (B, 3, 16, 16, 2048), "
                f"got {tuple(tokens.shape)}"
            )
        if not tokens.is_floating_point():
            raise TypeError("U-IGAKF tokens must be floating point")

    def _state_key(
        self, tokens: torch.Tensor, stage: int
    ) -> Tuple[int, int, Tuple[int, ...], str, str]:
        # gbw____
        # A1 状态按 episode、stage、shape、device、dtype 隔离。
        # 当前默认 FILT3R_STAGES=1，所以正常情况下只维护 Stage-1 state。
        # ____
        return (
            self._episode_index,
            stage,
            tuple(tokens.shape),
            str(tokens.device),
            str(tokens.dtype),
        )

    def _context_fields(self) -> dict:
        return {
            "task_label": self._context.get("task_label"),
            "timestep": self._context.get("timestep"),
            "eval_seed": self._context.get("eval_seed"),
        }

    @staticmethod
    def _quantiles(value: Optional[torch.Tensor]) -> Optional[dict]:
        if value is None:
            return None
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return None
        q = torch.quantile(
            flat,
            torch.tensor([0.5, 0.9, 0.99], device=flat.device),
        )
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "p50": float(q[0].cpu()),
            "p90": float(q[1].cpu()),
            "p99": float(q[2].cpu()),
        }

    @staticmethod
    def _camera_means(value: Optional[torch.Tensor]) -> Optional[list]:
        if value is None or value.ndim < 2 or value.shape[1] != 3:
            return None
        reduce_dims = (0,) + tuple(range(2, value.ndim))
        return value.detach().float().mean(dim=reduce_dims).cpu().tolist()

    def _append_diagnostics(self, record: dict) -> None:
        if not self.diagnostics_path:
            return
        parent = os.path.dirname(self.diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.diagnostics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _set_update_diagnostics(
        self,
        *,
        stage: int,
        initialized: bool,
        tokens: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        delta_ema: Optional[torch.Tensor] = None,
        innovation: Optional[torch.Tensor] = None,
        local_score: Optional[torch.Tensor] = None,
        camera_score: Optional[torch.Tensor] = None,
        q_t: Optional[torch.Tensor] = None,
        drift_ema: Optional[torch.Tensor] = None,
        drift_ratio: Optional[torch.Tensor] = None,
        # gbw____
        q_drift_ratio: Optional[torch.Tensor] = None,
        # gbw____
        q_low_guard: Optional[torch.Tensor] = None,
        # gbw____
        q_selective_gate: Optional[torch.Tensor] = None,
        # gbw____
        q_selective_weight: Optional[torch.Tensor] = None,
        # ____
        # ____
        # gbw____
        drift_coherence: Optional[torch.Tensor] = None,
        # gbw____
        magnitude_coherence: Optional[torch.Tensor] = None,
        # gbw____
        raw_magnitude_coherence: Optional[torch.Tensor] = None,
        # ____
        # ____
        # gbw____
        spatial_coherence: Optional[torch.Tensor] = None,
        # ____
        # ____
        # gbw____
        innovation_ratio: Optional[torch.Tensor] = None,
        # ____
        q_scale: Optional[torch.Tensor] = None,
        # gbw____
        drift_reference_ema: Optional[torch.Tensor] = None,
        # ____
        # gbw____
        innovation_energy_ema: Optional[torch.Tensor] = None,
        apne_q_hat: Optional[torch.Tensor] = None,
        apne_q_evidence: Optional[torch.Tensor] = None,
        # gbw____
        apne_floor_break: Optional[torch.Tensor] = None,
        # ____
        # ____
        jitter_score: Optional[torch.Tensor] = None,
        p_prior: Optional[torch.Tensor] = None,
        gain: Optional[torch.Tensor] = None,
        # gbw____
        gain_target: Optional[torch.Tensor] = None,
        # ____
        covariance: Optional[torch.Tensor] = None,
        # gbw____
        measurement_r: Optional[torch.Tensor] = None,
        # ____
        normalized_innovation: Optional[torch.Tensor] = None,
        # gbw____
        robust_normalized_innovation: Optional[torch.Tensor] = None,
        # ____
        filtered_residual_ratio: Optional[torch.Tensor] = None,
        token_shift_ratio: Optional[torch.Tensor] = None,
        trust_region_clipped: Optional[torch.Tensor] = None,
        local_reset: Optional[torch.Tensor] = None,
        camera_reset: Optional[torch.Tensor] = None,
        residual_reset: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        outlier_gate: Optional[torch.Tensor] = None,
        persistent_reset: Optional[torch.Tensor] = None,
        jitter_mask: Optional[torch.Tensor] = None,
        jitter_reset_overlap: Optional[torch.Tensor] = None,
        raw_takeover_exact_mask: Optional[torch.Tensor] = None,
        takeover_error: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.diagnostics_enabled:
            self._last_diagnostics = None
            return

        def mean(value: Optional[torch.Tensor]) -> Optional[float]:
            return None if value is None else float(value.detach().float().mean().cpu())

        def fraction(value: Optional[torch.Tensor]) -> Optional[float]:
            return None if value is None else float(value.detach().float().mean().cpu())

        # gbw____
        def conditional_fraction(
            value: Optional[torch.Tensor],
            condition: Optional[torch.Tensor],
        ) -> Optional[float]:
            if value is None or condition is None:
                return None
            denominator = condition.detach().float().sum()
            if float(denominator.cpu()) == 0.0:
                return None
            numerator = value.detach().float().sum()
            return float((numerator / denominator).cpu())

        def conditional_mean(
            value: Optional[torch.Tensor],
            condition: Optional[torch.Tensor],
        ) -> Optional[float]:
            if value is None or condition is None:
                return None
            selected = value.detach().float()[condition.detach().bool()]
            if selected.numel() == 0:
                return None
            return float(selected.mean().cpu())
        # ____

        self._last_diagnostics = {
            "event": "update",
            "algorithm": "A1_U_IGAKF",
            "call_index": self._call_index,
            "filter_mode": self.mode,
            "stage": stage,
            "episode_index": self._episode_index,
            **self._context_fields(),
            "initialized": initialized,
            "token_shape": list(tokens.shape),
            "token_dtype": str(tokens.dtype),
            "target_gain": self.config.target_gain,
            "gamma_p": self.config.gamma_p,
            # gbw____
            "p_init": self.config.p_init,
            "p_init_over_r": self.config.p_init / self.config.r,
            "nominal_q_ratio": self.config.target_gain
            * (1.0 / (1.0 - self.config.target_gain) - self.config.gamma_p),
            "nominal_q": self.config.r
            * self.config.target_gain
            * (
                1.0 / (1.0 - self.config.target_gain)
                - self.config.gamma_p
            ),
            # ____
            # gbw____
            "q_mode": self.config.q_mode,
            "reset_mode": self.config.reset_mode,
            # gbw____
            "prediction_mode": self.config.prediction_mode,
            "velocity_ema_beta": self.config.velocity_ema_beta,
            "velocity_decay": self.config.velocity_decay,
            "velocity_gate_tau": self.config.velocity_gate_tau,
            # ____
            "jitter_weight": self.config.jitter_weight,
            "max_token_shift_ratio": self.config.max_token_shift_ratio,
            # gbw____
            "q_min": self.config.q_min,
            "q_max": self.config.q_max,
            "q_sigmoid_alpha": self.config.q_sigmoid_alpha,
            "q_sigmoid_tau": self.config.q_sigmoid_tau,
            # gbw____
            "gain_k_delta": self.config.gain_k_delta,
            "gain_sigmoid_alpha": self.config.gain_sigmoid_alpha,
            "gain_sigmoid_tau": self.config.gain_sigmoid_tau,
            "gain_k_min": self.config.gain_k_min,
            "gain_k_max": self.config.gain_k_max,
            "gain_q_min": self.config.gain_q_min,
            "gain_q_max": self.config.gain_q_max,
            # ____
            # gbw____
            "q_calibrate_nominal": self.config.q_calibrate_nominal,
            # ____
            # gbw____
            "drift_reference_beta": self.config.drift_reference_beta,
            # ____
            # gbw____
            "q_coherence_weight": self.config.q_coherence_weight,
            # ____
            # gbw____
            "q_floor_nominal": self.config.q_floor_nominal,
            # ____
            # gbw____
            "q_positive_boost": self.config.q_positive_boost,
            # ____
            # gbw____
            "q_positive_deadband": self.config.q_positive_deadband,
            # ____
            # gbw____
            "q_innovation_weight": self.config.q_innovation_weight,
            "q_innovation_tau": self.config.q_innovation_tau,
            # gbw____
            "q_jitter_penalty": self.config.q_jitter_penalty,
            "q_jitter_tau": self.config.q_jitter_tau,
            # gbw____
            "q_jitter_breaks_floor": self.config.q_jitter_breaks_floor,
            # ____
            # gbw____
            "q_innovation_penalty": self.config.q_innovation_penalty,
            "q_innovation_penalty_tau": self.config.q_innovation_penalty_tau,
            # gbw____
            "q_selective_gate": self.config.q_selective_gate,
            "q_selective_jitter_tau": self.config.q_selective_jitter_tau,
            "q_selective_innovation_tau": self.config.q_selective_innovation_tau,
            "q_selective_coherence_tau": self.config.q_selective_coherence_tau,
            "q_selective_local_score_tau": self.config.q_selective_local_score_tau,
            "q_selective_q_ratio": self.config.q_selective_q_ratio,
            # gbw____
            "q_selective_soft_gate": self.config.q_selective_soft_gate,
            "q_selective_softness": self.config.q_selective_softness,
            # ____
            # gbw____
            "q_selective_positive_drift": self.config.q_selective_positive_drift,
            # ____
            # ____
            # ____
            # ____
            # ____
            # gbw____
            "q_innovation_guard_tau": self.config.q_innovation_guard_tau,
            # ____
            # gbw____
            "q_innovation_reference_beta": self.config.q_innovation_reference_beta,
            "q_innovation_power": self.config.q_innovation_power,
            # ____
            # gbw____
            "apne_ema_beta": self.config.apne_ema_beta,
            "apne_log_alpha": self.config.apne_log_alpha,
            "apne_innovation_clip": self.config.apne_innovation_clip,
            # gbw____
            "apne_innovation_reference": self.config.apne_innovation_reference,
            # gbw____
            "apne_robust": self.config.apne_robust,
            # gbw____
            "apne_robust_stat": self.config.apne_robust_stat,
            # ____
            # gbw____
            "apne_physical": self.config.apne_physical,
            # gbw____
            "apne_drift_blend": self.config.apne_drift_blend,
            "apne_drift_clip": self.config.apne_drift_clip,
            # gbw____
            "apne_drift_coherence": self.config.apne_drift_coherence,
            # ____
            # gbw____
            "apne_evidence_coherence_weight": (
                self.config.apne_evidence_coherence_weight
            ),
            # ____
            # gbw____
            "apne_floor_break_low_evidence": (
                self.config.apne_floor_break_low_evidence
            ),
            "apne_floor_break_evidence_tau": (
                self.config.apne_floor_break_evidence_tau
            ),
            "apne_floor_break_coherence_tau": (
                self.config.apne_floor_break_coherence_tau
            ),
            # gbw____
            "apne_floor_break_jitter_tau": (
                self.config.apne_floor_break_jitter_tau
            ),
            # ____
            # ____
            # ____
            # ____
            # ____
            # ____
            # gbw____
            "apne_jitter_penalty": self.config.apne_jitter_penalty,
            "apne_jitter_tau": self.config.apne_jitter_tau,
            # ____
            # gbw____
            "apne_q_floor_nominal": self.config.apne_q_floor_nominal,
            # ____
            # ____
            # gbw____
            "drift_ema_scope": (
                self.config.drift_scope
                if self.config.q_mode == "drift_adaptive_sigmoid"
                else "global_token_median+innovation_ema"
                if self.config.q_mode == "apne_bounded_sigmoid"
                else "token"
            ),
            # ____
            # ____
            # ____
            "r": self.config.r,
            # gbw____
            "r_mode": self.config.r_mode,
            "jitter_r_lambda": self.config.jitter_r_lambda,
            "jitter_r_max": self.config.jitter_r_max,
            "jitter_r_tau": self.config.jitter_r_tau,
            "jitter_r_coherence_weight": self.config.jitter_r_coherence_weight,
            # gbw____
            "jitter_r_coherence_mode": self.config.jitter_r_coherence_mode,
            # ____
            # gbw____
            "jitter_r_channel_mode": self.config.jitter_r_channel_mode,
            # ____
            "q_use_adaptive_r": self.config.q_use_adaptive_r,
            "r_t_mean": mean(measurement_r),
            "r_t_quantiles": self._quantiles(measurement_r),
            # ____
            "r_channel_count": 2048,
            "r_shape": list(tokens.shape),
            "reset_local_tau": self.config.reset_local_tau,
            "reset_camera_quantile": self.config.reset_camera_quantile,
            "reset_camera_tau": self.config.reset_camera_tau,
            # gbw____
            "reset_coherence_tau": self.config.reset_coherence_tau,
            # ____
            "max_filtered_residual_ratio": self.config.max_filtered_residual_ratio,
            # gbw____
            "diagnostic_jitter_threshold": DIAGNOSTIC_JITTER_THRESHOLD,
            "diagnostic_raw_match_tolerance": DIAGNOSTIC_RAW_MATCH_TOLERANCE,
            # ____
            "delta_mean": mean(delta),
            "delta_ema_mean": mean(delta_ema),
            "raw_delta_mean": mean(delta),
            "raw_delta_quantiles": self._quantiles(delta),
            # gbw____
            "drift_ema_mean": mean(drift_ema),
            "drift_ratio_mean": mean(drift_ratio),
            "drift_ratio_quantiles": self._quantiles(drift_ratio),
            # gbw____
            "drift_reference_ema_mean": mean(drift_reference_ema),
            "drift_reference_ema_quantiles": self._quantiles(
                drift_reference_ema
            ),
            # ____
            # gbw____
            "q_drift_ratio_mean": mean(q_drift_ratio),
            "q_drift_ratio_quantiles": self._quantiles(q_drift_ratio),
            # gbw____
            "q_low_guard_fraction": fraction(q_low_guard),
            # gbw____
            "q_selective_gate_fraction": fraction(q_selective_gate),
            # gbw____
            "q_selective_weight_mean": mean(q_selective_weight),
            "q_selective_weight_quantiles": self._quantiles(
                q_selective_weight
            ),
            # ____
            # ____
            # gbw____
            "drift_coherence_mean": mean(drift_coherence),
            "drift_coherence_quantiles": self._quantiles(drift_coherence),
            # gbw____
            "magnitude_coherence_mean": mean(magnitude_coherence),
            "magnitude_coherence_quantiles": self._quantiles(
                magnitude_coherence
            ),
            # gbw____
            "raw_magnitude_coherence_mean": mean(
                raw_magnitude_coherence
            ),
            "raw_magnitude_coherence_quantiles": self._quantiles(
                raw_magnitude_coherence
            ),
            # ____
            # ____
            # ____
            # gbw____
            "spatial_coherence_mean": mean(spatial_coherence),
            "spatial_coherence_quantiles": self._quantiles(spatial_coherence),
            # ____
            # ____
            # gbw____
            "innovation_ratio_mean": mean(innovation_ratio),
            "innovation_ratio_quantiles": self._quantiles(innovation_ratio),
            # ____
            "q_scale_mean": mean(q_scale),
            "q_scale_quantiles": self._quantiles(q_scale),
            # gbw____
            "innovation_energy_ema_mean": mean(innovation_energy_ema),
            "innovation_energy_ema_quantiles": self._quantiles(
                innovation_energy_ema
            ),
            "apne_q_hat_mean": mean(apne_q_hat),
            "apne_q_hat_quantiles": self._quantiles(apne_q_hat),
            "apne_q_evidence_mean": mean(apne_q_evidence),
            "apne_q_evidence_quantiles": self._quantiles(apne_q_evidence),
            # gbw____
            "apne_floor_break_fraction": fraction(apne_floor_break),
            # ____
            # ____
            "jitter_score_mean": mean(jitter_score),
            "jitter_score_quantiles": self._quantiles(jitter_score),
            # ____
            "innovation_mean": mean(innovation),
            "innovation_quantiles": self._quantiles(innovation),
            # gbw____
            "normalized_innovation_mean": mean(normalized_innovation),
            "normalized_innovation_quantiles": self._quantiles(
                normalized_innovation
            ),
            # gbw____
            "robust_normalized_innovation_mean": mean(
                robust_normalized_innovation
            ),
            "robust_normalized_innovation_quantiles": self._quantiles(
                robust_normalized_innovation
            ),
            # ____
            "innovation_camera_mean": self._camera_means(innovation),
            "local_score_mean": mean(local_score),
            "local_score_quantiles": self._quantiles(local_score),
            "camera_score_mean": mean(camera_score),
            "camera_score_camera_mean": self._camera_means(camera_score),
            "q_mean": mean(q_t),
            "p_prior_mean": mean(p_prior),
            "p_mean": mean(covariance),
            "r_mean": self.config.r,
            "gain_mean": mean(gain),
            "gain_min": None if gain is None else float(gain.detach().float().min().cpu()),
            "gain_max": None if gain is None else float(gain.detach().float().max().cpu()),
            # gbw____
            "gain_target_mean": mean(gain_target),
            "gain_target_quantiles": self._quantiles(gain_target),
            # ____
            "filtered_residual_ratio_mean": mean(filtered_residual_ratio),
            "filtered_residual_ratio_quantiles": self._quantiles(
                filtered_residual_ratio
            ),
            # gbw____
            "token_shift_ratio_mean": mean(token_shift_ratio),
            "token_shift_ratio_quantiles": self._quantiles(token_shift_ratio),
            "trust_region_clipped_fraction": fraction(trust_region_clipped),
            # ____
            "local_reset_fraction": fraction(local_reset),
            "camera_reset_fraction": fraction(camera_reset),
            # gbw____
            "residual_reset_fraction": fraction(residual_reset),
            "measurement_takeover_fraction": fraction(reset_mask),
            "jitter_token_fraction": fraction(jitter_mask),
            "jitter_reset_overlap_fraction": fraction(jitter_reset_overlap),
            "reset_given_jitter": conditional_fraction(
                jitter_reset_overlap, jitter_mask
            ),
            "jitter_given_reset": conditional_fraction(
                jitter_reset_overlap, reset_mask
            ),
            "raw_takeover_exact_fraction": fraction(raw_takeover_exact_mask),
            "raw_filtered_relative_error_mean": mean(takeover_error),
            "raw_filtered_relative_error_reset_mean": conditional_mean(
                takeover_error, reset_mask
            ),
            # ____
            "local_reset_camera_mean": self._camera_means(local_reset),
            "camera_reset_camera_mean": self._camera_means(camera_reset),
            # gbw____
            "outlier_gate_fraction": fraction(outlier_gate),
            "persistent_reset_fraction": fraction(persistent_reset),
            # ____
            "finite": bool(torch.isfinite(tokens).all().item()),
        }
        self._append_diagnostics(self._last_diagnostics)

    # gbw____
    @staticmethod
    def _heatmap_stats(logits: torch.Tensor) -> dict:
        flat = logits.detach().float().flatten(-2)
        probs = torch.softmax(flat, dim=-1)
        topk = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        entropy = -(
            probs.clamp_min(1e-6) * probs.clamp_min(1e-6).log()
        ).sum(dim=-1) / torch.log(
            torch.tensor(float(probs.shape[-1]), device=probs.device)
        )
        peak_index = probs.argmax(dim=-1)
        peak_xy = torch.stack(
            (peak_index % logits.shape[-1], peak_index // logits.shape[-1]),
            dim=-1,
        )
        return {
            "peak": topk[..., 0],
            "margin": (
                topk[..., 0] - topk[..., 1]
                if topk.shape[-1] > 1
                else None
            ),
            "entropy": entropy,
            "peak_xy": peak_xy,
        }
    # ____

    # gbw____
    def record_output_diagnostics(
        self,
        logits: torch.Tensor,
        stage: int,
        raw_logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Record filtered output and optional diagnostics-only raw shadow output."""

        if not self.diagnostics_enabled or not self.enabled_for_stage(stage):
            return
        if logits.ndim != 4:
            raise ValueError(f"expected logits (B,C,H,W), got {tuple(logits.shape)}")
        if raw_logits is not None:
            if not self.shadow_raw_enabled:
                raise ValueError(
                    "raw shadow diagnostics require "
                    "FILTER_DIAGNOSTICS_SHADOW_RAW=1"
                )
            if raw_logits.shape != logits.shape:
                raise ValueError(
                    "raw shadow logits must match filtered logits, got "
                    f"{tuple(raw_logits.shape)} vs {tuple(logits.shape)}"
                )

        filtered_stats = self._heatmap_stats(logits)
        record = {
            "event": "output",
            "algorithm": "A1_U_IGAKF",
            "call_index": self._call_index,
            "episode_index": self._episode_index,
            "filter_mode": self.mode,
            "stage": stage,
            "logit_shape": list(logits.shape),
            "heatmap_peak": filtered_stats["peak"].cpu().tolist(),
            "heatmap_margin": (
                filtered_stats["margin"].cpu().tolist()
                if filtered_stats["margin"] is not None
                else None
            ),
            "heatmap_entropy": filtered_stats["entropy"].cpu().tolist(),
            "heatmap_peak_xy": filtered_stats["peak_xy"].cpu().tolist(),
            "raw_shadow_enabled": self.shadow_raw_enabled,
            **self._context_fields(),
        }
        # gbw____
        if raw_logits is not None:
            raw_stats = self._heatmap_stats(raw_logits)
            peak_shift = torch.linalg.vector_norm(
                filtered_stats["peak_xy"].float() - raw_stats["peak_xy"].float(),
                dim=-1,
            )
            record.update(
                {
                    "raw_heatmap_peak": raw_stats["peak"].cpu().tolist(),
                    "raw_heatmap_margin": (
                        raw_stats["margin"].cpu().tolist()
                        if raw_stats["margin"] is not None
                        else None
                    ),
                    "raw_heatmap_entropy": raw_stats["entropy"].cpu().tolist(),
                    "raw_heatmap_peak_xy": raw_stats["peak_xy"].cpu().tolist(),
                    "raw_filtered_peak_shift_mean": float(
                        peak_shift.mean().cpu()
                    ),
                    "raw_filtered_peak_shift": peak_shift.cpu().tolist(),
                }
            )
        # ____
        self._append_diagnostics(record)
    # ____

    def record_token_selection_diagnostics(
        self,
        *,
        image_token_id: int,
        selected_count: int,
        expected_count: int,
        sequence_length: int,
        stage: int = 1,
    ) -> None:
        if not self.diagnostics_enabled or not self.enabled_for_stage(stage):
            return
        self._append_diagnostics(
            {
                "event": "token_selection",
                "algorithm": "A1_U_IGAKF",
                "call_index": self._call_index,
                "episode_index": self._episode_index,
                "filter_mode": self.mode,
                "stage": stage,
                "image_token_id": int(image_token_id),
                "selected_count": int(selected_count),
                "expected_count": int(expected_count),
                "sequence_length": int(sequence_length),
                **self._context_fields(),
            }
        )

    # gbw____
    def record_waypoint_diagnostics(
        self,
        waypoint: torch.Tensor,
        stage: int = 1,
        raw_waypoint: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.diagnostics_enabled or not self.enabled_for_stage(stage):
            return
        record = {
            "event": "waypoint",
            "algorithm": "A1_U_IGAKF",
            "call_index": self._call_index,
            "episode_index": self._episode_index,
            "filter_mode": self.mode,
            "stage": stage,
            "waypoint": waypoint.detach().float().cpu().tolist(),
            **self._context_fields(),
        }
        # gbw____
        if raw_waypoint is not None:
            if not self.shadow_raw_enabled:
                raise ValueError(
                    "raw shadow waypoint diagnostics require "
                    "FILTER_DIAGNOSTICS_SHADOW_RAW=1"
                )
            if raw_waypoint.shape != waypoint.shape:
                raise ValueError(
                    "raw shadow waypoint must match filtered waypoint, got "
                    f"{tuple(raw_waypoint.shape)} vs {tuple(waypoint.shape)}"
                )
            waypoint_shift = torch.linalg.vector_norm(
                waypoint.detach().float() - raw_waypoint.detach().float(), dim=-1
            )
            record.update(
                {
                    "raw_waypoint": raw_waypoint.detach().float().cpu().tolist(),
                    "raw_filtered_waypoint_shift_mean": float(
                        waypoint_shift.mean().cpu()
                    ),
                    "raw_filtered_waypoint_shift": waypoint_shift.cpu().tolist(),
                }
            )
        # ____
        self._append_diagnostics(record)
    # ____

    # gbw____
    @staticmethod
    def _temporal_metrics(
        candidate: torch.Tensor,
        state: _TokenState,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Compute causal measurements shared by A1-clean and A2."""

        feature_scale = float(candidate.shape[-1]) ** 0.5
        r_t = torch.full_like(candidate, cfg.r)
        delta_vector = candidate - state.previous_candidate # raw_delta: 当前 raw token - 上一帧 raw token.  raw token 的时间变化，用于 drift/jitter 分析和 B1 的自适应 Q
        # gbw____
        # 预测只读取上一帧 state。constant_velocity 的衰减速度来自已经
        # 滤波过的 token，不把当前 raw observation 偷换成历史状态；hold
        # 分支严格等价于原来的 random-walk prediction。gated_velocity
        # 只在当前 raw drift 与历史速度方向一致时使用预测，反向 spike
        # 自动退回 hold，避免速度外推放大 jitter。
        if (
            cfg.prediction_mode in {"constant_velocity", "gated_velocity"}
            and state.velocity is not None
        ):
            prediction_scale = torch.ones_like(
                torch.linalg.vector_norm(delta_vector, dim=-1)
            )
            if cfg.prediction_mode == "gated_velocity":
                current_norm = torch.linalg.vector_norm(delta_vector, dim=-1)
                velocity_norm = torch.linalg.vector_norm(state.velocity, dim=-1)
                cosine = torch.sum(
                    delta_vector * state.velocity, dim=-1
                ) / (current_norm * velocity_norm + cfg.eps)
                prediction_scale = (
                    (cosine - cfg.velocity_gate_tau)
                    / (1.0 - cfg.velocity_gate_tau + cfg.eps)
                ).clamp(min=0.0, max=1.0)
            prediction = state.filtered + (
                cfg.velocity_decay
                * prediction_scale.unsqueeze(-1)
                * state.velocity
            )
        else:
            prediction = state.filtered
        # ____
        innovation_vector = candidate - prediction # innovation    当前观测相对于滤波预测的偏差，用于 reset
        raw_delta = torch.linalg.vector_norm(delta_vector, dim=-1) / feature_scale 
        innovation = ( # 把 2048 个 channel 聚合成一个变化强度：innovation.shape = (B, 3, 16, 16)
            torch.linalg.vector_norm(innovation_vector, dim=-1) / feature_scale
        )
        normalized_innovation = torch.sqrt(
            torch.mean(
                innovation_vector.square()
                / (state.covariance + r_t + cfg.eps),
                dim=-1,
            ).clamp_min(cfg.eps)
        )

        # gbw____
        # innovation-aligned scope 使用 t-1 的 normalized-innovation EMA
        # 作为无监督、因果的局部尺度。当前帧只读取旧 reference 计算
        # innovation_ratio，随后才把当前值写入新的 EMA，避免当前异常帧
        # 同时改变证据基线和 Q。
        innovation_aligned = (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_innovation_aligned"
        )
        if innovation_aligned:
            if state.innovation_reference_ema is None:
                innovation_ratio = torch.ones_like(normalized_innovation)
                innovation_reference_ema = normalized_innovation.clamp_min(
                    cfg.delta_floor
                )
            else:
                innovation_ratio = normalized_innovation / (
                    state.innovation_reference_ema + cfg.eps
                )
                innovation_reference_ema = (
                    (1.0 - cfg.q_innovation_reference_beta)
                    * state.innovation_reference_ema
                    + cfg.q_innovation_reference_beta * normalized_innovation
                ).clamp_min(cfg.delta_floor)
        else:
            innovation_ratio = None
            innovation_reference_ema = None
        # ____

        camera_delta = torch.quantile(
            raw_delta.flatten(-2), cfg.reset_camera_quantile, dim=-1
        )
        if state.camera_delta_ema is None:
            camera_delta_ema = camera_delta.clamp_min(cfg.delta_floor)
            camera_score = torch.zeros_like(camera_delta)
        else:
            camera_score = camera_delta / (state.camera_delta_ema + cfg.eps)
            camera_delta_ema = (
                (1.0 - cfg.reset_camera_ema_beta) * state.camera_delta_ema
                + cfg.reset_camera_ema_beta * camera_delta
            ).clamp_min(cfg.delta_floor)

        # gbw____
        # innovation_coherent 与 innovation_repartition_norm 使用同一个
        # normalized innovation 统计量；差别只在 reset 是否需要时间
        # 方向一致性约束。
        # ____
        reset_metric = (
            normalized_innovation
            if cfg.reset_mode in {
                "innovation_repartition_norm",
                "innovation_coherent",
            }
            else innovation
        )
        local_values = reset_metric.flatten(-2)
        local_center = torch.median(local_values, dim=-1).values[..., None, None]
        local_deviation = (reset_metric - local_center).abs().flatten(-2)
        local_mad = torch.median(local_deviation, dim=-1).values[..., None, None]
        local_scale = (1.4826 * local_mad).clamp_min(cfg.delta_floor)
        local_score = (reset_metric - local_center) / (local_scale + cfg.eps)
        local_reset = local_score > cfg.reset_local_tau
        camera_reset = camera_score > cfg.reset_camera_tau
        camera_reset_map = camera_reset[..., None, None].expand_as(local_reset)

        # gbw____
        # A2 使用一个跨 view/spatial token 的 robust global drift EMA。
        # 这样 Q 的尺度由整帧历史变化决定，而不是让每个 token 自己的噪声
        # 把协方差推高；g_t 仍然保留到 token level，用来表达局部变化。
        # gbw____
        if (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope in {
                "global_token_median",
                "global_token_mean",
                "global_token_mean_persistent",
            }
        ):
            if cfg.drift_scope in {
                "global_token_mean",
                "global_token_mean_persistent",
            }:
                drift_stat = raw_delta.flatten(1).mean(dim=-1)
            else:
                drift_stat = torch.median(raw_delta.flatten(1), dim=-1).values
            if state.drift_ema is None:
                drift_ratio = torch.ones_like(raw_delta)
                drift_ema = drift_stat.clamp_min(cfg.delta_floor)
                previous_drift = drift_ema
            else:
                previous_drift = state.drift_ema
                drift_ratio = raw_delta / (
                    previous_drift[..., None, None, None] + cfg.eps
                )
                drift_ema = (
                    (1.0 - cfg.drift_ema_beta) * previous_drift
                    + cfg.drift_ema_beta * drift_stat
                ).clamp_min(cfg.delta_floor)
        else:
            if state.drift_ema is None:
                drift_ratio = torch.ones_like(raw_delta)
                drift_ema = raw_delta.clamp_min(cfg.delta_floor)
                previous_drift = drift_ema
            else:
                previous_drift = state.drift_ema
                drift_ratio = raw_delta / (previous_drift + cfg.eps)
                drift_ema = (
                    (1.0 - cfg.drift_ema_beta) * previous_drift
                    + cfg.drift_ema_beta * raw_delta
                ).clamp_min(cfg.delta_floor)
                previous_drift = state.drift_ema

        # gbw____
        # magnitude coherence 只比较相邻两帧 raw drift 的幅度连续性。
        # 第一帧没有历史时返回 0，令 Q 回到 sigmoid 的中性输入；后续
        # 幅度相近时接近 1，幅度突变时接近 0。它不比较 hidden channel
        # 的方向，因此比 2048 维 cosine 更适合视觉特征的旋转变化。
        if state.drift_ema is None:
            magnitude_coherence = torch.zeros_like(raw_delta)
        else:
            previous_drift_for_token = previous_drift
            if previous_drift_for_token.ndim < raw_delta.ndim:
                previous_drift_for_token = previous_drift_for_token[
                    ..., None, None, None
                ]
            magnitude_coherence = (
                1.0
                - (raw_delta - previous_drift_for_token).abs()
                / (
                    raw_delta
                    + previous_drift_for_token
                    + cfg.eps
                )
            ).clamp(min=0.0, max=1.0)
        # ____

        # gbw____
        # APNE 的 magnitude coherence 使用真正的 t-1 raw displacement，
        # 而不是 global scope 下的上一帧全局 EMA。这样每个 spatial token
        # 比较的是自己的 ||Z_t-Z_{t-1}|| 与 ||Z_{t-1}-Z_{t-2}||：连续
        # 运动接近 1，单帧幅度 spike 接近 0。它只读取已有的
        # previous_delta，不新增非因果信息，也不改变历史 magnitude
        # coherence 字段，便于和旧实验做严格对照。
        # ____
        if state.previous_delta is None:
            raw_magnitude_coherence = torch.zeros_like(raw_delta)
        else:
            previous_raw_delta = (
                torch.linalg.vector_norm(state.previous_delta, dim=-1)
                / feature_scale
            )
            raw_magnitude_coherence = (
                1.0
                - (raw_delta - previous_raw_delta).abs()
                / (raw_delta + previous_raw_delta + cfg.eps)
            ).clamp(min=0.0, max=1.0)
        # ____

        # gbw____
        # 双时间尺度 drift：fast EMA 响应当前变化，slow EMA 描述该 token
        # 的长期运动基线。两者的比值只在 token_ema_double 中作为 Q 证据，
        # 其余旧 scope 不增加额外状态。
        # gbw____
        # gain-space A2 同样需要 fast/slow drift reference；旧
        # drift_adaptive_sigmoid 的计算路径不变，只扩展到新的 Q 模式。
        # ____
        if (
            cfg.q_mode in {"drift_adaptive_sigmoid", "gain_space_adaptive"}
            and cfg.drift_scope == "token_ema_double"
        ):
            if state.drift_reference_ema is None:
                drift_reference_ema = raw_delta.clamp_min(cfg.delta_floor)
            else:
                drift_reference_ema = (
                    (1.0 - cfg.drift_reference_beta)
                    * state.drift_reference_ema
                    + cfg.drift_reference_beta * raw_delta
                ).clamp_min(cfg.delta_floor)
        elif (
            cfg.q_mode in {"drift_adaptive_sigmoid", "gain_space_adaptive"}
            and cfg.drift_scope == "global_token_mean_persistent"
        ):
            # gbw____
            # global mean 的 fast EMA 与 slow reference 都只使用当前帧和
            # t-1 状态；slow reference 抑制单帧 spike 对 Q 的放大。
            if state.drift_reference_ema is None:
                drift_reference_ema = drift_stat.clamp_min(cfg.delta_floor)
            else:
                drift_reference_ema = (
                    (1.0 - cfg.drift_reference_beta)
                    * state.drift_reference_ema
                    + cfg.drift_reference_beta * drift_stat
                ).clamp_min(cfg.delta_floor)
            # ____
        else:
            drift_reference_ema = None
        # ____

        # gbw____
        # Q-only persistence gate：同向 raw drift 的 cosine 接近 1，
        # 反向或孤立变化接近 0。该量只读取 t-1 历史，不修改 R、reset
        # 或模型参数，因此仍然是因果且 training-free 的过程噪声证据。
        if state.previous_delta is None:
            drift_coherence = torch.ones_like(raw_delta)
            # gbw____
            # 没有有效的上一帧方向时，不把首个真实运动误判为 jitter。
            # 这只控制 coherence penalty 的适用范围，不改变 q_drift_ratio
            # 的原始因果定义。
            coherence_history_valid = torch.zeros_like(raw_delta)
            # ____
        else:
            current_norm = torch.linalg.vector_norm(delta_vector, dim=-1)
            previous_norm = torch.linalg.vector_norm(
                state.previous_delta, dim=-1
            )
            cosine = torch.sum(
                delta_vector * state.previous_delta, dim=-1
            ) / (current_norm * previous_norm + cfg.eps)
            drift_coherence = cosine.clamp(min=0.0, max=1.0)
            # gbw____
            # previous_delta≈0 出现在 episode 开始或静止→运动的首个变化，
            # 此时方向余弦没有统计意义，不应施加“非持续 drift”惩罚。
            coherence_history_valid = (
                previous_norm > cfg.delta_floor
            ).to(raw_delta.dtype)
            # ____
        # ____
        if state.previous_delta is None or state.drift_ema is None:
            jitter_score = torch.zeros_like(raw_delta)
        else:
            acceleration_vector = delta_vector - state.previous_delta
            acceleration = (
                torch.linalg.vector_norm(acceleration_vector, dim=-1)
                / feature_scale
            )
            if (
                cfg.q_mode == "drift_adaptive_sigmoid"
                and cfg.drift_scope in {
                    "global_token_median",
                    "global_token_mean",
                    "global_token_mean_persistent",
                }
            ):
                jitter_denominator = previous_drift[..., None, None, None]
            else:
                jitter_denominator = previous_drift
        # ____
            jitter_score = acceleration / (jitter_denominator + cfg.eps)
        # gbw____
        # A3：先由当前帧与 t-1/t-2 raw token 的二阶变化得到 jitter，再
        # 生成 full-channel measurement covariance。jitter 只改变 R_t；
        # 是否让 Q 同步使用 R_t 由 q_use_adaptive_r 单独控制。
        # ____
        if cfg.r_mode == "jitter_aware":
            jitter_for_r = (jitter_score - cfg.jitter_r_tau).clamp_min(0.0)
            # gbw____
            # A3 的 coherence 也允许使用相邻 raw displacement 幅度；
            # direction 模式保留旧实验路径，magnitude 模式只改变 R_t
            # 的异常抑制证据，不改变 Q 的尺度或 reset。
            r_coherence = (
                raw_magnitude_coherence
                if cfg.jitter_r_coherence_mode == "magnitude"
                else drift_coherence
            )
            coherence_factor = 1.0 - (
                cfg.jitter_r_coherence_weight * r_coherence
            )
            jitter_for_r = jitter_for_r * coherence_factor.clamp(
                min=0.0, max=1.0
            )
            jitter_for_r = jitter_for_r.clamp(
                min=0.0, max=cfg.jitter_r_max
            )
            r_multiplier = 1.0 + cfg.jitter_r_lambda * jitter_for_r
            # gbw____
            # token 模式保持历史路径；channel 模式把 standardized
            # innovation^2 相对 token 内 median 做稳健归一化，令异常
            # channel 的观测协方差增大而不牵连同一 token 的正常 channel。
            # token 内 median 保证中性 channel 的 R 标尺仍接近原倍率。
            # ____
            if cfg.jitter_r_channel_mode == "channel":
                standardized_sq_for_r = innovation_vector.square() / (
                    state.covariance + r_t + cfg.eps
                )
                channel_weight = standardized_sq_for_r / (
                    torch.median(
                        standardized_sq_for_r, dim=-1, keepdim=True
                    ).values
                    + cfg.eps
                )
                channel_weight = channel_weight.clamp(min=0.0, max=4.0)
                r_multiplier = 1.0 + cfg.jitter_r_lambda * (
                    jitter_for_r.unsqueeze(-1) * channel_weight
                )
                r_t = torch.full_like(candidate, cfg.r) * r_multiplier
            else:
                r_t = torch.full_like(candidate, cfg.r) * r_multiplier.unsqueeze(
                    -1
                )
        # gbw____
        # normalized innovation 要与实际 measurement covariance 同尺度；
        # 对 fixed R，这与原有计算完全一致。
        normalized_innovation = torch.sqrt(
            torch.mean(
                innovation_vector.square()
                / (state.covariance + r_t + cfg.eps),
                dim=-1,
            ).clamp_min(cfg.eps)
        )
        # gbw____
        # channel-wise MAD 版本的无量纲 innovation。对标准正态残差而言，
        # median(|z|)=0.67449，因此除以该常数后，robust energy 在中性
        # 情况下仍约为 1。它只给 robust APNE 使用，不改变普通 normalized
        # innovation、reset 或已有 A1/A2 候选。
        # ____
        robust_normalized_innovation = None
        if cfg.apne_robust:
            # gbw____
            # MAD 保留历史路径；winsorized_mean 在 channel 维截断 z^2
            # 的极端尾部后求均值，使少量但一致的有效 channel 仍能提高
            # APNE evidence。两种统计都只用当前 innovation 与历史 P/R。
            # ____
            standardized_sq = innovation_vector.square() / (
                state.covariance + r_t + cfg.eps
            )
            if cfg.apne_robust_stat == "winsorized_mean":
                robust_normalized_innovation = torch.sqrt(
                    standardized_sq.clamp(max=4.0).mean(dim=-1).clamp_min(
                        cfg.eps
                    )
                ).clamp_min(cfg.delta_floor)
            else:
                standardized_abs = torch.sqrt(standardized_sq)
                robust_sigma = torch.median(
                    standardized_abs, dim=-1
                ).values / 0.67448975
                robust_normalized_innovation = robust_sigma.clamp_min(
                    cfg.delta_floor
                )
        # gbw____
        # APNE-Q：对每个 spatial token，用 innovation 的 channel-wise
        # 二阶矩估计过程噪声。随机游走模型下
        # E[nu^2]≈P_{t-1}+Q_t+R_t，因此先扣除历史 P 与当前固定 R，
        # 再除以 R 得到无量纲 Q/R 证据。证据先做因果 EMA，避免单帧
        # innovation spike 直接把 gain 推到上界。
        process_noise_ratio_ema = None
        if (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope in {
                "token_ema_apne",
                "token_ema_difference_apne",
            }
        ):
            innovation_energy = innovation_vector.square().mean(dim=-1)
            previous_covariance = state.covariance.mean(dim=-1)
            measurement_energy = r_t.mean(dim=-1)
            if cfg.drift_scope == "token_ema_difference_apne":
                # gbw____
                # z_t-z_{t-1}=w_t+v_t-v_{t-1}。在独立 measurement noise
                # 假设下，差分方差约为 Q+2R；用 raw delta 估计过程噪声，
                # 不把 state.filtered 的历史滞后重复计入 Q。
                raw_difference_energy = delta_vector.square().mean(dim=-1)
                process_noise = (
                    raw_difference_energy - 2.0 * measurement_energy
                )
                # ____
            else:
                # gbw____
                # 旧 token_ema_apne 保持 innovation-P-R 估计，便于与既有
                # APNE 结果直接比较；新差分分支不复用该量。
                process_noise = (
                    innovation_energy
                    - previous_covariance
                    - measurement_energy
                )
                # ____
            process_noise_ratio = process_noise.clamp_min(
                cfg.delta_floor
            ) / (measurement_energy + cfg.eps)
            if state.process_noise_ratio_ema is None:
                process_noise_ratio_ema = process_noise_ratio
            else:
                process_noise_ratio_ema = (
                    (1.0 - cfg.drift_ema_beta)
                    * state.process_noise_ratio_ema
                    + cfg.drift_ema_beta * process_noise_ratio
                ).clamp_min(cfg.delta_floor)
        # ____
        # ____
        # gbw____
        # 空间一致性只作为 Q 的证据：在每个 view 内对当前 raw drift 做
        # 3x3 无参数局部均值，孤立 spike 与邻域差异越大，spatial
        # coherence 越低。该量只读取当前帧，不改变 token 或 R。
        spatial_coherence = None
        if (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_spatial_coherent"
        ):
            height, width = raw_delta.shape[-2:]
            local_mean = F.avg_pool2d(
                raw_delta.reshape(-1, 1, height, width),
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ).reshape_as(raw_delta)
            local_deviation = (raw_delta - local_mean).abs()
            local_scale = F.avg_pool2d(
                local_deviation.reshape(-1, 1, height, width),
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ).reshape_as(raw_delta)
            spatial_coherence = torch.exp(
                -local_deviation / (local_scale + cfg.delta_floor)
            ).clamp(0.0, 1.0)
        # ____
        # gbw____
        # token_ema_relative 只让长期、空间上相对突出的 drift 提高 Q：
        # 当前 raw_delta 先进入逐 token EMA，再以本帧 EMA 的 median 做
        # robust 标定，避免单帧局部 outlier 直接把 Q 调度推向上界。
        if (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_relative"
        ):
            q_drift_baseline = torch.median(
                drift_ema.flatten(1), dim=-1
            ).values.clamp_min(cfg.delta_floor)
            q_drift_ratio = drift_ema / (
                q_drift_baseline[..., None, None, None] + cfg.eps
            )
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_persistent"
        ):
            # gbw____
            # 先以逐 token drift EMA 相对本帧 spatial median 表示长期
            # 局部运动，再用方向一致性收缩 g-1：
            # g_persistent = 1 + c_t * (g_relative - 1)。孤立 spike
            # 不会把 Q 推向上界，持续运动才会提高 Q。
            q_drift_baseline = torch.median(
                drift_ema.flatten(1), dim=-1
            ).values.clamp_min(cfg.delta_floor)
            relative_drift = drift_ema / (
                q_drift_baseline[..., None, None, None] + cfg.eps
            )
            q_drift_ratio = 1.0 + drift_coherence * (relative_drift - 1.0)
            # gbw____
            # coherence=1 表示连续帧方向一致，不惩罚有效运动；coherence
            # 越低，越倾向把变化视为瞬时扰动并降低 Q。该项仍然只使用
            # 当前 raw drift 与 t-1 raw drift，且最终继续经过 bounded sigmoid。
            q_drift_ratio = q_drift_ratio + cfg.q_coherence_weight * (
                drift_coherence - 1.0
            )
            # ____
            # ____
        elif (
            cfg.q_mode in {"drift_adaptive_sigmoid", "gain_space_adaptive"}
            and cfg.drift_scope == "token_ema_double"
        ):
            # gbw____
            # 只有 fast EMA 相对 slow reference 上升时，bounded sigmoid
            # 才会提高 Q；screen 阶段再由 q_floor_nominal 禁止降到 R0 以下。
            q_drift_ratio = drift_ema / (
                drift_reference_ema + cfg.eps
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "global_token_mean_persistent"
        ):
            # gbw____
            # 全局 fast/slow ratio 在 token grid 上广播；只有整帧 drift
            # 相对长期基线持续增大时，bounded sigmoid 才提高 Q。
            persistent_ratio = drift_ema / (
                drift_reference_ema + cfg.eps
            )
            q_drift_ratio = persistent_ratio[..., None, None, None].expand_as(
                raw_delta
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_coherent"
        ):
            # gbw____
            # 与 token_ema 的区别：raw_delta/previous_drift 只表示当前
            # 变化相对自身历史基线是变大还是变小；方向一致性再决定这
            # 个变化是否可信。单帧方向反转时 q_drift_ratio 回到 1，
            # 因而不会把 outlier 立即放大为高 Q；连续同向变化则保留
            # 正向 drift 的高 Q 证据。整个量仍只依赖 t-1 状态和当前帧。
            # ____
            q_drift_ratio = 1.0 + drift_coherence * (
                drift_ratio - 1.0
            )
            # gbw____
            # 对“高 drift 但方向不连续”的帧施加有限的 Q penalty。只有
            # drift_ratio>1 的异常增量会被惩罚；低 drift 不会因为 coherence
            # 偏低而被重复压低。这样同向持续运动仍提高 Q，孤立高 drift
            # 则降低 gain、更多保留历史状态，但不使用 hard reset。
            incoherent_excess = (drift_ratio - 1.0).clamp_min(0.0)
            q_drift_ratio = q_drift_ratio - (
                cfg.q_coherence_weight
                * (1.0 - drift_coherence)
                * coherence_history_valid
                * incoherent_excess
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_magnitude_coherent"
        ):
            # gbw____
            # 以 drift magnitude 的时间连续性代替 channel cosine：真实
            # 运动的幅度通常连续，而单帧 spike 会降低 coherence，使 Q
            # 下降并更多融合历史；持续变化仍保留提高 Q 的能力。
            # ____
            q_drift_ratio = 1.0 + magnitude_coherence * (
                drift_ratio - 1.0
            )
            incoherent_excess = (drift_ratio - 1.0).clamp_min(0.0)
            q_drift_ratio = q_drift_ratio - (
                cfg.q_coherence_weight
                * (1.0 - magnitude_coherence)
                * coherence_history_valid
                * incoherent_excess
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_spatial_coherent"
        ):
            # gbw____
            # 同时要求时间方向和空间邻域一致。对于局部孤立的高 drift，
            # coherence 会把 Q 从 nominal 向下压；对于连续、邻域一致的
            # 运动，q_drift_ratio 仍随 drift_ratio 增大，使滤波跟随 raw。
            coherence = drift_coherence * spatial_coherence
            q_drift_ratio = 1.0 + coherence * (drift_ratio - 1.0)
            incoherent_excess = (drift_ratio - 1.0).clamp_min(0.0)
            q_drift_ratio = q_drift_ratio - (
                cfg.q_coherence_weight
                * (1.0 - coherence)
                * coherence_history_valid
                * incoherent_excess
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope == "token_ema_innovation_aligned"
        ):
            # gbw____
            # 先以方向一致性过滤 raw drift，再用 innovation 相对其慢速
            # baseline 的比值调节“偏离 nominal 的幅度”：高 innovation
            # 允许持续 drift 更快跟随 raw，低 innovation 则收缩正向
            # drift；对负向偏离使用倒数，避免高 innovation 把历史保持
            # 误压得过低。最终仍由 bounded sigmoid 产生有界 Q/R。
            coherent_drift = 1.0 + drift_coherence * (
                drift_ratio - 1.0
            )
            innovation_factor = innovation_ratio.clamp(
                min=0.5, max=2.0
            ).pow(cfg.q_innovation_power)
            coherent_excess = coherent_drift - 1.0
            q_drift_ratio = 1.0 + torch.where(
                coherent_excess >= 0.0,
                coherent_excess * innovation_factor,
                coherent_excess / (innovation_factor + cfg.eps),
            )
            # ____
        elif (
            cfg.q_mode == "drift_adaptive_sigmoid"
            and cfg.drift_scope in {
                "token_ema_apne",
                "token_ema_difference_apne",
            }
        ):
            # gbw____
            # APNE-Q 已将因果的过程噪声证据转换为 Q/R；这里仅把 EMA
            # 送入统一 bounded sigmoid，仍不直接替换 token。
            q_drift_ratio = process_noise_ratio_ema
            # ____
        else:
            q_drift_ratio = drift_ratio
        # ____
        # gbw____
        # APNE 的观测是当前 innovation 的逐 channel 平方均值。先以预测
        # innovation 方差作为上限做稳健裁剪，再做因果 EMA；这样单个视觉
        # outlier 不会把后续多帧的 Q 永久推到上界。
        if cfg.q_mode in {
            "apne_bounded_sigmoid",
            "apne_normalized_sigmoid",
        }:
            if cfg.q_mode == "apne_normalized_sigmoid":
                # gbw____
                # 用无量纲 normalized innovation 作为 APNE evidence；e≈1
                # 表示观测与当前预测协方差同量级，不再减去绝对 R。
                # ____
                # gbw____
                # robust APNE 用 MAD 标准化后的 channel-wise 尺度，避免
                # 少量异常 channel 通过 mean(square) 把整 个 token 的 Q
                # 推向 raw；普通 S4 路径仍使用原始均方证据。
                # ____
                innovation_energy = (
                    robust_normalized_innovation.square()
                    if cfg.apne_robust
                    else normalized_innovation.square()
                )
                clipped_innovation_energy = innovation_energy.clamp(
                    min=0.0, max=cfg.apne_innovation_clip
                )
            else:
                innovation_energy = innovation_vector.square().mean(dim=-1)
                previous_covariance = state.covariance.mean(dim=-1)
                predicted_innovation_energy = (
                    cfg.gamma_p * previous_covariance + cfg.r
                )
                clipped_innovation_energy = torch.minimum(
                    innovation_energy,
                    predicted_innovation_energy * cfg.apne_innovation_clip,
                )
            # gbw____
            # 对 normalized evidence 做物理 Q/R 还原。若
            # e≈(P+Q+R)/(P+R)，则 Q/R≈(e-1)(P+R)/R；负估计只说明
            # 当前帧没有足够的过程噪声证据，裁到最小正值而不产生负 Q。
            # ____
            if cfg.apne_physical and cfg.q_mode == "apne_normalized_sigmoid":
                prior_variance = (
                    state.covariance + r_t
                ).mean(dim=-1)
                measurement_variance = r_t.mean(dim=-1)
                process_noise_evidence = (
                    (clipped_innovation_energy - 1.0)
                    * prior_variance
                    / (measurement_variance + cfg.eps)
                ).clamp_min(cfg.delta_floor)
            else:
                process_noise_evidence = clipped_innovation_energy
            if state.innovation_energy_ema is None:
                innovation_energy_ema = process_noise_evidence
            else:
                innovation_energy_ema = (
                    (1.0 - cfg.apne_ema_beta)
                    * state.innovation_energy_ema
                    + cfg.apne_ema_beta * process_noise_evidence
                )
            if cfg.q_mode == "apne_normalized_sigmoid":
                apne_q_hat = innovation_energy_ema.clamp_min(cfg.eps)
            else:
                apne_q_hat = (
                    innovation_energy_ema
                    - cfg.gamma_p * previous_covariance
                    - cfg.r
                ).clamp_min(cfg.eps)
            # gbw____
            # APNE-jitter 只改变 Q 的证据，不改变观测协方差 R：高于
            # tau 的瞬时加速度越大，当前 Q_hat 越保守；持续 drift 的
            # 低加速度帧不受影响。由于 jitter_score 只读取 t-1 状态
            # 和当前观测，该项仍然是因果、training-free 的 Q 更新。
            if cfg.apne_jitter_penalty > 0.0:
                jitter_excess = (
                    jitter_score - cfg.apne_jitter_tau
                ).clamp_min(0.0)
                apne_q_hat = apne_q_hat / (
                    1.0 + cfg.apne_jitter_penalty * jitter_excess
                )
            # ____
            # gbw____
            # 物理 APNE 的 q_hat 已经是无量纲 Q/R，不再除以 normalized
            # energy reference；普通 normalized APNE 保持原有 evidence=1
            # 中性点，绝对 APNE 保持以 R 为尺度。
            # ____
            if cfg.q_mode == "apne_normalized_sigmoid" and cfg.apne_physical:
                apne_q_evidence = apne_q_hat
            else:
                apne_q_evidence = apne_q_hat / (
                    cfg.apne_innovation_reference + cfg.eps
                    if cfg.q_mode == "apne_normalized_sigmoid"
                    else cfg.r + cfg.eps
                )
        else:
            innovation_energy_ema = None
            apne_q_hat = None
            apne_q_evidence = None
        # ____

        return {
            "r_t": r_t,
            # gbw____
            "prediction": prediction,
            # ____
            "innovation_vector": innovation_vector,
            "delta_vector": delta_vector,
            "raw_delta": raw_delta,
            "innovation": innovation,
            "normalized_innovation": normalized_innovation,
            "camera_delta_ema": camera_delta_ema,
            "camera_score": camera_score,
            "reset_metric": reset_metric,
            "local_center": local_center,
            "local_scale": local_scale,
            "local_score": local_score,
            "local_reset": local_reset,
            "camera_reset_map": camera_reset_map,
            "drift_ema": drift_ema,
            "drift_ratio": drift_ratio,
            # gbw____
            "drift_reference_ema": drift_reference_ema,
            # ____
            # gbw____
            "q_drift_ratio": q_drift_ratio,
            # gbw____
            "process_noise_ratio_ema": process_noise_ratio_ema,
            # ____
            "drift_coherence": drift_coherence,
            # gbw____
            "coherence_history_valid": coherence_history_valid,
            # ____
            # gbw____
            "magnitude_coherence": magnitude_coherence,
            # gbw____
            "raw_magnitude_coherence": raw_magnitude_coherence,
            # ____
            # ____
            # gbw____
            "spatial_coherence": spatial_coherence,
            # ____
            # gbw____
            "innovation_ratio": innovation_ratio,
            "innovation_reference_ema": innovation_reference_ema,
            # gbw____
            "robust_normalized_innovation": robust_normalized_innovation,
            # ____
            # ____
            "jitter_score": jitter_score,
            # gbw____
            "innovation_energy_ema": innovation_energy_ema,
            "apne_q_hat": apne_q_hat,
            "apne_q_evidence": apne_q_evidence,
            # ____
        }

    @staticmethod
    def _kalman_update(
        metrics: dict,
        state: _TokenState,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Compute Q, prediction covariance and gain for A1 or A2."""

        # gbw____
        # A3 默认固定 Q 的 R0 标尺，避免 jitter-aware R 同时把 Q 放大；
        # 旧 A1/A2 的 fixed-R 路径两种选择数值相同。
        q_reference_r = (
            metrics["r_t"]
            if cfg.q_use_adaptive_r
            else torch.full_like(metrics["r_t"], cfg.r)
        )
        # ____
        # gbw____
        # 非 drift-adaptive 分支没有 innovation guard；先给出与 token
        # spatial grid 同形状的全 False mask，保证 A1/B1/APNE 兼容。
        # ____
        q_low_guard = torch.zeros_like(metrics["raw_delta"], dtype=torch.bool)
        # gbw____
        # selective gate 也保持 token-level 诊断形状；默认全 False，确保
        # 所有旧候选的数值路径完全不变。
        q_selective_gate = torch.zeros_like(
            metrics["raw_delta"], dtype=torch.bool
        )
        # gbw____
        # soft gate 的连续权重只用于当前 Q 更新；默认路径保持全零，便于
        # diagnostics 明确区分硬 gate 与 soft gate。
        # ____
        q_selective_weight = torch.zeros_like(metrics["raw_delta"])
        # gbw____
        # floor-break 只在显式 APNE 候选中打开；默认全 False，保证旧
        # A1/A2/A3/A4 diagnostics 的数值路径不变。
        # ____
        apne_floor_break = torch.zeros_like(
            metrics["raw_delta"], dtype=torch.bool
        )
        # gbw____
        # 仅 gain-space 分支提供逐 token 的目标 gain；其他历史模式保持
        # None，避免改变既有 diagnostics 的含义。
        # ____
        gain_target = None
        # ____
        if cfg.q_mode == "constant":
            q_scale = torch.ones_like(metrics["raw_delta"])
            q_t = q_reference_r * (
                cfg.target_gain
                * (1.0 / (1.0 - cfg.target_gain) - cfg.gamma_p)
            )
        elif cfg.q_mode == "motion_jitter_adaptive":
            motion_scale = metrics["drift_ratio"] / (
                1.0 + cfg.jitter_weight * metrics["jitter_score"]
            )
            q_scale = torch.clamp(
                motion_scale.clamp_min(cfg.eps).pow(cfg.q_power),
                min=cfg.q_scale_min,
                max=cfg.q_scale_max,
            )
            q_ratio = cfg.target_gain * (
                1.0 / (1.0 - cfg.target_gain) - cfg.gamma_p
            )
            q_t = q_reference_r * q_ratio * q_scale.unsqueeze(-1)
        else:
            # gbw____
            # Gain-space adaptive Q：先由 causal drift 产生目标 gain，再用
            # 当前 posterior covariance 反解 Q/R。设 p̄_{t-1}=P_{t-1}/R，
            # 则 q̄_t=K*/(1-K*)-gamma_p*p̄_{t-1}，这样中性 drift g=1
            # 且 K*=0.95 时严格回到 Q/R=18.05，而不是依赖 sigmoid
            # 在 Q 坐标中的偶然标定。q̄ 只做安全裁剪，避免数值不稳定。
            # ____
            if cfg.q_mode == "gain_space_adaptive":
                q_input = metrics.get(
                    "q_drift_ratio", metrics["drift_ratio"]
                )
                gain_target = torch.clamp(
                    cfg.target_gain
                    + cfg.gain_k_delta
                    * torch.tanh(
                        cfg.gain_sigmoid_alpha
                        * (q_input - cfg.gain_sigmoid_tau)
                    ),
                    min=cfg.gain_k_min,
                    max=cfg.gain_k_max,
                )
                p_previous_ratio = state.covariance.mean(dim=-1) / (
                    q_reference_r.mean(dim=-1) + cfg.eps
                )
                q_scale = (
                    gain_target / (1.0 - gain_target + cfg.eps)
                    - cfg.gamma_p * p_previous_ratio
                ).clamp(min=cfg.gain_q_min, max=cfg.gain_q_max)
            elif cfg.q_mode == "drift_adaptive_sigmoid":
                # gbw____
                # A2：将归一化 drift g_t 通过有界 sigmoid 映射到 Q/R；
                # jitter 只记录，不参与 Q，也不触发 reset。
                q_input = metrics.get(
                    "q_drift_ratio", metrics["drift_ratio"]
                )
                # gbw____
                # 对潜在低 Q 的动态帧做保护：只有低 innovation 帧才允许
                # q_input<1；q_input>1 的正向 drift 不被 guard 截断。
                q_low_guard = torch.zeros_like(q_input, dtype=torch.bool)
                if cfg.q_innovation_guard_tau > 0.0:
                    q_low_guard = (q_input < 1.0) & (
                        metrics["normalized_innovation"]
                        > cfg.q_innovation_guard_tau
                    )
                    q_input = torch.where(
                        q_low_guard, torch.ones_like(q_input), q_input
                    )
                # ____
                # gbw____
                # 新的 sparse A2 gate：正常 token 不经过连续 penalty，而是
                # 直接回到 q_input=1 的 nominal gain；只有四个因果证据同时
                # 通过时才使用低 Q/R。这样不会把大量正常运动误判为 jitter。
                if cfg.q_selective_gate:
                    # gbw____
                    # magnitude-coherent 候选使用幅度连续性作为同一个
                    # selective coherence 证据；其余 scope 使用方向
                    # coherence。这样“同方向但突然放大”的 spike 也能
                    # 被识别为异常，而不是误认为持续有效运动。
                    selective_coherence = (
                        metrics["magnitude_coherence"]
                        if cfg.drift_scope
                        == "token_ema_magnitude_coherent"
                        else metrics["spatial_coherence"]
                        if cfg.drift_scope == "token_ema_spatial_coherent"
                        else metrics["drift_coherence"]
                    )
                    # gbw____
                    # 空间一致性只用于该 scope 的 gate 证据；其他 scope
                    # 的值始终为时间一致性，避免把 None 广播进布尔表达式。
                    # ____
                    if selective_coherence is None:
                        selective_coherence = metrics["drift_coherence"]
                    # ____
                    q_selective_gate = (
                        (metrics["jitter_score"]
                         >= cfg.q_selective_jitter_tau)
                        & (
                            metrics["normalized_innovation"]
                            >= cfg.q_selective_innovation_tau
                        )
                        & (
                            selective_coherence
                            <= cfg.q_selective_coherence_tau
                        )
                        & (
                            metrics["local_score"]
                            >= cfg.q_selective_local_score_tau
                        )
                    )
                    # gbw____
                    # 硬 gate 保留旧候选的完全离散行为；soft gate 则把四个
                    # 证据的 sigmoid 权重相乘，得到连续的 abnormality。这样
                    # token 不会因为刚好跨过一个阈值而突然改变 Q。
                    # ____
                    low_q_input = torch.full_like(
                        q_input, cfg.q_selective_q_ratio
                    )
                    if cfg.q_selective_soft_gate:
                        soft_width = cfg.q_selective_softness
                        jitter_weight = torch.sigmoid(
                            (
                                metrics["jitter_score"]
                                - cfg.q_selective_jitter_tau
                            )
                            / soft_width
                        )
                        innovation_weight = torch.sigmoid(
                            (
                                metrics["normalized_innovation"]
                                - cfg.q_selective_innovation_tau
                            )
                            / soft_width
                        )
                        incoherence_weight = torch.sigmoid(
                            (
                                cfg.q_selective_coherence_tau
                                - selective_coherence
                            )
                            / soft_width
                        )
                        local_weight = torch.sigmoid(
                            (
                                metrics["local_score"]
                                - cfg.q_selective_local_score_tau
                            )
                            / soft_width
                        )
                        q_selective_weight = (
                            jitter_weight
                            * innovation_weight
                            * incoherence_weight
                            * local_weight
                        ).clamp(min=0.0, max=1.0)
                        q_input = torch.ones_like(q_input)
                    else:
                        q_selective_weight = q_selective_gate.to(
                            q_input.dtype
                        )
                        q_input = torch.where(
                            q_selective_gate,
                            low_q_input,
                            torch.ones_like(q_input),
                        )
                    # gbw____
                    # 事件触发式的“双向”Q：异常 gate 只降低 Q；非异常
                    # token 对正向、持续的 q_drift_ratio 保留提高 Q 的能力，
                    # 对低 drift 仍固定为 nominal。这样不会因稳定帧的
                    # q_input<1 而额外滞后，也不会把真实持续运动强行压平。
                    # ____
                    if cfg.q_selective_positive_drift:
                        positive_input = metrics.get(
                            "q_drift_ratio", torch.ones_like(q_input)
                        ).clamp_min(1.0)
                        if cfg.q_positive_deadband > 0.0:
                            positive_excess = (
                                positive_input - 1.0 - cfg.q_positive_deadband
                            ).clamp_min(0.0)
                            positive_input = 1.0 + positive_excess
                        positive_input = 1.0 + cfg.q_positive_boost * (
                            positive_input - 1.0
                        )
                        if cfg.q_selective_soft_gate:
                            q_input = positive_input + q_selective_weight * (
                                low_q_input - positive_input
                            )
                        else:
                            q_input = torch.where(
                                q_selective_gate, q_input, positive_input
                            )
                # ____
                # gbw____
                # 仅允许正向 drift 提高 Q；负向 drift 不再制造比 R0 更
                # 强的滞后。这是 training-free 的保守边界，不引入新状态。
                if cfg.q_floor_nominal and not cfg.q_selective_gate:
                    q_input = q_input.clamp_min(1.0)
                    # gbw____
                    if cfg.q_positive_deadband > 0.0:
                        positive_excess = (
                            q_input - 1.0 - cfg.q_positive_deadband
                        ).clamp_min(0.0)
                        q_input = 1.0 + positive_excess
                    # ____
                    q_input = 1.0 + cfg.q_positive_boost * (q_input - 1.0)
                    # ____
                    # gbw____
                    if cfg.q_innovation_weight > 0.0:
                        innovation_excess = (
                            metrics["normalized_innovation"]
                            - cfg.q_innovation_tau
                        ).clamp_min(0.0)
                        q_input = q_input + (
                            cfg.q_innovation_weight
                            * innovation_excess
                            * metrics.get(
                                "drift_coherence", torch.ones_like(q_input)
                            )
                        )
                    # ____
                # gbw____
                # A2 jitter-gated Q：对高于 q_jitter_tau 的瞬时加速度，按
                # 方向不一致程度降低 q_input。它只改变 Q/R 的 sigmoid 输入；
                # q_floor_nominal 候选保持 nominal 下界，因此本机制只在
                # 非 floor 候选中用于检验“异常帧应更多依赖历史”的假设。
                if (
                    cfg.q_jitter_penalty > 0.0
                    and not cfg.q_selective_gate
                    and (
                        not cfg.q_floor_nominal
                        or cfg.q_jitter_breaks_floor
                    )
                ):
                    jitter_excess = (
                        metrics["jitter_score"] - cfg.q_jitter_tau
                    ).clamp_min(0.0)
                    incoherence = (
                        1.0 - metrics.get(
                            "drift_coherence",
                            torch.ones_like(q_input),
                        )
                    ).clamp(min=0.0, max=1.0)
                    q_input = q_input - (
                        cfg.q_jitter_penalty
                        * jitter_excess
                        * incoherence
                    )
                # gbw____
                # 以 normalized innovation 作为第二个、独立的异常证据。
                # 只有“偏离历史状态很大”且“raw drift 方向不连续”时才
                # 降低 Q；持续同向运动的 coherence≈1，不会被误判为噪声。
                if (
                    cfg.q_innovation_penalty > 0.0
                    and not cfg.q_selective_gate
                ):
                    innovation_excess = (
                        metrics["normalized_innovation"]
                        - cfg.q_innovation_penalty_tau
                    ).clamp_min(0.0)
                    incoherence = (
                        1.0
                        - metrics.get(
                            "drift_coherence",
                            torch.ones_like(q_input),
                        )
                    ).clamp(min=0.0, max=1.0)
                    q_input = q_input - (
                        cfg.q_innovation_penalty
                        * innovation_excess
                        * incoherence
                    )
                # ____
                # ____
                q_scale = cfg.q_min + (cfg.q_max - cfg.q_min) * torch.sigmoid(
                    cfg.q_sigmoid_alpha * (q_input - cfg.q_sigmoid_tau)
                )
                # gbw____
                # 对新候选执行 nominal-centered calibration。先计算 raw
                # sigmoid 在 q_input=1 处的值，再用一个有界仿射映射把它
                # 对齐到 nominal Q/R；这样 q_floor + g=1 的 gain 不会因
                # q_min/q_max 选择而整体漂移，且高 drift 仍能向 q_max 过渡。
                if cfg.q_calibrate_nominal:
                    nominal_q_ratio = cfg.target_gain * (
                        1.0 / (1.0 - cfg.target_gain) - cfg.gamma_p
                    )
                    neutral_raw = cfg.q_min + (
                        cfg.q_max - cfg.q_min
                    ) * torch.sigmoid(
                        torch.as_tensor(
                            cfg.q_sigmoid_alpha * (1.0 - cfg.q_sigmoid_tau),
                            device=q_scale.device,
                            dtype=q_scale.dtype,
                        )
                    )
                    upper_span = max(cfg.q_max - nominal_q_ratio, cfg.eps)
                    raw_span = (cfg.q_max - neutral_raw).clamp_min(cfg.eps)
                    q_scale = nominal_q_ratio + (
                        q_scale - neutral_raw
                    ) * (upper_span / raw_span)
                    q_scale = q_scale.clamp(
                        min=cfg.q_min, max=cfg.q_max
                    )
                # ____
                # ____
            else:
                # gbw____
                # APNE：以 innovation 方差估计的 q_evidence 为输入。对
                # normalized APNE，reference=1 表示中性证据；对物理
                # APNE，evidence 本身就是 Q/R，仍以 reference=1 为中性点。
                # log-ratio 后映射到有界 [q_min,q_max]，全程只使用当前及
                # 历史观测。
                nominal_q_ratio = cfg.target_gain * (
                    1.0 / (1.0 - cfg.target_gain) - cfg.gamma_p
                )
                # gbw____
                # 物理 APNE 的 evidence 已经是无量纲 Q/R，故其 sigmoid
                # 中性点必须是配置的 evidence reference（默认 1.0）。
                # 不能再拿 nominal Q/R=18.05 作除数；否则正常 evidence
                # 会被压到 q_min，而加 floor 的候选又会退化成固定增益。
                evidence_reference = (
                    cfg.apne_innovation_reference
                    if cfg.q_mode == "apne_normalized_sigmoid"
                    else nominal_q_ratio
                )
                log_q_evidence = torch.log(
                    metrics["apne_q_evidence"].clamp_min(cfg.eps)
                    / (evidence_reference + cfg.eps)
                )
                # gbw____
                # APNE-drift 融合：把长期 drift ratio 作为 log-evidence
                # 的有界修正，而不是直接覆盖 APNE 的物理估计。方向一致
                # 的持续变化可以提高 Q；方向不连续的高 drift 只获得很小
                # 的修正，避免再次把 jitter 当作有效运动。blend=0 时
                # 该分支严格退化为纯 APNE。
                if cfg.apne_drift_blend > 0.0:
                    drift_ratio = metrics.get("drift_ratio")
                    if drift_ratio is not None:
                        drift_log = torch.log(
                            drift_ratio.clamp(
                                min=1.0 / cfg.apne_drift_clip,
                                max=cfg.apne_drift_clip,
                            )
                        )
                        # gbw____
                        # hidden feature 的方向在相邻观测间可能发生旋转；
                        # magnitude 模式改用真正的 raw displacement
                        # 幅度连续性，避免方向 cosine 偏低使 APNE-drift
                        # 修正失效。默认 direction 保持旧路径。
                        # ____
                        coherence_key = (
                            "raw_magnitude_coherence"
                            if cfg.apne_drift_coherence == "magnitude"
                            else "drift_coherence"
                        )
                        coherence = metrics.get(
                            coherence_key, torch.ones_like(drift_log)
                        )
                        history_valid = metrics.get(
                            "coherence_history_valid",
                            torch.ones_like(drift_log),
                        )
                        coherence = torch.where(
                            history_valid > 0.0,
                            coherence,
                            torch.ones_like(coherence),
                        )
                        log_q_evidence = log_q_evidence + (
                            cfg.apne_drift_blend * coherence * drift_log
                        )
                # ____
                # gbw____
                # APNE evidence coherence gate：异常帧可能同时产生较大
                # innovation，但如果相邻 raw displacement 的幅度不连续，
                # 不应把全部正向 evidence 都解释成有效过程噪声。这里只
                # 收缩 log evidence 的正半轴；负半轴交给 nominal floor，
                # 因而不会把稳定 token 进一步压成过低 gain。第一帧没有
                # 历史 coherence 时按 1 处理，保持因果初始化连续。
                # ____
                if cfg.apne_evidence_coherence_weight > 0.0:
                    coherence_key = (
                        "raw_magnitude_coherence"
                        if cfg.apne_drift_coherence == "magnitude"
                        else "drift_coherence"
                    )
                    evidence_coherence = metrics.get(
                        coherence_key, torch.ones_like(log_q_evidence)
                    )
                    history_valid = metrics.get(
                        "coherence_history_valid",
                        torch.ones_like(log_q_evidence),
                    )
                    evidence_coherence = torch.where(
                        history_valid > 0.0,
                        evidence_coherence,
                        torch.ones_like(evidence_coherence),
                    ).clamp(min=0.0, max=1.0)
                    positive_log_evidence = log_q_evidence.clamp_min(0.0)
                    negative_log_evidence = log_q_evidence.clamp_max(0.0)
                    positive_weight = (
                        1.0
                        - cfg.apne_evidence_coherence_weight
                        + cfg.apne_evidence_coherence_weight
                        * evidence_coherence
                    )
                    log_q_evidence = negative_log_evidence + (
                        positive_log_evidence * positive_weight
                    )
                # ____
                # ____
                q_scale = cfg.q_min + (cfg.q_max - cfg.q_min) * torch.sigmoid(
                    cfg.apne_log_alpha * log_q_evidence
                )
                # gbw____
                # APNE-clean 的保护：创新证据不足时保持 A1 的 nominal
                # Q/R=18.05，只允许证据充分的状态变化提高 Q。该 floor
                # 只作用于 Q，不修改 R、reset、token 或 decoder。
                if cfg.apne_q_floor_nominal:
                    nominal_q_ratio = cfg.target_gain * (
                        1.0 / (1.0 - cfg.target_gain) - cfg.gamma_p
                    )
                    nominal_q_tensor = torch.full_like(
                        q_scale, nominal_q_ratio
                    )
                    q_scale = torch.maximum(q_scale, nominal_q_tensor)
                    # gbw____
                    # APNE-clean 的细分门控：高 process evidence 或高
                    # raw-motion coherence 的 token 不允许突破 floor；
                    # 只有低 evidence 且幅度不连续的 token 才恢复 APNE
                    # 的低 Q。这是 token-level 的统一因果规则，不读取
                    # task label，也不改变 R/reset。
                    # ____
                    if cfg.apne_floor_break_low_evidence:
                        floor_coherence = (
                            metrics.get(
                                "raw_magnitude_coherence",
                                metrics["drift_coherence"],
                            )
                            if cfg.apne_drift_coherence == "magnitude"
                            else metrics["drift_coherence"]
                        )
                        floor_break = (
                            (metrics["apne_q_evidence"]
                             <= cfg.apne_floor_break_evidence_tau)
                            & (
                                floor_coherence
                                <= cfg.apne_floor_break_coherence_tau
                            )
                            & (metrics["coherence_history_valid"] > 0.0)
                        )
                        # gbw____
                        # jitter_tau>0 时增加保守的二阶变化上限；默认 0
                        # 关闭，保持已有 floor-break 候选路径不变。
                        # ____
                        if cfg.apne_floor_break_jitter_tau > 0.0:
                            floor_break = floor_break & (
                                metrics["jitter_score"]
                                <= cfg.apne_floor_break_jitter_tau
                            )
                        apne_floor_break = floor_break
                        q_scale = torch.where(
                            floor_break, q_scale, nominal_q_tensor
                        )
                # ____
                # ____
            q_t = q_reference_r * q_scale.unsqueeze(-1)
        p_prior = cfg.gamma_p * state.covariance + q_t
        gain = torch.clamp(
            p_prior / (p_prior + metrics["r_t"] + cfg.eps),
            min=cfg.k_min,
            max=cfg.k_max,
        )
        return {
            "q_scale": q_scale,
            "q_t": q_t,
            "p_prior": p_prior,
            "gain": gain,
            # gbw____
            "gain_target": gain_target,
            # ____
            # gbw____
            "q_low_guard": q_low_guard,
            # gbw____
            "q_selective_gate": q_selective_gate,
            # gbw____
            "q_selective_weight": q_selective_weight,
            # gbw____
            "apne_floor_break": apne_floor_break,
            # ____
            # ____
        }

    @staticmethod
    def _prepare_reset(
        candidate: torch.Tensor,
        state: _TokenState,
        metrics: dict,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Choose normal, bounded, or persistent-reset measurement handling."""

        local_reset = metrics["local_reset"]
        camera_reset_map = metrics["camera_reset_map"]
        # gbw____
        # A1-clean/A2 明确禁止 hard reset。状态仍然在 episode 边界由 reset()
        # 清除，但单帧 token 异常只能通过正常 Kalman gain 被吸收，不能 raw takeover。
        if cfg.reset_mode == "none":
            return {
                "candidate": candidate,
                "outlier_gate": torch.zeros_like(local_reset),
                "persistent_reset": torch.zeros_like(local_reset),
                "local_high_count": None,
                "camera_high_count": None,
            }
        # ____
        # gbw____
        # 新 A4 reset 的核心分工：只有 normalized innovation 异常且
        # 与上一帧 raw motion 方向一致的 token 才能进入 patience 计数。
        # 方向反复的瞬时 jitter 不会触发 raw takeover；camera/global
        # 异常仍由原有 camera gate 处理。
        if cfg.reset_mode == "innovation_coherent":
            local_reset = local_reset & (
                metrics["drift_coherence"] >= cfg.reset_coherence_tau
            ) & (metrics["coherence_history_valid"] > 0.0)
        # ____
        if cfg.reset_mode == "current":
            return {
                "candidate": candidate,
                "outlier_gate": torch.zeros_like(local_reset),
                "persistent_reset": local_reset | camera_reset_map,
                "local_high_count": None,
                "camera_high_count": None,
            }

        previous_local_count = (
            torch.zeros_like(local_reset, dtype=torch.int64)
            if state.local_high_count is None
            else state.local_high_count
        )
        previous_camera_count = (
            torch.zeros_like(metrics["camera_score"], dtype=torch.int64)
            if state.camera_high_count is None
            else state.camera_high_count
        )
        local_high_count = torch.where(
            local_reset,
            previous_local_count + 1,
            torch.zeros_like(previous_local_count),
        )
        camera_high_count = torch.where(
            metrics["camera_score"] > cfg.reset_camera_tau,
            previous_camera_count + 1,
            torch.zeros_like(previous_camera_count),
        )
        persistent_local = local_high_count >= cfg.reset_patience
        persistent_camera = camera_high_count >= cfg.reset_patience
        persistent_reset = persistent_local | persistent_camera[..., None, None]

        local_high_count = torch.where(
            persistent_reset,
            torch.zeros_like(local_high_count),
            local_high_count,
        )
        camera_high_count = torch.where(
            persistent_camera,
            torch.zeros_like(camera_high_count),
            camera_high_count,
        )

        isolated_local = local_reset & ~persistent_local
        innovation_cap = (
            metrics["local_center"] + cfg.innovation_gate_tau * metrics["local_scale"]
        ).clamp_min(cfg.delta_floor)
        bounded_scale = torch.minimum(
            torch.ones_like(metrics["reset_metric"]),
            innovation_cap / (metrics["reset_metric"] + cfg.eps),
        )
        bounded_candidate = state.filtered + metrics["innovation_vector"] * (
            bounded_scale.unsqueeze(-1)
        )
        candidate_for_update = torch.where(
            isolated_local.unsqueeze(-1), bounded_candidate, candidate
        )
        return {
            "candidate": candidate_for_update,
            "outlier_gate": isolated_local,
            "persistent_reset": persistent_reset,
            "local_high_count": local_high_count,
            "camera_high_count": camera_high_count,
        }

    @staticmethod
    def _apply_trust_region(
        filtered: torch.Tensor,
        candidate: torch.Tensor,
        covariance: torch.Tensor,
        gain: torch.Tensor,
        r_t: torch.Tensor,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Apply B3's optional pre-decoder token displacement bound."""

        token_shift_ratio = torch.linalg.vector_norm(
            filtered - candidate, dim=-1
        ) / (torch.linalg.vector_norm(candidate, dim=-1) + cfg.eps)
        clipped = torch.zeros_like(token_shift_ratio, dtype=torch.bool)
        if cfg.max_token_shift_ratio > 0.0:
            clipped = token_shift_ratio > cfg.max_token_shift_ratio
            trust_scale = torch.minimum(
                torch.ones_like(token_shift_ratio),
                cfg.max_token_shift_ratio / (token_shift_ratio + cfg.eps),
            )
            filtered = candidate + (filtered - candidate) * trust_scale.unsqueeze(-1)
            clipped_expand = clipped.unsqueeze(-1)
            covariance = torch.where(clipped_expand, r_t, covariance)
            gain = torch.where(clipped_expand, torch.ones_like(gain), gain)
            token_shift_ratio = torch.linalg.vector_norm(
                filtered - candidate, dim=-1
            ) / (torch.linalg.vector_norm(candidate, dim=-1) + cfg.eps)
        return {
            "filtered": filtered,
            "covariance": covariance,
            "gain": gain,
            "token_shift_ratio": token_shift_ratio,
            "trust_region_clipped": clipped,
        }
    # ____

    # gbw____
    def apply(self, tokens: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply the selected A1/B1/B2/B3 update before ConvexUpSample."""

        # A0 或未启用的 stage 是严格旁路，不创建 state，也不改变输入对象。
        if not self.enabled_for_stage(stage):
            return tokens
        self._validate(tokens)
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("U-IGAKF tokens must be finite")

        self._call_index += 1
        key = self._state_key(tokens, stage)
        candidate = tokens.detach().float()
        state = self._states.get(key)
        if state is None:
            # 第一帧直接初始化：S_0=Z_0，P_0=p_init。
            covariance = torch.full_like(candidate, self.config.p_init)
            self._states[key] = _TokenState(
                filtered=candidate.clone(),
                covariance=covariance,
                previous_candidate=candidate.clone(),
                camera_delta_ema=None,
                # gbw____
                velocity=(
                    torch.zeros_like(candidate)
                    if self.config.prediction_mode
                    in {"constant_velocity", "gated_velocity"}
                    else None
                ),
                # ____
                drift_ema=None,
                # gbw____
                drift_reference_ema=None,
                # ____
                # gbw____
                innovation_energy_ema=None,
                # ____
                # gbw____
                process_noise_ratio_ema=None,
                # ____
                # gbw____
                innovation_reference_ema=None,
                # ____
                previous_delta=None,
                local_high_count=None,
                camera_high_count=None,
            )
            self._set_update_diagnostics(
                stage=stage,
                initialized=True,
                tokens=tokens,
                covariance=covariance,
            )
            return tokens

        cfg = self.config
        metrics = self._temporal_metrics(candidate, state, cfg)
        kalman = self._kalman_update(metrics, state, cfg)
        reset = self._prepare_reset(candidate, state, metrics, cfg)
        candidate_for_update = reset["candidate"]
        outlier_gate = reset["outlier_gate"]
        persistent_reset = reset["persistent_reset"]
        local_high_count = reset["local_high_count"]
        camera_high_count = reset["camera_high_count"]
        filtered_normal = ( # 执行 kalman filter update
            kalman["gain"] * candidate_for_update
            + (1.0 - kalman["gain"]) * metrics["prediction"]
        )
        filtered_residual_ratio = torch.linalg.vector_norm(
            filtered_normal - candidate_for_update, dim=-1
        ) / (torch.linalg.vector_norm(candidate_for_update, dim=-1) + cfg.eps)
        # gbw____
        residual_reset = torch.zeros_like(metrics["local_reset"])
        if cfg.reset_mode == "current":
            residual_reset = (
                filtered_residual_ratio > cfg.max_filtered_residual_ratio
            )
            reset_mask = (
                metrics["local_reset"]
                | metrics["camera_reset_map"]
                | residual_reset
            )
            persistent_reset = reset_mask
        else:
            reset_mask = persistent_reset
        # ____

        covariance_normal = (# 计算后验 P_t
            (1.0 - kalman["gain"]) ** 2 * kalman["p_prior"]
            + kalman["gain"].square() * metrics["r_t"]
        )
        reset_expand = reset_mask.unsqueeze(-1)
        filtered = torch.where(reset_expand, candidate, filtered_normal)
        covariance = torch.where(reset_expand, metrics["r_t"], covariance_normal)
        effective_gain = torch.where(
            reset_expand, torch.ones_like(kalman["gain"]), kalman["gain"]
        )
        trust = self._apply_trust_region(
            filtered=filtered,
            candidate=candidate,
            covariance=covariance,
            gain=effective_gain,
            r_t=metrics["r_t"],
            cfg=cfg,
        )
        filtered = trust["filtered"]
        covariance = torch.clamp(trust["covariance"], min=cfg.eps)

        # gbw____
        # 用实际送入 decoder 的 filtered 位移更新速度 EMA。异常帧若被
        # Q/R 抑制，速度也同步被抑制，下一帧不会继续沿着 raw spike 外推。
        # 速度状态只在显式 constant_velocity 候选中分配。
        velocity = None
        if cfg.prediction_mode in {"constant_velocity", "gated_velocity"}:
            previous_velocity = state.velocity
            if previous_velocity is None:
                previous_velocity = torch.zeros_like(filtered)
            observed_step = filtered - state.filtered
            velocity = (
                cfg.velocity_ema_beta * observed_step
                + (1.0 - cfg.velocity_ema_beta) * previous_velocity
            ).detach()
        # ____

        # gbw____
        # 诊断必须比较实际返回给 ConvexUpSample 的 token，而不是 trust-region
        # 之前的 float state。这样 raw takeover 的判断也覆盖了 dtype cast。
        returned_tokens = filtered.to(dtype=tokens.dtype)
        raw_reference = tokens.detach()
        takeover_error = torch.linalg.vector_norm(
            returned_tokens.float() - raw_reference.float(), dim=-1
        ) / (torch.linalg.vector_norm(raw_reference.float(), dim=-1) + cfg.eps)
        raw_takeover_exact_mask = reset_mask & (
            takeover_error <= DIAGNOSTIC_RAW_MATCH_TOLERANCE
        )
        jitter_mask = (
            metrics["jitter_score"] > DIAGNOSTIC_JITTER_THRESHOLD
        )
        jitter_reset_overlap = jitter_mask & reset_mask
        # ____

        self._states[key] = _TokenState(
            filtered=filtered.detach(),
            covariance=covariance.detach(),
            previous_candidate=candidate.detach(),
            camera_delta_ema=metrics["camera_delta_ema"].detach(),
            # gbw____
            velocity=velocity,
            # ____
            drift_ema=metrics["drift_ema"].detach(),
            # gbw____
            drift_reference_ema=(
                None
                if metrics["drift_reference_ema"] is None
                else metrics["drift_reference_ema"].detach()
            ),
            # ____
            # gbw____
            innovation_energy_ema=(
                None
                if metrics["innovation_energy_ema"] is None
                else metrics["innovation_energy_ema"].detach()
            ),
            # ____
            # gbw____
            process_noise_ratio_ema=(
                None
                if metrics["process_noise_ratio_ema"] is None
                else metrics["process_noise_ratio_ema"].detach()
            ),
            # ____
            # gbw____
            innovation_reference_ema=(
                None
                if metrics["innovation_reference_ema"] is None
                else metrics["innovation_reference_ema"].detach()
            ),
            # ____
            previous_delta=(
                metrics["delta_vector"].detach()
                if cfg.q_mode in {
                    "motion_jitter_adaptive",
                    "drift_adaptive_sigmoid",
                    # gbw____
                    "gain_space_adaptive",
                    # ____
                    "apne_bounded_sigmoid",
                    "apne_normalized_sigmoid",
                }
                or cfg.r_mode == "jitter_aware"
                else None
            ),
            local_high_count=(
                None
                if local_high_count is None
                else local_high_count.detach()
            ),
            camera_high_count=(
                None
                if camera_high_count is None
                else camera_high_count.detach()
            ),
        )
        self._set_update_diagnostics(
            stage=stage,
            initialized=False,
            tokens=tokens,
            delta=metrics["raw_delta"],
            delta_ema=metrics["camera_delta_ema"],
            innovation=metrics["innovation"],
            drift_ema=metrics["drift_ema"],
            drift_ratio=metrics["drift_ratio"],
            # gbw____
            drift_reference_ema=metrics["drift_reference_ema"],
            # ____
            # gbw____
            q_drift_ratio=metrics["q_drift_ratio"],
            # gbw____
            q_low_guard=kalman["q_low_guard"],
            # gbw____
            q_selective_gate=kalman["q_selective_gate"],
            # gbw____
            q_selective_weight=kalman.get("q_selective_weight"),
            # ____
            drift_coherence=metrics["drift_coherence"],
            # gbw____
            magnitude_coherence=metrics["magnitude_coherence"],
            # gbw____
            raw_magnitude_coherence=metrics["raw_magnitude_coherence"],
            # ____
            # ____
            # gbw____
            spatial_coherence=metrics["spatial_coherence"],
            # ____
            # ____
            # gbw____
            innovation_ratio=metrics["innovation_ratio"],
            # ____
            q_scale=kalman["q_scale"],
            # gbw____
            innovation_energy_ema=metrics["innovation_energy_ema"],
            apne_q_hat=metrics["apne_q_hat"],
            apne_q_evidence=metrics["apne_q_evidence"],
            # gbw____
            apne_floor_break=kalman["apne_floor_break"],
            # ____
            # ____
            jitter_score=metrics["jitter_score"],
            local_score=metrics["local_score"],
            camera_score=metrics["camera_score"],
            q_t=kalman["q_t"],
            p_prior=kalman["p_prior"],
            gain=trust["gain"],
            # gbw____
            gain_target=kalman.get("gain_target"),
            # ____
            covariance=covariance,
            # gbw____
            measurement_r=metrics["r_t"],
            # ____
            normalized_innovation=metrics["normalized_innovation"],
            # gbw____
            robust_normalized_innovation=metrics[
                "robust_normalized_innovation"
            ],
            # ____
            filtered_residual_ratio=filtered_residual_ratio,
            token_shift_ratio=trust["token_shift_ratio"],
            trust_region_clipped=trust["trust_region_clipped"],
            local_reset=metrics["local_reset"],
            camera_reset=metrics["camera_reset_map"],
            residual_reset=residual_reset,
            reset_mask=reset_mask,
            outlier_gate=outlier_gate,
            persistent_reset=persistent_reset,
            jitter_mask=jitter_mask,
            jitter_reset_overlap=jitter_reset_overlap,
            raw_takeover_exact_mask=raw_takeover_exact_mask,
            takeover_error=takeover_error,
        )
        return returned_tokens
    # ____
