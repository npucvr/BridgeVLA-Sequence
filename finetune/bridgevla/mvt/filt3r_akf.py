"""Causal, inference-only AKF between PaliGemma tokens and ConvexUpSample."""

# gbw____
from dataclasses import dataclass
import json
import os
from typing import Callable, Dict, Optional, Tuple

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
    """Runtime configuration kept for the active event-neutral path."""

    # gbw____
    p_init: float = 1.5
    gamma_p: float = 1.0
    r: float = 1.0
    # gbw____
    # A3-QR-token-mild-v2：默认 fixed 保持所有历史 A0/A1/A2 路径不变；
    # jitter_aware 只改变 measurement covariance，不改 checkpoint、decoder
    # 或 Stage-2。R 先以 token scalar 广播到 2048 channels。
    r_mode: str = "fixed"
    jitter_r_lambda: float = 1.0
    jitter_r_max: float = 2.0
    jitter_r_tau: float = 1.25
    jitter_r_beta: float = 0.20
    jitter_r_confirm: int = 2
    jitter_r_release_tau: float = 0.15
    jitter_r_trigger_tau: float = 0.25
    jitter_r_coherence_weight: float = 0.75
    jitter_r_coherence_mode: str = "hybrid"
    jitter_r_channel_mode: str = "token"
    jitter_r_topk: int = 16
    jitter_r_exceedance_tau: float = 2.5
    q_use_adaptive_r: bool = False
    q_conflict_veto: bool = False
    # ____
    k_min: float = 0.01
    k_max: float = 0.99
    delta_floor: float = 1e-2
    eps: float = 1e-6
    stages: Tuple[int, ...] = (1,)

    q_mode: str = "event_neutral"
    drift_ema_beta: float = 0.2
    drift_metric: str = "diagonal_robust_mahalanobis"

    reset_mode: str = "current"
    reset_patience: int = 2
    reset_local_tau: float = 3.0
    reset_camera_quantile: float = 0.75
    reset_camera_tau: float = 2.0
    reset_camera_ema_beta: float = 0.2
    reset_coherence_tau: float = 0.5
    innovation_gate_tau: float = 6.0
    max_filtered_residual_ratio: float = 0.05

    event_q_reference_ratio: float = 18.05
    event_q_min_ratio: float = 12.0
    event_q_max_ratio: float = 32.0
    event_q_motion_strength: float = 0.30
    event_q_jitter_strength: float = 0.35
    event_q_motion_confirm: int = 1
    event_q_evidence_mode: str = "temporal"
    event_q_log_alpha: float = 4.0
    event_q_motion_tau: float = 0.15
    event_q_jitter_tau: float = 0.35
    event_q_innovation_tau: float = 1.5
    event_q_innovation_alpha: float = 1.5
    event_q_temporal_tau: float = 0.5
    event_q_spatial_tau: float = 0.5

    event_q_sparse_mode: str = "none"
    event_q_sparse_weight: float = 0.0
    event_q_sparse_tau: float = 3.0
    event_q_sparse_alpha: float = 1.0
    event_q_sparse_topk: int = 16
    event_q_sparse_spatial: bool = False

    event_reset_motion_tau: float = 0.35
    event_reset_jitter_tau: float = 0.35
    event_reset_innovation_tau: float = 1.5
    # ____

    def __post_init__(self) -> None:
        if self.p_init <= 0 or self.gamma_p < 0:
            raise ValueError("p_init must be positive and gamma_p non-negative")
        if self.r <= 0 or self.delta_floor <= 0 or self.eps <= 0:
            raise ValueError("r, delta_floor, and eps must be positive")
        # gbw____
        if self.r_mode not in {"fixed", "jitter_aware"}:
            raise ValueError("unsupported r_mode")
        if self.jitter_r_lambda < 0 or self.jitter_r_max < 1.0:
            raise ValueError("adaptive R strength/max must be non-negative and >= 1")
        if self.jitter_r_beta <= 0 or self.jitter_r_beta > 1:
            raise ValueError("jitter_r_beta must be in (0, 1]")
        if self.jitter_r_confirm < 1:
            raise ValueError("jitter_r_confirm must be >= 1")
        if not 0 <= self.jitter_r_release_tau <= self.jitter_r_trigger_tau <= 1:
            raise ValueError("adaptive R release/trigger thresholds are invalid")
        if self.jitter_r_coherence_mode not in {"magnitude", "direction", "hybrid"}:
            raise ValueError("unsupported jitter_r_coherence_mode")
        if self.jitter_r_channel_mode not in {"token", "channel"}:
            raise ValueError("unsupported jitter_r_channel_mode")
        if not 1 <= self.jitter_r_topk <= 2048:
            raise ValueError("jitter_r_topk must be in [1, 2048]")
        if self.jitter_r_exceedance_tau < 0 or self.jitter_r_tau < 0:
            raise ValueError("adaptive R thresholds must be non-negative")
        if not 0 <= self.jitter_r_coherence_weight <= 1:
            raise ValueError("jitter_r_coherence_weight must be in [0, 1]")
        if self.q_use_adaptive_r:
            raise ValueError("q_use_adaptive_r must remain false for independent A3 R")
        # ____
        if not 0 <= self.k_min <= self.k_max <= 1:
            raise ValueError("gain clamps must satisfy 0 <= k_min <= k_max <= 1")
        if not self.stages or any(stage not in (1, 2) for stage in self.stages):
            raise ValueError("stages must contain only 1 and/or 2")
        if self.q_mode != "event_neutral":
            raise ValueError("q_mode must be event_neutral; legacy Q modes are archived")
        if self.drift_metric != "diagonal_robust_mahalanobis":
            raise ValueError(
                "drift_metric must be diagonal_robust_mahalanobis; "
                "legacy metrics are archived"
            )
        if not 0 < self.drift_ema_beta <= 1:
            raise ValueError("drift_ema_beta must be in (0, 1]")
        if self.reset_mode not in {
            "none",
            "current",
            "innovation_repartition_norm",
            "innovation_coherent",
            "event_coherent",
        }:
            raise ValueError("unsupported reset_mode")
        if self.reset_patience < 1:
            raise ValueError("reset_patience must be at least one")
        if self.reset_local_tau < 0 or self.reset_camera_tau < 0:
            raise ValueError("reset thresholds must be non-negative")
        if not 0 < self.reset_camera_quantile < 1:
            raise ValueError("reset_camera_quantile must be in (0, 1)")
        if not 0 < self.reset_camera_ema_beta <= 1:
            raise ValueError("reset_camera_ema_beta must be in (0, 1]")
        if self.reset_coherence_tau < 0 or self.reset_coherence_tau > 1:
            raise ValueError("reset_coherence_tau must be in [0, 1]")
        if self.innovation_gate_tau <= 0:
            raise ValueError("innovation_gate_tau must be positive")
        if self.max_filtered_residual_ratio < 0:
            raise ValueError("max_filtered_residual_ratio must be non-negative")
        if (
            self.event_q_reference_ratio <= 0
            or self.event_q_min_ratio <= 0
            or self.event_q_min_ratio > self.event_q_max_ratio
            or not self.event_q_min_ratio
            <= self.event_q_reference_ratio
            <= self.event_q_max_ratio
        ):
            raise ValueError("event Q/R bounds must contain a positive reference ratio")
        if self.event_q_motion_strength < 0 or self.event_q_jitter_strength < 0:
            raise ValueError("event Q strengths must be non-negative")
        if self.event_q_motion_confirm < 1:
            raise ValueError("event_q_motion_confirm must be >= 1")
        if self.event_q_evidence_mode not in {
            "temporal",
            "spatial",
            "innovation",
            "guarded",
        }:
            raise ValueError("unsupported event_q_evidence_mode")
        if self.event_q_log_alpha <= 0:
            raise ValueError("event_q_log_alpha must be positive")
        if self.event_q_motion_tau < 0 or self.event_q_jitter_tau < 0:
            raise ValueError("event Q drift thresholds must be non-negative")
        if self.event_q_innovation_tau < 0 or self.event_q_innovation_alpha <= 0:
            raise ValueError("event Q innovation settings are invalid")
        if not 0 <= self.event_q_temporal_tau <= 1:
            raise ValueError("event_q_temporal_tau must be in [0, 1]")
        if not 0 <= self.event_q_spatial_tau <= 1:
            raise ValueError("event_q_spatial_tau must be in [0, 1]")
        if self.event_q_sparse_mode not in {"none", "topk", "exceedance"}:
            raise ValueError("unsupported event_q_sparse_mode")
        if not 0 <= self.event_q_sparse_weight <= 1:
            raise ValueError("event_q_sparse_weight must be in [0, 1]")
        if self.event_q_sparse_tau < 0 or self.event_q_sparse_alpha <= 0:
            raise ValueError("event sparse tau/alpha are invalid")
        if not 1 <= self.event_q_sparse_topk <= 2048:
            raise ValueError("event_q_sparse_topk must be in [1, 2048]")

    @classmethod
    def from_environment(cls) -> "Filt3rAKFConfig":
        # gbw____
        return cls(
            p_init=_env_float("FILT3R_P_INIT", cls.p_init),
            gamma_p=_env_float("FILT3R_GAMMA_P", cls.gamma_p),
            r=_env_float("FILT3R_R", cls.r),
            # gbw____
            r_mode=os.environ.get("FILT3R_R_MODE", cls.r_mode).strip().lower(),
            jitter_r_lambda=_env_float(
                "FILT3R_JITTER_R_LAMBDA", cls.jitter_r_lambda
            ),
            jitter_r_max=_env_float(
                "FILT3R_JITTER_R_MAX", cls.jitter_r_max
            ),
            jitter_r_tau=_env_float(
                "FILT3R_JITTER_R_TAU", cls.jitter_r_tau
            ),
            jitter_r_beta=_env_float(
                "FILT3R_JITTER_R_BETA", cls.jitter_r_beta
            ),
            jitter_r_confirm=_env_int(
                "FILT3R_JITTER_R_CONFIRM", cls.jitter_r_confirm
            ),
            jitter_r_release_tau=_env_float(
                "FILT3R_JITTER_R_RELEASE_TAU", cls.jitter_r_release_tau
            ),
            jitter_r_trigger_tau=_env_float(
                "FILT3R_JITTER_R_TRIGGER_TAU", cls.jitter_r_trigger_tau
            ),
            jitter_r_coherence_weight=_env_float(
                "FILT3R_JITTER_R_COHERENCE_WEIGHT",
                cls.jitter_r_coherence_weight,
            ),
            jitter_r_coherence_mode=os.environ.get(
                "FILT3R_JITTER_R_COHERENCE_MODE",
                cls.jitter_r_coherence_mode,
            ).strip().lower(),
            jitter_r_channel_mode=os.environ.get(
                "FILT3R_JITTER_R_CHANNEL_MODE",
                cls.jitter_r_channel_mode,
            ).strip().lower(),
            jitter_r_topk=_env_int(
                "FILT3R_JITTER_R_TOPK", cls.jitter_r_topk
            ),
            jitter_r_exceedance_tau=_env_float(
                "FILT3R_JITTER_R_EXCEEDANCE_TAU",
                cls.jitter_r_exceedance_tau,
            ),
            q_use_adaptive_r=_env_bool(
                "FILT3R_Q_USE_ADAPTIVE_R", cls.q_use_adaptive_r
            ),
            q_conflict_veto=_env_bool(
                "FILT3R_Q_CONFLICT_VETO", cls.q_conflict_veto
            ),
            # ____
            k_min=_env_float("FILT3R_K_MIN", cls.k_min),
            k_max=_env_float("FILT3R_K_MAX", cls.k_max),
            delta_floor=_env_float("FILT3R_DELTA_FLOOR", cls.delta_floor),
            eps=_env_float("FILT3R_EPS", cls.eps),
            stages=_env_stages(),
            q_mode=os.environ.get("FILT3R_Q_MODE", cls.q_mode).strip().lower(),
            drift_ema_beta=_env_float(
                "FILT3R_DRIFT_EMA_BETA", cls.drift_ema_beta
            ),
            drift_metric=os.environ.get(
                "FILT3R_DRIFT_METRIC", cls.drift_metric
            ).strip().lower(),
            reset_mode=os.environ.get(
                "FILT3R_RESET_MODE", cls.reset_mode
            ).strip().lower(),
            reset_patience=_env_int(
                "FILT3R_RESET_PATIENCE", cls.reset_patience
            ),
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
            reset_coherence_tau=_env_float(
                "FILT3R_RESET_COHERENCE_TAU", cls.reset_coherence_tau
            ),
            innovation_gate_tau=_env_float(
                "FILT3R_INNOVATION_GATE_TAU", cls.innovation_gate_tau
            ),
            max_filtered_residual_ratio=_env_float(
                "FILT3R_MAX_FILTERED_RESIDUAL_RATIO",
                cls.max_filtered_residual_ratio,
            ),
            event_q_reference_ratio=_env_float(
                "FILT3R_EVENT_Q_REFERENCE_RATIO",
                cls.event_q_reference_ratio,
            ),
            event_q_min_ratio=_env_float(
                "FILT3R_EVENT_Q_MIN_RATIO", cls.event_q_min_ratio
            ),
            event_q_max_ratio=_env_float(
                "FILT3R_EVENT_Q_MAX_RATIO", cls.event_q_max_ratio
            ),
            event_q_motion_strength=_env_float(
                "FILT3R_EVENT_Q_MOTION_STRENGTH",
                cls.event_q_motion_strength,
            ),
            event_q_jitter_strength=_env_float(
                "FILT3R_EVENT_Q_JITTER_STRENGTH",
                cls.event_q_jitter_strength,
            ),
            event_q_motion_confirm=_env_int(
                "FILT3R_EVENT_Q_MOTION_CONFIRM", cls.event_q_motion_confirm
            ),
            event_q_evidence_mode=os.environ.get(
                "FILT3R_EVENT_Q_EVIDENCE_MODE",
                cls.event_q_evidence_mode,
            ).strip().lower(),
            event_q_log_alpha=_env_float(
                "FILT3R_EVENT_Q_LOG_ALPHA", cls.event_q_log_alpha
            ),
            event_q_motion_tau=_env_float(
                "FILT3R_EVENT_Q_MOTION_TAU", cls.event_q_motion_tau
            ),
            event_q_jitter_tau=_env_float(
                "FILT3R_EVENT_Q_JITTER_TAU", cls.event_q_jitter_tau
            ),
            event_q_innovation_tau=_env_float(
                "FILT3R_EVENT_Q_INNOVATION_TAU",
                cls.event_q_innovation_tau,
            ),
            event_q_innovation_alpha=_env_float(
                "FILT3R_EVENT_Q_INNOVATION_ALPHA",
                cls.event_q_innovation_alpha,
            ),
            event_q_temporal_tau=_env_float(
                "FILT3R_EVENT_Q_TEMPORAL_TAU",
                cls.event_q_temporal_tau,
            ),
            event_q_spatial_tau=_env_float(
                "FILT3R_EVENT_Q_SPATIAL_TAU",
                cls.event_q_spatial_tau,
            ),
            event_q_sparse_mode=os.environ.get(
                "FILT3R_EVENT_Q_SPARSE_MODE",
                cls.event_q_sparse_mode,
            ).strip().lower(),
            event_q_sparse_weight=_env_float(
                "FILT3R_EVENT_Q_SPARSE_WEIGHT",
                cls.event_q_sparse_weight,
            ),
            event_q_sparse_tau=_env_float(
                "FILT3R_EVENT_Q_SPARSE_TAU", cls.event_q_sparse_tau
            ),
            event_q_sparse_alpha=_env_float(
                "FILT3R_EVENT_Q_SPARSE_ALPHA", cls.event_q_sparse_alpha
            ),
            event_q_sparse_topk=_env_int(
                "FILT3R_EVENT_Q_SPARSE_TOPK", cls.event_q_sparse_topk
            ),
            event_q_sparse_spatial=_env_bool(
                "FILT3R_EVENT_Q_SPARSE_SPATIAL",
                cls.event_q_sparse_spatial,
            ),
            event_reset_motion_tau=_env_float(
                "FILT3R_EVENT_RESET_MOTION_TAU",
                cls.event_reset_motion_tau,
            ),
            event_reset_jitter_tau=_env_float(
                "FILT3R_EVENT_RESET_JITTER_TAU",
                cls.event_reset_jitter_tau,
            ),
            event_reset_innovation_tau=_env_float(
                "FILT3R_EVENT_RESET_INNOVATION_TAU",
                cls.event_reset_innovation_tau,
            ),
        )
        # ____
@dataclass
class _TokenState:
    # gbw____
    filtered: torch.Tensor
    covariance: torch.Tensor
    previous_candidate: torch.Tensor
    camera_delta_ema: Optional[torch.Tensor]
    drift_ema: Optional[torch.Tensor]
    previous_delta: Optional[torch.Tensor]
    event_jitter_count: Optional[torch.Tensor]
    event_motion_count: Optional[torch.Tensor]
    # gbw____
    adaptive_r_ema: Optional[torch.Tensor]
    adaptive_r_count: Optional[torch.Tensor]
    adaptive_r_active: Optional[torch.Tensor]
    # ____
    local_high_count: Optional[torch.Tensor]
    camera_high_count: Optional[torch.Tensor]
    # ____


class A1UnifiedTokenKalmanFilter:
    """Causal event-neutral AKF for Stage-1 image tokens."""

    VALID_MODES = frozenset(("none", "filt3r_akf"))

    # gbw____
    def __init__(
        self,
        mode: Optional[str] = None,
        config: Optional[Filt3rAKFConfig] = None,
        diagnostics_enabled: Optional[bool] = None,
    ) -> None:
        self.mode = mode or os.environ.get("FILTER_MODE", "none")
        if self.mode not in self.VALID_MODES:
            raise ValueError("FILTER_MODE must be none or filt3r_akf")
        self.config = config or Filt3rAKFConfig.from_environment()
        self.diagnostics_enabled = (
            _env_bool("FILTER_DIAGNOSTICS")
            if diagnostics_enabled is None
            else diagnostics_enabled
        )
        self.shadow_raw_enabled = self.diagnostics_enabled and _env_bool(
            "FILTER_DIAGNOSTICS_SHADOW_RAW"
        )
        self.decoder_protection_enabled = _env_bool(
            "FILT3R_DECODER_PROTECTION"
        )
        self.diagnostics_path = os.environ.get("FILTER_DIAGNOSTICS_PATH")
        self.metric_audit_enabled = False
        self._episode_index = 0
        self._states: Dict[Tuple[int, int, Tuple[int, ...], str, str], _TokenState] = {}
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
        self._context = {
            "task_label": None if task_label is None else str(task_label),
            "timestep": None if timestep is None else int(timestep),
            "eval_seed": os.environ.get("EVAL_SEED"),
        }

    def reset(self, reason: str = "agent_reset") -> int:
        count = len(self._states)
        self._states.clear()
        self._last_diagnostics = None
        if self.mode != "none":
            episode = self._episode_index
            self._episode_index += 1
            self._append_diagnostics(
                {
                    "event": "reset",
                    "algorithm": "A1_U_IGAKF",
                    "reason": reason,
                    "cleared_states": count,
                    "filter_mode": self.mode,
                    "call_index": self._call_index,
                    "episode_index": episode,
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
                "expected token shape (B, 3, 16, 16, 2048), "
                f"got {tuple(tokens.shape)}"
            )
        if not tokens.is_floating_point():
            raise TypeError("U-IGAKF tokens must be floating point")

    def _state_key(
        self, tokens: torch.Tensor, stage: int
    ) -> Tuple[int, int, Tuple[int, ...], str, str]:
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
        if not flat.numel():
            return None
        q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
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
        return value.detach().float().mean(dim=(0, *range(2, value.ndim))).cpu().tolist()

    def _append_diagnostics(self, record: dict) -> None:
        if not self.diagnostics_path:
            return
        parent = os.path.dirname(self.diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.diagnostics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    # ____
    # gbw____
    def record_metric_audit(
        self, tokens: torch.Tensor, stage: int = 1
    ) -> None:
        """Archived audit hook kept as a no-op for old callers."""

        return
    # ____
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
        drift_ema: Optional[torch.Tensor] = None,
        drift_ratio: Optional[torch.Tensor] = None,
        event_motion_score: Optional[torch.Tensor] = None,
        event_jitter_score: Optional[torch.Tensor] = None,
        sparse_event_score: Optional[torch.Tensor] = None,
        event_jitter_count: Optional[torch.Tensor] = None,
        drift_coherence: Optional[torch.Tensor] = None,
        magnitude_coherence: Optional[torch.Tensor] = None,
        raw_magnitude_coherence: Optional[torch.Tensor] = None,
        spatial_coherence: Optional[torch.Tensor] = None,
        jitter_score: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        q_t: Optional[torch.Tensor] = None,
        p_prior: Optional[torch.Tensor] = None,
        gain: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
        measurement_r: Optional[torch.Tensor] = None,
        adaptive_r_evidence: Optional[torch.Tensor] = None,
        adaptive_r_score: Optional[torch.Tensor] = None,
        adaptive_r_scale: Optional[torch.Tensor] = None,
        adaptive_r_active: Optional[torch.Tensor] = None,
        adaptive_r_count: Optional[torch.Tensor] = None,
        q_conflict_veto: Optional[torch.Tensor] = None,
        normalized_innovation: Optional[torch.Tensor] = None,
        filtered_residual_ratio: Optional[torch.Tensor] = None,
        token_shift_ratio: Optional[torch.Tensor] = None,
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
        decoder_protection_alpha: Optional[torch.Tensor] = None,
        decoder_protection_mask: Optional[torch.Tensor] = None,
        decoder_raw_margin: Optional[torch.Tensor] = None,
        decoder_filtered_margin: Optional[torch.Tensor] = None,
        decoder_peak_shift: Optional[torch.Tensor] = None,
        decoder_waypoint_shift: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.diagnostics_enabled:
            self._last_diagnostics = None
            return

        def mean(value: Optional[torch.Tensor]) -> Optional[float]:
            return (
                None
                if value is None
                else float(value.detach().float().mean().cpu())
            )

        def fraction(value: Optional[torch.Tensor]) -> Optional[float]:
            return mean(value)

        record = {
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
            "q_mode": self.config.q_mode,
            "drift_metric": self.config.drift_metric,
            "reset_mode": self.config.reset_mode,
            "r": self.config.r,
            # gbw____
            "r_mode": self.config.r_mode,
            "jitter_r_max": self.config.jitter_r_max,
            "jitter_r_beta": self.config.jitter_r_beta,
            "jitter_r_confirm": self.config.jitter_r_confirm,
            "jitter_r_channel_mode": self.config.jitter_r_channel_mode,
            "q_conflict_veto_enabled": self.config.q_conflict_veto,
            # ____
            "p_init": self.config.p_init,
            "gamma_p": self.config.gamma_p,
            "event_q_evidence_mode": self.config.event_q_evidence_mode,
            "event_q_sparse_mode": self.config.event_q_sparse_mode,
            "event_q_sparse_weight": self.config.event_q_sparse_weight,
            "delta_mean": mean(delta),
            "delta_ema_mean": mean(delta_ema),
            "innovation_mean": mean(innovation),
            "normalized_innovation_mean": mean(normalized_innovation),
            "drift_ema_mean": mean(drift_ema),
            "drift_ratio_mean": mean(drift_ratio),
            "drift_ratio_quantiles": self._quantiles(drift_ratio),
            "event_motion_score_mean": mean(event_motion_score),
            "event_motion_score_quantiles": self._quantiles(event_motion_score),
            "event_jitter_score_mean": mean(event_jitter_score),
            "event_jitter_score_quantiles": self._quantiles(event_jitter_score),
            "sparse_event_score_mean": mean(sparse_event_score),
            "sparse_event_score_quantiles": self._quantiles(sparse_event_score),
            "event_jitter_count_mean": mean(event_jitter_count),
            "drift_coherence_mean": mean(drift_coherence),
            "magnitude_coherence_mean": mean(magnitude_coherence),
            "raw_magnitude_coherence_mean": mean(raw_magnitude_coherence),
            "spatial_coherence_mean": mean(spatial_coherence),
            "jitter_score_mean": mean(jitter_score),
            "q_scale_mean": mean(q_scale),
            "q_mean": mean(q_t),
            "p_prior_mean": mean(p_prior),
            "p_mean": mean(covariance),
            "r_mean": self.config.r,
            "r_t_mean": mean(measurement_r),
            # gbw____
            "adaptive_r_evidence_mean": mean(adaptive_r_evidence),
            "adaptive_r_evidence_quantiles": self._quantiles(adaptive_r_evidence),
            "adaptive_r_score_mean": mean(adaptive_r_score),
            "adaptive_r_score_quantiles": self._quantiles(adaptive_r_score),
            "adaptive_r_scale_mean": mean(adaptive_r_scale),
            "adaptive_r_scale_quantiles": self._quantiles(adaptive_r_scale),
            "adaptive_r_activation_fraction": fraction(adaptive_r_active),
            "adaptive_r_count_mean": mean(adaptive_r_count),
            "q_conflict_veto_fraction": fraction(q_conflict_veto),
            # ____
            "gain_mean": mean(gain),
            "gain_min": None if gain is None else float(gain.min().cpu()),
            "gain_max": None if gain is None else float(gain.max().cpu()),
            "filtered_residual_ratio_mean": mean(filtered_residual_ratio),
            "token_shift_ratio_mean": mean(token_shift_ratio),
            "local_reset_fraction": fraction(local_reset),
            "camera_reset_fraction": fraction(camera_reset),
            "residual_reset_fraction": fraction(residual_reset),
            "measurement_takeover_fraction": fraction(reset_mask),
            "outlier_gate_fraction": fraction(outlier_gate),
            "persistent_reset_fraction": fraction(persistent_reset),
            "jitter_token_fraction": fraction(jitter_mask),
            "jitter_reset_overlap_fraction": fraction(jitter_reset_overlap),
            "raw_takeover_exact_fraction": fraction(raw_takeover_exact_mask),
            "raw_filtered_relative_error_mean": mean(takeover_error),
            "decoder_protection_alpha_mean": mean(decoder_protection_alpha),
            "decoder_protection_fraction": fraction(decoder_protection_mask),
            "decoder_raw_margin_mean": mean(decoder_raw_margin),
            "decoder_filtered_margin_mean": mean(decoder_filtered_margin),
            "decoder_peak_shift_mean": mean(decoder_peak_shift),
            "decoder_waypoint_shift_mean": mean(decoder_waypoint_shift),
            "finite": bool(torch.isfinite(tokens).all().item()),
        }
        self._last_diagnostics = record
        self._append_diagnostics(record)
    # ____
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
    def _pairwise_drift_metric(
        previous: torch.Tensor,
        current: torch.Tensor,
        previous_step: Optional[torch.Tensor],
        cfg: Filt3rAKFConfig,
    ) -> torch.Tensor:
        """Convert one raw-token transition into the active causal drift metric."""

        if cfg.drift_metric != "diagonal_robust_mahalanobis":
            raise ValueError(
                "only diagonal_robust_mahalanobis is available in the runtime filter"
            )
        step = current - previous
        if previous_step is None:
            raw_scale = previous.abs()
        else:
            raw_scale = previous_step.abs()
        scale_center = torch.median(raw_scale, dim=-1, keepdim=True).values
        scale_mad = torch.median(
            (raw_scale - scale_center).abs(), dim=-1, keepdim=True
        ).values
        robust_floor = (
            scale_center + 1.4826 * scale_mad
        ).clamp_min(cfg.delta_floor)
        channel_scale = torch.maximum(
            raw_scale,
            0.25 * robust_floor,
        )
        standardized = (step / (channel_scale + cfg.eps)).clamp(
            min=-6.0, max=6.0
        )
        return torch.sqrt(standardized.square().mean(dim=-1))
    # ____
    @staticmethod
    def _event_spatial_support(
        drift_value: torch.Tensor,
        cfg: Filt3rAKFConfig,
    ) -> torch.Tensor:
        """Measure causal 3x3 support for one scalar drift map."""

        if drift_value.ndim != 4:
            return torch.ones_like(drift_value)
        height, width = drift_value.shape[-2:]
        local_mean = F.avg_pool2d(
            drift_value.reshape(-1, 1, height, width),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        ).reshape_as(drift_value)
        deviation = (drift_value - local_mean).abs()
        local_scale = F.avg_pool2d(
            deviation.reshape(-1, 1, height, width),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        ).reshape_as(drift_value)
        return torch.exp(
            -deviation / (local_scale + cfg.delta_floor)
        ).clamp(min=0.0, max=1.0)
    # ____

    # gbw____
    @staticmethod
    def _adaptive_r_update(
        *,
        candidate: torch.Tensor,
        innovation_vector: torch.Tensor,
        covariance: torch.Tensor,
        reference_r: torch.Tensor,
        normalized_innovation: torch.Tensor,
        jitter_score: torch.Tensor,
        drift_coherence: torch.Tensor,
        magnitude_coherence: torch.Tensor,
        coherence_history_valid: torch.Tensor,
        spatial_support: torch.Tensor,
        state: _TokenState,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Compute independent, token-level measurement reliability evidence."""

        fixed_r = torch.full_like(candidate, cfg.r)
        zero_score = torch.zeros_like(normalized_innovation)
        zero_active = torch.zeros_like(normalized_innovation, dtype=torch.bool)
        if cfg.r_mode == "fixed":
            return {
                "r_t": fixed_r,
                "adaptive_r_evidence": zero_score,
                "adaptive_r_score": zero_score,
                "adaptive_r_ema": None,
                "adaptive_r_count": None,
                "adaptive_r_active": zero_active,
            }
        if cfg.jitter_r_channel_mode != "token":
            raise ValueError(
                "A3-QR-token-mild-v2 only supports token-level adaptive R"
            )

        # The dense diagonal metric remains the base metric. Sparse evidence is
        # an auxiliary standardized top-k statistic, not a semantic mask.
        standardized_abs = torch.sqrt(
            innovation_vector.square()
            / (covariance + reference_r + cfg.eps)
        )
        topk = torch.topk(
            standardized_abs,
            k=min(cfg.jitter_r_topk, standardized_abs.shape[-1]),
            dim=-1,
        ).values
        topk_rms = torch.sqrt(topk.square().mean(dim=-1).clamp_min(cfg.eps))
        sparse_signal = torch.sigmoid(
            cfg.jitter_r_lambda
            * (topk_rms - cfg.jitter_r_exceedance_tau)
        )
        innovation_signal = torch.sigmoid(
            cfg.jitter_r_lambda
            * (normalized_innovation - cfg.event_q_innovation_tau)
        )

        direction_bad = (1.0 - drift_coherence).clamp(0.0, 1.0)
        magnitude_bad = (1.0 - magnitude_coherence).clamp(0.0, 1.0)
        if cfg.jitter_r_coherence_mode == "magnitude":
            coherence_bad = magnitude_bad
        elif cfg.jitter_r_coherence_mode == "direction":
            coherence_bad = direction_bad
        else:
            coherence_bad = (
                (1.0 - cfg.jitter_r_coherence_weight) * magnitude_bad
                + cfg.jitter_r_coherence_weight * direction_bad
            ).clamp(0.0, 1.0)

        second_order_signal = torch.sigmoid(
            cfg.jitter_r_lambda * (jitter_score - cfg.jitter_r_tau)
        )
        history = coherence_history_valid.clamp(0.0, 1.0)
        spatial_isolation = (1.0 - spatial_support).clamp(0.0, 1.0)
        raw_score = (
            sparse_signal
            * innovation_signal
            * second_order_signal
            * coherence_bad
            * spatial_isolation
            * history
        ).clamp(0.0, 1.0)

        previous_ema = (
            torch.zeros_like(raw_score)
            if state.adaptive_r_ema is None
            else state.adaptive_r_ema
        )
        adaptive_r_ema = (
            (1.0 - cfg.jitter_r_beta) * previous_ema
            + cfg.jitter_r_beta * raw_score
        ).clamp(0.0, 1.0)
        # Confirmation is based on the current causal evidence, not on the
        # smoothed EMA.  With beta=0.20 an EMA-triggered counter would need
        # many frames before it could ever satisfy the explicit two-frame
        # confirmation contract.  The EMA still controls the scale once the
        # evidence is confirmed and provides hysteresis on release.
        trigger = raw_score >= cfg.jitter_r_trigger_tau
        previous_count = (
            torch.zeros_like(trigger, dtype=torch.int64)
            if state.adaptive_r_count is None
            else state.adaptive_r_count
        )
        adaptive_r_count = torch.where(
            trigger,
            (previous_count + 1).clamp(max=cfg.jitter_r_confirm),
            torch.zeros_like(previous_count),
        )
        previous_active = (
            torch.zeros_like(trigger)
            if state.adaptive_r_active is None
            else state.adaptive_r_active
        )
        adaptive_r_active = (
            (adaptive_r_count >= cfg.jitter_r_confirm)
            | (previous_active & (adaptive_r_ema >= cfg.jitter_r_release_tau))
        )
        # Use the larger of the causal EMA and the just-confirmed evidence for
        # the active scale.  This preserves beta=0.20 smoothing while ensuring
        # a confirmed two-frame event actually increases R on its first active
        # frame instead of being clamped back to the nominal floor.
        active_score = torch.where(
            adaptive_r_active,
            torch.maximum(adaptive_r_ema, raw_score),
            torch.zeros_like(adaptive_r_ema),
        )
        scale = 1.0 + (cfg.jitter_r_max - 1.0) * (
            (active_score - cfg.jitter_r_trigger_tau)
            / max(1.0 - cfg.jitter_r_trigger_tau, cfg.eps)
        ).clamp(0.0, 1.0)
        r_t = cfg.r * scale.unsqueeze(-1).expand_as(candidate)
        return {
            "r_t": r_t,
            "adaptive_r_evidence": raw_score,
            "adaptive_r_score": active_score,
            "adaptive_r_ema": adaptive_r_ema,
            "adaptive_r_count": adaptive_r_count,
            "adaptive_r_active": adaptive_r_active,
        }
    # ____

    # gbw____
    @staticmethod
    def _temporal_metrics(
        candidate: torch.Tensor,
        state: _TokenState,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Compute the active causal evidence for one raw-token transition."""

        feature_scale = float(candidate.shape[-1]) ** 0.5
        # Keep the reference R fixed while computing reliability evidence;
        # adaptive R is applied only after this causal score is formed.
        reference_r = torch.full_like(candidate, cfg.r)
        delta_vector = candidate - state.previous_candidate
        prediction = state.filtered
        innovation_vector = candidate - prediction
        raw_delta = torch.linalg.vector_norm(
            delta_vector, dim=-1
        ) / feature_scale
        innovation = torch.linalg.vector_norm(
            innovation_vector, dim=-1
        ) / feature_scale
        normalized_innovation = torch.sqrt(
            torch.mean(
                innovation_vector.square()
                / (state.covariance + reference_r + cfg.eps),
                dim=-1,
            ).clamp_min(cfg.eps)
        )

        camera_delta = torch.quantile(
            raw_delta.flatten(-2), cfg.reset_camera_quantile, dim=-1
        )
        if state.camera_delta_ema is None:
            camera_delta_ema = camera_delta.clamp_min(cfg.delta_floor)
            camera_score = torch.zeros_like(camera_delta)
        else:
            camera_score = camera_delta / (
                state.camera_delta_ema + cfg.eps
            )
            camera_delta_ema = (
                (1.0 - cfg.reset_camera_ema_beta) * state.camera_delta_ema
                + cfg.reset_camera_ema_beta * camera_delta
            ).clamp_min(cfg.delta_floor)

        reset_metric = (
            normalized_innovation
            if cfg.reset_mode in {
                "innovation_repartition_norm",
                "innovation_coherent",
                "event_coherent",
            }
            else innovation
        )
        local_values = reset_metric.flatten(-2)
        local_center = torch.median(
            local_values, dim=-1
        ).values[..., None, None]
        local_deviation = (
            reset_metric - local_center
        ).abs().flatten(-2)
        local_mad = torch.median(
            local_deviation, dim=-1
        ).values[..., None, None]
        local_scale = (1.4826 * local_mad).clamp_min(cfg.delta_floor)
        local_score = (reset_metric - local_center) / (
            local_scale + cfg.eps
        )
        local_reset = local_score > cfg.reset_local_tau
        camera_reset = camera_score > cfg.reset_camera_tau
        camera_reset_map = camera_reset[..., None, None].expand_as(
            local_reset
        )

        current_drift = A1UnifiedTokenKalmanFilter._pairwise_drift_metric(
            state.previous_candidate,
            candidate,
            state.previous_delta,
            cfg,
        )
        if state.drift_ema is None:
            drift_ratio = torch.ones_like(current_drift)
            drift_ema = current_drift.clamp_min(cfg.delta_floor)
        else:
            drift_ratio = current_drift / (state.drift_ema + cfg.eps)
            drift_ema = (
                (1.0 - cfg.drift_ema_beta) * state.drift_ema
                + cfg.drift_ema_beta * current_drift
            ).clamp_min(cfg.delta_floor)

        if state.previous_delta is None:
            drift_coherence = torch.ones_like(raw_delta)
            coherence_history_valid = torch.zeros_like(raw_delta)
            raw_magnitude_coherence = torch.zeros_like(raw_delta)
            jitter_score = torch.zeros_like(raw_delta)
        else:
            current_norm = torch.linalg.vector_norm(delta_vector, dim=-1)
            previous_norm = torch.linalg.vector_norm(
                state.previous_delta, dim=-1
            )
            cosine = torch.sum(
                delta_vector * state.previous_delta, dim=-1
            ) / (current_norm * previous_norm + cfg.eps)
            drift_coherence = cosine.clamp(min=0.0, max=1.0)
            coherence_history_valid = (
                previous_norm > cfg.delta_floor
            ).to(raw_delta.dtype)
            previous_raw_delta = previous_norm / feature_scale
            raw_magnitude_coherence = (
                1.0
                - (raw_delta - previous_raw_delta).abs()
                / (raw_delta + previous_raw_delta + cfg.eps)
            ).clamp(min=0.0, max=1.0)
            acceleration = torch.linalg.vector_norm(
                delta_vector - state.previous_delta, dim=-1
            ) / feature_scale
            jitter_score = acceleration / (
                state.drift_ema + cfg.eps
            )
        event_temporal_consistency = torch.where(
            coherence_history_valid > 0.0,
            raw_magnitude_coherence,
            torch.ones_like(raw_magnitude_coherence),
        ).clamp(min=0.0, max=1.0)
        event_spatial_support = (
            A1UnifiedTokenKalmanFilter._event_spatial_support(
                current_drift, cfg
            )
        )

        sparse_event_score = torch.zeros_like(current_drift)
        if (
            cfg.event_q_sparse_mode != "none"
            and cfg.event_q_sparse_weight > 0.0
        ):
            standardized_abs = torch.sqrt(
                    innovation_vector.square()
                    / (state.covariance + reference_r + cfg.eps)
            )
            if cfg.event_q_sparse_mode == "topk":
                topk_values = torch.topk(
                    standardized_abs,
                    k=min(
                        cfg.event_q_sparse_topk,
                        standardized_abs.shape[-1],
                    ),
                    dim=-1,
                ).values
                sparse_excess = (
                    torch.sqrt(topk_values.square().mean(dim=-1))
                    - cfg.event_q_sparse_tau
                ).clamp_min(0.0)
            else:
                sparse_excess = (
                    (standardized_abs > cfg.event_q_sparse_tau)
                    .to(standardized_abs.dtype)
                    .mean(dim=-1)
                )
            sparse_signal = torch.sigmoid(
                cfg.event_q_sparse_alpha * sparse_excess
            )
            sparse_support = event_temporal_consistency
            if cfg.event_q_sparse_spatial:
                sparse_support = sparse_support * event_spatial_support
            sparse_event_score = (
                sparse_signal * sparse_support
            ).clamp(min=0.0, max=1.0)

        log_drift_ratio = torch.log(drift_ratio.clamp_min(cfg.eps))
        motion_excess = (
            log_drift_ratio - cfg.event_q_motion_tau
        ).clamp_min(0.0)
        motion_signal = (
            motion_excess
            * torch.sigmoid(
                cfg.event_q_log_alpha
                * (log_drift_ratio - cfg.event_q_motion_tau)
            )
        ).clamp(min=0.0, max=1.0)
        if cfg.event_q_evidence_mode == "temporal":
            motion_support = event_temporal_consistency
        elif cfg.event_q_evidence_mode == "spatial":
            motion_support = (
                event_temporal_consistency * event_spatial_support
            )
        elif cfg.event_q_evidence_mode == "innovation":
            innovation_support = torch.sigmoid(
                cfg.event_q_innovation_alpha
                * (
                    normalized_innovation
                    - cfg.event_q_innovation_tau
                )
            )
            motion_support = event_temporal_consistency * innovation_support
        else:
            motion_support = (
                event_temporal_consistency * event_spatial_support
            )
        event_motion_score = (
            motion_signal * motion_support
        ).clamp(min=0.0, max=1.0)
        if cfg.event_q_sparse_weight > 0.0:
            event_motion_score = (
                (1.0 - cfg.event_q_sparse_weight) * event_motion_score
                + cfg.event_q_sparse_weight * sparse_event_score
            ).clamp(min=0.0, max=1.0)

        previous_motion_count = (
            torch.zeros_like(event_motion_score, dtype=torch.int64)
            if state.event_motion_count is None
            else state.event_motion_count
        )
        motion_candidate = event_motion_score > 0.0
        event_motion_count = torch.where(
            motion_candidate,
            (previous_motion_count + 1).clamp(
                max=cfg.event_q_motion_confirm
            ),
            torch.zeros_like(previous_motion_count),
        )
        if cfg.event_q_motion_confirm > 1:
            event_motion_score = torch.where(
                event_motion_count >= cfg.event_q_motion_confirm,
                event_motion_score,
                torch.zeros_like(event_motion_score),
            )

        if cfg.event_q_evidence_mode == "guarded":
            innovation_support = torch.sigmoid(
                cfg.event_q_innovation_alpha
                * (normalized_innovation - cfg.event_q_innovation_tau)
            )
            jitter_signal = torch.sigmoid(
                cfg.event_q_log_alpha
                * (log_drift_ratio - cfg.event_q_jitter_tau)
            )
            jitter_candidate = (
                (log_drift_ratio > cfg.event_q_jitter_tau)
                & (normalized_innovation >= cfg.event_q_innovation_tau)
                & (event_temporal_consistency <= cfg.event_q_temporal_tau)
                & (event_spatial_support <= cfg.event_q_spatial_tau)
                & (coherence_history_valid > 0.0)
            )
            previous_count = (
                torch.zeros_like(jitter_candidate, dtype=torch.int64)
                if state.event_jitter_count is None
                else state.event_jitter_count
            )
            confirmed_jitter = jitter_candidate & (previous_count >= 1)
            event_jitter_count = torch.where(
                jitter_candidate,
                (previous_count + 1).clamp(max=2),
                torch.zeros_like(previous_count),
            )
            event_jitter_score = (
                confirmed_jitter.to(normalized_innovation.dtype)
                * jitter_signal
                * innovation_support
                * (1.0 - event_temporal_consistency)
                * (1.0 - event_spatial_support)
            ).clamp(min=0.0, max=1.0)
        else:
            event_jitter_count = None
            event_jitter_score = torch.zeros_like(event_motion_score)

        adaptive_r = A1UnifiedTokenKalmanFilter._adaptive_r_update(
            candidate=candidate,
            innovation_vector=innovation_vector,
            covariance=state.covariance,
            reference_r=reference_r,
            normalized_innovation=normalized_innovation,
            jitter_score=jitter_score,
            drift_coherence=drift_coherence,
            magnitude_coherence=raw_magnitude_coherence,
            coherence_history_valid=coherence_history_valid,
            spatial_support=event_spatial_support,
            state=state,
            cfg=cfg,
        )

        return {
            "r_t": adaptive_r["r_t"],
            "prediction": prediction,
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
            "event_motion_score": event_motion_score,
            "event_jitter_score": event_jitter_score,
            "event_jitter_count": event_jitter_count,
            "sparse_event_score": sparse_event_score,
            "event_motion_count": event_motion_count,
            "drift_coherence": drift_coherence,
            "coherence_history_valid": coherence_history_valid,
            "magnitude_coherence": raw_magnitude_coherence,
            "raw_magnitude_coherence": raw_magnitude_coherence,
            "spatial_coherence": event_spatial_support,
            "jitter_score": jitter_score,
            # gbw____
            "adaptive_r_evidence": adaptive_r["adaptive_r_evidence"],
            "adaptive_r_score": adaptive_r["adaptive_r_score"],
            "adaptive_r_ema": adaptive_r["adaptive_r_ema"],
            "adaptive_r_count": adaptive_r["adaptive_r_count"],
            "adaptive_r_active": adaptive_r["adaptive_r_active"],
            # ____
        }
    # ____
    # gbw____
    @staticmethod
    def _kalman_update(
        metrics: dict,
        state: _TokenState,
        cfg: Filt3rAKFConfig,
    ) -> dict:
        """Update the active event-neutral diagonal Kalman model."""

        q_scale = (
            cfg.event_q_reference_ratio
            * torch.exp(
                cfg.event_q_motion_strength * metrics["event_motion_score"]
                - cfg.event_q_jitter_strength * metrics["event_jitter_score"]
            )
        ).clamp(
            min=cfg.event_q_min_ratio,
            max=cfg.event_q_max_ratio,
        )
        # gbw____
        # Independent A3 R evidence can veto a simultaneous Q boost. This is
        # a safety partition, not a second Q search: Q remains measured from
        # the frozen A2 anchor and is only clipped back to its nominal ratio
        # when the current observation is confirmed unreliable.
        q_conflict_veto = (
            cfg.q_conflict_veto
            & (metrics["adaptive_r_active"])
        )
        q_scale = torch.where(
            q_conflict_veto,
            torch.minimum(
                q_scale,
                torch.full_like(q_scale, cfg.event_q_reference_ratio),
            ),
            q_scale,
        )
        # ____
        q_t = torch.full_like(metrics["r_t"], cfg.r) * q_scale.unsqueeze(-1)
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
            "q_conflict_veto": q_conflict_veto,
            # ____
        }
    # ____
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
        # reset() 清理 episode 状态；单帧 reset 只在显式 reset_mode=current
        # 时生效，其他模式保留因果状态。
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
        # innovation_coherent 需要异常 innovation 与历史方向一致。
        if cfg.reset_mode == "innovation_coherent":
            local_reset = local_reset & (
                metrics["drift_coherence"] >= cfg.reset_coherence_tau
            ) & (metrics["coherence_history_valid"] > 0.0)
        # gbw____
        # event_coherent 只接收持续 motion、低 jitter 和高 innovation。
        # ____
        if cfg.reset_mode == "event_coherent":
            local_reset = local_reset & (
                metrics["event_motion_score"]
                >= cfg.event_reset_motion_tau
            ) & (
                metrics["event_jitter_score"]
                <= cfg.event_reset_jitter_tau
            ) & (
                metrics["normalized_innovation"]
                >= cfg.event_reset_innovation_tau
            ) & (metrics["coherence_history_valid"] > 0.0)
            camera_reset_map = torch.zeros_like(camera_reset_map)
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

    def apply(
        self,
        tokens: torch.Tensor,
        stage: int,
        decoder_gate: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict]
        ] = None,
    ) -> torch.Tensor:
        """Apply the active causal update before ConvexUpSample."""

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
                drift_ema=None,
                previous_delta=None,
                event_jitter_count=None,
                event_motion_count=None,
                # gbw____
                adaptive_r_ema=None,
                adaptive_r_count=None,
                adaptive_r_active=None,
                # ____
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
        covariance = torch.clamp(covariance, min=cfg.eps)
        token_shift_ratio = torch.linalg.vector_norm(
            filtered - candidate, dim=-1
        ) / (torch.linalg.vector_norm(candidate, dim=-1) + cfg.eps)
        # gbw____
        # decoder gate 在 proposal 提交前保护异常 token；alpha=1 保持 proposal。
        # ____
        decoder_protection_alpha = None
        decoder_protection_mask = None
        decoder_raw_margin = None
        decoder_filtered_margin = None
        decoder_peak_shift = None
        decoder_waypoint_shift = None
        if decoder_gate is not None:
            gate_allowed = (
                ~reset_mask
                & ~outlier_gate
            )
            gate_info = decoder_gate(
                candidate.detach(), filtered.detach(), gate_allowed.detach()
            )
            if not isinstance(gate_info, dict):
                raise TypeError("decoder_gate must return a dict")
            alpha = gate_info.get("alpha")
            if not isinstance(alpha, torch.Tensor):
                raise TypeError("decoder_gate result must contain tensor alpha")
            if tuple(alpha.shape) != tuple(candidate.shape[:-1]):
                raise ValueError(
                    "decoder_gate alpha must have token shape "
                    f"{tuple(candidate.shape[:-1])}, got {tuple(alpha.shape)}"
                )
            if not torch.isfinite(alpha).all():
                raise FloatingPointError("decoder_gate alpha must be finite")
            alpha = alpha.detach().float().clamp(min=0.0, max=1.0)
            alpha = torch.where(gate_allowed, alpha, torch.ones_like(alpha))
            alpha_expand = alpha.unsqueeze(-1)
            proposal = filtered
            filtered = candidate + alpha_expand * (proposal - candidate)
            proposal_gain = effective_gain
            gated_gain = 1.0 - alpha_expand * (1.0 - proposal_gain)
            gated_covariance = (
                (1.0 - gated_gain).square() * kalman["p_prior"]
                + gated_gain.square() * metrics["r_t"]
            )
            allowed_expand = gate_allowed.unsqueeze(-1)
            covariance = torch.where(
                allowed_expand, gated_covariance, covariance
            ).clamp_min(cfg.eps)
            effective_gain = torch.where(
                allowed_expand, gated_gain, effective_gain
            )
            decoder_protection_alpha = alpha
            decoder_protection_mask = gate_info.get(
                "protection_mask", alpha < (1.0 - 1e-6)
            )
            if not isinstance(decoder_protection_mask, torch.Tensor):
                raise TypeError("decoder_gate protection_mask must be a tensor")
            decoder_protection_mask = decoder_protection_mask.detach().bool()
            decoder_raw_margin = gate_info.get("raw_margin")
            decoder_filtered_margin = gate_info.get("filtered_margin")
            decoder_peak_shift = gate_info.get("peak_shift")
            decoder_waypoint_shift = gate_info.get("waypoint_shift")
        # ____

        # gbw____
        # 诊断比较实际返回给 ConvexUpSample 的 token，而不是中间 proposal。
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
            drift_ema=metrics["drift_ema"].detach(),
            previous_delta=metrics["delta_vector"].detach(),
            event_jitter_count=(
                None
                if metrics.get("event_jitter_count") is None
                else metrics["event_jitter_count"].detach()
            ),
            event_motion_count=(
                None
                if metrics.get("event_motion_count") is None
                else metrics["event_motion_count"].detach()
            ),
            # gbw____
            adaptive_r_ema=(
                None
                if metrics.get("adaptive_r_ema") is None
                else metrics["adaptive_r_ema"].detach()
            ),
            adaptive_r_count=(
                None
                if metrics.get("adaptive_r_count") is None
                else metrics["adaptive_r_count"].detach()
            ),
            adaptive_r_active=(
                None
                if metrics.get("adaptive_r_active") is None
                else metrics["adaptive_r_active"].detach()
            ),
            # ____
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
            local_score=metrics["local_score"],
            camera_score=metrics["camera_score"],
            drift_ema=metrics["drift_ema"],
            drift_ratio=metrics["drift_ratio"],
            event_motion_score=metrics["event_motion_score"],
            event_jitter_score=metrics["event_jitter_score"],
            sparse_event_score=metrics["sparse_event_score"],
            event_jitter_count=metrics["event_jitter_count"],
            drift_coherence=metrics["drift_coherence"],
            magnitude_coherence=metrics["magnitude_coherence"],
            raw_magnitude_coherence=metrics["raw_magnitude_coherence"],
            spatial_coherence=metrics["spatial_coherence"],
            jitter_score=metrics["jitter_score"],
            q_scale=kalman["q_scale"],
            q_t=kalman["q_t"],
            p_prior=kalman["p_prior"],
            gain=effective_gain,
            covariance=covariance,
            measurement_r=metrics["r_t"],
            # gbw____
            adaptive_r_evidence=metrics["adaptive_r_evidence"],
            adaptive_r_score=metrics["adaptive_r_score"],
            adaptive_r_scale=(metrics["r_t"][..., 0] / cfg.r),
            adaptive_r_active=metrics["adaptive_r_active"],
            adaptive_r_count=metrics["adaptive_r_count"],
            q_conflict_veto=kalman["q_conflict_veto"],
            # ____
            normalized_innovation=metrics["normalized_innovation"],
            filtered_residual_ratio=filtered_residual_ratio,
            token_shift_ratio=token_shift_ratio,
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
            decoder_protection_alpha=decoder_protection_alpha,
            decoder_protection_mask=decoder_protection_mask,
            decoder_raw_margin=decoder_raw_margin,
            decoder_filtered_margin=decoder_filtered_margin,
            decoder_peak_shift=decoder_peak_shift,
            decoder_waypoint_shift=decoder_waypoint_shift,
        )
        return returned_tokens
    # ____
