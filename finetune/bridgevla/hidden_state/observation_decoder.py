"""Causal prior-observation decoder for the hidden-state route.

The decoder models a compact observation target from the pre-observation
hidden state ``y_t^-``.  It is intentionally separate from ``U_omega``:
current visual tokens are targets only and never inputs to this module.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F



def pool_visual_tokens(current_tokens, num_views):
    """Return a normalized per-view pooled visual-token target.

    Args:
        current_tokens: Tensor with shape ``[B, num_views * S, D]``.
        num_views: Number of rendered camera views in the token sequence.

    Returns:
        Tensor with shape ``[B, num_views * D]``.  Each view is pooled over
        its patch tokens and normalized independently.  The caller decides
        whether to detach the target from the visual encoder graph.
    """
    if not isinstance(current_tokens, torch.Tensor) or current_tokens.ndim != 3:
        raise ValueError(
            "current_tokens must be a tensor with shape [B, S, D], "
            f"got {type(current_tokens).__name__}"
        )
    if int(num_views) < 1:
        raise ValueError(f"num_views must be positive, got {num_views}")
    num_views = int(num_views)
    batch_size, sequence_size, token_dim = current_tokens.shape
    if sequence_size % num_views != 0:
        raise ValueError(
            "visual token sequence length must be divisible by num_views: "
            f"sequence_size={sequence_size}, num_views={num_views}"
        )

    tokens = current_tokens.float().reshape(
        batch_size, num_views, sequence_size // num_views, token_dim
    )
    per_view = tokens.mean(dim=2)
    per_view = F.normalize(per_view, dim=-1)
    return per_view.reshape(batch_size, num_views * token_dim)


class ObservationDecoder(nn.Module):
    """Diagonal-Gaussian decoder ``p(target | y_t^-)``.

    The mean depends only on the prior hidden state.  A learned global
    diagonal log-scale gives the auxiliary objective a likelihood
    interpretation while keeping the output head small and stable.
    """

    def __init__(
        self,
        hidden_dim=128,
        target_dim=2048,
        decoder_hidden_dim=256,
        min_log_scale=-5.0,
        max_log_scale=3.0,
    ):
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(target_dim) < 1:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        if int(decoder_hidden_dim) < 1:
            raise ValueError(
                "decoder_hidden_dim must be positive, "
                f"got {decoder_hidden_dim}"
            )
        if float(min_log_scale) >= float(max_log_scale):
            raise ValueError(
                "min_log_scale must be smaller than max_log_scale, got "
                f"{min_log_scale} >= {max_log_scale}"
            )

        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_dim, self.target_dim),
        )
        self.log_scale = nn.Parameter(torch.zeros(self.target_dim))

    def forward(self, prior_hidden_state, target=None):
        """Predict the target from ``y_t^-`` only.

        When ``target`` is provided, the NLL is computed inside this forward
        call.  This keeps the learned scale parameter inside the enclosing
        DDP forward graph during chronological multi-step training.
        """
        if not isinstance(prior_hidden_state, torch.Tensor):
            raise ValueError("prior_hidden_state must be a tensor")
        if prior_hidden_state.ndim != 2:
            raise ValueError(
                "prior_hidden_state must have shape [B, hidden_dim], got "
                f"{tuple(prior_hidden_state.shape)}"
            )
        if prior_hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                "prior_hidden_state has the wrong width: "
                f"expected {self.hidden_dim}, got {prior_hidden_state.shape[-1]}"
            )
        parameter = next(self.parameters())
        prior_hidden_state = prior_hidden_state.to(
            device=parameter.device, dtype=parameter.dtype
        )
        prediction = self.network(prior_hidden_state)
        if target is None:
            return prediction
        return prediction, self.nll(prediction, target)

    def nll(self, prediction, target):
        """Return per-sample diagonal-Gaussian negative log likelihood."""
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 2:
            raise ValueError(
                "prediction must have shape [B, target_dim], got "
                f"{tuple(prediction.shape) if isinstance(prediction, torch.Tensor) else type(prediction).__name__}"
            )
        if prediction.shape[-1] != self.target_dim:
            raise ValueError(
                "prediction has the wrong width: "
                f"expected {self.target_dim}, got {prediction.shape[-1]}"
            )
        if not isinstance(target, torch.Tensor) or target.shape != prediction.shape:
            raise ValueError(
                "target must have the same shape as prediction: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape) if isinstance(target, torch.Tensor) else type(target).__name__}"
            )

        target = target.to(device=prediction.device, dtype=prediction.dtype)
        log_scale = self.log_scale.to(
            device=prediction.device, dtype=prediction.dtype
        ).clamp(self.min_log_scale, self.max_log_scale)
        standardized_error = (target - prediction) * torch.exp(-log_scale)
        per_dimension = 0.5 * (
            standardized_error.square()
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        )
        return per_dimension.mean(dim=-1)
