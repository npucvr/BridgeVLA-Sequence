"""History-aware losses for current-token Stage-1 correction."""

import torch
import torch.nn.functional as F


def masked_history_cosine_loss(current_tokens, history_tokens, history_mask):
    """Match corrected current tokens to a causal historical token target.

    Args:
        current_tokens: Corrected current tokens, ``[B, S, D]``.
        history_tokens: Cached preceding tokens, ``[B, H, S, D]``.
        history_mask: Valid-history mask, ``[B, H]``.

    The history target is detached and averaged only over valid preceding
    observations. Samples without valid history contribute a differentiable
    zero, so callers can safely use mixed batches at episode boundaries.
    """
    if current_tokens.ndim != 3:
        raise ValueError(
            "current_tokens must have shape [B, S, D], "
            f"got {tuple(current_tokens.shape)}"
        )
    if history_tokens.ndim != 4:
        raise ValueError(
            "history_tokens must have shape [B, H, S, D], "
            f"got {tuple(history_tokens.shape)}"
        )
    if history_mask.ndim != 2:
        raise ValueError(
            "history_mask must have shape [B, H], "
            f"got {tuple(history_mask.shape)}"
        )
    batch_size, sequence, token_dim = current_tokens.shape
    if history_tokens.shape[0] != batch_size:
        raise ValueError("current/history batch sizes do not match")
    if history_tokens.shape[2:] != (sequence, token_dim):
        raise ValueError(
            "current/history token shapes do not match: "
            f"current={tuple(current_tokens.shape)}, "
            f"history={tuple(history_tokens.shape)}"
        )
    if history_mask.shape != history_tokens.shape[:2]:
        raise ValueError(
            "history_mask shape does not match history_tokens: "
            f"mask={tuple(history_mask.shape)}, "
            f"history={tuple(history_tokens.shape)}"
        )

    # Keep a zero connected to the correction graph when no past observation
    # exists (e.g. the first frame of an episode).
    if history_tokens.shape[1] == 0:
        return current_tokens.sum() * 0.0

    valid = history_mask.to(device=current_tokens.device, dtype=current_tokens.dtype)
    counts = valid.sum(dim=1)
    has_history = counts > 0
    if not has_history.any():
        return current_tokens.sum() * 0.0

    history = history_tokens.to(
        device=current_tokens.device,
        dtype=current_tokens.dtype,
    ).detach()
    target = (
        history * valid[:, :, None, None]
    ).sum(dim=1) / counts.clamp_min(1.0)[:, None, None]

    current_norm = F.normalize(current_tokens.float(), dim=-1)
    target_norm = F.normalize(target.float(), dim=-1)
    per_sample = (1.0 - (current_norm * target_norm).sum(dim=-1)).mean(dim=-1)
    return per_sample[has_history].mean()
