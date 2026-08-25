"""Small temporal adapter for frozen Stage-1 PaliGemma visual tokens."""

import math

import torch
from torch import nn


class Stage1TemporalTokenAdapter(nn.Module):
    """Fuse a short causal window of Stage-1 visual tokens.

    Args:
        token_dim: Dimension of one PaliGemma visual token.
        bottleneck_dim: Internal temporal-fusion dimension.
        max_history: Maximum number of tokens in the time window, including
            the current token.

    Input tokens have shape ``[B, K, S, D]`` where K is ordered from oldest to
    newest and S is the flattened view/patch dimension. The last token is the
    current observation. ``valid_mask`` has shape ``[B, K]`` and masks padded
    history at the beginning of an episode.
    """

    def __init__(self, token_dim=2048, bottleneck_dim=128, max_history=4):
        super().__init__()
        if max_history < 1:
            raise ValueError(f"max_history must be >= 1, got {max_history}")
        self.token_dim = token_dim
        self.bottleneck_dim = bottleneck_dim
        self.max_history = max_history

        self.norm = nn.LayerNorm(token_dim)
        self.down = nn.Linear(token_dim, bottleneck_dim)
        self.time_embedding = nn.Parameter(
            torch.zeros(max_history, bottleneck_dim)
        )
        self.up = nn.Linear(bottleneck_dim, token_dim)

        # Start as an exact identity adapter. This preserves the released
        # checkpoint's behavior before the first optimization step.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, valid_mask=None):
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 4:
            raise ValueError(
                "tokens must have shape [B, K, S, D] or [B, S, D], "
                f"got {tuple(tokens.shape)}"
            )

        batch_size, history, sequence, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(
                f"expected token dim {self.token_dim}, got {token_dim}"
            )
        if history > self.max_history:
            raise ValueError(
                f"history length {history} exceeds max_history {self.max_history}"
            )

        # PaliGemma normally emits bfloat16 tokens while the small adapter is
        # kept in float32 for stable updates.
        tokens = tokens.to(dtype=self.norm.weight.dtype)

        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size,
                history,
                dtype=torch.bool,
                device=tokens.device,
            )
        else:
            if valid_mask.shape != (batch_size, history):
                raise ValueError(
                    "valid_mask must have shape "
                    f"[{batch_size}, {history}], got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)

        # The newest token is always the current observation. Keeping this
        # assertion explicit prevents accidentally training with future tokens.
        if not valid_mask[:, -1].all():
            raise ValueError("the current token must be valid for every sample")

        z = self.down(self.norm(tokens))
        z = z + self.time_embedding[:history].view(1, history, 1, -1)

        query = z[:, -1:]  # [B, 1, S, R]
        scores = (query * z).sum(dim=-1) / math.sqrt(self.bottleneck_dim)
        scores = scores.masked_fill(
            ~valid_mask[:, :, None], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * z).sum(dim=1)

        return tokens[:, -1] + self.up(torch.nn.functional.gelu(context))
