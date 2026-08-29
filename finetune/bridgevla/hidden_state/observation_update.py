"""Observation-conditioned hidden-state update :math:`U_\\omega(y_t^-, H_t)`."""

import torch
from torch import nn


class U_omega(nn.Module):
    """Update a predicted hidden state from current PaliGemma tokens.

    The module treats the PaliGemma token sequence as a learned observation
    representation.  A single query derived from the predicted hidden state
    attends to the current tokens, and a GRUCell applies the resulting
    observation summary as a measurement update.  The token reduction is
    deliberately kept inside this module so the public route does not commit
    to a particular pooling or projection strategy.

    Args:
        token_dim: Width of each PaliGemma token.
        hidden_state_dim: Width of the recurrent hidden state.
        num_heads: Number of attention heads used by the observation update.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        token_dim=2048,
        hidden_state_dim=128,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        if token_dim < 1:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if hidden_state_dim < 1:
            raise ValueError(
                f"hidden_state_dim must be positive, got {hidden_state_dim}"
            )
        if num_heads < 1 or hidden_state_dim % num_heads != 0:
            raise ValueError(
                "hidden_state_dim must be divisible by num_heads, got "
                f"hidden_state_dim={hidden_state_dim}, num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.token_dim = int(token_dim)
        self.hidden_state_dim = int(hidden_state_dim)
        self.num_heads = int(num_heads)

        self.token_norm = nn.LayerNorm(self.token_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_state_dim)
        self.observation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_state_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
            kdim=self.token_dim,
            vdim=self.token_dim,
        )
        self.update = nn.GRUCell(
            input_size=self.hidden_state_dim,
            hidden_size=self.hidden_state_dim,
        )

    def initial_hidden_state(self, batch_size, device=None, dtype=None):
        """Return the zero prior used at the beginning of a sequence."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_state_dim,
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def forward(self, hidden_state_y, current_tokens):
        """Return the posterior hidden state for the current observation.

        Args:
            hidden_state_y: Predicted state with shape ``[B, hidden_state_dim]``.
                ``None`` means a zero prior.
            current_tokens: PaliGemma tokens with shape ``[B, S, token_dim]``.
        """
        if current_tokens.ndim != 3:
            raise ValueError(
                "current_tokens must have shape [B, S, D]; "
                f"got {tuple(current_tokens.shape)}"
            )
        if current_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected token dim {self.token_dim}, "
                f"got {current_tokens.shape[-1]}"
            )

        parameter = next(self.parameters())
        batch_size = current_tokens.shape[0]
        if hidden_state_y is None:
            hidden_state_y = self.initial_hidden_state(
                batch_size,
                device=current_tokens.device,
                dtype=parameter.dtype,
            )
        else:
            if hidden_state_y.ndim != 2:
                raise ValueError(
                    "hidden_state_y must have shape [B, hidden_state_dim], "
                    f"got {tuple(hidden_state_y.shape)}"
                )
            if hidden_state_y.shape != (batch_size, self.hidden_state_dim):
                raise ValueError(
                    "hidden_state_y shape does not match current tokens: "
                    f"hidden={tuple(hidden_state_y.shape)}, "
                    f"current={tuple(current_tokens.shape)}"
                )
            hidden_state_y = hidden_state_y.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

        current_tokens = current_tokens.to(
            device=parameter.device,
            dtype=self.token_norm.weight.dtype,
        )
        token_features = self.token_norm(current_tokens)
        query = self.hidden_norm(hidden_state_y).unsqueeze(1)
        context, _ = self.observation_attention(
            query=query,
            key=token_features,
            value=token_features,
            need_weights=False,
        )
        return self.update(context.squeeze(1), hidden_state_y)
