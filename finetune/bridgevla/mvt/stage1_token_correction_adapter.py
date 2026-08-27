"""Lightweight current-token correction conditioned on a hidden state."""

from torch import nn
from torch.nn import functional as F


class Stage1TokenCorrectionAdapter(nn.Module):
    """Apply a small residual correction to current observation tokens.

    Args:
        token_dim: Width of each PaliGemma token.
        bottleneck_dim: Width of the trainable residual bottleneck.
        hidden_state_dim: Width of the optional episode-local hidden state.

    The adapter accepts only the current token tensor and one compact hidden
    state. It never concatenates historical token windows with the current
    observation.
    """

    def __init__(self, token_dim=2048, bottleneck_dim=128, hidden_state_dim=0):
        super().__init__()
        if token_dim < 1:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if bottleneck_dim < 1:
            raise ValueError(
                f"bottleneck_dim must be positive, got {bottleneck_dim}"
            )
        if hidden_state_dim < 0:
            raise ValueError(
                f"hidden_state_dim must be non-negative, got {hidden_state_dim}"
            )

        self.token_dim = int(token_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_state_dim = int(hidden_state_dim)
        self.norm = nn.LayerNorm(self.token_dim)
        self.down = nn.Linear(self.token_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, self.token_dim)
        self.hidden_to_bottleneck = (
            nn.Linear(self.hidden_state_dim, self.bottleneck_dim)
            if self.hidden_state_dim > 0
            else None
        )

        # Preserve the released model's behavior at the start of fine-tuning.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if self.hidden_to_bottleneck is not None:
            # The hidden-state branch is also initially a no-op, including for
            # a zero episode state and for old checkpoints loaded without it.
            nn.init.zeros_(self.hidden_to_bottleneck.weight)
            nn.init.zeros_(self.hidden_to_bottleneck.bias)

    def forward(self, current_tokens, hidden_state=None):
        """Return corrected current tokens with the same shape as the input."""
        if current_tokens.ndim != 3:
            raise ValueError(
                "current_tokens must have shape [B, S, D]; historical token "
                f"windows are not accepted, got {tuple(current_tokens.shape)}"
            )
        if current_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected token dim {self.token_dim}, "
                f"got {current_tokens.shape[-1]}"
            )
        if self.hidden_state_dim == 0 and hidden_state is not None:
            raise ValueError("hidden state conditioning is disabled for this adapter")

        # PaliGemma tokens are usually bfloat16; keep the small trainable
        # correction path in float32 for stable updates.
        current_tokens = current_tokens.to(dtype=self.norm.weight.dtype)
        bottleneck = self.down(self.norm(current_tokens))

        if self.hidden_state_dim > 0:
            if hidden_state is None:
                hidden_state = current_tokens.new_zeros(
                    current_tokens.shape[0], self.hidden_state_dim
                )
            if hidden_state.ndim != 2:
                raise ValueError(
                    "hidden_state must have shape [B, hidden_state_dim], "
                    f"got {tuple(hidden_state.shape)}"
                )
            if hidden_state.shape != (
                current_tokens.shape[0],
                self.hidden_state_dim,
            ):
                raise ValueError(
                    "hidden_state shape does not match current tokens: "
                    f"hidden={tuple(hidden_state.shape)}, "
                    f"current={tuple(current_tokens.shape)}"
                )
            hidden_state = hidden_state.to(
                device=current_tokens.device,
                dtype=bottleneck.dtype,
            )
            bottleneck = bottleneck + self.hidden_to_bottleneck(hidden_state).unsqueeze(1)

        correction = self.up(F.gelu(bottleneck))
        return current_tokens + correction
