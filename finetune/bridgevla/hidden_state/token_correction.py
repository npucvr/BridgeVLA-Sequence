"""The token correction module :math:`A_\\psi(H_t, y_t)`."""

from torch import nn
from torch.nn import functional as F


class A_psi(nn.Module):
    """Apply :math:`A_\\psi` to the current observation tokens.

    Args:
        token_dim: Width of each PaliGemma token.
        token_bottleneck_dim: Width of the trainable residual bottleneck.
        hidden_state_dim: Width of the episode-local hidden state.

    Only the current token sequence and the current hidden state are consumed;
    no historical token window is part of this module's interface.
    """

    def __init__(
        self, token_dim=2048, token_bottleneck_dim=128, hidden_state_dim=0
    ):
        super().__init__()
        if token_dim < 1:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if token_bottleneck_dim < 1:
            raise ValueError(
                "token_bottleneck_dim must be positive, "
                f"got {token_bottleneck_dim}"
            )
        if hidden_state_dim < 0:
            raise ValueError(
                f"hidden_state_dim must be non-negative, got {hidden_state_dim}"
            )

        self.token_dim = int(token_dim)
        self.token_bottleneck_dim = int(token_bottleneck_dim)
        self.hidden_state_dim = int(hidden_state_dim)
        self.norm = nn.LayerNorm(self.token_dim)
        self.down = nn.Linear(self.token_dim, self.token_bottleneck_dim)
        self.up = nn.Linear(self.token_bottleneck_dim, self.token_dim)
        self.hidden_to_bottleneck = (
            nn.Linear(self.hidden_state_dim, self.token_bottleneck_dim)
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

    def forward(self, current_tokens, hidden_state_y=None):
        """Return corrected current tokens with the same shape as the input."""
        if current_tokens.ndim != 3:
            raise ValueError(
                "current_tokens must have shape [B, S, D]; token windows are "
                f"not accepted, got {tuple(current_tokens.shape)}"
            )
        if current_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected token dim {self.token_dim}, "
                f"got {current_tokens.shape[-1]}"
            )
        if self.hidden_state_dim == 0 and hidden_state_y is not None:
            raise ValueError("hidden state conditioning is disabled for A_psi")

        # PaliGemma tokens are usually bfloat16; keep the small trainable
        # correction path in float32 for stable updates.
        current_tokens = current_tokens.to(dtype=self.norm.weight.dtype)
        bottleneck = self.down(self.norm(current_tokens))

        if self.hidden_state_dim > 0:
            if hidden_state_y is None:
                hidden_state_y = current_tokens.new_zeros(
                    current_tokens.shape[0], self.hidden_state_dim
                )
            if hidden_state_y.ndim != 2:
                raise ValueError(
                    "hidden_state_y must have shape [B, hidden_state_dim], "
                    f"got {tuple(hidden_state_y.shape)}"
                )
            if hidden_state_y.shape != (
                current_tokens.shape[0],
                self.hidden_state_dim,
            ):
                raise ValueError(
                    "hidden_state_y shape does not match current tokens: "
                    f"hidden={tuple(hidden_state_y.shape)}, "
                    f"current={tuple(current_tokens.shape)}"
                )
            hidden_state_y = hidden_state_y.to(
                device=current_tokens.device,
                dtype=bottleneck.dtype,
            )
            bottleneck = bottleneck + self.hidden_to_bottleneck(
                hidden_state_y
            ).unsqueeze(1)

        correction = self.up(F.gelu(bottleneck))
        return current_tokens + correction
