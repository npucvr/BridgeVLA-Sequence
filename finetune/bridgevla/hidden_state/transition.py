"""The hidden-state transition module :math:`F_\\phi(y_t, u_t)`."""

import torch
from torch import nn


class F_phi(nn.Module):
    """Update the policy hidden state after one waypoint action.

    The state is an episode-local runtime value. The module parameters are
    shared across decision steps and are trained through the downstream action
    loss when a sequence is unrolled.
    """

    def __init__(self, hidden_dim=128, action_dim=8):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.transition = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def initial_hidden_state(self, batch_size, device=None, dtype=None):
        """Return the zero state used at the beginning of an episode."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def forward(self, hidden_state_y, action):
        """Return the next state for ``action``.

        Args:
            hidden_state_y: ``[B, hidden_dim]`` or ``None`` for a zero state.
            action: ``[B, action_dim]`` waypoint action.
        """
        if action.ndim != 2:
            raise ValueError(
                "action must have shape [B, action_dim], "
                f"got {tuple(action.shape)}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action dim {self.action_dim}, got {action.shape[-1]}"
            )

        parameter = next(self.parameters())
        action = action.to(device=parameter.device, dtype=parameter.dtype)
        if hidden_state_y is None:
            hidden_state_y = self.initial_hidden_state(
                action.shape[0], device=parameter.device, dtype=parameter.dtype
            )
        else:
            if hidden_state_y.ndim != 2:
                raise ValueError(
                    "hidden_state_y must have shape [B, hidden_dim], "
                    f"got {tuple(hidden_state_y.shape)}"
                )
            if hidden_state_y.shape != (action.shape[0], self.hidden_dim):
                raise ValueError(
                    "hidden_state_y shape does not match action: "
                    f"hidden={tuple(hidden_state_y.shape)}, "
                    f"action={tuple(action.shape)}"
                )
            hidden_state_y = hidden_state_y.to(
                device=parameter.device, dtype=parameter.dtype
            )

        action_embedding = self.action_encoder(action)
        return self.transition(action_embedding, hidden_state_y)
