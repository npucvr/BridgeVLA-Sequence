"""Small LoRA adapters for the legacy discrete action classifiers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` with a zero-initialized low-rank weight update.

    The base ``weight`` and ``bias`` retain their original state-dict names so
    an existing BridgeVLA checkpoint can be loaded before adapters are added.
    """

    def __init__(self, in_features, out_features, bias=True, rank=8, alpha=16):
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        super().__init__(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(self.weight.new_empty(self.rank, in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @classmethod
    def from_linear(cls, linear, rank=8, alpha=16):
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        return wrapped

    def forward(self, input):
        base = F.linear(input, self.weight, self.bias)
        lora_input = input.to(dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        return base + delta.to(dtype=base.dtype) * self.scaling


def inject_discrete_action_lora(mvt, rank=8, alpha=16):
    """Attach LoRA to the final projections of the discrete pose/action heads."""
    if int(getattr(mvt, "rot_ver", -1)) != 1:
        raise ValueError("discrete action LoRA requires mvt.rot_ver=1")
    if bool(getattr(mvt, "continuous_rotation", False)):
        raise ValueError("discrete action LoRA cannot be used with continuous rotation")

    target_names = ("feat_fc_x", "feat_fc_y", "feat_fc_z", "feat_fc_ex_rot")
    injected = []
    for name in target_names:
        head = getattr(mvt, name, None)
        if not isinstance(head, nn.Sequential) or len(head) < 1:
            raise ValueError(f"discrete action head {name} is missing")
        layer_index = len(head) - 1
        final = head[layer_index]
        if not isinstance(final, nn.Linear):
            raise TypeError(f"{name}[{layer_index}] must be nn.Linear")
        head[layer_index] = LoRALinear.from_linear(
            final, rank=rank, alpha=alpha
        )
        injected.append(f"{name}.{layer_index}")
    return tuple(injected)


def merged_lora_state_dict(model):
    """Return an evaluation-compatible state dict with LoRA deltas merged."""
    state = model.state_dict()
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not lora_modules:
        return state

    state = dict(state)
    for name, module in lora_modules:
        prefix = f"{name}." if name else ""
        weight_key = f"{prefix}weight"
        a_key = f"{prefix}lora_A"
        b_key = f"{prefix}lora_B"
        delta = module.lora_B @ module.lora_A
        state[weight_key] = state[weight_key] + delta.to(
            dtype=state[weight_key].dtype
        ) * module.scaling
        state.pop(a_key, None)
        state.pop(b_key, None)
    return state
