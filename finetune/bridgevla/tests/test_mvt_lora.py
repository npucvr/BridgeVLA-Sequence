import unittest

import torch
from torch import nn
from torch.nn import functional as F

from bridgevla.mvt.lora import (
    LoRALinear,
    inject_discrete_action_lora,
    merged_lora_state_dict,
)


class DiscreteActionHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot_ver = 1
        self.continuous_rotation = False
        for name, output_size in (
            ("feat_fc_x", 72),
            ("feat_fc_y", 72),
            ("feat_fc_z", 72),
            ("feat_fc_ex_rot", 4),
        ):
            setattr(
                self,
                name,
                nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, output_size)),
            )


class DiscreteActionLoRATests(unittest.TestCase):
    def test_injection_starts_as_exact_identity_and_only_adapters_train(self):
        torch.manual_seed(7)
        model = DiscreteActionHeads().eval()
        inputs = torch.randn(3, 5)
        baseline = {
            name: getattr(model, name)(inputs).detach().clone()
            for name in ("feat_fc_x", "feat_fc_y", "feat_fc_z", "feat_fc_ex_rot")
        }

        targets = inject_discrete_action_lora(model, rank=2, alpha=4)
        self.assertEqual(
            targets,
            ("feat_fc_x.2", "feat_fc_y.2", "feat_fc_z.2", "feat_fc_ex_rot.2"),
        )
        for name, expected in baseline.items():
            torch.testing.assert_close(getattr(model, name)(inputs), expected, rtol=0, atol=0)
            adapter = getattr(model, name)[-1]
            self.assertIsInstance(adapter, LoRALinear)
            self.assertFalse(adapter.weight.requires_grad)
            self.assertTrue(adapter.lora_A.requires_grad)
            self.assertTrue(adapter.lora_B.requires_grad)

        sum(getattr(model, name)(inputs).sum() for name in baseline).backward()
        self.assertIsNotNone(model.feat_fc_x[-1].lora_B.grad)
        self.assertGreater(model.feat_fc_x[-1].lora_B.grad.abs().sum().item(), 0)

    def test_merged_state_dict_matches_adapter_forward(self):
        torch.manual_seed(11)
        model = DiscreteActionHeads().eval()
        inject_discrete_action_lora(model, rank=2, alpha=3)
        adapter = model.feat_fc_y[-1]
        with torch.no_grad():
            adapter.lora_B.normal_(mean=0.0, std=0.03)

        inputs = torch.randn(4, 5)
        hidden = model.feat_fc_y[:2](inputs)
        adapted_output = model.feat_fc_y(inputs)
        merged = merged_lora_state_dict(model)
        merged_output = F.linear(
            hidden,
            merged["feat_fc_y.2.weight"],
            merged["feat_fc_y.2.bias"],
        )

        torch.testing.assert_close(adapted_output, merged_output, rtol=1e-6, atol=1e-6)
        self.assertNotIn("feat_fc_y.2.lora_A", merged)
        self.assertNotIn("feat_fc_y.2.lora_B", merged)
        self.assertIn("feat_fc_y.2.weight", merged)


if __name__ == "__main__":
    unittest.main()
