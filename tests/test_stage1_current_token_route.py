import unittest

import torch
from torch import nn

from bridgevla.mvt.mvt import MVT
from bridgevla.mvt.stage1_hidden_state import Stage1HiddenStateTransition
from bridgevla.mvt.stage1_temporal_loss import masked_history_cosine_loss
from bridgevla.mvt.stage1_token_correction_adapter import (
    Stage1TokenCorrectionAdapter,
)


class CurrentTokenRouteTest(unittest.TestCase):
    def test_mvt_forwards_hidden_state_to_stage1_only(self):
        class FakeStage1(nn.Module):
            def forward(self, **kwargs):
                self.received_hidden_state = kwargs.get("stage1_hidden_state")
                return {}

        class FakeMVT(MVT):
            def __init__(self):
                nn.Module.__init__(self)
                self.stage_two = False
                self.img_aug_2 = 0.0
                self.mvt1 = FakeStage1()

            def verify_inp(self, **kwargs):
                return None

            def render(self, **kwargs):
                return torch.zeros(1, 1, 6, 4, 4)

        network = FakeMVT().eval()
        hidden = torch.randn(1, 5)
        network(
            pc=[torch.zeros(1, 3)],
            img_feat=[torch.zeros(1, 3)],
            language_goal=[["goal"]],
            stage1_hidden_state=hidden,
        )

        self.assertIs(network.mvt1.received_hidden_state, hidden)

    def test_correction_adapter_is_current_only_and_starts_identity(self):
        torch.manual_seed(0)
        adapter = Stage1TokenCorrectionAdapter(token_dim=8, bottleneck_dim=4)
        current = torch.randn(2, 6, 8)

        corrected = adapter(current)

        self.assertEqual(corrected.shape, current.shape)
        self.assertTrue(torch.allclose(corrected, current))
        with self.assertRaisesRegex(ValueError, "historical token windows"):
            adapter(current.unsqueeze(1))

    def test_history_loss_uses_detached_history_target(self):
        current = torch.randn(2, 4, 8, requires_grad=True)
        history = torch.randn(2, 3, 4, 8, requires_grad=True)
        mask = torch.tensor([[True, True, False], [False, True, False]])

        loss = masked_history_cosine_loss(current, history, mask)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(current.grad)
        self.assertIsNone(history.grad)

    def test_history_loss_is_zero_without_valid_history(self):
        current = torch.randn(2, 4, 8, requires_grad=True)
        history = torch.randn(2, 0, 4, 8)
        mask = torch.zeros(2, 0, dtype=torch.bool)

        loss = masked_history_cosine_loss(current, history, mask)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(current.grad)

    def test_hidden_conditioned_adapter_preserves_initial_identity(self):
        adapter = Stage1TokenCorrectionAdapter(
            token_dim=8,
            bottleneck_dim=4,
            hidden_state_dim=3,
        )
        current = torch.randn(2, 6, 8)
        hidden = torch.randn(2, 3)

        corrected = adapter(current, hidden)

        self.assertTrue(torch.allclose(corrected, current))
        with torch.no_grad():
            adapter.up.weight.fill_(0.1)
            adapter.hidden_to_bottleneck.weight.fill_(0.1)
        changed = adapter(current, hidden)
        zero_hidden = adapter(current, torch.zeros_like(hidden))
        self.assertFalse(torch.allclose(changed, current))
        self.assertFalse(torch.allclose(changed, zero_hidden))

    def test_hidden_state_transition_is_action_conditioned(self):
        transition = Stage1HiddenStateTransition(hidden_dim=5, action_dim=8)
        state = transition.initial_state(batch_size=2)
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        action_a = torch.zeros(2, 8)
        action_b = torch.ones(2, 8)

        next_a = transition(state, action_a)
        next_b = transition(state, action_b)

        self.assertEqual(next_a.shape, (2, 5))
        self.assertFalse(torch.allclose(next_a, next_b))

    def test_two_step_unroll_backpropagates_into_hidden_transition(self):
        torch.manual_seed(1)
        transition = Stage1HiddenStateTransition(hidden_dim=5, action_dim=8)
        adapter = Stage1TokenCorrectionAdapter(
            token_dim=8,
            bottleneck_dim=4,
            hidden_state_dim=5,
        )
        with torch.no_grad():
            adapter.up.weight.normal_(mean=0.0, std=0.05)

        state_0 = transition.initial_state(batch_size=2)
        tokens_0 = torch.randn(2, 6, 8)
        tokens_1 = torch.randn(2, 6, 8)
        action_0 = torch.randn(2, 8)
        state_1 = transition(state_0, action_0)
        corrected_1 = adapter(tokens_1, state_1)
        loss = corrected_1.square().mean()
        loss.backward()

        self.assertIsNotNone(transition.action_encoder[0].weight.grad)
        self.assertIsNotNone(transition.transition.weight_hh.grad)
        self.assertIsNotNone(adapter.hidden_to_bottleneck.weight.grad)


if __name__ == "__main__":
    unittest.main()
