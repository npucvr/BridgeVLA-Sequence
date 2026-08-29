"""Focused tests for the H-token hidden-state route."""

import numpy as np
import torch

from bridgevla.hidden_state import F_phi, U_omega
from bridgevla.mvt.mvt import MVT
from yarr.replay_buffer.replay_buffer import ReplayElement
from yarr.replay_buffer.sequence_replay_buffer import SequenceReplayBuffer


class _Count:
    def __init__(self, value):
        self.value = value


class _FakeReplay:
    """Small in-memory replay exposing the fields used by the sequence wrapper."""

    timesteps = 1
    batch_size = 2
    replay_capacity = 5
    _disk_saving = False
    _task_names = ["task"]
    _task_replay_storage_folders = [""]
    _task_add_count = [_Count(5)]
    _index_mapping = np.asarray(
        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64
    )
    _storage_signature = [
        ReplayElement("action", (2,), np.float32),
        ReplayElement("reward", (), np.float32),
        ReplayElement("terminal", (), np.int8),
        ReplayElement("timeout", (), bool),
        ReplayElement("observation", (1,), np.float32),
    ]
    _store = {
        "action": np.arange(10, dtype=np.float32).reshape(5, 2),
        "reward": np.arange(5, dtype=np.float32),
        "terminal": np.asarray([0, 0, 1, 0, -1], dtype=np.int8),
        "timeout": np.zeros(5, dtype=bool),
        "observation": np.arange(5, dtype=np.float32).reshape(5, 1),
    }

    def get_storage_signature(self):
        return self._storage_signature, [self._storage_signature[-1]]

    def sequence_task_local_index(self, global_index):
        task_idx, local_idx = self._index_mapping[int(global_index)]
        if task_idx < 0 or local_idx < 0:
            raise RuntimeError("invalid mapping")
        return int(task_idx), int(local_idx)

    def sequence_global_index(self, task_idx, local_idx):
        for index, pair in enumerate(self._index_mapping):
            if tuple(pair) == (task_idx, local_idx):
                return index
        return None

    def sequence_task_length(self, task_idx):
        return self._task_add_count[task_idx].value

    def sequence_task_name(self, task_idx):
        return self._task_names[task_idx]

    def sequence_transition_record(self, task_idx, local_idx):
        global_index = self.sequence_global_index(task_idx, local_idx)
        return {
            element.name: np.array(self._store[element.name][global_index])
            for element in self._storage_signature
        }

    def sample_index_batch(self, batch_size, distribution_mode="transition_uniform"):
        del distribution_mode
        return [0, 3][:batch_size]


def test_observation_update_and_transition_unroll_have_gradients():
    torch.manual_seed(0)
    observation_update = U_omega(token_dim=16, hidden_state_dim=8, num_heads=2)
    transition = F_phi(hidden_dim=8, action_dim=4)
    policy_head = torch.nn.Linear(8, 3)

    hidden = torch.zeros(2, 8)
    observations = torch.randn(2, 3, 11, 16)
    actions = torch.randn(2, 3, 4)
    loss = 0.0
    for timestep in range(observations.shape[1]):
        hidden = observation_update(hidden, observations[:, timestep])
        loss = loss + policy_head(hidden).square().mean()
        hidden = transition(hidden, actions[:, timestep])
    loss.backward()

    assert hidden.shape == (2, 8)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in observation_update.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in transition.parameters()
    )


def test_stage_two_reuses_one_observation_posterior():
    class _FakeSingle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_state_enabled = True
            self.calls = []

        def forward(
            self,
            img,
            wpt_local=None,
            rot_x_y=None,
            language_goal=None,
            forward_no_feat=False,
            hidden_state_y=None,
            hidden_state_update=True,
            **kwargs,
        ):
            del img, wpt_local, rot_x_y, language_goal, kwargs
            self.calls.append(
                {
                    "hidden_state_y": hidden_state_y.clone(),
                    "hidden_state_update": hidden_state_update,
                    "forward_no_feat": forward_no_feat,
                }
            )
            posterior = hidden_state_y + 1.0
            return {
                "trans": torch.zeros(2, 3, 4, 4),
                "hidden_state_y": posterior,
                **({} if forward_no_feat else {"feat": torch.zeros(2, 4)}),
            }

    mvt = MVT.__new__(MVT)
    torch.nn.Module.__init__(mvt)
    mvt.stage_two = True
    mvt.img_aug_2 = 0.0
    mvt.st_wpt_loc_aug = 0.0
    mvt.st_wpt_loc_inp_no_noise = False
    mvt.st_sca = 1.0
    mvt.num_img = 3
    mvt.img_size = 4
    mvt.mvt1 = _FakeSingle()
    mvt.verify_inp = lambda **kwargs: None
    mvt.render = lambda **kwargs: torch.zeros(2, 3, 6, 4, 4)
    mvt.train()

    prior = torch.zeros(2, 8)
    output = mvt(
        pc=[torch.zeros(5, 3), torch.zeros(5, 3)],
        img_feat=[torch.zeros(5, 3), torch.zeros(5, 3)],
        wpt_local=torch.zeros(2, 3),
        language_goal=[["pick"], ["pick"]],
        hidden_state_y=prior,
    )

    assert len(mvt.mvt1.calls) == 2
    assert mvt.mvt1.calls[0]["hidden_state_update"] is True
    assert mvt.mvt1.calls[0]["forward_no_feat"] is True
    assert mvt.mvt1.calls[1]["hidden_state_update"] is False
    assert mvt.mvt1.calls[1]["forward_no_feat"] is False
    torch.testing.assert_close(mvt.mvt1.calls[0]["hidden_state_y"], prior)
    torch.testing.assert_close(
        mvt.mvt1.calls[1]["hidden_state_y"], prior + 1.0
    )
    torch.testing.assert_close(output["hidden_state_y"], prior + 1.0)


def test_sequence_replay_is_forward_and_masks_terminal_padding():
    replay = SequenceReplayBuffer(_FakeReplay(), sequence_length=4)
    batch = replay.sample_transition_batch()

    assert batch["action"].shape == (2, 4, 2)
    assert batch["observation"].shape == (2, 4, 1)
    np.testing.assert_array_equal(
        batch["valid_mask"],
        np.asarray([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch["action"][0],
        np.asarray([[0, 1], [2, 3], [4, 5], [4, 5]], dtype=np.float32),
    )

    timeout_replay = _FakeReplay()
    timeout_replay._store = {
        key: value.copy() for key, value in _FakeReplay._store.items()
    }
    timeout_replay._store["timeout"] = np.asarray(
        [False, True, False, False, False], dtype=bool
    )
    timeout_batch = SequenceReplayBuffer(timeout_replay, sequence_length=4).sample_transition_batch()
    np.testing.assert_array_equal(
        timeout_batch["valid_mask"],
        np.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    )
