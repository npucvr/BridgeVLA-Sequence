"""Forward, episode-bounded sequence sampling for recurrent policy training."""

import numpy as np

from yarr.replay_buffer.uniform_replay_buffer import (
    ACTION,
    REWARD,
    TERMINAL,
    TIMEOUT,
)


class SequenceReplayBuffer:
    """Expose forward sequences without changing the legacy replay API.

    The wrapped replay buffer remains configured with ``timesteps=1``.  By
    default, a target transition is sampled using its existing distribution
    policy.  Optional preceding records are read from the same task replay as a
    no-loss burn-in prefix, followed by ``sequence_length`` forward target
    records.  With ``full_episode=True``, one complete episode is sampled from
    its first transition instead and padded to the longest retained episode.
    Boundary padding is repeated from a valid record and marked with
    ``valid_mask=0``; burn-in records are additionally marked with
    ``loss_mask=0``.

    This wrapper intentionally uses the replay's existing terminal and timeout
    markers.  A keypoint-augmented trajectory is terminated before another
    augmented suffix can be crossed, while old replay files remain readable.
    """

    def __init__(
        self,
        replay_buffer,
        sequence_length,
        burn_in_length=0,
        full_episode=False,
    ):
        if sequence_length < 2:
            raise ValueError(
                f"sequence_length must be at least 2, got {sequence_length}"
            )
        if burn_in_length < 0:
            raise ValueError(
                f"burn_in_length must be non-negative, got {burn_in_length}"
            )
        if full_episode and burn_in_length:
            raise ValueError(
                "full_episode sampling already starts at the episode boundary; "
                "burn_in_length must be zero"
            )
        if getattr(replay_buffer, "timesteps", 1) != 1:
            raise ValueError(
                "SequenceReplayBuffer requires the wrapped replay buffer to "
                "use timesteps=1"
            )
        required_methods = (
            "get_storage_signature",
            "sequence_task_local_index",
            "sequence_task_length",
            "sequence_task_name",
            "sequence_transition_record",
        )
        missing_methods = [
            name for name in required_methods if not hasattr(replay_buffer, name)
        ]
        missing_attributes = [
            name
            for name in ("batch_size", "replay_capacity")
            if not hasattr(replay_buffer, name)
        ]
        if missing_methods or missing_attributes:
            raise TypeError(
                "SequenceReplayBuffer requires the replay sequence adapter; "
                f"missing methods: {missing_methods}, "
                f"missing attributes: {missing_attributes}"
            )
        self._replay_buffer = replay_buffer
        self._sequence_length = int(sequence_length)
        self._burn_in_length = int(burn_in_length)
        self._full_episode = bool(full_episode)
        self._batch_size = replay_buffer.batch_size
        self._storage_signature = replay_buffer.get_storage_signature()[0]
        self._global_index = self._build_global_index()
        self._episodes_by_task = {}
        self._episode_padded_length = self._sequence_length
        if self._full_episode:
            self._episodes_by_task = self._build_episode_index()
            all_episodes = [
                episode
                for episodes in self._episodes_by_task.values()
                for episode in episodes
            ]
            if not all_episodes:
                raise RuntimeError("full_episode sampling found no complete episodes")
            self._episode_padded_length = max(
                episode[1] - episode[0] for episode in all_episodes
            )

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def sequence_length(self):
        return self._episode_padded_length if self._full_episode else self._sequence_length

    @property
    def burn_in_length(self):
        return self._burn_in_length

    @property
    def full_episode(self):
        return self._full_episode

    @property
    def batch_size(self):
        return self._batch_size

    def _build_global_index(self):
        """Map ``(task_idx, local_idx)`` to the replay's global index."""
        result = {}
        for global_idx in range(self._replay_buffer.replay_capacity):
            try:
                task_idx, local_idx = self._replay_buffer.sequence_task_local_index(
                    global_idx
                )
            except (IndexError, RuntimeError):
                continue
            result[(task_idx, local_idx)] = int(global_idx)
        return result

    def _build_episode_index(self):
        """Scan retained records into ``(start_local, end_local)`` episodes."""
        episodes_by_task = {}
        task_indices = sorted({task_idx for task_idx, _ in self._global_index})
        for task_idx in task_indices:
            task_length = int(self._task_length(task_idx))
            task_episodes = []
            episode_start = None
            contiguous = True
            for local_idx in range(task_length):
                global_idx = self._global_index.get((task_idx, local_idx))
                if global_idx is None:
                    # Never let a circular-replay hole join two retained
                    # fragments into one apparently complete episode.
                    episode_start = None
                    contiguous = False
                    continue

                terminal_value, timeout_value = self._record_markers(
                    task_idx, local_idx
                )
                if terminal_value == -1:
                    # ``add_final`` stores an observation-only sentinel.  It
                    # is never an action transition and never closes an
                    # otherwise unfinished suffix.
                    episode_start = None
                    contiguous = True
                    continue

                if not contiguous:
                    # The first record after a retained-replay hole is not a
                    # trustworthy episode start.  Wait for its boundary.
                    if terminal_value == 1 or timeout_value:
                        contiguous = True
                    continue

                if episode_start is None:
                    episode_start = local_idx
                if terminal_value == 1 or timeout_value:
                    task_episodes.append((episode_start, local_idx + 1))
                    episode_start = None

            # A complete replay normally has terminal=1 followed by -1.  Drop
            # an unfinished suffix instead of silently treating it as an episode.
            if task_episodes:
                episodes_by_task[task_idx] = task_episodes
        return episodes_by_task

    def _task_and_local_index(self, global_idx):
        return self._replay_buffer.sequence_task_local_index(global_idx)

    def _read_record(self, task_idx, local_idx):
        return self._replay_buffer.sequence_transition_record(task_idx, local_idx)

    def _record_markers(self, task_idx, local_idx):
        marker_reader = getattr(
            self._replay_buffer,
            "sequence_transition_markers",
            None,
        )
        if marker_reader is not None:
            return marker_reader(task_idx, local_idx)
        record = self._read_record(task_idx, local_idx)
        return (
            int(np.asarray(record[TERMINAL]).item()),
            bool(np.asarray(record[TIMEOUT]).item()),
        )

    def _task_length(self, task_idx):
        return self._replay_buffer.sequence_task_length(task_idx)

    def _episode_for_local_index(self, task_idx, local_idx):
        for start_local_idx, end_local_idx in self._episodes_by_task.get(task_idx, []):
            if start_local_idx <= local_idx < end_local_idx:
                return start_local_idx, end_local_idx
        raise RuntimeError(
            "Could not locate a complete episode for "
            f"task={task_idx}, local_index={local_idx}"
        )

    def _sample_episode_anchor_indices(self, batch_size, distribution_mode):
        """Sample episode starts while retaining task/length distribution choices."""
        if distribution_mode == "task_uniform":
            task_indices = list(self._episodes_by_task)
            sampled = []
            for _ in range(batch_size):
                task_idx = task_indices[np.random.randint(len(task_indices))]
                episodes = self._episodes_by_task[task_idx]
                episode = episodes[np.random.randint(len(episodes))]
                sampled.append(self._global_index[(task_idx, episode[0])])
            return sampled

        all_episodes = [
            (task_idx, episode)
            for task_idx, episodes in self._episodes_by_task.items()
            for episode in episodes
        ]
        if distribution_mode in {"episode_uniform", "uniform"}:
            choices = np.random.randint(len(all_episodes), size=batch_size)
        elif distribution_mode in {"transition_uniform", "episode_length_weighted"}:
            lengths = np.asarray(
                [episode[1] - episode[0] for _, episode in all_episodes],
                dtype=np.float64,
            )
            probabilities = lengths / lengths.sum()
            choices = np.random.choice(
                len(all_episodes), size=batch_size, p=probabilities
            )
        else:
            raise ValueError(
                "Unsupported full-episode distribution mode: "
                f"{distribution_mode!r}; use task_uniform, episode_uniform, "
                "or episode_length_weighted"
            )
        return [
            self._global_index[all_episodes[int(choice)][0], all_episodes[int(choice)][1][0]]
            for choice in choices
        ]

    def _read_full_episode(self, anchor_global_idx):
        """Read one complete episode and pad it to the batch maximum length."""
        task_idx, local_idx = self._task_and_local_index(anchor_global_idx)
        start_local_idx, end_local_idx = self._episode_for_local_index(
            task_idx, local_idx
        )
        local_indices = list(range(start_local_idx, end_local_idx))
        records = []
        global_indices = []
        for current_local_idx in local_indices:
            global_idx = self._global_index.get((task_idx, current_local_idx))
            if global_idx is None:
                raise RuntimeError(
                    "A complete episode left the retained replay window: "
                    f"task={task_idx}, local_index={current_local_idx}"
                )
            record = self._read_record(task_idx, current_local_idx)
            if int(np.asarray(record[TERMINAL]).item()) == -1:
                raise RuntimeError(
                    "Final-observation sentinel appeared inside an episode: "
                    f"task={task_idx}, local_index={current_local_idx}"
                )
            records.append(record)
            global_indices.append(int(global_idx))

        if not records:
            raise RuntimeError("Cannot sample an empty episode")
        valid_length = len(records)
        pad_count = self._episode_padded_length - valid_length
        pad_record = records[-1]
        pad_index = global_indices[-1]
        records.extend([pad_record] * pad_count)
        global_indices.extend([pad_index] * pad_count)
        valid_mask = np.asarray(
            [1.0] * valid_length + [0.0] * pad_count,
            dtype=np.float32,
        )
        return (
            task_idx,
            records,
            np.asarray(global_indices, dtype=np.int32),
            valid_mask,
            valid_mask.copy(),
            valid_length,
            int(global_indices[0]),
            int(global_indices[valid_length - 1]),
        )

    def _episode_context_start(self, task_idx, target_local_idx):
        """Find a burn-in start without crossing an episode boundary."""
        context_start = max(0, target_local_idx - self._burn_in_length)
        for local_idx in range(target_local_idx - 1, context_start - 1, -1):
            record = self._read_record(task_idx, local_idx)
            terminal_value = int(np.asarray(record[TERMINAL]).item())
            timeout_value = bool(np.asarray(record[TIMEOUT]).item())
            if terminal_value in (-1, 1) or timeout_value:
                context_start = local_idx + 1
                break
        return context_start

    def _read_forward(self, task_idx, start_local_idx, length):
        """Read a fixed-length forward sequence from one task-local index."""
        task_length = self._task_length(task_idx)
        records = []
        indices = []
        valid_mask = []
        local_idx = int(start_local_idx)
        terminal_seen = False
        last_record = None
        last_global_idx = None

        for _ in range(length):
            if terminal_seen or local_idx >= task_length:
                if last_record is None:
                    raise RuntimeError("Cannot pad an empty replay sequence")
                record = last_record
                global_idx = last_global_idx
                valid = 0.0
            else:
                global_idx = self._global_index.get((task_idx, local_idx))
                if global_idx is None:
                    if last_record is None:
                        raise RuntimeError(
                            "A sampled sequence started outside the retained "
                            "replay window: "
                            f"task={task_idx}, local_index={local_idx}"
                        )
                    # A circular replay may retain the start but not all later
                    # task-local records.  Stop rather than reading stale disk
                    # records or crossing into another retained segment.
                    record = last_record
                    global_idx = last_global_idx
                    valid = 0.0
                    terminal_seen = True
                else:
                    record = self._read_record(task_idx, local_idx)
                    terminal_value = int(np.asarray(record[TERMINAL]).item())
                    if terminal_value == -1:
                        if last_record is None:
                            raise RuntimeError(
                                "A sampled sequence started at a final observation "
                                f"record: task={task_idx}, local_index={local_idx}"
                            )
                        # A well-formed replay has terminal=1 before add_final.
                        # Treat a dangling final observation as padding rather than
                        # crossing into the next augmented trajectory.
                        record = last_record
                        global_idx = last_global_idx
                        valid = 0.0
                        terminal_seen = True
                    else:
                        valid = 1.0
                        last_record = record
                        last_global_idx = int(global_idx)
                        timeout_value = bool(np.asarray(record[TIMEOUT]).item())
                        terminal_seen = terminal_value == 1 or timeout_value
                        local_idx += 1

            records.append(record)
            indices.append(int(global_idx))
            valid_mask.append(valid)

        return records, np.asarray(indices, dtype=np.int32), np.asarray(
            valid_mask, dtype=np.float32
        )

    def _read_sequence(self, start_global_idx):
        """Read burn-in context followed by the target sequence."""
        task_idx, target_local_idx = self._task_and_local_index(start_global_idx)
        context_start = self._episode_context_start(task_idx, target_local_idx)
        prefix_local_indices = list(range(context_start, target_local_idx))
        if len(prefix_local_indices) > self._burn_in_length:
            raise RuntimeError("burn-in context is longer than configured length")

        prefix_records = [
            self._read_record(task_idx, local_idx)
            for local_idx in prefix_local_indices
        ]
        prefix_indices = [
            self._global_index.get((task_idx, local_idx))
            for local_idx in prefix_local_indices
        ]
        if any(global_idx is None for global_idx in prefix_indices):
            # The sampled target is retained, but an older burn-in record is not.
            # Do not read stale records from a circular replay window.
            prefix_records = []
            prefix_indices = []

        target_records, target_indices, target_valid_mask = self._read_forward(
            task_idx,
            target_local_idx,
            self._sequence_length,
        )

        pad_record = prefix_records[0] if prefix_records else target_records[0]
        pad_index = prefix_indices[0] if prefix_indices else target_indices[0]
        left_padding = self._burn_in_length - len(prefix_records)
        records = [pad_record] * left_padding + prefix_records + target_records
        indices = [pad_index] * left_padding + prefix_indices + list(target_indices)
        valid_mask = (
            [0.0] * left_padding
            + [1.0] * len(prefix_records)
            + list(target_valid_mask)
        )
        loss_mask = [0.0] * self._burn_in_length + list(target_valid_mask)
        return (
            task_idx,
            records,
            np.asarray(indices, dtype=np.int32),
            np.asarray(valid_mask, dtype=np.float32),
            np.asarray(loss_mask, dtype=np.float32),
            -1,
            -1,
            -1,
        )

    def _stack_field(self, records, field):
        try:
            return np.stack([record[field] for record in records], axis=0)
        except KeyError as error:
            raise RuntimeError(
                f"Replay record is missing required field {field!r}"
            ) from error

    def sample_transition_batch(
        self,
        batch_size=None,
        indices=None,
        pack_in_dict=True,
        distribution_mode="transition_uniform",
    ):
        """Sample forward windows or complete episodes as ``[B, L, ...]``."""
        if not pack_in_dict:
            raise NotImplementedError(
                "SequenceReplayBuffer only supports pack_in_dict=True"
            )
        if indices is None:
            batch_size = self._batch_size if batch_size is None else int(batch_size)
            if self._full_episode:
                indices = self._sample_episode_anchor_indices(
                    batch_size, distribution_mode
                )
            else:
                indices = self._replay_buffer.sample_index_batch(
                    batch_size, distribution_mode=distribution_mode
                )
        else:
            indices = [int(index) for index in indices]
            batch_size = len(indices)

        if self._full_episode:
            sequences = [self._read_full_episode(index) for index in indices]
        else:
            sequences = [self._read_sequence(index) for index in indices]
        (
            task_indices,
            records,
            sequence_indices,
            valid_masks,
            loss_masks,
            episode_lengths,
            episode_starts,
            episode_ends,
        ) = zip(*sequences)

        batch = {
            ACTION: np.stack(
                [self._stack_field(record_list, ACTION) for record_list in records],
                axis=0,
            ),
            REWARD: np.stack(
                [self._stack_field(record_list, REWARD) for record_list in records],
                axis=0,
            ),
            TERMINAL: np.stack(
                [self._stack_field(record_list, TERMINAL) for record_list in records],
                axis=0,
            ),
            TIMEOUT: np.stack(
                [self._stack_field(record_list, TIMEOUT) for record_list in records],
                axis=0,
            ),
            "indices": np.stack(sequence_indices, axis=0),
            "valid_mask": np.stack(valid_masks, axis=0),
            "loss_mask": np.stack(loss_masks, axis=0),
            "episode_lengths": np.asarray(episode_lengths, dtype=np.int32),
            "episode_starts": np.asarray(episode_starts, dtype=np.int32),
            "episode_ends": np.asarray(episode_ends, dtype=np.int32),
            "tasks": [
                self._replay_buffer.sequence_task_name(task_idx)
                for task_idx in task_indices
            ],
        }

        for element in self._storage_signature:
            if element.name in {ACTION, REWARD, TERMINAL, TIMEOUT}:
                continue
            batch[element.name] = np.stack(
                [self._stack_field(record_list, element.name) for record_list in records],
                axis=0,
            )

        return batch
