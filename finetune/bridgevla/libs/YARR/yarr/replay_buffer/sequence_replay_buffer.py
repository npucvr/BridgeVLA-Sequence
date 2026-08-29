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

    The wrapped replay buffer remains configured with ``timesteps=1``.  A start
    transition is sampled using its existing distribution policy, then records
    are read forward from the same task replay until a terminal/timeout
    transition or ``sequence_length`` is reached.  Boundary padding is repeated
    from the last valid record and marked with ``valid_mask=0``.

    This wrapper intentionally uses the replay's existing terminal and timeout
    markers.  A keypoint-augmented trajectory is terminated before another
    augmented suffix can be crossed, while old replay files remain readable.
    """

    def __init__(self, replay_buffer, sequence_length):
        if sequence_length < 2:
            raise ValueError(
                f"sequence_length must be at least 2, got {sequence_length}"
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
        self._batch_size = replay_buffer.batch_size
        self._storage_signature = replay_buffer.get_storage_signature()[0]
        self._global_index = self._build_global_index()

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def sequence_length(self):
        return self._sequence_length

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

    def _task_and_local_index(self, global_idx):
        return self._replay_buffer.sequence_task_local_index(global_idx)

    def _read_record(self, task_idx, local_idx):
        return self._replay_buffer.sequence_transition_record(task_idx, local_idx)

    def _task_length(self, task_idx):
        return self._replay_buffer.sequence_task_length(task_idx)

    def _read_sequence(self, start_global_idx):
        task_idx, local_idx = self._task_and_local_index(start_global_idx)
        task_length = self._task_length(task_idx)
        records = []
        indices = []
        valid_mask = []
        terminal_seen = False
        last_record = None
        last_global_idx = int(start_global_idx)

        for offset in range(self._sequence_length):
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

        return task_idx, records, np.asarray(indices, dtype=np.int32), np.asarray(
            valid_mask, dtype=np.float32
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
        """Sample a batch shaped ``[B, L, ...]`` in forward time order."""
        if not pack_in_dict:
            raise NotImplementedError(
                "SequenceReplayBuffer only supports pack_in_dict=True"
            )
        if indices is None:
            batch_size = self._batch_size if batch_size is None else int(batch_size)
            indices = self._replay_buffer.sample_index_batch(
                batch_size, distribution_mode=distribution_mode
            )
        else:
            indices = [int(index) for index in indices]
            batch_size = len(indices)

        sequences = [self._read_sequence(index) for index in indices]
        task_indices, records, sequence_indices, valid_masks = zip(*sequences)

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
