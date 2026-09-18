# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer


class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sample_mode,
        sample_distribution_mode="transition_uniform",
        distributed_rank=0,
        distributed_world_size=1,
    ):
        self._replay_buffer = replay_buffer
        self._sample_mode = sample_mode
        if self._sample_mode == 'enumerate':
            self._num_samples = self._replay_buffer.prepare_enumeration()
        self._sample_distribution_mode = sample_distribution_mode
        self._distributed_rank = int(distributed_rank)
        self._distributed_world_size = int(distributed_world_size)
        if self._distributed_world_size < 1:
            raise ValueError(
                "distributed_world_size must be positive, got "
                f"{self._distributed_world_size}"
            )
        if not 0 <= self._distributed_rank < self._distributed_world_size:
            raise ValueError(
                "distributed_rank must be in [0, world_size), got "
                f"rank={self._distributed_rank}, "
                f"world_size={self._distributed_world_size}"
            )

    def _seed_distributed_worker(self):
        """Give each DDP rank/worker an independent replay RNG stream.

        Replay sampling is implemented with NumPy's process-global RNG rather
        than a torch ``Sampler``.  Without an explicit offset, forked loader
        workers (and DDP ranks started from the same seed) can walk the same
        random stream.  We only alter the seed in distributed mode so the
        single-GPU legacy path keeps its existing behavior.
        """
        if self._distributed_world_size <= 1:
            return

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        seed = (
            int(torch.initial_seed())
            + 1000003 * self._distributed_rank
            + 9176 * worker_id
        ) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)

    def _generator(self):
        self._seed_distributed_worker()
        while True:
            if self._sample_mode == 'random':
                yield self._replay_buffer.sample_transition_batch(pack_in_dict=True, distribution_mode = self._sample_distribution_mode)
            elif self._sample_mode == 'enumerate':
                yield self._replay_buffer.enumerate_next_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())

    def __len__(self): # enumeration will throw away the last incomplete batch
        return self._num_samples // self._replay_buffer._batch_size

class PyTorchReplayBuffer(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
      sample_mode: the mode to sample data, choose from ['random', 'enumerate']
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        num_workers: int = 2,
        sample_mode="random",
        sample_distribution_mode="transition_uniform",
        distributed_rank=None,
        distributed_world_size=None,
    ):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers
        self._sample_mode = sample_mode
        self._sample_distribution_mode = sample_distribution_mode
        if distributed_rank is None or distributed_world_size is None:
            if dist.is_available() and dist.is_initialized():
                distributed_rank = dist.get_rank()
                distributed_world_size = dist.get_world_size()
            else:
                distributed_rank = 0
                distributed_world_size = 1
        self._distributed_rank = int(distributed_rank)
        self._distributed_world_size = int(distributed_world_size)

    def dataset(self) -> DataLoader:
        d = PyTorchIterableReplayDataset(
            self._replay_buffer,
            self._sample_mode,
            self._sample_distribution_mode,
            distributed_rank=self._distributed_rank,
            distributed_world_size=self._distributed_world_size,
        )

        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=None, pin_memory=True,
                          num_workers=self._num_workers)
