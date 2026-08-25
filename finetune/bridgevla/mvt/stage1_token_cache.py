"""Disk-backed keypoint token cache for the frozen Stage-1 encoder."""

from pathlib import Path

import torch


class Stage1KeypointTokenCache:
    """Load per-episode PaliGemma tokens and build a causal short window.

    Each cache file is expected at ``root/task/episode{index}.pt`` and stores:

    ``{"keypoint_frames": LongTensor[num_keypoints],
       "tokens": Tensor[num_keypoints, num_view_patches, token_dim]}``.

    ``keypoint_idx`` identifies the current action target. The returned history
    contains only keypoints strictly before the current replay observation (or
    the target as a fallback), so it cannot leak the target observation.
    """

    def __init__(self, root, max_history=4):
        self.root = Path(root)
        self.max_history = int(max_history)
        if self.max_history < 1:
            raise ValueError("max_history must be >= 1")
        self._episodes = {}

    def _episode_path(self, task, episode_idx):
        return self.root / str(task) / f"episode{int(episode_idx)}.pt"

    @staticmethod
    def _validate_episode(entry, path):
        if not isinstance(entry, dict) or "keypoint_frames" not in entry or "tokens" not in entry:
            raise ValueError(
                f"invalid Stage-1 token cache format: {path}; expected keypoint_frames and tokens"
            )
        tokens = entry["tokens"]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim < 2:
            raise ValueError(f"invalid token tensor in Stage-1 cache: {path}")
        keypoint_frames = torch.as_tensor(entry["keypoint_frames"])
        if keypoint_frames.ndim != 1:
            raise ValueError(f"keypoint_frames must be 1-D in Stage-1 cache: {path}")
        if keypoint_frames.numel() != tokens.shape[0]:
            raise ValueError(f"keypoint frame/token count mismatch in cache: {path}")
        if keypoint_frames.numel() > 1 and not torch.all(
            keypoint_frames[1:] > keypoint_frames[:-1]
        ):
            raise ValueError(f"keypoint_frames must be strictly increasing: {path}")
        return entry

    def _load_episode(self, task, episode_idx):
        key = (str(task), int(episode_idx))
        if key not in self._episodes:
            path = self._episode_path(*key)
            if not path.is_file():
                raise FileNotFoundError(f"missing Stage-1 token cache: {path}")
            entry = torch.load(path, map_location="cpu", weights_only=True)
            self._episodes[key] = self._validate_episode(entry, path)
        return self._episodes[key]

    def get_batch(
        self,
        tasks,
        episode_indices,
        keypoint_indices,
        sample_frames=None,
    ):
        """Return padded past tokens and a valid mask for a replay batch.

        When ``sample_frames`` is supplied, only keyframes strictly before the
        current replay observation are used. This avoids duplicating the latest
        keyframe when the replay sample itself is that keyframe.
        """

        if self.max_history == 1:
            return (
                torch.empty(len(tasks), 0, 0, 0),
                torch.empty(len(tasks), 0, dtype=torch.bool),
            )

        episode_indices = torch.as_tensor(episode_indices).reshape(-1).tolist()
        keypoint_indices = torch.as_tensor(keypoint_indices).reshape(-1).tolist()
        if sample_frames is None:
            raise ValueError(
                "sample_frames are required when using the Stage-1 token cache"
            )
        sample_frames = torch.as_tensor(sample_frames).reshape(-1).tolist()
        if (
            len(tasks) != len(episode_indices)
            or len(tasks) != len(keypoint_indices)
            or len(tasks) != len(sample_frames)
        ):
            raise ValueError("Stage-1 cache metadata batch lengths do not match")

        entries = [
            self._load_episode(task, episode_idx)
            for task, episode_idx in zip(tasks, episode_indices)
        ]
        token_shape = tuple(entries[0]["tokens"].shape[1:])
        token_dtype = entries[0]["tokens"].dtype
        history = self.max_history - 1
        output = torch.zeros(
            len(entries), history, *token_shape, dtype=token_dtype
        )
        valid = torch.zeros(len(entries), history, dtype=torch.bool)

        for batch_idx, (entry, keypoint_idx, sample_frame) in enumerate(
            zip(entries, keypoint_indices, sample_frames)
        ):
            tokens = entry["tokens"]
            if tuple(tokens.shape[1:]) != token_shape:
                raise ValueError("inconsistent Stage-1 token shapes in cache")
            keypoint_idx = int(keypoint_idx)
            if keypoint_idx < 0 or keypoint_idx >= tokens.shape[0]:
                raise ValueError(
                    f"invalid keypoint_idx={keypoint_idx} for {tokens.shape[0]} keypoints"
                )
            keypoint_frames = torch.as_tensor(
                entry["keypoint_frames"], dtype=torch.long
            )
            sample_frame = int(sample_frame)
            target_frame = int(keypoint_frames[keypoint_idx])
            if sample_frame >= target_frame:
                raise ValueError(
                    "replay sample_frame must be strictly before its target "
                    f"frame: sample_frame={sample_frame}, target_frame={target_frame}"
                )
            eligible_count = int(
                torch.searchsorted(
                    keypoint_frames,
                    torch.tensor(sample_frame, dtype=torch.long),
                    right=False,
                )
            )
            if eligible_count > keypoint_idx:
                raise ValueError(
                    "replay sample_frame is not strictly before its target "
                    f"keypoint_idx={keypoint_idx}"
                )
            past = tokens[:eligible_count][-history:]
            count = past.shape[0]
            if count:
                output[batch_idx, -count:] = past
                valid[batch_idx, -count:] = True

        return output, valid
