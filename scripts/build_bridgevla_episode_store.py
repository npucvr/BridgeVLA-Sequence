#!/usr/bin/env python3
"""Build the first episode-aware training store directly from raw RLBench data.

The command intentionally accepts only the shared ``/remote_databuffer``
paths.  It reads one raw episode at a time, turns the existing keypoint/action
semantics into a chronological payload, and writes immutable chunks with
``EpisodeStoreWriter``.  FilterCorrection state is never serialized.

This is an encoder/smoke entry point for ``encoded_train_v2``.  The training
adapter is a separate change; until it is wired and validated, the normal
legacy replay path remains the only formal training path.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_ROOT = REPO_ROOT / "finetune" / "RLBench" / "utils"
sys.path.insert(0, str(UTILS_ROOT))

from dataset import _clip_encode_text, _get_action  # noqa: E402
from episode_store import EpisodeStoreWriter  # noqa: E402
from peract_colab.rlbench.utils import get_stored_demo  # noqa: E402
from peract_utils_rlbench import (  # noqa: E402
    CAMERAS,
    SCENE_BOUNDS,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from bridgevla.libs.peract.helpers.demo_loading_utils import (  # noqa: E402
    keypoint_discovery,
)
from bridgevla.libs.peract.helpers.utils import extract_obs  # noqa: E402


_EPISODE_RE = re.compile(r"^episode([0-9]+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="raw RLBench_TRAIN_DATA under /remote_databuffer",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="new encoded_train_v2 directory under /remote_databuffer",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=0,
        help="limit for smoke runs; zero means all episodes",
    )
    parser.add_argument("--max-chunk-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="time-feature horizon retained from the current replay contract",
    )
    parser.add_argument("--clip-cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def _assert_shared_nfs(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    shared_root = Path("/remote_databuffer").resolve()
    if path != shared_root and shared_root not in path.parents:
        raise ValueError(
            f"{label} must be under /remote_databuffer; local data paths are forbidden: {path}"
        )
    best_mount = None
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 5:
            continue
        mountpoint = Path(fields[4].replace("\\040", " ").replace("\\011", "\t"))
        if path != mountpoint and mountpoint not in path.parents:
            continue
        right_fields = right.split()
        if right_fields and (
            best_mount is None or len(str(mountpoint)) > len(str(best_mount[0]))
        ):
            best_mount = (mountpoint, right_fields[0])
    if best_mount is None or best_mount[1] not in {"nfs", "nfs4"}:
        filesystem = best_mount[1] if best_mount else "unknown"
        raise RuntimeError(
            f"{label} is not on an NFS mount: {path} ({filesystem})"
        )
    return path


def _episode_ids(episodes_root: Path) -> list[int]:
    result = []
    for path in episodes_root.iterdir():
        match = _EPISODE_RE.match(path.name)
        if match and path.is_dir():
            result.append(int(match.group(1)))
    return sorted(result)


def _stack(values: Sequence[object], field: str) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    try:
        return np.stack(arrays, axis=0)
    except ValueError as exc:
        shapes = [tuple(array.shape) for array in arrays]
        raise ValueError(f"inconsistent shapes for {field}: {shapes}") from exc


def _description(episode_dir: Path) -> str:
    path = episode_dir / "variation_descriptions.pkl"
    with path.open("rb") as handle:
        descriptions = pickle.load(handle)
    if not descriptions:
        raise ValueError(f"episode has no variation description: {episode_dir}")
    return str(descriptions[0])


def _language_embedding(clip_model, description: str, device: torch.device) -> np.ndarray:
    import clip

    tokens = clip.tokenize([description]).to(device)
    with torch.no_grad():
        _, embeddings = _clip_encode_text(clip_model, tokens)
    return embeddings[0].float().detach().cpu().numpy()


def _encode_episode(
    *,
    task: str,
    episode_id: int,
    episodes_root: Path,
    clip_model,
    device: torch.device,
    episode_length: int,
) -> Mapping[str, object]:
    episode_dir = episodes_root / f"episode{episode_id}"
    demo = get_stored_demo(data_path=str(episodes_root), index=episode_id)
    description = _description(episode_dir)
    language_embedding = _language_embedding(clip_model, description, device)
    keypoints = list(keypoint_discovery(demo))
    if not keypoints:
        raise ValueError(f"episode has no keypoints: {episode_dir}")

    observation_rows: dict[str, list[np.ndarray]] = {}
    actions: list[np.ndarray] = []
    trans_indices: list[np.ndarray] = []
    rot_grip_indices: list[np.ndarray] = []
    ignore_collisions: list[int] = []
    gripper_poses: list[np.ndarray] = []
    keypoint_frames: list[int] = []
    next_keypoint_frames: list[int] = []
    previous_action = None
    observation = demo[0]

    for keypoint_index, frame in enumerate(keypoints):
        target = demo[frame]
        previous = demo[max(0, frame - 1)]
        (
            translation,
            rotation_grip,
            ignore_collision,
            action,
            _attention_coordinates,
        ) = _get_action(
            target,
            previous,
            SCENE_BOUNDS,
            VOXEL_SIZES,
            ROTATION_RESOLUTION,
            False,
        )
        obs_dict = extract_obs(
            observation,
            CAMERAS,
            t=keypoint_index,
            prev_action=previous_action,
            episode_length=episode_length,
        )
        for field, value in obs_dict.items():
            if field in {"ignore_collisions", "lang_goal_embs", "lang_goal"}:
                continue
            if not isinstance(value, (np.ndarray, list, tuple)):
                raise TypeError(f"unsupported observation field {field}: {type(value)}")
            observation_rows.setdefault(field, []).append(
                np.array(value, copy=True)
            )

        actions.append(np.asarray(action, dtype=np.float32))
        trans_indices.append(np.asarray(translation, dtype=np.int32))
        rot_grip_indices.append(np.asarray(rotation_grip, dtype=np.int32))
        ignore_collisions.append(int(ignore_collision))
        gripper_poses.append(np.asarray(target.gripper_pose, dtype=np.float32))
        keypoint_frames.append(-1 if keypoint_index == 0 else keypoints[keypoint_index - 1])
        next_keypoint_frames.append(int(frame))
        previous_action = np.array(action, copy=True)
        observation = target

    valid_length = len(keypoints)
    labels = {
        "action": _stack(actions, "action").astype(np.float32, copy=False),
        "reward": np.asarray(
            [0.0] * (valid_length - 1) + [1.0], dtype=np.float32
        ),
        "terminal": np.asarray(
            [False] * (valid_length - 1) + [True], dtype=np.bool_
        ),
        "timeout": np.zeros(valid_length, dtype=np.bool_),
        "trans_action_indicies": _stack(
            trans_indices, "trans_action_indicies"
        ).astype(np.int32, copy=False),
        "rot_grip_action_indicies": _stack(
            rot_grip_indices, "rot_grip_action_indicies"
        ).astype(np.int32, copy=False),
        "ignore_collisions": np.asarray(ignore_collisions, dtype=np.int32)[:, None],
        "gripper_pose": _stack(gripper_poses, "gripper_pose").astype(
            np.float32, copy=False
        ),
    }
    observations = {
        field: _stack(values, field) for field, values in observation_rows.items()
    }
    # The legacy replay exposes the action's collision flag as an observation
    # field.  Keep that exact contract for the future adapter.
    observations["ignore_collisions"] = labels["ignore_collisions"].astype(
        np.float32, copy=False
    )
    return {
        "format": "bridgevla_episode_payload",
        "version": 1,
        "task": task,
        "episode_id": int(episode_id),
        "valid_length": valid_length,
        "observations": observations,
        "labels": labels,
        "language": {
            "description": description,
            "lang_goal_embs": language_embedding,
        },
        "metadata": {
            "raw_frame_count": len(demo),
            "keypoint_frame": np.asarray(keypoint_frames, dtype=np.int32),
            "next_keypoint_frame": np.asarray(
                next_keypoint_frames, dtype=np.int32
            ),
            "time_feature_horizon": int(episode_length),
        },
    }


def _task_names(data_root: Path, requested: Iterable[str] | None) -> list[str]:
    if requested:
        names = [str(item) for item in requested]
    else:
        names = sorted(
            path.name
            for path in data_root.iterdir()
            if (path / "all_variations" / "episodes").is_dir()
        )
    result = []
    for task in names:
        episodes_root = data_root / task / "all_variations" / "episodes"
        if not episodes_root.is_dir():
            raise FileNotFoundError(f"missing raw episode root for task={task}: {episodes_root}")
        result.append(task)
    return result


def main() -> int:
    args = _parse_args()
    if args.max_episodes_per_task < 0:
        raise ValueError("max-episodes-per-task must be non-negative")
    if args.episode_length < 2:
        raise ValueError("episode-length must be at least 2")
    data_root = _assert_shared_nfs(Path(args.data_root), "data-root")
    destination = _assert_shared_nfs(Path(args.destination), "destination")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing destination: {destination}")

    import clip

    device = torch.device(args.device)
    clip_kwargs = {}
    if args.clip_cache_dir:
        clip_kwargs["download_root"] = args.clip_cache_dir
    clip_model, _ = clip.load("RN50", device=device, **clip_kwargs)
    clip_model.eval()
    tasks = _task_names(data_root, args.tasks)
    source = {
        "root": str(data_root),
        "split": "train",
        "layout": "task/all_variations/episodes/episodeN",
        "keypoint_method": "heuristic",
        "rotation_resolution": int(ROTATION_RESOLUTION),
        "voxel_sizes": [int(value) for value in VOXEL_SIZES],
        "episode_length": int(args.episode_length),
        "seed": int(args.seed),
    }
    schema = {
        "payload": "bridgevla_episode_payload",
        "version": 1,
        "cameras": list(CAMERAS),
        "sequence_unit": "heuristic_keypoint",
        "hidden_state": "runtime_only",
    }
    writer = EpisodeStoreWriter(
        destination,
        split="train",
        max_chunk_bytes=args.max_chunk_bytes,
        schema=schema,
        source=source,
    )
    encoded = 0
    started = time.perf_counter()
    try:
        for task in tasks:
            episodes_root = data_root / task / "all_variations" / "episodes"
            episode_ids = _episode_ids(episodes_root)
            if args.max_episodes_per_task:
                episode_ids = episode_ids[: args.max_episodes_per_task]
            for episode_id in episode_ids:
                payload = _encode_episode(
                    task=task,
                    episode_id=episode_id,
                    episodes_root=episodes_root,
                    clip_model=clip_model,
                    device=device,
                    episode_length=args.episode_length,
                )
                writer.add_episode(
                    task,
                    episode_id,
                    payload,
                    valid_length=int(payload["valid_length"]),
                )
                encoded += 1
                print(
                    json.dumps(
                        {
                            "task": task,
                            "episode_id": episode_id,
                            "valid_length": int(payload["valid_length"]),
                            "encoded": encoded,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
        manifest = writer.finalize()
    except Exception:
        writer.abort()
        raise

    print(
        json.dumps(
            {
                "destination": str(destination),
                "tasks": tasks,
                "episodes": encoded,
                "chunks": len(manifest["chunks"]),
                "seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
