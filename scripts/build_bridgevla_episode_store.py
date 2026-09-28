#!/usr/bin/env python3
"""Build the episode-aware training store from packed RLBench task archives.

The command intentionally accepts only the shared ``/remote_databuffer``
paths.  Source episodes are streamed from per-task archives (no expanded
small-file tree), turned into chronological keypoint/action payloads, and
written as immutable chunks with ``EpisodeStoreWriter``.  FilterCorrection
state is never serialized.

This is an encoder/smoke entry point for ``encoded_train_v2``.  The training
adapter is a separate change; until it is wired and validated, the normal
legacy replay path remains the only formal training path.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
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
from tar_episode_source import (  # noqa: E402
    archive_path_for_task,
    discover_task_archives,
    iter_episodes_from_archive,
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archives-root",
        required=True,
        help="packed RLBench_TRAIN_DATA archives under /remote_databuffer",
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="number of task-parallel encode workers",
    )
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


def _stack(values: Sequence[object], field: str) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    try:
        return np.stack(arrays, axis=0)
    except ValueError as exc:
        shapes = [tuple(array.shape) for array in arrays]
        raise ValueError(f"inconsistent shapes for {field}: {shapes}") from exc


def _language_embedding(clip_model, description: str, device: torch.device) -> np.ndarray:
    import clip

    tokens = clip.tokenize([description]).to(device)
    with torch.no_grad():
        _, embeddings = _clip_encode_text(clip_model, tokens)
    return embeddings[0].float().detach().cpu().numpy()


def _encode_demo(
    *,
    task: str,
    episode_id: int,
    demo,
    description: str,
    clip_model,
    device: torch.device,
    episode_length: int,
) -> Mapping[str, object]:
    language_embedding = _language_embedding(clip_model, description, device)
    keypoints = list(keypoint_discovery(demo))
    if not keypoints:
        raise ValueError(
            f"episode has no keypoints: {task}#{episode_id}"
        )

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


def _task_names(archives_root: Path, requested: Iterable[str] | None) -> list[str]:
    if requested:
        names = [str(item) for item in requested]
    else:
        names = discover_task_archives(archives_root)
    result = []
    for task in names:
        archive_path_for_task(archives_root, task)
        result.append(task)
    return result


def _source_and_schema(archives_root: Path, episode_length: int, seed: int):
    source = {
        "root": str(archives_root),
        "split": "train",
        "layout": "task.tar.xz!all_variations/episodes/episodeN",
        "keypoint_method": "heuristic",
        "rotation_resolution": int(ROTATION_RESOLUTION),
        "voxel_sizes": [int(value) for value in VOXEL_SIZES],
        "episode_length": int(episode_length),
        "seed": int(seed),
    }
    schema = {
        "payload": "bridgevla_episode_payload",
        "version": 1,
        "cameras": list(CAMERAS),
        "sequence_unit": "heuristic_keypoint",
        "hidden_state": "runtime_only",
    }
    return source, schema


def _encode_task_to_part(job: Mapping[str, object]) -> dict:
    """Worker: encode one task archive into a standalone part store."""
    task = str(job["task"])
    archives_root = Path(str(job["archives_root"]))
    part_dir = Path(str(job["part_dir"]))
    max_episodes = int(job["max_episodes"])
    episode_length = int(job["episode_length"])
    max_chunk_bytes = int(job["max_chunk_bytes"])
    device_name = str(job["device"])
    clip_cache_dir = job.get("clip_cache_dir")
    source = dict(job["source"])
    schema = dict(job["schema"])

    import clip

    device = torch.device(device_name)
    clip_kwargs = {}
    if clip_cache_dir:
        clip_kwargs["download_root"] = str(clip_cache_dir)
    clip_model, _ = clip.load("RN50", device=device, **clip_kwargs)
    clip_model.eval()

    archive_path = archive_path_for_task(archives_root, task)
    if part_dir.exists():
        raise FileExistsError(f"part store already exists: {part_dir}")

    writer = EpisodeStoreWriter(
        part_dir,
        split="train",
        max_chunk_bytes=max_chunk_bytes,
        schema=schema,
        source=source,
    )
    encoded = 0
    started = time.perf_counter()
    try:
        for episode_id, demo, description in iter_episodes_from_archive(archive_path):
            if max_episodes and encoded >= max_episodes:
                break
            payload = _encode_demo(
                task=task,
                episode_id=episode_id,
                demo=demo,
                description=description,
                clip_model=clip_model,
                device=device,
                episode_length=episode_length,
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
    return {
        "task": task,
        "episodes": encoded,
        "chunks": len(manifest["chunks"]),
        "seconds": round(time.perf_counter() - started, 3),
        "part_dir": str(part_dir),
    }


def _merge_part_stores(
    *,
    tasks: Sequence[str],
    parts_dir: Path,
    destination: Path,
    split: str,
    schema: Mapping[str, object],
    source: Mapping[str, object],
) -> dict:
    """Merge per-task part stores into one episode store directory."""
    building = destination.with_name(f".{destination.name}.building-merge")
    if building.exists():
        shutil.rmtree(building)
    chunks_dir = building / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=False)

    merged_tasks: dict[str, dict] = {}
    merged_chunks: list[dict] = []
    episode_total = 0

    for task in sorted(tasks):
        part_manifest_path = parts_dir / task / "manifest.json"
        with part_manifest_path.open("r", encoding="utf-8") as handle:
            part_manifest = json.load(handle)
        local_chunks = part_manifest["chunks"]
        chunk_remap: dict[int, int] = {}
        for local_index, chunk_spec in enumerate(local_chunks):
            global_index = len(merged_chunks)
            chunk_remap[local_index] = global_index
            src = parts_dir / task / chunk_spec["file"]
            dst_name = f"chunks/chunk-{global_index:06d}.bin"
            dst = building / dst_name
            os.link(src, dst)
            merged_chunks.append(
                {
                    "file": dst_name,
                    "bytes": int(chunk_spec["bytes"]),
                    "sha256": str(chunk_spec["sha256"]),
                }
            )

        task_spec = part_manifest["tasks"][task]
        remapped_episodes = []
        for episode in task_spec["episodes"]:
            remapped = dict(episode)
            remapped["chunk"] = chunk_remap[int(episode["chunk"])]
            remapped_episodes.append(remapped)
        merged_tasks[task] = {
            "episode_count": int(task_spec["episode_count"]),
            "episodes": remapped_episodes,
        }
        episode_total += int(task_spec["episode_count"])

    manifest = {
        "format": "bridgevla_episode_store",
        "version": 1,
        "split": split,
        "schema": dict(schema),
        "source": dict(source),
        "chunks": merged_chunks,
        "tasks": merged_tasks,
    }
    manifest_path = building / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    if destination.exists():
        raise FileExistsError(f"refusing to replace existing destination: {destination}")
    os.replace(building, destination)
    return {
        "tasks": sorted(merged_tasks),
        "episodes": episode_total,
        "chunks": len(merged_chunks),
    }


def main() -> int:
    args = _parse_args()
    if args.max_episodes_per_task < 0:
        raise ValueError("max-episodes-per-task must be non-negative")
    if args.episode_length < 2:
        raise ValueError("episode-length must be at least 2")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    archives_root = _assert_shared_nfs(Path(args.archives_root), "archives-root")
    destination = _assert_shared_nfs(Path(args.destination), "destination")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing destination: {destination}")

    tasks = _task_names(archives_root, args.tasks)
    source, schema = _source_and_schema(archives_root, args.episode_length, args.seed)
    parts_dir = destination.with_name(f".{destination.name}.parts-{os.getpid()}")
    if parts_dir.exists():
        raise FileExistsError(f"temporary parts directory already exists: {parts_dir}")
    parts_dir.mkdir(parents=True, exist_ok=False)

    jobs = [
        {
            "task": task,
            "archives_root": str(archives_root),
            "part_dir": str(parts_dir / task),
            "max_episodes": int(args.max_episodes_per_task),
            "episode_length": int(args.episode_length),
            "max_chunk_bytes": int(args.max_chunk_bytes),
            "device": args.device,
            "clip_cache_dir": args.clip_cache_dir,
            "source": source,
            "schema": schema,
        }
        for task in tasks
    ]

    started = time.perf_counter()
    results = []
    context = mp.get_context("spawn")
    try:
        with context.Pool(processes=min(args.jobs, len(jobs))) as pool:
            for result in pool.imap_unordered(_encode_task_to_part, jobs):
                results.append(result)
                print(
                    json.dumps(
                        {
                            "task_done": result["task"],
                            "episodes": result["episodes"],
                            "seconds": result["seconds"],
                            "finished_tasks": len(results),
                            "total_tasks": len(jobs),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
        merged = _merge_part_stores(
            tasks=tasks,
            parts_dir=parts_dir,
            destination=destination,
            split="train",
            schema=schema,
            source=source,
        )
    except Exception:
        shutil.rmtree(parts_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(parts_dir, ignore_errors=True)

    print(
        json.dumps(
            {
                "destination": str(destination),
                "tasks": merged["tasks"],
                "episodes": merged["episodes"],
                "chunks": merged["chunks"],
                "jobs": min(args.jobs, len(jobs)),
                "seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
