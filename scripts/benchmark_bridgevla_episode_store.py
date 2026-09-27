#!/usr/bin/env python3
"""Benchmark direct NFS reads from an episode-aware train store.

This benchmark intentionally has no local-copy mode.  Run it with the store
root under ``/remote_databuffer`` and record the same command, host and mount
for each comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="encoded_train_v2 root")
    parser.add_argument("--task", default="", help="task name, or all tasks")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cache-episodes", type=int, default=0)
    parser.add_argument("--verify-chunks", action="store_true")
    return parser.parse_args()


def _mount_info(path: Path) -> dict[str, str]:
    """Return the filesystem mount serving ``path``.

    The benchmark is deliberately for the shared NFS path.  Refusing a local
    filesystem here prevents an accidental local-copy benchmark from being
    presented as evidence for the direct-NFS data path.
    """
    path = path.resolve()
    best = None
    mountinfo = Path("/proc/self/mountinfo")
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
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
        if not right_fields:
            continue
        if best is None or len(str(mountpoint)) > len(str(best["mountpoint"])):
            best = {"mountpoint": str(mountpoint), "fstype": right_fields[0]}
    if best is None:
        raise RuntimeError(f"could not resolve filesystem mount for {path}")
    if best["fstype"] not in {"nfs", "nfs4"}:
        raise RuntimeError(
            f"episode store must be read directly from NFS; {path} is on "
            f"{best['fstype']} at {best['mountpoint']}"
        )
    return best


def main() -> int:
    args = _parse_args()
    if args.episodes < 1 or args.repeats < 1:
        raise ValueError("episodes and repeats must be positive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune" / "RLBench" / "utils"))
    from episode_store import EpisodeStoreReader

    store = Path(args.store).expanduser().resolve()
    mount = _mount_info(store)
    with EpisodeStoreReader(
        store,
        expected_split="train",
        cache_episodes=args.cache_episodes,
        verify_chunks=args.verify_chunks,
    ) as reader:
        tasks = [args.task] if args.task else reader.task_names
        for task in tasks:
            if task not in reader.task_names:
                raise ValueError(f"unknown task: {task}")
        refs = []
        for task in tasks:
            refs.extend((task, episode_id) for episode_id in reader.episode_ids(task))
        if not refs:
            raise ValueError("selected tasks contain no episodes")
        rng = random.Random(args.seed)
        if len(refs) > args.episodes:
            refs = rng.sample(refs, args.episodes)
        else:
            refs = [refs[index % len(refs)] for index in range(args.episodes)]

        samples = 0
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            payloads = reader.read_batch(refs)
            elapsed = time.perf_counter() - start
            if len(payloads) != len(refs):
                raise RuntimeError("reader returned the wrong number of episodes")
            samples += len(payloads)
            timings.append(elapsed)
        result = {
            "hostname": socket.gethostname(),
            "store": str(store),
            "mount": mount,
            "tasks": tasks,
            "episodes_per_repeat": len(refs),
            "repeats": args.repeats,
            "cache_episodes": args.cache_episodes,
            "seconds": [round(value, 6) for value in timings],
            "episodes_per_second": [round(len(refs) / value, 3) for value in timings],
            "reader_stats": dict(reader.stats),
            "samples": samples,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
