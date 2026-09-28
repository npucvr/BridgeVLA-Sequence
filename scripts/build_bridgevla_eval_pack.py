#!/usr/bin/env python3
"""Build NFS-efficient eval packs from packed RLBench_EVAL_DATA archives.

Each episode becomes one small tar containing only the metadata files RLBench
eval reset needs (``low_dim_obs.pkl`` / variation number / descriptions).
This avoids expanding millions of camera PNGs onto NFS while keeping Demo
semantics identical to official ``get_stored_demos``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_ROOT = REPO_ROOT / "finetune" / "RLBench" / "utils"
sys.path.insert(0, str(UTILS_ROOT))

from eval_pack import (  # noqa: E402
    FORMAT_NAME,
    FORMAT_VERSION,
    LOW_DIM_PICKLE,
    VARIATION_DESCRIPTIONS_PICKLE,
    VARIATION_NUMBER_PICKLE,
    write_episode_pack,
    write_manifest,
)
from tar_episode_source import (  # noqa: E402
    _episode_id_from_member,
    _open_tar,
    archive_path_for_task,
    discover_task_archives,
)


_META = (
    LOW_DIM_PICKLE,
    VARIATION_NUMBER_PICKLE,
    VARIATION_DESCRIPTIONS_PICKLE,
)


def _assert_shared_nfs(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    shared_root = Path("/remote_databuffer").resolve()
    if path != shared_root and shared_root not in path.parents:
        raise ValueError(
            f"{label} must be under /remote_databuffer; local data paths are forbidden: {path}"
        )
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archives-root",
        required=True,
        help="packed RLBench_EVAL_DATA archives under /remote_databuffer",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="eval pack root under /remote_databuffer",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=6)
    return parser.parse_args()


def _task_names(archives_root: Path, requested: Iterable[str] | None) -> list[str]:
    if requested:
        names = [str(item) for item in requested]
    else:
        names = discover_task_archives(archives_root)
    for task in names:
        archive_path_for_task(archives_root, task)
    return names


def _build_task(job: Mapping[str, object]) -> dict:
    task = str(job["task"])
    archive_path = Path(str(job["archive_path"]))
    destination = Path(str(job["destination"]))
    started = time.perf_counter()

    current_id = None
    current_files: Dict[str, bytes] = {}
    written = 0

    def flush():
        nonlocal current_id, current_files, written
        if current_id is None:
            return
        write_episode_pack(
            root=destination,
            task=task,
            episode_id=current_id,
            files=current_files,
        )
        written += 1
        current_id = None
        current_files = {}

    with _open_tar(archive_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = _episode_id_from_member(member.name)
            if parsed is None:
                continue
            _episode_dir, episode_id, relative = parsed
            name = Path(relative).name if relative else ""
            if name not in _META:
                continue
            if current_id is not None and episode_id != current_id:
                flush()
            current_id = episode_id
            handle = tar.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"archive member is not a file: {member.name}")
            with handle:
                current_files[name] = handle.read()
        flush()

    return {
        "task": task,
        "episodes": written,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    args = _parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    archives_root = _assert_shared_nfs(Path(args.archives_root), "archives-root")
    destination = _assert_shared_nfs(Path(args.destination), "destination")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing destination: {destination}")
    destination.mkdir(parents=True, exist_ok=False)

    tasks = _task_names(archives_root, args.tasks)
    jobs = [
        {
            "task": task,
            "archive_path": str(archive_path_for_task(archives_root, task)),
            "destination": str(destination),
        }
        for task in tasks
    ]

    started = time.perf_counter()
    results = []
    context = mp.get_context("spawn")
    try:
        with context.Pool(processes=min(args.jobs, len(jobs))) as pool:
            for result in pool.imap_unordered(_build_task, jobs):
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
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "split": "held_out",
            "source": {
                "root": str(archives_root),
                "layout": "task.tar!all_variations/episodes/episodeN",
                "fields": list(_META),
            },
            "tasks": {
                item["task"]: {"episode_count": int(item["episodes"])}
                for item in sorted(results, key=lambda row: row["task"])
            },
        }
        write_manifest(destination, manifest)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    episodes = sum(int(item["episodes"]) for item in results)
    print(
        json.dumps(
            {
                "destination": str(destination),
                "tasks": sorted(manifest["tasks"]),
                "episodes": episodes,
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
