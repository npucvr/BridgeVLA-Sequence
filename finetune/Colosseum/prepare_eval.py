#!/usr/bin/env python3
# gbw____
"""Build a read-only, auditable evaluation manifest for Colosseum."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


COLOSSEUM_TASKS: Tuple[str, ...] = (
    "basketball_in_hoop",
    "close_box",
    "close_laptop_lid",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "insert_onto_square_peg",
    "meat_on_grill",
    "move_hanger",
    "open_drawer",
    "place_wine_at_rack_location",
    "put_money_in_safe",
    "reach_and_drag",
    "scoop_with_spatula",
    "setup_chess",
    "slide_block_to_target",
    "stack_cups",
    "straighten_rope",
    "turn_oven_on",
    "wipe_desk",
)

# gbw____
# Colosseum 的论文平均值由 variation 1--14 组成；variation 0 是额外的
# no_variations sanity check，不应混入 paper Average。manifest 仍保留所有
# 已验证 job，便于之后分别运行 strict pass 和 clean sanity pass。
OFFICIAL_VARIATIONS: Tuple[int, ...] = tuple(range(1, 15))
SANITY_VARIATIONS: Tuple[int, ...] = (0,)
# ____

CAMERA_DIRS: Tuple[str, ...] = (
    "front_rgb",
    "front_depth",
    "left_shoulder_rgb",
    "left_shoulder_depth",
    "right_shoulder_rgb",
    "right_shoulder_depth",
    "wrist_rgb",
    "wrist_depth",
)


def _natural_episode_ids(episodes_dir: Path) -> List[int]:
    pattern = re.compile(r"^episode(\d+)$")
    result = []
    if not episodes_dir.is_dir():
        return result
    for entry in episodes_dir.iterdir():
        match = pattern.match(entry.name)
        if match and entry.is_dir():
            result.append(int(match.group(1)))
    return sorted(result)


def _frame_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and not item.name.startswith("."))


def _inspect_episode(episode_dir: Path) -> Dict[str, object]:
    missing = []
    frame_counts: Dict[str, int] = {}
    if not (episode_dir / "low_dim_obs.pkl").is_file():
        missing.append("low_dim_obs.pkl")
    for camera_dir in CAMERA_DIRS:
        path = episode_dir / camera_dir
        count = _frame_count(path)
        frame_counts[camera_dir] = count
        if not path.is_dir():
            missing.append(camera_dir)
        elif count == 0:
            missing.append(f"{camera_dir}:empty")
    return {
        "valid": not missing,
        "missing": missing,
        "frame_counts": frame_counts,
    }


def _variant_dirs(task_dir: Path, task: str) -> List[Tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(task)}_(\d+)$")
    found = []
    if not task_dir.is_dir():
        return found
    for entry in task_dir.iterdir():
        match = pattern.match(entry.name)
        if match and entry.is_dir():
            found.append((int(match.group(1)), entry))
    return sorted(found, key=lambda item: item[0])


def _inspect_variant(
    base_task: str,
    variation: int,
    variant_dir: Path,
    expected_episodes: int,
) -> Dict[str, object]:
    episodes_dir = variant_dir / "variation0" / "episodes"
    episode_ids = _natural_episode_ids(episodes_dir)
    expected_ids = list(range(expected_episodes))
    missing_ids = [episode_id for episode_id in expected_ids if episode_id not in episode_ids]
    extra_ids = [episode_id for episode_id in episode_ids if episode_id not in expected_ids]
    episode_checks = {}
    for episode_id in expected_ids:
        episode_dir = episodes_dir / f"episode{episode_id}"
        if episode_dir.is_dir():
            episode_checks[str(episode_id)] = _inspect_episode(episode_dir)
        else:
            episode_checks[str(episode_id)] = {
                "valid": False,
                "missing": ["episode_directory"],
                "frame_counts": {},
            }
    bad_episode_ids = [
        int(episode_id)
        for episode_id, check in episode_checks.items()
        if not bool(check["valid"])
    ]
    reasons = []
    if not episodes_dir.is_dir():
        reasons.append("missing variation0/episodes")
    if missing_ids:
        reasons.append(f"missing episodes {missing_ids}")
    if bad_episode_ids:
        reasons.append(f"invalid episode contents {bad_episode_ids}")
    return {
        "base_task": base_task,
        "variation": variation,
        "task_name": f"{base_task}_{variation}",
        "variant_dir": str(variant_dir),
        "episodes_dir": str(episodes_dir),
        "episode_ids": expected_ids,
        "discovered_episode_ids": episode_ids,
        "extra_episode_ids": extra_ids,
        "episode_checks": episode_checks,
        "valid": not reasons,
        "reasons": reasons,
    }


def build_manifest(data_root: Path, expected_episodes: int = 25) -> Dict[str, object]:
    # Keep the user-facing symlink path in the manifest.  Path.resolve() would
    # erase the repository-local data entry point even though traversal itself
    # works correctly through the symlink.
    data_root = data_root.expanduser().absolute()
    task_records = []
    valid_jobs = []
    invalid_records = []
    for task in COLOSSEUM_TASKS:
        task_dir = data_root / task
        variants = []
        if not task_dir.is_dir():
            invalid_records.append({"base_task": task, "reasons": ["missing task directory"]})
        for variation, variant_dir in _variant_dirs(task_dir, task):
            record = _inspect_variant(task, variation, variant_dir, expected_episodes)
            variants.append(record)
            if record["valid"]:
                valid_jobs.append(
                    {
                        "base_task": task,
                        "variation": variation,
                        "task_name": record["task_name"],
                        "eval_datafolder": str(task_dir),
                        "episodes": record["episode_ids"],
                    }
                )
            else:
                invalid_records.append(record)
        task_records.append(
            {
                "base_task": task,
                "task_dir": str(task_dir),
                "variation_ids": [record["variation"] for record in variants],
                "variation_count": len(variants),
                "valid_variation_ids": [
                    record["variation"] for record in variants if record["valid"]
                ],
                "variants": variants,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "protocol": {
            "task_count": len(COLOSSEUM_TASKS),
            "tasks": list(COLOSSEUM_TASKS),
            "episode_policy": "natural episode0 through episode24",
            "start_episode": 0,
            "eval_episodes": expected_episodes,
            "episode_length": 25,
            # gbw____
            "variation_policy": "paper scope is variation 1 through 14; variation 0 is a separate sanity check",
            "official_variation_ids": list(OFFICIAL_VARIATIONS),
            "sanity_variation_ids": list(SANITY_VARIATIONS),
            # ____
            "selection_is_random": False,
        },
        "tasks": task_records,
        "valid_jobs": valid_jobs,
        "invalid_records": invalid_records,
        "summary": {
            "task_count": len(COLOSSEUM_TASKS),
            "tasks_with_data": sum(bool(record["variation_ids"]) for record in task_records),
            "valid_job_count": len(valid_jobs),
            "invalid_record_count": len(invalid_records),
        },
    }


def _print_summary(manifest: Dict[str, object]) -> None:
    summary = manifest["summary"]
    print(
        "Colosseum manifest: "
        f"{summary['task_count']} tasks, "
        f"{summary['tasks_with_data']} with data, "
        f"{summary['valid_job_count']} valid task-variation jobs, "
        f"{summary['invalid_record_count']} invalid/missing records"
    )
    for task in manifest["tasks"]:
        print(
            f"  {task['base_task']}: "
            f"variations={task['variation_ids']} "
            f"valid={task['valid_variation_ids']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/data2/local_userdata/gaobowen/VLA/BridgeVLA-Sequence/data/datasets/colosseum_eval",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/colosseum_a0_official/manifest.json"),
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return failure if any expected task/variation/episode is invalid",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    manifest = build_manifest(args.data_root, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(manifest)
    if args.strict and manifest["invalid_records"]:
        print("strict validation failed; see manifest.invalid_records", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ____
