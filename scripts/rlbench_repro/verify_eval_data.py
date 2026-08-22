#!/usr/bin/env python3
"""Verify held-out RLBench eval episodes and camera-file consistency.

For each episode, load low_dim_obs.pkl and compare observation length with
the number of files in the camera directories checked by RLBench's
get_stored_demos(). This is the same consistency check eval will perform.
Use --exact-episodes to reject accidentally passing the 100-demo training
split to the paper's 25-episode evaluation protocol.
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
COPPELIA = REPO_ROOT / "finetune" / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = str(COPPELIA) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)

TASKS = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

DIRS = [
    "left_shoulder_rgb", "left_shoulder_depth",
    "right_shoulder_rgb", "right_shoulder_depth",
    "overhead_rgb", "overhead_depth",
    "wrist_rgb", "wrist_depth",
    "front_rgb", "front_depth",
    "left_shoulder_mask", "right_shoulder_mask",
    "overhead_mask", "wrist_mask", "front_mask",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-folder", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument(
        "--exact-episodes",
        action="store_true",
        help="also fail if a task directory contains episodes outside the requested range",
    )
    args = ap.parse_args()

    root = Path(args.data_folder)
    all_ok = True
    expected = set(range(args.start, args.start + args.episodes))
    for task in args.tasks:
        task_ok = True
        if args.exact_episodes:
            episodes_dir = root / task / "all_variations" / "episodes"
            actual = {
                int(p.name[len("episode"):])
                for p in episodes_dir.glob("episode*")
                if p.is_dir() and p.name[len("episode"):].isdigit()
            }
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            if missing:
                print(f"[MISSING-EPISODES] {task}: {missing}")
                all_ok = task_ok = False
            if extra:
                print(f"[EXTRA-EPISODES] {task}: {extra}")
                all_ok = task_ok = False
        for ep in range(args.start, args.start + args.episodes):
            ep_dir = root / task / "all_variations" / "episodes" / f"episode{ep}"
            if not ep_dir.is_dir():
                print(f"[MISSING-DIR] {task}/episode{ep}")
                all_ok = task_ok = False
                continue
            low_dim = ep_dir / "low_dim_obs.pkl"
            if not low_dim.exists():
                print(f"[MISSING] {task}/episode{ep}/low_dim_obs.pkl")
                all_ok = task_ok = False
                continue
            with low_dim.open("rb") as f:
                obs = pickle.load(f)
            n = len(obs)
            bad = False
            for d in DIRS:
                dpath = ep_dir / d
                if not dpath.is_dir():
                    print(f"[MISSING-DIR] {task}/episode{ep}/{d}")
                    all_ok = task_ok = bad = True
                    continue
                nfiles = len(list(dpath.iterdir()))
                if nfiles != n:
                    print(f"[COUNT] {task}/episode{ep}/{d}: obs={n}, files={nfiles}")
                    all_ok = task_ok = bad = True
            if not bad:
                print(f"[OK] {task}/episode{ep}: {n} steps")
        print(f"[TASK {'OK' if task_ok else 'FAIL'}] {task}")
    print("ALL-OK" if all_ok else "ALL-FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
