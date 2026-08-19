#!/usr/bin/env python3
"""Aggregate repeated RLBench eval runs into paper-style mean +- std.

Reads eval_results.csv files produced by eval.py under
<model_folder>/eval/rlbench_repro/run_<id>/model_80/eval_results.csv
and prints per-task mean/std plus the 18-task average.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

PAPER_ORDER = [
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


def load_run(csv_path: Path) -> dict[str, float]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        task = row["task"].strip()
        sr = row["success rate"].strip()
        if sr in ("", "None", "nan"):
            out[task] = float("nan")
        else:
            out[task] = float(sr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--runs", default="1,2,3,4,5", help="comma-separated run ids")
    ap.add_argument("--log-dir", default="rlbench_repro")
    args = ap.parse_args()

    model_folder = Path(args.model_folder).expanduser().resolve()
    run_ids = [x.strip() for x in args.runs.split(",") if x.strip()]
    all_runs = []
    for run_id in run_ids:
        csv_path = model_folder / "eval" / args.log_dir / f"run_{run_id}" / "model_80" / "eval_results.csv"
        if not csv_path.exists():
            print(f"[missing] {csv_path}")
            continue
        all_runs.append((run_id, load_run(csv_path)))

    if not all_runs:
        raise SystemExit("No eval_results.csv files found")

    task_sets = [set(rows.keys()) for _, rows in all_runs]
    common = set.intersection(*task_sets)
    missing = set.union(*task_sets) - common
    if missing:
        print(f"[warn] tasks present only in some runs: {sorted(missing)}")

    tasks = [t for t in PAPER_ORDER if t in common]
    tasks += sorted(common - set(PAPER_ORDER))

    print(f"\nAggregating {len(all_runs)} runs: {', '.join(r for r, _ in all_runs)}\n")
    print(f"{'task':32s} {'mean':>8s} {'std':>8s}   runs")
    print("-" * 62)
    means = []
    for task in tasks:
        vals = np.array([rows.get(task, np.nan) for _, rows in all_runs], dtype=float)
        if np.isnan(vals).any():
            print(f"{task:32s} {np.nanmean(vals):8.2f} {np.nanstd(vals):8.2f}   (contains nan)")
        else:
            print(f"{task:32s} {vals.mean():8.2f} {vals.std(ddof=1):8.2f}   {vals.tolist()}")
        if not np.isnan(vals).all():
            means.append(np.nanmean(vals))
    avg = float(np.mean(means))
    print("-" * 62)
    print(f"{'18-task average':32s} {avg:8.2f}")
    print("\nNote: std is sample std (ddof=1) across repeated runs.")


if __name__ == "__main__":
    main()
