#!/usr/bin/env python3
"""Compare reproduced RLBench success rates with the BridgeVLA paper table.

Paper values are the released BridgeVLA mean +- std over repeated runs for
the model_80 checkpoint.
"""
import argparse
from pathlib import Path

PAPER = {
    "close_jar": (100.0, 0.0),
    "reach_and_drag": (100.0, 0.0),
    "insert_onto_square_peg": (88.0, 2.8),
    "meat_off_grill": (100.0, 0.0),
    "open_drawer": (100.0, 0.0),
    "place_cups": (58.4, 10.0),
    "place_wine_at_rack_location": (88.0, 2.8),
    "push_buttons": (98.4, 2.2),
    "put_groceries_in_cupboard": (73.6, 4.6),
    "put_item_in_drawer": (99.2, 1.8),
    "put_money_in_safe": (99.2, 1.8),
    "light_bulb_in": (87.2, 6.6),
    "slide_block_to_color_target": (96.0, 2.8),
    "place_shape_in_shape_sorter": (60.8, 7.7),
    "stack_blocks": (76.8, 8.7),
    "stack_cups": (81.6, 3.6),
    "sweep_to_dustpan_of_size": (87.2, 1.8),
    "turn_tap": (92.8, 3.3),
}
PAPER_AVG = 88.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="path to aggregate_runs.py output txt")
    args = ap.parse_args()

    txt = Path(args.ours).read_text()
    ours = {}
    in_table = False
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("---") or "runs" in line or "Note:" in line or "average" in line.lower():
            if line.startswith("-"):
                in_table = True
            continue
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            task = parts[0]
            try:
                mean = float(parts[1])
                std = float(parts[2])
            except ValueError:
                continue
            if task in PAPER:
                ours[task] = (mean, std)

    print(f"{'task':32s} {'paper':>14s} {'ours':>14s} {'delta':>7s}  verdict")
    print("-" * 80)
    deltas = []
    for task, (pm, ps) in PAPER.items():
        if task not in ours:
            print(f"{task:32s} {pm:6.2f}+-{ps:<6.2f} {'MISSING':>14s}")
            continue
        om, os_ = ours[task]
        d = om - pm
        deltas.append(d)
        within = abs(d) <= max(ps, os_, 1e-9) + 0.5
        verdict = "OK-ish" if within else "GAP"
        print(f"{task:32s} {pm:6.2f}+-{ps:<6.2f} {om:6.2f}+-{os_:<6.2f} {d:+7.2f}  {verdict}")
    if ours:
        avg = sum(m for m, _ in ours.values()) / len(ours)
        print("-" * 80)
        print(f"18-task average: paper={PAPER_AVG:.2f}, ours={avg:.2f}, delta={avg-PAPER_AVG:+.2f}")
        print(f"mean absolute per-task delta: {sum(abs(d) for d in deltas)/len(deltas):.2f}")


if __name__ == "__main__":
    main()
