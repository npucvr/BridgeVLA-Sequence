#!/usr/bin/env python3
# gbw____
"""Collect 25 valid trials for each known Colosseum data-error cell.

The official workaround excludes simulator/data errors, not policy failures.
Every completed rollout with a numeric reward is therefore a valid trial and
its actual reward remains part of the final success-rate calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# gbw____
SPECIAL_JOBS = frozenset(
    {
        ("close_laptop_lid", 1),
        ("close_laptop_lid", 6),
        ("wipe_desk", 1),
        ("wipe_desk", 6),
        ("insert_onto_square_peg", 1),
        ("insert_onto_square_peg", 6),
    }
)
# ____


# gbw____
def _read_single_result(path: Path) -> Tuple[float, float, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {path}, got {len(rows)}")
    row = rows[0]
    score = float(row["success rate"])
    length = float(row.get("length", "0"))
    transitions = int(float(row.get("total_transitions", "0")))
    return score, length, transitions


def _episode_result_path(attempt_dir: Path, model_stem: str) -> Path:
    return attempt_dir / model_stem / "eval_results.csv"


def _run_one(
    args: argparse.Namespace,
    task: str,
    variation: int,
    episode: int,
    attempt: int,
    attempt_dir: Path,
) -> Tuple[bool, Dict[str, object]]:
    # gbw____
    attempt_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{task}_{variation}"
    command = [
        args.python_bin,
        args.eval_script,
        "--model-folder",
        str(args.model_folder),
        "--eval-datafolder",
        str(args.data_root / task),
        "--tasks",
        task_name,
        "--eval-episodes",
        "1",
        "--start-episode",
        str(episode),
        "--episode-length",
        str(args.episode_length),
        "--log-name",
        str(attempt_dir),
        "--device",
        str(args.device),
        "--headless",
        "--model-name",
        args.model_name,
    ]
    if args.save_video:
        command.append("--save-video")
    env = os.environ.copy()
    # gbw____
    # workaround 的单 episode eval 必须与普通 eval 使用同一方法配置。
    # A0 传入 none，A1 传入 filt3r_akf；其余 FILT3R_* 参数继续从父进程
    # 环境继承，避免异常单元悄悄退回 A0。
    env["FILTER_MODE"] = args.filter_mode
    env["FILTER_DIAGNOSTICS"] = args.filter_diagnostics
    env["FILTER_DIAGNOSTICS_SHADOW_RAW"] = args.filter_diagnostics_shadow_raw
    # ____
    stdout_path = attempt_dir / "stdout.log"
    with stdout_path.open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command,
            cwd=args.eval_script.parent,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = _episode_result_path(attempt_dir, args.model_stem)
    record: Dict[str, object] = {
        "attempt": attempt,
        "source_episode": episode,
        "return_code": completed.returncode,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
    }
    if completed.returncode != 0 or not result_path.is_file():
        record["status"] = "process_error"
        return False, record
    try:
        score, length, transitions = _read_single_result(result_path)
    except (OSError, ValueError, KeyError) as exc:
        record["status"] = "malformed_result"
        record["error"] = str(exc)
        return False, record
    record.update(
        {
            "score": score,
            "length": length,
            "total_transitions": transitions,
            # A numeric score means the rollout completed and is therefore a
            # valid trial.  Keep policy success separate so a 0-score rollout
            # is not discarded by the data-error workaround.
            "valid_trial": True,
            "task_success": score >= 99.999,
            "status": (
                "valid_trial_success"
                if score >= 99.999
                else "valid_trial_failure"
            ),
        }
    )
    return True, record
    # ____


# gbw____
def _write_final_result(
    output_dir: Path,
    task_name: str,
    valid_trials: Sequence[Dict[str, object]],
    model_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = [float(item["length"]) for item in valid_trials]
    transitions = sum(int(item["total_transitions"]) for item in valid_trials)
    scores = [float(item["score"]) for item in valid_trials]
    path = output_dir / model_stem / "eval_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "success rate", "length", "total_transitions"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": task_name,
                # Only invalid simulator/data attempts are discarded.  The
                # task reward observed on the retained trials is the metric.
                "success rate": statistics.fmean(scores) if scores else 0.0,
                "length": statistics.fmean(lengths) if lengths else 0.0,
                "total_transitions": transitions,
            }
        )


def run(args: argparse.Namespace) -> int:
    # gbw____
    if (args.task, args.variation) not in SPECIAL_JOBS:
        raise ValueError(
            f"unsupported workaround cell: {args.task}_{args.variation}; "
            "only the six official data-error cells are allowed"
        )
    if args.successes <= 0 or args.max_attempts < args.successes:
        raise ValueError("successes must be positive and max_attempts >= successes")
    variant_dir = args.data_root / args.task / f"{args.task}_{args.variation}"
    if not (variant_dir / "variation0" / "episodes").is_dir():
        raise FileNotFoundError(f"missing Colosseum variant data: {variant_dir}")

    task_name = f"{args.task}_{args.variation}"
    output_dir = (
        args.output_root
        / f"repeat_{args.repeat}"
        / str(args.variation)
        / args.task
    )
    # gbw____
    # 每次 helper 进程使用独立审计命名空间。旧版本的 attempt 目录可能含有
    # unknown/空 CSV；不能复用它们，否则修复后的 valid-trial 统计会被旧记录
    # 混淆。使用 PID 足以区分本次并行启动的每个异常单元进程。
    # ____
    run_tag = args.run_tag or f"run_pid_{os.getpid()}"
    work_dir = output_dir / "official_workaround_attempts" / run_tag
    valid_trials: List[Dict[str, object]] = []
    attempts: List[Dict[str, object]] = []
    for attempt in range(args.max_attempts):
        episode = attempt % args.source_episodes
        attempt_dir = work_dir / f"attempt_{attempt:04d}_episode_{episode:02d}"
        ok, record = _run_one(
            args,
            args.task,
            args.variation,
            episode,
            attempt,
            attempt_dir,
        )
        attempts.append(record)
        if ok:
            valid_trials.append(record)
            print(
                f"[workaround] {task_name}: "
                f"valid trial {len(valid_trials)}/{args.successes} "
                f"(episode={episode}, attempt={attempt})",
                flush=True,
            )
        if len(valid_trials) >= args.successes:
            break

    metadata = {
        "task": args.task,
        "variation": args.variation,
        "task_name": task_name,
        "target_valid_trials": args.successes,
        "source_episode_count": args.source_episodes,
        "valid_trial_count": len(valid_trials),
        "task_success_count": sum(
            bool(item.get("task_success", False)) for item in valid_trials
        ),
        "attempt_count": len(attempts),
        # gbw____
        "run_tag": run_tag,
        # ____
        "valid_trials": valid_trials,
        "attempts": attempts,
        "protocol": (
            "retry only simulator/data-error attempts; retain the first "
            "25 valid rollouts and compute SR from their observed task rewards"
        ),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if len(valid_trials) < args.successes:
        print(
            f"[error] {task_name}: only {len(valid_trials)}/"
            f"{args.successes} valid trials after {len(attempts)} attempts",
            file=sys.stderr,
        )
        return 2
    _write_final_result(output_dir, task_name, valid_trials, args.model_stem)
    print(f"[workaround] wrote {output_dir / args.model_stem / 'eval_results.csv'}")
    return 0
    # ____


# gbw____
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--model-name", default="model_80.pth")
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--source-episodes", type=int, default=25)
    parser.add_argument("--successes", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=250)
    # gbw____
    parser.add_argument(
        "--filter-mode",
        default=os.environ.get("FILTER_MODE", "none"),
        choices=("none", "filt3r_akf"),
    )
    parser.add_argument(
        "--filter-diagnostics",
        default=os.environ.get("FILTER_DIAGNOSTICS", "0"),
        choices=("0", "1"),
    )
    parser.add_argument(
        "--filter-diagnostics-shadow-raw",
        default=os.environ.get("FILTER_DIAGNOSTICS_SHADOW_RAW", "0"),
        choices=("0", "1"),
    )
    # ____
    # gbw____
    parser.add_argument(
        "--run-tag",
        default=None,
        help="独立审计目录名；省略时自动使用当前 helper PID",
    )
    # ____
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.absolute()
    args.output_root = args.output_root.absolute()
    args.model_folder = args.model_folder.absolute()
    args.python_bin = args.python_bin.absolute()
    args.eval_script = args.eval_script.absolute()
    args.model_stem = Path(args.model_name).stem
    if args.source_episodes <= 0:
        parser.error("--source-episodes must be positive")
    try:
        return run(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
# ____
