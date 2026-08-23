#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


PAIR_TYPES = ("S->S", "S->F", "F->S", "F->F")
PAIR_FIELDS = [
    "task",
    "episode",
    "shadow_reward",
    "shadow_success",
    "shadow_episode_length",
    "pmf_reward",
    "pmf_success",
    "pmf_episode_length",
    "pair_type",
]
PAIR_SUMMARY_FIELDS = [
    "task",
    "num_pairs",
    "shadow_success_rate",
    "pmf_success_rate",
    "S_to_S",
    "S_to_F",
    "F_to_S",
    "F_to_F",
]


def read_csv(path):
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_jobs(run_root):
    jobs_root = run_root / "jobs"
    jobs = []
    for episodes_path in sorted(jobs_root.rglob("pmf_diagnostics_episodes.csv")):
        relative = episodes_path.relative_to(jobs_root)
        if len(relative.parts) != 4:
            raise ValueError(f"unexpected diagnostics path: {episodes_path}")
        mode, task, _, _ = relative.parts
        model_dir = episodes_path.parent
        jobs.append(
            {
                "mode": mode,
                "task": task,
                "model_dir": model_dir,
                "episodes": episodes_path,
                "steps": model_dir / "pmf_diagnostics_steps.csv",
                "eval_results": model_dir / "eval_results.csv",
            }
        )
    if not jobs:
        raise ValueError(f"no diagnostics jobs found under {jobs_root}")
    return jobs


def eval_success_rate(path, task):
    _, rows = read_csv(path)
    matches = [row for row in rows if row.get("task") == task]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one eval_results row for {task} in {path}, got {len(matches)}"
        )
    try:
        return float(matches[0]["success rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid success rate in {path}") from exc


def load_and_validate_jobs(jobs):
    all_steps = []
    all_episodes = []
    step_fields = None
    episode_fields = None
    seen_jobs = set()

    for job in jobs:
        key = (job["mode"], job["task"])
        if key in seen_jobs:
            raise ValueError(f"duplicate diagnostics job: {key[0]} + {key[1]}")
        seen_jobs.add(key)
        for required in (job["steps"], job["episodes"], job["eval_results"]):
            if not required.is_file():
                raise ValueError(f"missing diagnostics input: {required}")

        current_step_fields, step_rows = read_csv(job["steps"])
        current_episode_fields, episode_rows = read_csv(job["episodes"])
        if step_fields is None:
            step_fields = current_step_fields
        elif step_fields != current_step_fields:
            raise ValueError(f"step CSV header mismatch: {job['steps']}")
        if episode_fields is None:
            episode_fields = current_episode_fields
        elif episode_fields != current_episode_fields:
            raise ValueError(f"episode CSV header mismatch: {job['episodes']}")
        if not episode_rows:
            raise ValueError(f"diagnostics episode CSV is empty: {job['episodes']}")

        for row in step_rows:
            if row.get("mode") != job["mode"] or row.get("task") != job["task"]:
                raise ValueError(f"step context mismatch in {job['steps']}")
        successes = []
        for row in episode_rows:
            if row.get("mode") != job["mode"] or row.get("task") != job["task"]:
                raise ValueError(f"episode context mismatch in {job['episodes']}")
            try:
                successes.append(int(row["success"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid episode success in {job['episodes']}") from exc
        if any(success not in (0, 1) for success in successes):
            raise ValueError(f"episode success must be 0 or 1: {job['episodes']}")

        diagnostic_rate = 100.0 * sum(successes) / len(successes)
        result_rate = eval_success_rate(job["eval_results"], job["task"])
        if abs(diagnostic_rate - result_rate) > 1e-6:
            raise ValueError(
                f"success-rate mismatch for {job['mode']} + {job['task']}: "
                f"diagnostics={diagnostic_rate:.8f}, eval_results={result_rate:.8f}"
            )

        all_steps.extend(step_rows)
        all_episodes.extend(episode_rows)

    all_steps.sort(
        key=lambda row: (
            row["mode"], row["task"], int(row["episode"]), int(row["action_idx"])
        )
    )
    all_episodes.sort(
        key=lambda row: (row["mode"], row["task"], int(row["episode"]))
    )
    return step_fields, all_steps, episode_fields, all_episodes


def pair_episodes(episode_rows):
    by_mode = {"shadow": {}, "pmf": {}}
    for row in episode_rows:
        mode = row["mode"]
        if mode not in by_mode:
            continue
        key = (row["task"], int(row["episode"]))
        if key in by_mode[mode]:
            raise ValueError(f"duplicate {mode} episode: {key[0]} episode {key[1]}")
        by_mode[mode][key] = row

    if not by_mode["shadow"] or not by_mode["pmf"]:
        return []

    shadow_keys = set(by_mode["shadow"])
    pmf_keys = set(by_mode["pmf"])
    if shadow_keys != pmf_keys:
        missing_pmf = sorted(shadow_keys - pmf_keys)
        missing_shadow = sorted(pmf_keys - shadow_keys)
        raise ValueError(
            "shadow/pmf episode keys do not match; "
            f"missing_pmf={missing_pmf}, missing_shadow={missing_shadow}"
        )

    pairs = []
    for task, episode in sorted(shadow_keys):
        shadow = by_mode["shadow"][(task, episode)]
        pmf = by_mode["pmf"][(task, episode)]
        shadow_success = int(shadow["success"])
        pmf_success = int(pmf["success"])
        pair_type = (
            ("S" if shadow_success else "F")
            + "->"
            + ("S" if pmf_success else "F")
        )
        pairs.append(
            {
                "task": task,
                "episode": episode,
                "shadow_reward": shadow["reward"],
                "shadow_success": shadow_success,
                "shadow_episode_length": shadow["episode_length"],
                "pmf_reward": pmf["reward"],
                "pmf_success": pmf_success,
                "pmf_episode_length": pmf["episode_length"],
                "pair_type": pair_type,
            }
        )
    return pairs


def summarize_pairs(pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["task"]].append(pair)

    def summarize(task, task_pairs):
        counts = Counter(pair["pair_type"] for pair in task_pairs)
        count = len(task_pairs)
        return {
            "task": task,
            "num_pairs": count,
            "shadow_success_rate": 100.0
            * sum(int(pair["shadow_success"]) for pair in task_pairs)
            / count,
            "pmf_success_rate": 100.0
            * sum(int(pair["pmf_success"]) for pair in task_pairs)
            / count,
            "S_to_S": counts["S->S"],
            "S_to_F": counts["S->F"],
            "F_to_S": counts["F->S"],
            "F_to_F": counts["F->F"],
        }

    rows = [summarize(task, grouped[task]) for task in sorted(grouped)]
    if pairs:
        rows.append(summarize("ALL", pairs))
    return rows


def print_pair_summary(rows):
    if not rows:
        print("Pair summary not generated: both shadow and pmf episodes are required.")
        return
    print()
    print(
        f"{'Task':34s} {'Shadow SR':>10s} {'PMF SR':>8s} "
        f"{'S->S':>6s} {'S->F':>6s} {'F->S':>6s} {'F->F':>6s}"
    )
    print("-" * 86)
    for row in rows:
        print(
            f"{row['task']:34s} {float(row['shadow_success_rate']):9.2f}% "
            f"{float(row['pmf_success_rate']):7.2f}% "
            f"{int(row['S_to_S']):6d} {int(row['S_to_F']):6d} "
            f"{int(row['F_to_S']):6d} {int(row['F_to_F']):6d}"
        )
    print()


def analyze(run_root):
    jobs = discover_jobs(run_root)
    step_fields, steps, episode_fields, episodes = load_and_validate_jobs(jobs)

    steps_output = run_root / "diagnostics_steps_all.csv"
    episodes_output = run_root / "diagnostics_episodes_all.csv"
    write_csv(steps_output, step_fields, steps)
    write_csv(episodes_output, episode_fields, episodes)

    pairs = pair_episodes(episodes)
    summary_rows = summarize_pairs(pairs)
    outputs = [steps_output, episodes_output]
    if pairs:
        pairs_output = run_root / "diagnostics_episode_pairs.csv"
        summary_output = run_root / "diagnostics_pair_summary.csv"
        write_csv(pairs_output, PAIR_FIELDS, pairs)
        write_csv(summary_output, PAIR_SUMMARY_FIELDS, summary_rows)
        outputs.extend([pairs_output, summary_output])

    print_pair_summary(summary_rows)
    print("Saved:")
    for output in outputs:
        print(f"  {output}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate PMF diagnostics jobs")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.run_root.is_dir():
        parser.error(f"run root does not exist: {args.run_root}")
    try:
        analyze(args.run_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
