#!/usr/bin/env python3
# gbw____
"""等待两套五 repeat 评测完成，并生成统一的中文 metric 选择报告。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "outputs/non_euclidean_metric_campaign/online_5repeat"
CO_ROOT = ROOT / "outputs/colosseum_non_euclidean_metric_5repeat_v2"
MANIFEST = ROOT / "outputs/colosseum_a0_official/manifest.json"
FINAL_ROOT = ROOT / "outputs/non_euclidean_metric_5repeat_comparison"
METRICS = ("js_signed_simplex", "local_fused_ot_js")
VARIATIONS = (4, 8, 11)
EXPECTED_COLOSSEUM_CELLS_PER_REPEAT = 50


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rl_complete(metric: str) -> bool:
    for repeat in range(5):
        path = RL_ROOT / "screen" / f"metric_{metric}" / f"repeat_{repeat}" / "summary.json"
        if not path.is_file():
            return False
        if int(_read(path).get("task_count") or 0) != 18:
            return False
    return True


def _col_complete(metric: str) -> bool:
    expected: dict[str, set[tuple[int, str]]] = {
        f"repeat_{repeat}": set() for repeat in range(5)
    }
    for job in _read(MANIFEST).get("valid_jobs", []):
        variation = int(job["variation"])
        if variation in VARIATIONS:
            expected["repeat_0"].add((variation, str(job["base_task"])))

    observed: dict[str, set[tuple[int, str]]] = {
        f"repeat_{repeat}": set() for repeat in range(5)
    }
    observed_counts: dict[str, dict[tuple[int, str], int]] = {
        f"repeat_{repeat}": {} for repeat in range(5)
    }
    for path in (CO_ROOT / metric).glob("repeat_*/**/eval_results.csv"):
        relative = path.relative_to(CO_ROOT / metric)
        repeat_name = relative.parts[0]
        if repeat_name not in observed or len(relative.parts) < 4:
            continue
        try:
            variation = int(relative.parts[1])
        except ValueError:
            continue
        if variation not in VARIATIONS or path.parent.name != "model_80":
            continue
        try:
            rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        except OSError:
            continue
        for row in rows:
            task_name = str(row.get("task") or "")
            if not task_name.endswith(f"_{variation}"):
                continue
            try:
                float(row.get("success rate", ""))
            except (TypeError, ValueError):
                continue
            key = (variation, task_name[: -(len(str(variation)) + 1)])
            observed[repeat_name].add(key)
            observed_counts[repeat_name][key] = observed_counts[repeat_name].get(key, 0) + 1

    return all(
        len(observed[repeat_name]) >= EXPECTED_COLOSSEUM_CELLS_PER_REPEAT
        and expected["repeat_0"] <= observed[repeat_name]
        and all(count == 1 for count in observed_counts[repeat_name].values())
        for repeat_name in observed
    )


def _run_col_summary(metric: str) -> Path:
    out = FINAL_ROOT / "colosseum" / metric
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            str(ROOT / "finetune/Colosseum/summarize_eval.py"),
            "--manifest",
            str(MANIFEST),
            "--output-root",
            str(CO_ROOT / metric),
            "--model-stem",
            "model_80",
            "--repeats",
            "5",
            "--variation-filter",
            "4,8,11",
            "--json-output",
            str(out / "summary_5repeat.json"),
            "--markdown-output",
            str(out / "summary_5repeat_zh.md"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return out / "summary_5repeat.json"


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    return (fmean(values), pstdev(values)) if values else (None, None)


def _diagnostic_mean(metric: str) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for path in (CO_ROOT / metric).glob("repeat_*/filt3r_diagnostics.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("peak_shift_mean_px", "waypoint_shift_mean", "raw_delta_mean", "q_mean", "gain_mean"):
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    totals.setdefault(key, []).append(float(value))
    return {key: fmean(values) for key, values in totals.items() if values}


def _baseline_colosseum(path: Path) -> dict[str, dict[str, float | None]]:
    data = _read(path)
    result: dict[str, dict[str, float | None]] = {}
    for variation in VARIATIONS:
        values = []
        for repeat_data in data.get("per_repeat", {}).values():
            value = repeat_data.get("variation_stats", {}).get(str(variation), {}).get("mean")
            if isinstance(value, (int, float)):
                values.append(float(value))
        mean, std = _mean_std(values)
        result[str(variation)] = {"mean": mean, "std": std}
    return result


def _load_rl(metric: str) -> dict[str, Any]:
    rows = []
    for repeat in range(5):
        rows.append(_read(RL_ROOT / "screen" / f"metric_{metric}" / f"repeat_{repeat}" / "summary.json"))
    values = [float(row["macro_success_rate"]) for row in rows]
    mean, std = _mean_std(values)
    task_scores: dict[str, list[float]] = {}
    for row in rows:
        for task, value in row.get("task_scores", {}).items():
            if isinstance(value, (int, float)):
                task_scores.setdefault(task, []).append(float(value))
    return {
        "repeats": values,
        "mean": mean,
        "std": std,
        "task_mean": {task: fmean(values) for task, values in task_scores.items()},
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _write_report() -> Path:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    col_paths = {metric: _run_col_summary(metric) for metric in METRICS}
    rl = {metric: _load_rl(metric) for metric in METRICS}
    col = {metric: _read(path) for metric, path in col_paths.items()}
    a0 = _baseline_colosseum(ROOT / "outputs/colosseum_a0_official/summary_repaired_3repeat.json")
    a1 = _baseline_colosseum(ROOT / "outputs/colosseum_a1_official/summary_a1_3repeat.json")

    rows = []
    for metric in METRICS:
        across = col[metric]["across_repeats"]
        variation = {}
        for vid in VARIATIONS:
            values = []
            for rep in col[metric]["per_repeat"].values():
                value = rep.get("variation_stats", {}).get(str(vid), {}).get("mean")
                if isinstance(value, (int, float)):
                    values.append(float(value))
            mean, std = _mean_std(values)
            variation[str(vid)] = {
                "mean": mean,
                "std": std,
                "delta_vs_a0": mean - a0[str(vid)]["mean"] if mean is not None and a0[str(vid)]["mean"] is not None else None,
                "delta_vs_a1": mean - a1[str(vid)]["mean"] if mean is not None and a1[str(vid)]["mean"] is not None else None,
            }
        rows.append(
            {
                "metric": metric,
                "rlbench": rl[metric],
                "colosseum": {
                    "official_mean": across.get("official_average_across_repeats"),
                    "official_std": across.get("official_average_population_std_across_repeats"),
                    "variation": variation,
                    "diagnostics": _diagnostic_mean(metric),
                },
            }
        )

    # 当前证据只允许给出“候选选择”，不能把 Colosseum 三 variation 外推成全量 benchmark claim。
    best = max(rows, key=lambda row: row["colosseum"]["official_mean"] or -math.inf)
    report = {
        "protocol": {
            "rlbench": "18 tasks, natural episodes 0-24, 25 episodes/task, 25 steps, 5 repeats",
            "colosseum": "variations 4,8,11 (MO-TEXTURE, Light Color, Distractor), 20 tasks/cell, 25 episodes/cell, 25 steps, 5 repeats",
            "filter": "Stage-1 image tokens -> one W4 gain_space_adaptive Kalman filter -> unchanged decoder; Stage-2 raw; R=1.0; P_init=1.5; reset=none",
        },
        "a0_colosseum": a0,
        "a1_colosseum": a1,
        "rows": rows,
        "recommended_by_representative_colosseum_mean": best["metric"],
        "caveat": "代表 variation 只能用于候选排序，最终选择仍需同时满足 RLBench regression gate 和后续 Colosseum 全量验证。",
    }
    (FINAL_ROOT / "comparison_5repeat.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 非欧式 metric 五重复 RLBench + Colosseum 代表 variation 对比",
        "",
        "## 评测结论",
        "",
        f"按代表性 Colosseum variation 4/8/11 的五重复官方宏平均，当前暂选 `{best['metric']}`；这只是候选排序，不等同于全量 Colosseum 结论。",
        "两个 metric 使用完全相同的 W4 父配置、checkpoint、decoder、episode 计划和 reset=none。",
        "",
        "## RLBench 五重复",
        "",
        "| metric | repeat SR (%) | mean ± std (%) |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['metric']}` | {', '.join(f'{v:.2f}' for v in row['rlbench']['repeats'])} | {_fmt(row['rlbench']['mean'])} ± {_fmt(row['rlbench']['std'])} |")
    lines += [
        "",
        "## Colosseum 代表 variation 五重复",
        "",
        "| metric | variation 4 MO-TEXTURE | variation 8 Light Color | variation 11 Distractor | 三 variation 官方宏平均 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row["colosseum"]["variation"]
        lines.append(
            f"| `{row['metric']}` | {_fmt(values['4']['mean'])} ± {_fmt(values['4']['std'])} | "
            f"{_fmt(values['8']['mean'])} ± {_fmt(values['8']['std'])} | {_fmt(values['11']['mean'])} ± {_fmt(values['11']['std'])} | "
            f"{_fmt(row['colosseum']['official_mean'])} ± {_fmt(row['colosseum']['official_std'])} |"
        )
    lines += [
        "",
        "## 相对既有 Colosseum A0/A1",
        "",
        "| metric | variation | Δ vs A0 (pp) | Δ vs A1 (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        for vid in VARIATIONS:
            item = row["colosseum"]["variation"][str(vid)]
            lines.append(f"| `{row['metric']}` | {vid} | {_fmt(item['delta_vs_a0'])} | {_fmt(item['delta_vs_a1'])} |")
    lines += [
        "",
        "## 解释与限制",
        "",
        "- RLBench 是 regression gate；不能用 Colosseum 代表 variation 的偶然高分抵消 RLBench 的稳定下降。",
        "- variation 4 主要检验局部纹理域偏移，variation 8 检验较 dense 的光照变化，variation 11 检验局部 distractor/错误 peak 风险。",
        "- 代表 variation 结果不能外推到 Colosseum 全部 14 variation；晋级最终 metric 前仍需全量 variation 验证。",
        "- diagnostics 中的 peak/waypoint displacement 只用于解释 filter 如何影响 decoder，不等价于 success rate。",
        "",
        "## 结果路径",
        "",
        f"- RLBench：`{RL_ROOT}`",
        f"- Colosseum：`{CO_ROOT}`",
        f"- 本报告：`{FINAL_ROOT}`",
    ]
    md = FINAL_ROOT / "comparison_5repeat_zh.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.wait:
        while not all(_rl_complete(metric) and _col_complete(metric) for metric in METRICS):
            time.sleep(max(30, args.poll_seconds))
    if not all(_rl_complete(metric) and _col_complete(metric) for metric in METRICS):
        raise SystemExit("五重复评测尚未完整，未生成最终比较报告")
    print(_write_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ____
