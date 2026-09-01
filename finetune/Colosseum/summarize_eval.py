#!/usr/bin/env python3
# gbw____
"""Summarize official-style Colosseum task/variation evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# gbw____
# variation 0 仅用于 clean sanity；论文中的 Colosseum Average 使用 1--14。
OFFICIAL_VARIATION_IDS = tuple(range(1, 15))
# ____


# gbw____
def _parse_variation_filter(raw: str) -> Optional[Set[int]]:
    """Parse ``all``, comma-separated IDs, and inclusive ranges."""
    if raw.strip().lower() == "all":
        return None
    selected: Set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            if not left.isdigit() or not right.isdigit() or int(left) > int(right):
                raise ValueError(f"invalid variation range: {token}")
            selected.update(range(int(left), int(right) + 1))
        elif token.isdigit():
            selected.add(int(token))
        else:
            raise ValueError(f"invalid variation token: {token}")
    if not selected:
        raise ValueError("variation filter selected no variations")
    return selected
# ____


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pstdev(values: Sequence[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0 if values else None


def _read_result(csv_path: Path, model_stem: str) -> List[Dict[str, object]]:
    if csv_path.parent.name != model_stem:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        try:
            score = float(row["success rate"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append(
            {
                "task_name": row.get("task", ""),
                "success_rate": score,
                "csv_path": str(csv_path),
                "length": row.get("length", ""),
                "total_transitions": row.get("total_transitions", ""),
            }
        )
    return parsed


def _result_key(path: Path, output_root: Path, task_name: str) -> Tuple[str, int, str]:
    relative = path.relative_to(output_root)
    repeat_match = re.match(r"repeat_(\d+)", relative.parts[0])
    variation = int(relative.parts[1])
    # gbw____
    # 统一使用 repeat_0/repeat_1/... 作为内部 key，与 expected_repeat_names
    # 和完整性检查保持一致；此前只返回数字部分会造成混合 key 排序错误。
    repeat = repeat_match.group(0) if repeat_match else "unknown"
    # ____
    return repeat, variation, task_name


def summarize(
    manifest_path: Path,
    output_root: Path,
    model_stem: str,
    variation_filter: str = "1-14",
    repeats: int = 1,
) -> Dict[str, object]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # gbw____
    selected_variations = _parse_variation_filter(variation_filter)
    manifest_variations = {
        int(job["variation"]) for job in manifest.get("valid_jobs", [])
    }
    aggregate_variations = sorted(
        manifest_variations if selected_variations is None
        else manifest_variations.intersection(selected_variations)
    )
    # ____
    output_root = output_root.resolve()
    expected = {}
    # gbw____
    # 即使评测尚未创建 repeat_* 目录，也必须把“应有但尚未产生”的 job
    # 计入完整性检查，避免空输出被错误报告为完整。
    expected_repeat_names = [f"repeat_{repeat}" for repeat in range(repeats)]
    # ____
    for job in manifest.get("valid_jobs", []):
        if selected_variations is not None and int(job["variation"]) not in selected_variations:
            continue
        for repeat_name in expected_repeat_names:
            expected[(repeat_name, int(job["variation"]), job["base_task"])] = False

    records: List[Dict[str, object]] = []
    for csv_path in sorted(output_root.glob("repeat_*/**/eval_results.csv")):
        # gbw____
        # workaround 的每次单 episode 尝试也会写一份 CSV，但它们只是审计
        # 中间产物，不能与最终 cell 汇总重复计数。
        if "official_workaround_attempts" in csv_path.relative_to(output_root).parts:
            continue
        # ____
        for row in _read_result(csv_path, model_stem):
            task_name = str(row["task_name"])
            if not task_name:
                continue
            base_task, separator, variation_text = task_name.rpartition("_")
            if not separator or not variation_text.isdigit():
                continue
            repeat, variation, _ = _result_key(csv_path, output_root, task_name)
            if selected_variations is not None and variation not in selected_variations:
                continue
            row.update(
                {
                    "repeat": repeat,
                    "variation": variation,
                    "base_task": base_task,
                }
            )
            records.append(row)
            # gbw____
            # _result_key 已返回 repeat_0/repeat_1/...，这里直接使用该 key，
            # 避免生成 repeat_repeat_0 并把所有结果误判为缺失。
            expected[(repeat, variation, base_task)] = True
            # ____

    duplicates = []
    unique: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for record in records:
        key = (str(record["repeat"]), int(record["variation"]), str(record["base_task"]))
        if key in unique:
            duplicates.append({"key": list(key), "paths": [unique[key]["csv_path"], record["csv_path"]]})
        else:
            unique[key] = record

    # gbw____
    expected_keys = set(expected)
    unexpected = [list(key) for key in unique if key not in expected_keys]
    # ____

    # gbw____
    # 保留没有任何结果的 repeat 行，令汇总明确显示“0/N”而不是把未启动
    # 的 repeat 从统计中隐去。跨 repeat 均值仍只使用有完整 official_average
    # 的 repeat，完整性由 missing_jobs 单独判定。
    repeat_names = sorted(
        set(expected_repeat_names).union(str(record["repeat"]) for record in records),
        key=lambda name: int(name.split("_", 1)[1]) if name.startswith("repeat_") and name.split("_", 1)[1].isdigit() else name,
    )
    # ____
    per_repeat = {}
    for repeat in repeat_names:
        repeat_records = [record for record in unique.values() if str(record["repeat"]) == repeat]
        task_stats = {}
        for task in sorted({str(record["base_task"]) for record in repeat_records}):
            values = [float(record["success_rate"]) for record in repeat_records if record["base_task"] == task]
            task_stats[task] = {
                "variation_count": len(values),
                "mean": _mean(values),
                "population_std": _pstdev(values),
            }
        # gbw____
        # 官方统计先在每个 variation 内对有效 task 求均值，再对 variation
        # 均值做宏平均；不能先按 task 平均，否则缺失 variation 的 task 会
        # 获得不同权重。
        variation_stats = {}
        for variation in sorted({int(record["variation"]) for record in repeat_records}):
            values = [
                float(record["success_rate"])
                for record in repeat_records
                if int(record["variation"]) == variation
            ]
            variation_stats[str(variation)] = {
                "task_count": len(values),
                "mean": _mean(values),
                "population_std": _pstdev(values),
            }
        official_variation_values = [
            variation_stats[str(variation)]["mean"]
            for variation in aggregate_variations
            if str(variation) in variation_stats
            and variation_stats[str(variation)]["mean"] is not None
        ]
        official_average = _mean(official_variation_values)
        # ____
        all_values = [float(record["success_rate"]) for record in repeat_records]
        per_repeat[repeat] = {
            "result_count": len(repeat_records),
            "task_count": len(task_stats),
            "task_stats": task_stats,
            # gbw____
            "variation_stats": variation_stats,
            "official_average": official_average,
            "official_variation_count": len(official_variation_values),
            # ____
            "task_macro_mean": _mean(
                [float(item["mean"]) for item in task_stats.values() if item["mean"] is not None]
            ),
            "cell_macro_mean": _mean(all_values),
        }

    all_task_stats: Dict[str, List[float]] = {}
    for repeat_data in per_repeat.values():
        for task, stats in repeat_data["task_stats"].items():
            if stats["mean"] is not None:
                all_task_stats.setdefault(task, []).append(float(stats["mean"]))
    repeat_task_macro = [data["task_macro_mean"] for data in per_repeat.values() if data["task_macro_mean"] is not None]

    missing = [list(key) for key, done in sorted(expected.items()) if not done]
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "model_stem": model_stem,
        # gbw____
        "protocol": {
            **manifest.get("protocol", {}),
            "selected_variation_filter": variation_filter,
            "selected_variation_ids": aggregate_variations,
            "official_aggregation": "mean of per-variation means",
            "expected_repeats": repeats,
        },
        # ____
        "records": sorted(unique.values(), key=lambda row: (str(row["repeat"]), int(row["variation"]), str(row["base_task"]))),
        "per_repeat": per_repeat,
        "across_repeats": {
            "repeat_count": len(repeat_task_macro),
            "task_macro_mean_across_repeats": _mean(repeat_task_macro),
            "task_macro_std_across_repeats": _pstdev(repeat_task_macro),
            "task_mean_across_repeats": {
                task: {"mean": _mean(values), "population_std": _pstdev(values), "repeat_count": len(values)}
                for task, values in sorted(all_task_stats.items())
            },
            # gbw____
            "official_average_across_repeats": _mean(
                [
                    float(data["official_average"])
                    for data in per_repeat.values()
                    if data["official_average"] is not None
                ]
            ),
            "official_average_population_std_across_repeats": _pstdev(
                [
                    float(data["official_average"])
                    for data in per_repeat.values()
                    if data["official_average"] is not None
                ]
            ),
            # ____
        },
        "integrity": {
            "expected_job_count": len(expected),
            "observed_unique_result_count": len(unique),
            "missing_jobs": missing,
            "duplicate_rows": duplicates,
            "unexpected_jobs": unexpected,
        },
    }


def _pct(value: float | None) -> str:
    # gbw____
    # evaluator 写入的 success rate 已经是百分数（例如 64.0），不是
    # [0, 1] 比例；这里只负责格式化，不能再次乘 100。
    return "-" if value is None else f"{value:.2f}%"
    # ____


def to_markdown(summary: Dict[str, object]) -> str:
    # gbw____
    protocol = summary["protocol"]
    selected_ids = protocol.get("selected_variation_ids", [])
    official_average = summary["across_repeats"].get("official_average_across_repeats")
    official_std = summary["across_repeats"].get(
        "official_average_population_std_across_repeats"
    )
    # ____
    lines = [
        "# Colosseum A0 评测汇总",
        "",
        f"- 输出目录：`{summary['output_root']}`",
        f"- 模型目录名：`{summary['model_stem']}`",
        f"- 唯一结果数：{summary['integrity']['observed_unique_result_count']} / {summary['integrity']['expected_job_count']}",
        # gbw____
        f"- variation 范围：`{protocol.get('selected_variation_filter', 'unknown')}`，实际纳入 `{selected_ids}`。",
        "- 官方口径：每个 variation 先对该 variation 下的有效 task 求均值，再对 variation 均值做宏平均；variation 0 不进入论文 Average。",
        f"- 官方口径 Average（跨 repeat）：{_pct(official_average)} ± {_pct(official_std)}。",
        # ____
        "",
        "## 每个 repeat",
        "",
        "| repeat | 结果单元数 | task 数 | 官方 variation mean | task macro mean（诊断） | cell macro mean（诊断） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for repeat, data in summary["per_repeat"].items():
        lines.append(
            f"| {repeat} | {data['result_count']} | {data['task_count']} | "
            f"{_pct(data['official_average'])} | {_pct(data['task_macro_mean'])} | {_pct(data['cell_macro_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## variation 统计（官方聚合基础）",
            "",
            "| repeat | variation | 有效 task 数 | mean | population std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for repeat, data in summary["per_repeat"].items():
        for variation, stats in sorted(
            data["variation_stats"].items(), key=lambda item: int(item[0])
        ):
            lines.append(
                f"| {repeat} | {variation} | {stats['task_count']} | "
                f"{_pct(stats['mean'])} | {_pct(stats['population_std'])} |"
            )
    lines.extend(["", "## task 统计", "", "| task | variation 数 | mean | variation population std |", "|---|---:|---:|---:|"])
    task_names = sorted({task for data in summary["per_repeat"].values() for task in data["task_stats"]})
    for task in task_names:
        # The table is intentionally per-repeat so that missing or unstable runs remain visible.
        for repeat, data in summary["per_repeat"].items():
            stats = data["task_stats"].get(task)
            if stats is not None:
                lines.append(
                    f"| {task} ({repeat}) | {stats['variation_count']} | {_pct(stats['mean'])} | {_pct(stats['population_std'])} |"
                )
    integrity = summary["integrity"]
    lines.extend(["", "## 完整性", ""])
    if not integrity["missing_jobs"] and not integrity["duplicate_rows"] and not integrity["unexpected_jobs"]:
        lines.append("- 所有 manifest 中的 valid task-variation job 均有唯一结果行，未发现缺失或重复。")
    else:
        lines.append(f"- 缺失 job：{len(integrity['missing_jobs'])}")
        lines.append(f"- 重复结果：{len(integrity['duplicate_rows'])}")
        lines.append(f"- 非预期结果：{len(integrity['unexpected_jobs'])}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-stem", default="model_80")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="期望的 repeat 数；默认 1。空输出也会按此数量报告缺失 job。",
    )
    # gbw____
    parser.add_argument(
        "--variation-filter",
        default="1-14",
        help="variation 范围：默认 1-14；支持 all、逗号列表和闭区间",
    )
    # ____
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize(
            args.manifest,
            args.output_root,
            args.model_stem,
            variation_filter=args.variation_filter,
            repeats=args.repeats,
        )
    except ValueError as exc:
        parser.error(str(exc))
    json_output = args.json_output or args.output_root / "summary.json"
    markdown_output = args.markdown_output or args.output_root / "summary_zh.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_output.write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0 if (
        not summary["integrity"]["missing_jobs"]
        and not summary["integrity"]["duplicate_rows"]
        and not summary["integrity"]["unexpected_jobs"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
# ____
