"""Summaries and figures generated only from recorded benchmark artifacts."""
from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _json(path: Path, default: Any):
    return json.loads(path.read_text()) if path.exists() else default


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verified(row: dict) -> bool:
    value = row.get("verification")
    if isinstance(value, dict):
        return value.get("ok", value.get("status") in ("ok", "passed")) is True
    return value in (True, "ok", "passed")


def _ok(row: dict) -> bool:
    return row.get("status") == "ok"


def _median(values: list[float]):
    return float(statistics.median(values)) if values else None


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        succeeded = [row for row in members if _ok(row)]
        verified = [row for row in succeeded if _verified(row)]
        result.append({
            **dict(zip(keys, key)),
            "records": len(members),
            "successful_records": len(succeeded),
            "verified_records": len(verified),
            "success_rate": len(succeeded) / len(members),
            "verified_rate": len(verified) / len(members),
            "all_successful": len(succeeded) == len(members),
            "all_verified": len(verified) == len(members),
            "median_total_s": _median([float(r["total_s"]) for r in succeeded if r.get("total_s") is not None]),
            "median_peak_rss_bytes": _median([float(r["peak_rss_bytes"]) for r in succeeded if r.get("peak_rss_bytes") is not None]),
            "status_counts": dict(Counter(row.get("status", "missing") for row in members)),
        })
    return result


def _task_measurements(rows: list[dict], metric: str, dimensions: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in dimensions) + (row.get("task_id"),)
        grouped[key].append(row)
    result = {}
    for key, members in grouped.items():
        values = [float(row[metric]) for row in members
                  if _ok(row) and row.get(metric) is not None
                  and (metric != "total_s" or _verified(row))]
        result[key] = {"median": _median(values), "all_verified": bool(members) and all(
            _ok(row) and _verified(row) for row in members)}
    return result


def _save_figure(fig, output: Path, stem: str, layout_rect=None):
    fig.tight_layout(rect=layout_rect)
    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _empty_axis(ax, message="No records yet"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#555")
    ax.set_axis_off()


def _preprocess_figure(rows: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"adaptive": "#287271", "fixed": "#e09f3e", "native": "#5271a4"}
    time_rows = _task_measurements(rows, "total_s", ("strategy",))
    memory_rows = _task_measurements(rows, "peak_rss_bytes", ("strategy",))
    strategies = {key[0] for key in time_rows}
    if "native" not in strategies or not (strategies & {"adaptive", "fixed"}):
        for ax in axes:
            _empty_axis(ax)
    else:
        ax = axes[0]
        plotted = False
        for strategy in ("fixed", "adaptive"):
            factors = []
            for key, value in time_rows.items():
                if key[0] != strategy or value["median"] is None or not value["all_verified"]:
                    continue
                base = time_rows.get(("native", key[-1]))
                if base and base["all_verified"] and base["median"]:
                    factors.append(base["median"] / value["median"])
            if factors:
                plotted = True
                ax.boxplot(factors, positions=[1 if strategy == "fixed" else 2], widths=.5,
                           patch_artist=True, boxprops={"facecolor": colors[strategy], "alpha": .7})
                x = 1 if strategy == "fixed" else 2
                ax.scatter([x] * len(factors), factors, s=13, alpha=.48, color=colors[strategy])
        if plotted:
            ax.axhline(1.0, color="#444", linestyle="--", linewidth=1)
            ax.set_xticks([1, 2], ["fixed", "adaptive"])
            ax.set_ylabel("Native / strategy time (×)")
            ax.set_title("Paired, verified amplitude runs")
            ax.grid(axis="y", alpha=.22)
            ax.text(.02, .02, "Above 1 means faster. One point per matched circuit.", transform=ax.transAxes, fontsize=8)
        else:
            _empty_axis(ax, "No fully verified paired runs")

        ax = axes[1]
        ratios = []
        labels = []
        for strategy in ("fixed", "adaptive"):
            vals = []
            for key, value in memory_rows.items():
                if key[0] != strategy or value["median"] is None or not value["all_verified"]:
                    continue
                base = memory_rows.get(("native", key[-1]))
                if base and base["all_verified"] and base["median"]:
                    vals.append(base["median"] / value["median"])
            if vals:
                ratios.append(vals)
                labels.append(strategy)
        if ratios:
            ax.boxplot(ratios, tick_labels=labels, showfliers=False)
            ax.axhline(1.0, color="#444", linestyle="--", linewidth=1)
            ax.set_ylabel("Native / strategy peak RSS (×)")
            ax.set_title("Paired process peak memory")
            ax.grid(axis="y", alpha=.22)
            ax.text(.02, .02, "Above 1 means lower peak RSS.", transform=ax.transAxes, fontsize=8)
        else:
            _empty_axis(ax, "No matched memory rows")
    fig.suptitle("Preprocessing: total cost and memory")
    return _save_figure(fig, output, "preprocess")


def _selector_figure(rows: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    policies = ("fixed", "raw", "effective")
    labels, medians, feature_medians, rates = [], [], [], []
    for policy in policies:
        members = [row for row in rows if row.get("policy") == policy]
        if not members:
            continue
        labels.append(policy)
        successful = [row for row in members if _ok(row)]
        medians.append(_median([float(row["total_s"]) for row in successful if row.get("total_s") is not None]))
        overhead = []
        for row in members:
            route = row.get("routing", {})
            times = route.get("timings_s", {})
            overhead.append(float(times.get("features", times.get("raw_features", 0.0))) +
                            float(times.get("inference", 0.0)))
        feature_medians.append(_median(overhead) or 0.0)
        rates.append(len(successful) / len(members))
    if labels:
        ax = axes[0]
        x = np.arange(len(labels))
        base = np.array([value or 0.0 for value in medians])
        overhead = np.array(feature_medians)
        ax.bar(x, np.maximum(0, base - overhead), color="#5271a4", label="selected backend work")
        ax.bar(x, np.minimum(base, overhead), bottom=np.maximum(0, base - overhead), color="#e09f3e", label="routing features + prediction")
        ax.set_ylim(0, max(float(np.max(base)) * 1.15, 0.005))
        ax.set_xticks(x, labels)
        ax.set_ylabel("Median end-to-end time (s)")
        ax.set_title("Online routing cost")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=.22)
        ax2 = axes[1]
        ax2.bar(labels, rates, color="#287271")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Completed replay fraction")
        ax2.set_title("Selected route completion")
        ax2.grid(axis="y", alpha=.22)
        for idx, (rate, policy) in enumerate(zip(rates, labels)):
            ax2.text(idx, rate + .025, f"{rate:.0%}", ha="center", fontsize=9)
    else:
        for ax in axes:
            _empty_axis(ax)
    fig.suptitle("Effective-structure backend selection")
    return _save_figure(fig, output, "selector")


def _batch_figure(rows: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    methods = ("independent", "shared_graph", "quimb_cached")
    colors = {"independent": "#5271a4", "shared_graph": "#287271", "quimb_cached": "#e09f3e"}
    series = {}
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        if not method_rows:
            continue
        points = []
        for size in sorted({int(row["batch_size"]) for row in method_rows}):
            group = [row for row in method_rows if int(row["batch_size"]) == size]
            successful = [row for row in group if _ok(row) and row.get("total_s") is not None]
            points.append((size, _median([float(row["total_s"]) for row in successful]), len(successful), len(group)))
        series[method] = points
    if not series:
        for ax in axes:
            _empty_axis(ax)
    else:
        ax = axes[0]
        for method, points in series.items():
            xs = [p[0] for p in points if p[1] is not None]
            ys = [p[1] for p in points if p[1] is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=method, color=colors[method])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Queries sharing the same skeleton")
        ax.set_ylabel("Median total batch time (s)")
        ax.set_title("Cumulative batch cost")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(which="both", alpha=.22)

        ax = axes[1]
        for method, points in series.items():
            xs = [p[0] for p in points if p[1] is not None]
            ys = [p[1] / p[0] for p in points if p[1] is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=method, color=colors[method])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Queries sharing the same skeleton")
        ax.set_ylabel("Median time per query (s)")
        ax.set_title("Preparation amortization")
        ax.grid(which="both", alpha=.22)
    fig.suptitle("Repeated T/T† filling queries")
    return _save_figure(fig, output, "batch")


def _batch_by_task_figure(rows: list[dict], tasks: list[dict], output: Path):
    task_by_id = {task["id"]: task for task in tasks}
    task_ids = sorted({row["task_id"] for row in rows})
    if not task_ids:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _empty_axis(ax)
        fig.suptitle("Batch cost by circuit")
        return _save_figure(fig, output, "batch_by_task")

    cols = min(3, len(task_ids))
    rows_count = (len(task_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.7 * cols, 3.6 * rows_count), squeeze=False)
    colors = {"independent": "#5271a4", "shared_graph": "#287271", "quimb_cached": "#e09f3e"}
    methods = ("independent", "shared_graph", "quimb_cached")
    for index, task_id in enumerate(task_ids):
        ax = axes.flat[index]
        task = task_by_id.get(task_id, {})
        local = [row for row in rows if row["task_id"] == task_id]
        for method in methods:
            sizes = sorted({int(row["batch_size"]) for row in local if row.get("method") == method})
            xs, ys = [], []
            for size in sizes:
                repeats = [row for row in local
                           if row.get("method") == method and int(row["batch_size"]) == size
                           and _ok(row) and _verified(row) and row.get("total_s") is not None]
                if repeats:
                    xs.append(size)
                    ys.append(_median([float(row["total_s"]) for row in repeats]))
            if xs:
                ax.plot(xs, ys, marker="o", label=method, color=colors[method])
        n = task.get("n", "?")
        ax.set_title(f"{task.get('family', 'unknown')} · {task_id}\nphysical qubits: {n}", fontsize=9)
        ax.set_xlabel("Batch size (queries)")
        ax.set_ylabel("Median total batch time (s)")
        if len({row.get("batch_size") for row in local}) > 1:
            ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(which="both", alpha=.22)
        if index == 0:
            ax.legend(frameon=False, fontsize=8)
    for index in range(len(task_ids), rows_count * cols):
        axes.flat[index].set_axis_off()
    fig.suptitle("Batch cost by circuit (only verified runs)")
    return _save_figure(fig, output, "batch_by_task")


def _batch_break_even(rows: list[dict], tasks: list[dict]) -> list[dict]:
    task_by_id = {task["id"]: task for task in tasks}
    output = []
    for task_id in sorted({row["task_id"] for row in rows}):
        local = [row for row in rows if row["task_id"] == task_id]
        sizes = sorted({int(row["batch_size"]) for row in local})
        for size in sizes:
            rowsets = {}
            medians = {}
            for method in ("independent", "shared_graph", "quimb_cached"):
                repeats = [row for row in local if row.get("method") == method
                           and int(row["batch_size"]) == size]
                rowsets[method] = repeats
                if repeats and all(_ok(row) and _verified(row) for row in repeats):
                    medians[method] = _median([float(row["total_s"]) for row in repeats])
                else:
                    medians[method] = None
            metadata = next((row.get("metadata", {}) for row in rowsets["shared_graph"]), {})
            output.append({
                "task_id": task_id,
                "family": task_by_id.get(task_id, {}).get("family"),
                "physical_qubits": task_by_id.get(task_id, {}).get("n"),
                "batch_size": size,
                "unique_query_count": metadata.get("unique_query_count"),
                "repeats": len(rowsets["shared_graph"]),
                "independent_repeats_verified": bool(rowsets["independent"]) and medians["independent"] is not None,
                "shared_graph_repeats_verified": bool(rowsets["shared_graph"]) and medians["shared_graph"] is not None,
                "quimb_cached_repeats_verified": bool(rowsets["quimb_cached"]) and medians["quimb_cached"] is not None,
                "all_repeats_verified": all(bool(rowsets[method]) and medians[method] is not None
                                            for method in ("independent", "shared_graph", "quimb_cached")),
                "median_independent_total_s": medians["independent"],
                "median_shared_graph_total_s": medians["shared_graph"],
                "median_quimb_cached_total_s": medians["quimb_cached"],
                "shared_first_beats_independent": medians["shared_graph"] < medians["independent"]
                    if medians["shared_graph"] is not None and medians["independent"] is not None else None,
                "shared_first_beats_quimb_cached": medians["shared_graph"] < medians["quimb_cached"]
                    if medians["shared_graph"] is not None and medians["quimb_cached"] is not None else None,
                "independent_pair_scope": "Requires all independent and shared_graph repeats to pass verification.",
                "quimb_cached_pair_scope": "Requires all quimb_cached and shared_graph repeats to pass verification.",
            })

    by_task = defaultdict(list)
    for row in output:
        by_task[row["task_id"]].append(row)
    summaries = []
    for task_id, groups in sorted(by_task.items()):
        summaries.append({
            "task_id": task_id,
            "family": groups[0]["family"],
            "physical_qubits": groups[0]["physical_qubits"],
            "unique_query_counts_by_batch_size": {str(row["batch_size"]): row["unique_query_count"] for row in groups},
            "first_tested_shared_beats_independent_at_k": next(
                (row["batch_size"] for row in groups if row["independent_repeats_verified"]
                 and row["shared_graph_repeats_verified"]
                 and row["shared_first_beats_independent"]), None),
            "first_tested_shared_beats_quimb_cached_at_k": next(
                (row["batch_size"] for row in groups if row["quimb_cached_repeats_verified"]
                 and row["shared_graph_repeats_verified"]
                 and row["shared_first_beats_quimb_cached"]), None),
            "tested_batch_sizes": [row["batch_size"] for row in groups],
        })
    return {"detail": output, "summary": summaries}


def _stage_breakdown(rows: list[dict]):
    stages = ("prepare", "plan", "evaluate")
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("family"), row.get("strategy"))].append(row)
    summary = []
    for (family, strategy), members in sorted(groups.items()):
        good = [row for row in members if _ok(row)]
        residuals = [float(row["total_s"]) - sum(float(row.get("timings_s", {}).get(name, 0.0))
                                                   for name in stages)
                     for row in good if row.get("total_s") is not None]
        stage_medians = {
            name: _median([float(row.get("timings_s", {}).get(name, 0.0)) for row in good])
            for name in stages
        }
        stage_medians["other"] = _median(residuals)
        summary.append({
            "family": family,
            "strategy": strategy,
            "records": len(members),
            "successful_records": len(good),
            "verified_records": sum(_verified(row) for row in good),
            "median_prepare_s": stage_medians["prepare"],
            "median_plan_s": stage_medians["plan"],
            "median_evaluate_s": stage_medians["evaluate"],
            "median_other_s": stage_medians["other"],
            "sum_of_stage_medians_s": sum(stage_medians.values()) if good else None,
            "median_end_to_end_total_s": _median([float(row["total_s"]) for row in good if row.get("total_s") is not None]),
            "aggregation_note": "Stage medians are computed independently; their sum need not equal the median end-to-end time.",
        })
    return summary


def _raw_timing_residuals(record_sets: dict[str, list[dict]]):
    """Audit total_s against the sum of recorded top-level stage timers."""
    summary = {}
    all_values = []
    tolerance_s = 1e-9
    for name, rows in record_sets.items():
        values = [float(row["total_s"]) - sum(float(value) for value in (row.get("timings_s") or {}).values())
                  for row in rows if row.get("total_s") is not None]
        all_values.extend(values)
        summary[name] = {
            "records_with_total_s": len(values),
            "residual_definition": "total_s minus sum of top-level timings_s fields",
            "min_other_s": min(values) if values else None,
            "max_other_s": max(values) if values else None,
            "negative_below_tolerance_count": sum(value < -tolerance_s for value in values),
            "negative_tolerance_s": tolerance_s,
        }
    summary["all_experiments"] = {
        "records_with_total_s": len(all_values),
        "min_other_s": min(all_values) if all_values else None,
        "max_other_s": max(all_values) if all_values else None,
        "negative_below_tolerance_count": sum(value < -tolerance_s for value in all_values),
        "negative_tolerance_s": tolerance_s,
        "residual_definition": "total_s minus sum of top-level timings_s fields; nested selector routing timers are included in other_s",
    }
    return summary


def _structure_tradeoffs(rows: list[dict], tasks: list[dict]):
    """Summarize rank-step structure changes against complete adaptive/native timings."""
    adaptive_by_task = defaultdict(list)
    strategies_by_task = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task_id = row.get("task_id")
        strategy = row.get("strategy")
        strategies_by_task[task_id][strategy].append(row)
        if strategy == "adaptive" and isinstance(row.get("structure"), dict):
            adaptive_by_task[task_id].append(row)

    task_by_id = {task["id"]: task for task in tasks}
    counts = {
        "tasks_with_rank_step_metrics": 0,
        "rank_decrease_tasks": 0,
        "L_increase_at_rank_step_tasks": 0,
        "estimated_combo_comparable_at_rank_step_tasks": 0,
        "estimated_combo_increase_at_rank_step_tasks": 0,
        "estimated_combo_comparable_all_tasks": 0,
        "estimated_combo_increase_all_tasks": 0,
    }
    examples = []
    for task_id, adaptive_rows in sorted(adaptive_by_task.items()):
        row = adaptive_rows[0]
        structure = row["structure"]
        reduced = structure.get("reduced")
        ranked = structure.get("rank")
        if not reduced or not ranked:
            continue
        counts["tasks_with_rank_step_metrics"] += 1
        rank_decreased = int(ranked["r"]) < int(reduced["r"])
        if rank_decreased:
            counts["rank_decrease_tasks"] += 1
            if int(ranked["L"]) > int(reduced["L"]):
                counts["L_increase_at_rank_step_tasks"] += 1

        candidates = {candidate.get("name"): candidate for candidate in row.get("candidates", [])}
        rank_candidate = candidates.get("rank", {})
        reduced_candidate = candidates.get("reduced", {})
        rank_combo = rank_candidate.get("combo")
        reduced_combo = reduced_candidate.get("combo")
        if rank_combo is not None and reduced_combo is not None:
            counts["estimated_combo_comparable_all_tasks"] += 1
            combo_increased = float(rank_combo) > float(reduced_combo)
            if combo_increased:
                counts["estimated_combo_increase_all_tasks"] += 1
            if rank_decreased:
                counts["estimated_combo_comparable_at_rank_step_tasks"] += 1
                if combo_increased:
                    counts["estimated_combo_increase_at_rank_step_tasks"] += 1

        if not rank_decreased:
            continue
        strategy_rows = strategies_by_task[task_id]
        native_rows = strategy_rows.get("native", [])
        adaptive_rows = strategy_rows.get("adaptive", [])
        native_by_repeat = {r.get("repeat"): r for r in native_rows}
        adaptive_by_repeat = {r.get("repeat"): r for r in adaptive_rows}
        if (not native_by_repeat or set(native_by_repeat) != set(adaptive_by_repeat)
                or not all(_ok(r) and _verified(r) for r in native_by_repeat.values())
                or not all(_ok(r) and _verified(r) for r in adaptive_by_repeat.values())):
            continue
        native_times = [float(r["total_s"]) for r in native_by_repeat.values()]
        adaptive_times = [float(r["total_s"]) for r in adaptive_by_repeat.values()]
        native_median = _median(native_times)
        adaptive_median = _median(adaptive_times)
        if native_median is None or adaptive_median is None or adaptive_median <= native_median:
            continue
        examples.append({
            "task_id": task_id,
            "family": task_by_id.get(task_id, {}).get("family"),
            "physical_qubits": task_by_id.get(task_id, {}).get("n"),
            "verified_repeats_per_strategy": len(native_times),
            "r_reduced": int(reduced["r"]),
            "r_rank": int(ranked["r"]),
            "L_reduced": int(reduced["L"]),
            "L_rank": int(ranked["L"]),
            "median_native_total_s": native_median,
            "median_adaptive_total_s": adaptive_median,
            "adaptive_over_native": adaptive_median / native_median,
            "adaptive_minus_native_s": adaptive_median - native_median,
        })
    examples.sort(key=lambda item: item["adaptive_over_native"], reverse=True)
    return {
        "comparison": "structure reduced -> rank candidate; timings compare full adaptive and native workflows",
        "counts": counts,
        "verified_slowdown_examples": examples[:3],
        "example_selection": "Top three adaptive/native median-time ratios among rank-decrease tasks with all matched repeats verified.",
        "interpretation": "A smaller rank or L does not establish a runtime cause; adaptive total includes candidate generation, planning, evaluation, and task overhead.",
    }


def _selector_penalized_summary(rows: list[dict], timeout_s: float):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("policy")].append(row)
    summary = {}
    for policy, members in sorted(groups.items()):
        task_costs = defaultdict(float)
        success = []
        for row in members:
            completed = _ok(row) and _verified(row)
            cost = float(row.get("total_s", 0.0)) if completed else 2.0 * timeout_s
            task_costs[row.get("task_id")] += cost
            success.append(completed)
        summary[policy] = {
            "recorded_replays": len(members),
            "unique_tasks": len(task_costs),
            "successful_verified_replays": sum(success),
            "completion_rate": sum(success) / len(success) if success else None,
            "par2_penalty_s_per_failure": 2.0 * timeout_s,
            "total_par2_s_over_recorded_replays": sum(task_costs.values()),
            "median_per_task_par2_s": _median(list(task_costs.values())),
            "scope": "Each timed-out, unsupported, failed, or unverified replay is assigned twice the configured timeout.",
        }
    matched_keys = defaultdict(dict)
    for row in rows:
        matched_keys[(row.get("task_id"), row.get("repeat"))][row.get("policy")] = row
    common = [entries for entries in matched_keys.values()
              if all(policy in entries for policy in ("fixed", "raw", "effective"))]
    matched_totals = {policy: 0.0 for policy in ("fixed", "raw", "effective")}
    for entries in common:
        for policy in matched_totals:
            row = entries[policy]
            matched_totals[policy] += (float(row["total_s"]) if _ok(row) and _verified(row)
                                       else 2.0 * timeout_s)
    return {"per_policy": summary,
            "complete_matched_task_repeats": len(common),
            "par2_totals_on_common_task_repeats": matched_totals}


def _stage_figure(stage_rows: list[dict], output: Path):
    families = sorted({row["family"] for row in stage_rows})
    strategies = ("native", "fixed", "adaptive")
    stages = (("median_prepare_s", "Preparation", "#5271a4"),
              ("median_plan_s", "Planning", "#e09f3e"),
              ("median_evaluate_s", "Evaluation", "#287271"),
              ("median_other_s", "Other task overhead (incl. parsing/optimizer setup)", "#8b6f9c"))
    if not stage_rows:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _empty_axis(ax)
        fig.suptitle("Preprocessing stage medians")
    else:
        lookup = {(row["family"], row["strategy"]): row for row in stage_rows}
        cols = min(3, len(families))
        rows_count = (len(families) + cols - 1) // cols
        fig, axes = plt.subplots(rows_count, cols, figsize=(4.2 * cols, 3.1 * rows_count), squeeze=False)
        for index, family in enumerate(families):
            ax = axes.flat[index]
            bottoms = np.zeros(len(strategies))
            for field, label, color in stages:
                values = [float(lookup[(family, strategy)][field] or 0.0)
                          if (family, strategy) in lookup else 0.0 for strategy in strategies]
                ax.bar(np.arange(len(strategies)), values, bottom=bottoms, width=.62,
                       color=color, label=label)
                bottoms += np.asarray(values)
            ax.set_xticks(np.arange(len(strategies)), strategies)
            ax.set_title(family)
            ax.set_ylabel("Median stage time (s)")
            ax.grid(axis="y", alpha=.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncols=2, fontsize=8,
                   loc="upper center", bbox_to_anchor=(0.5, 0.93))
        for index in range(len(families), rows_count * cols):
            axes.flat[index].set_axis_off()
        fig.suptitle("Preprocessing stage medians (stack heights sum stage medians, not median total)")
        return _save_figure(fig, output, "preprocess_stages", layout_rect=[0, 0, 1, 0.86])
    return _save_figure(fig, output, "preprocess_stages")


def _write_csv(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _status_summary(rows: list[dict]):
    return {"records": len(rows), "status_counts": dict(Counter(row.get("status", "missing") for row in rows)),
            "verified_records": sum(_verified(row) for row in rows if _ok(row))}


def _paired(rows: list[dict], field: str, a: str, b: str, metric: str = "total_s"):
    grouped = _task_measurements(rows, metric, (field,))
    pairs = []
    ids = sorted({key[-1] for key in grouped})
    for task_id in ids:
        left = grouped.get((a, task_id))
        right = grouped.get((b, task_id))
        if (left and right and left["all_verified"] and right["all_verified"]
                and left["median"] is not None and right["median"] is not None and right["median"] > 0):
            pairs.append({"task_id": task_id, "baseline": left["median"], "candidate": right["median"],
                          "baseline_over_candidate": left["median"] / right["median"]})
    return {"matched_verified_tasks": len(pairs),
            "median_baseline_over_candidate": _median([r["baseline_over_candidate"] for r in pairs]),
            "faster_candidate_tasks": sum(r["baseline_over_candidate"] > 1 for r in pairs),
            "slower_candidate_tasks": sum(r["baseline_over_candidate"] < 1 for r in pairs),
            "pairs": pairs}


def generate_report(output_dir: str | Path) -> list[str]:
    """Generate portable summaries and publication-ready charts from run records."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _json(output / "config.json", {})
    tasks = _json(output / "tasks.json", [])
    env = _json(output / "environment.json", {})
    preprocess = _rows(output / "preprocess.jsonl")
    backends = _rows(output / "backends.jsonl")
    selector = _rows(output / "selection_replay.jsonl")
    batch = _rows(output / "batch.jsonl")
    references = _rows(output / "references.jsonl")
    models = _json(output / "selection_models.json", {})
    oof = _json(output / "selection_oof.json", {})
    timing_residuals = _raw_timing_residuals({
        "preprocess": preprocess,
        "backends": backends,
        "selection_replay": selector,
        "batch": batch,
    })

    aggregates = {
        "preprocess": _aggregate(preprocess, ("family", "strategy")),
        "backends": _aggregate(backends, ("family", "backend")),
        "selector": _aggregate(selector, ("family", "policy")),
        "batch": _aggregate(batch, ("family", "method", "batch_size")),
    }
    for row in aggregates["batch"]:
        matching = [item for item in batch if (item.get("family"), item.get("method"),
                    item.get("batch_size")) == (row["family"], row["method"], row["batch_size"])]
        unique_counts = [item.get("metadata", {}).get("unique_query_count") for item in matching
                         if item.get("metadata", {}).get("unique_query_count") is not None]
        row["median_unique_query_count"] = _median([float(value) for value in unique_counts])

    for row in aggregates["selector"]:
        matching = [item for item in selector if (item.get("family"), item.get("policy"))
                    == (row["family"], row["policy"])]
        features = [float(item.get("routing", {}).get("timings_s", {}).get("features", 0.0))
                    for item in matching]
        inference = [float(item.get("routing", {}).get("timings_s", {}).get("inference", 0.0))
                     for item in matching]
        row["median_feature_extraction_s"] = _median(features)
        row["median_model_inference_s"] = _median(inference)
        row["median_routing_overhead_s"] = _median([a + b for a, b in zip(features, inference)])
    for name, rows in aggregates.items():
        _write_csv(output / f"{name}_summary.csv", rows)

    stage_rows = _stage_breakdown(preprocess)
    _write_csv(output / "stage_breakdown.csv", stage_rows)
    batch_break_even = _batch_break_even(batch, tasks)
    _write_csv(output / "batch_break_even.csv", batch_break_even["detail"])
    _write_csv(output / "batch_by_task_summary.csv", batch_break_even["summary"])

    preprocess_pairs = {strategy: _paired(preprocess, "strategy", "native", strategy)
                        for strategy in ("fixed", "adaptive")}
    selector_pairs = {policy: _paired(selector, "policy", "fixed", policy)
                      for policy in ("raw", "effective")}
    selector_pairs["effective_vs_raw"] = _paired(selector, "policy", "raw", "effective")
    timeout_s = float(config.get("timeout_s", 60.0))
    selector_penalized = _selector_penalized_summary(selector, timeout_s)
    structure_tradeoffs = _structure_tradeoffs(preprocess, tasks)
    batch_pairs = {}
    batch_groups = sorted({(row.get("family"), int(row["batch_size"])) for row in batch})
    for family, size in batch_groups:
        local = [row for row in batch if row.get("family") == family and int(row["batch_size"]) == size]
        batch_pairs[f"{family}:batch={size}"] = {
            "shared_vs_independent": _paired(local, "method", "independent", "shared_graph"),
            "shared_vs_quimb_cached": _paired(local, "method", "quimb_cached", "shared_graph"),
        }
    summary = {
        "experiment": config.get("name"),
        "task_count": len(tasks),
        "family_counts": dict(Counter(task.get("family", "unknown") for task in tasks)),
        "environment": env,
        "records": {"preprocess": _status_summary(preprocess), "backends": _status_summary(backends),
                    "selector_replay": _status_summary(selector), "batch": _status_summary(batch),
                    "references": {"records": len(references),
                                   "reference_kinds": dict(Counter(r.get("reference_kind", "unknown") for r in references)),
                                   "crosscheck_failures": sum(r.get("status") == "error" for r in references)}},
        "timing_accounting": timing_residuals,
        "paired_verified_preprocess": preprocess_pairs,
        "paired_verified_selector": selector_pairs,
        "selector_completion_and_par2": selector_penalized,
        "paired_verified_batch": batch_pairs,
        "batch_by_task": batch_break_even["summary"],
        "preprocess_stage_breakdown": stage_rows,
        "structure_performance_tradeoffs": structure_tradeoffs,
        "selection_fold_count": len(models.get("folds", {})),
        "selection_oof_summary": oof.get("summary", {}) if isinstance(oof, dict) else {},
        "scope_note": "Run-local measurements on the recorded CPU and workload only; no general speedup claim is inferred from structure metrics.",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    figures = []
    figures.extend(_preprocess_figure(preprocess, output))
    figures.extend(_selector_figure(selector, output))
    figures.extend(_batch_figure(batch, output))
    figures.extend(_batch_by_task_figure(batch, tasks, output))
    figures.extend(_stage_figure(stage_rows, output))

    total_records = len(preprocess) + len(backends) + len(selector) + len(batch)
    verified = (summary["records"]["preprocess"]["verified_records"] +
                summary["records"]["backends"]["verified_records"] +
                summary["records"]["selector_replay"]["verified_records"])
    lines = [
        f"# Results: {config.get('name', output.name)}", "",
        "This report is generated from the JSONL run records. It does not treat a lower variable count or L value as an application speedup.", "",
        "## Run scope", "",
        f"- Workload: {len(tasks)} tasks across {len(summary['family_counts'])} circuit families.",
        f"- Recorded environment: Python {env.get('python', 'not recorded')}, {env.get('architecture', 'unknown architecture')}, {env.get('logical_cpus', 'unknown')} logical CPUs; experiment workers use one numerical thread.",
        f"- Timing includes task JSON parsing, feature extraction or preprocessing, path planning, and evaluation inside each worker. Shared dependency import and process startup are recorded separately.",
        "- Every per-run total is reconciled as the sum of its top-level recorded stages plus other_s = total_s minus the sum of timings_s. The residual contains work outside the detailed timers, including task parsing and dispatch. Nested selector feature and inference timers are included in its residual.",
        f"- Across {timing_residuals['all_experiments']['records_with_total_s']} recorded runs, other_s ranges from {timing_residuals['all_experiments']['min_other_s']:.6g} to {timing_residuals['all_experiments']['max_other_s']:.6g} seconds; {timing_residuals['all_experiments']['negative_below_tolerance_count']} residuals are below -{timing_residuals['all_experiments']['negative_tolerance_s']:.0e} seconds.",
        "- The preprocessing stage chart includes Other task overhead (incl. parsing/optimizer setup). The separate stage medians need not sum to the median end-to-end time.",
        f"- Process peak RSS is taken from OS high-water marks and sampled RSS; the run metadata documents the measurement scope and polling interval.",
        "- The backend dataset reuses matching native Quimb rows from `preprocess.jsonl` when the source field marks them; record counts therefore are not counts of independent executions. Online selector replays are timed separately.",
        "- The amplitude reference and per-run numerical verification are recorded separately. Failed, timed-out, unsupported, and unverified rows remain in the raw files.", "",
        "## Measured records", "",
        f"There are {total_records} experiment records; {verified} successful records across preprocessing, backend and selector runs carry a passing reference check.", "",
        "| Experiment | Family / method | Runs | Success | Verified | Median time (s) | Median peak RSS (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("preprocess", "backends", "selector", "batch"):
        for row in aggregates[name]:
            label = " / ".join(str(row.get(k, "")) for k in ("family", "strategy", "backend", "policy", "method", "batch_size") if row.get(k) is not None)
            timing = "—" if row["median_total_s"] is None else f"{row['median_total_s']:.6g}"
            memory = "—" if row["median_peak_rss_bytes"] is None else f"{row['median_peak_rss_bytes'] / 2**30:.3f}"
            lines.append(f"| {name} | {label} | {row['records']} | {row['successful_records']} | {row['verified_records']} | {timing} | {memory} |")
    lines += ["", "## Figures", "", "| Figure | PNG | SVG |", "|---|---|---|"]
    for stem, title in (("preprocess", "Preprocessing"), ("preprocess_stages", "Preprocessing stage costs"),
                        ("selector", "Backend selection"), ("batch", "Batch overview"),
                        ("batch_by_task", "Batch cost by circuit")):
        lines.append(f"| {title} | ![{title}]({stem}.png) | [SVG]({stem}.svg) |")
    lines += ["", "## Paired results", "", "A pair is included only when every repeat for both methods completed and passed reference verification.", ""]
    for strategy, value in preprocess_pairs.items():
        ratio = value["median_baseline_over_candidate"]
        result = "no verified pairs" if ratio is None else f"median native/strategy time = {ratio:.3f}× over {value['matched_verified_tasks']} tasks; candidate faster on {value['faster_candidate_tasks']} and slower on {value['slower_candidate_tasks']}"
        lines.append(f"- Preprocessing `{strategy}`: {result}.")
    for policy, value in selector_pairs.items():
        ratio = value["median_baseline_over_candidate"]
        baseline = "raw" if policy == "effective_vs_raw" else "fixed"
        candidate = "effective" if policy == "effective_vs_raw" else policy
        result = "no verified pairs" if ratio is None else f"median {baseline}/{candidate} end-to-end time = {ratio:.3f}× over {value['matched_verified_tasks']} tasks"
        lines.append(f"- Selector `{policy}`: {result}.")
    tradeoff_counts = structure_tradeoffs["counts"]
    rank_count = tradeoff_counts["rank_decrease_tasks"]
    l_up = tradeoff_counts["L_increase_at_rank_step_tasks"]
    combo_n = tradeoff_counts["estimated_combo_comparable_at_rank_step_tasks"]
    combo_up = tradeoff_counts["estimated_combo_increase_at_rank_step_tasks"]
    combo_all_n = tradeoff_counts["estimated_combo_comparable_all_tasks"]
    combo_all_up = tradeoff_counts["estimated_combo_increase_all_tasks"]
    lines += ["", "## Structure and end-to-end cost", "",
              f"Among {tradeoff_counts['tasks_with_rank_step_metrics']} tasks with rank-step metrics, rank decreased on {rank_count}; L increased on {l_up} of those. Estimated combo increased on {combo_up} of {combo_n} rank-decrease tasks with comparable candidate estimates. Across all comparable tasks, estimated combo increased on {combo_all_up} of {combo_all_n}; this includes tasks where rank did not decrease.",
              "The examples below compare paired medians for the complete adaptive and native workflows. Adaptive cost includes candidate generation, planning, evaluation, and task overhead; the smaller rank/L is a structural observation, not evidence that the rank change caused the runtime difference.", ""]
    if structure_tradeoffs["verified_slowdown_examples"]:
        lines += ["| Task | Verified repeats | Rank r | L | Native median (s) | Adaptive median (s) | Adaptive/native |", "|---|---:|---:|---:|---:|---:|---:|"]
        for example in structure_tradeoffs["verified_slowdown_examples"]:
            lines.append(f"| {example['task_id']} | {example['verified_repeats_per_strategy']} | {example['r_reduced']} → {example['r_rank']} | {example['L_reduced']} → {example['L_rank']} | {example['median_native_total_s']:.6g} | {example['median_adaptive_total_s']:.6g} | {example['adaptive_over_native']:.2f}× |")
    else:
        lines.append("No rank-decrease task had fully verified matched native and adaptive repeats with adaptive slower in this run set.")
    oof_summary = summary["selection_oof_summary"]
    if oof_summary:
        lines += ["", "## Offline backend-cost oracle regret", "",
                  "These are out-of-fold model predictions compared with the fastest measured backend per task. They are separate from measured online selector replays above. OOF prediction completion is not an independent amplitude verification.", "",
                  "| Policy | OOF tasks | Predictions completed | Median estimated total (s) | Median oracle regret (×) |", "|---|---:|---:|---:|---:|"]
        for policy, row in sorted(oof_summary.items()):
            regret = "—" if row.get("median_oracle_regret") is None else f"{row['median_oracle_regret']:.3f}"
            estimate = "—" if row.get("median_estimated_total_s") is None else f"{row['median_estimated_total_s']:.6g}"
            lines.append(f"| {policy} | {row.get('tasks', 0)} | {row.get('completed', 0)} | {estimate} | {regret} |")
    lines += ["", "Selector routes include feature extraction and inference. The PAR2 view assigns each failed, timed-out, unsupported, or unverified replay twice the configured timeout; it complements the verified-pair comparison.", "",
              "| Selector | Recorded replays | Correctly completed | Completion rate | Total PAR2 (s) |",
              "|---|---:|---:|---:|---:|"]
    for policy, row in selector_penalized["per_policy"].items():
        rate = "—" if row["completion_rate"] is None else f"{row['completion_rate']:.0%}"
        lines.append(f"| {policy} | {row['recorded_replays']} | {row['successful_verified_replays']} | {rate} | {row['total_par2_s_over_recorded_replays']:.6g} |")
    lines += ["", "## Batch reuse by circuit", "",
              "Each break-even entry is the first tested batch size for which the shared graph run is faster. It is reported only when every repeat for both compared methods passed verification.", "",
              "| Task | Physical qubits | Unique queries by batch size | First shared win vs independent | First shared win vs Quimb cache |",
              "|---|---:|---|---:|---:|"]
    for row in batch_break_even["summary"]:
        independent_k = "—" if row["first_tested_shared_beats_independent_at_k"] is None else row["first_tested_shared_beats_independent_at_k"]
        quimb_k = "—" if row["first_tested_shared_beats_quimb_cached_at_k"] is None else row["first_tested_shared_beats_quimb_cached_at_k"]
        counts = ", ".join(f"{size}: {count}" for size, count in row["unique_query_counts_by_batch_size"].items())
        lines.append(f"| {row['task_id']} | {row['physical_qubits']} | {counts} | {independent_k} | {quimb_k} |")
    lines += ["", "## Reproduction files", "",
              "The JSONL files preserve individual instances and failures. `summary.json` and the per-experiment CSV files are derived summaries. Re-run the report command after adding run records to rebuild this page and all charts.", "",
              "This is a small, machine-specific CPU benchmark. A benefit is claimed only for the tested workload, where the measured end-to-end results and correctness checks support it.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return [str(summary_path), str(output / "RESULTS.md"),
            *[str(output / name) for name in figures],
            *[str(output / f"{name}_summary.csv") for name in aggregates],
            str(output / "stage_breakdown.csv"), str(output / "batch_break_even.csv"),
            str(output / "batch_by_task_summary.csv")]
