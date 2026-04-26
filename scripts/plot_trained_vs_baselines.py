"""Plot the trained LLM's mean reward next to every scripted baseline.

Reads:
  - `artifacts/baseline_eval.json`      (scripted policies, train seeds)
  - `artifacts/holdout_eval.json`       (scripted policies, holdout seeds, optional)
  - `artifacts/trained_llm_eval.json`   (trained LLM rollouts, produced by
                                          `scripts/evaluate_trained_llm.py`)

Writes:
  - `plots/trained_llm_vs_baselines.svg` — grouped bar chart, one panel per task
  - `artifacts/trained_llm_summary.json` — flat row-format payload that the
    front end and README table render directly
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TASKS: tuple[str, ...] = ("single_trade", "market_round", "coalition_market")
TASK_LABELS = {
    "single_trade": "Single trade (easy)",
    "market_round": "Market round (medium)",
    "coalition_market": "Coalition market (hard)",
}

POLICY_ORDER: list[str] = [
    "random_validish",
    "no_negotiation_allocator",
    "base_instruct_naive",
    "base_instruct_llm",
    "always_accept",
    "greedy_hoarder",
    "rule_based_expert",
]
POLICY_LABEL = {
    "random_validish": "Random valid",
    "no_negotiation_allocator": "Allocator only",
    "base_instruct_naive": "Naive instruct",
    "base_instruct_llm": "Base Llama (untrained)",
    "always_accept": "Always-accept bot",
    "greedy_hoarder": "Greedy hoarder",
    "rule_based_expert": "Rule expert (ceiling)",
    "trained_llm": "Trained LLM (this work)",
    "trained_llm_grpo": "GRPO LLM (this work)",
}
POLICY_COLOR = {
    "random_validish": "#94A3B8",
    "no_negotiation_allocator": "#64748B",
    "base_instruct_naive": "#A8A29E",
    "base_instruct_llm": "#9CA3AF",
    "always_accept": "#F59E0B",
    "greedy_hoarder": "#10B981",
    "rule_based_expert": "#DC2626",
    "trained_llm": "#2563EB",
    "trained_llm_grpo": "#7C3AED",
}


def _safe_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _summary_value(summary: dict[str, Any], task: str, policy: str, key: str = "mean_reward") -> float | None:
    bucket = summary.get(task)
    if not isinstance(bucket, dict):
        return None
    inner = bucket.get(policy)
    if not isinstance(inner, dict):
        return None
    return float(inner.get(key, 0.0))


def _gather_summary(
    baseline: dict[str, Any] | None,
    trained: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return (mean_table, stdev_table) keyed as table[task][policy]."""
    mean: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    std: dict[str, dict[str, float]] = {task: {} for task in TASKS}

    for src in (baseline, holdout, trained):
        if src is None:
            continue
        summary = src.get("summary") or {}
        for task in TASKS:
            for policy, stats in (summary.get(task) or {}).items():
                if not isinstance(stats, dict):
                    continue
                mean[task][policy] = float(stats.get("mean_reward", 0.0))
                std[task][policy] = float(stats.get("stdev_reward", 0.0))
    return mean, std


def _trained_policy_names(trained: dict[str, Any] | None) -> list[str]:
    if not trained:
        return []
    summary = trained.get("summary") or {}
    seen: list[str] = []
    for task in TASKS:
        for policy in (summary.get(task) or {}).keys():
            if policy not in seen and policy not in POLICY_ORDER:
                seen.append(policy)
    return seen


def _plot_panels(
    mean: dict[str, dict[str, float]],
    std: dict[str, dict[str, float]],
    trained_names: list[str],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )

    ordered = [p for p in POLICY_ORDER if any(p in mean[t] for t in TASKS)]
    extras = [p for p in trained_names if any(p in mean[t] for t in TASKS)]
    bar_order = ordered + extras

    fig, axes = plt.subplots(1, len(TASKS), figsize=(5.6 * len(TASKS), 6.4), dpi=160)
    if len(TASKS) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F5F7FB")

    global_max = 0.0
    global_min = 0.0
    for task in TASKS:
        for policy in bar_order:
            value = mean[task].get(policy)
            if value is not None:
                global_max = max(global_max, value)
                global_min = min(global_min, value)

    for ax, task in zip(axes, TASKS, strict=False):
        ax.set_facecolor("white")
        ax.set_title(TASK_LABELS[task], fontsize=14, color="#0F172A", pad=14)

        values: list[float] = []
        errs: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        present_policies: list[str] = []
        for policy in bar_order:
            mean_val = mean[task].get(policy)
            if mean_val is None:
                continue
            values.append(mean_val)
            errs.append(std[task].get(policy, 0.0))
            labels.append(POLICY_LABEL.get(policy, policy))
            colors.append(POLICY_COLOR.get(policy, "#7C3AED"))
            present_policies.append(policy)

        positions = list(range(len(values)))
        bars = ax.bar(
            positions,
            values,
            yerr=errs,
            color=colors,
            edgecolor="white",
            linewidth=1.4,
            capsize=4,
            ecolor="#94A3B8",
            zorder=3,
        )

        for bar, val, policy in zip(bars, values, present_policies, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.012 if val >= 0 else -0.04),
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9.5,
                color="#0F172A" if policy not in {"trained_llm", "trained_llm_grpo"} else "#1D4ED8",
                fontweight="bold" if policy in {"trained_llm", "trained_llm_grpo"} else "normal",
                zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.6)
        ax.set_ylim(min(global_min - 0.05, -0.02), max(global_max * 1.18, 0.6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="y", color="#E2E8F0", linewidth=1.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")

    fig.suptitle(
        "Trained LLM vs scripted baselines: mean episode reward (higher is better)",
        fontsize=17,
        color="#0F172A",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "Bars are mean reward over the eval seeds; whiskers are population std-dev. "
        "Blue is the SFT'd Llama-3.2-3B, purple is the GRPO'd version.",
        ha="center",
        fontsize=10.5,
        color="#64748B",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _row_format(
    mean: dict[str, dict[str, float]],
    std: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    policies = set()
    for task in TASKS:
        policies.update(mean[task].keys())
    for policy in sorted(policies):
        row: dict[str, Any] = {"policy": policy, "label": POLICY_LABEL.get(policy, policy)}
        per_task_means: list[float] = []
        for task in TASKS:
            value = mean[task].get(policy)
            if value is None:
                continue
            row[f"{task}_mean"] = round(value, 4)
            row[f"{task}_stdev"] = round(std[task].get(policy, 0.0), 4)
            per_task_means.append(value)
        if per_task_means:
            row["overall_mean"] = round(statistics.mean(per_task_means), 4)
        rows.append(row)
    rows.sort(key=lambda r: r.get("overall_mean", 0.0), reverse=True)
    return {"tasks": list(TASKS), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval", default="artifacts/baseline_eval.json")
    parser.add_argument("--holdout-eval", default="artifacts/holdout_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/trained_llm_eval.json")
    parser.add_argument("--output", default="plots/trained_llm_vs_baselines.svg")
    parser.add_argument("--summary-output", default="artifacts/trained_llm_summary.json")
    args = parser.parse_args()

    baseline = _safe_load(Path(args.baseline_eval))
    trained = _safe_load(Path(args.trained_eval))
    holdout = _safe_load(Path(args.holdout_eval))

    if baseline is None and trained is None:
        print({"status": "skipped", "reason": "Neither baseline_eval.json nor trained_llm_eval.json found."})
        return

    mean, std = _gather_summary(baseline, trained, holdout)
    trained_names = _trained_policy_names(trained)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_panels(mean, std, trained_names, output_path)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = _row_format(mean, std)
    summary_payload["trained_policies"] = trained_names
    if trained and "parse_failure_rate" in trained:
        summary_payload["parse_failure_rate"] = trained["parse_failure_rate"]
    if trained and "model_path" in trained:
        summary_payload["model_path"] = trained["model_path"]
    if trained and "base_model" in trained:
        summary_payload["base_model"] = trained["base_model"]
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "plot": str(output_path),
                "summary": str(summary_path),
                "trained_policies": trained_names,
                "policies_in_chart": [p for p in POLICY_ORDER if any(p in mean[t] for t in TASKS)] + trained_names,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
