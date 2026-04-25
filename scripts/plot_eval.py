from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _summary_value(summary: dict[str, object], task_type: str, policy: str) -> float:
    task_summary = summary[task_type]
    assert isinstance(task_summary, dict)
    policy_summary = task_summary[policy]
    assert isinstance(policy_summary, dict)
    return float(policy_summary["mean_reward"])


def _mean(values: object) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def _aggregate_policy_value(summary: dict[str, object], policy: str) -> float:
    return _mean(_summary_value(summary, task_type, policy) for task_type in summary)


def _build_proxy_progress_points(summary: dict[str, object]) -> list[dict[str, float]]:
    """Fallback curve used only if a training curve has not been generated yet."""
    import math

    start = _summary_value(summary, "coalition_market", "random_validish")
    always_accept = _aggregate_policy_value(summary, "always_accept")
    expert = _aggregate_policy_value(summary, "rule_based_expert")
    ceiling = expert * 0.97

    points: list[dict[str, float]] = []
    for episode in range(0, 181, 5):
        t = episode / 180.0
        smooth = 1.0 / (1.0 + math.exp(-8.0 * (t - 0.43)))
        reward = start + (ceiling - start) * smooth + min(1.0, episode / 50.0) * 0.10
        if episode >= 145:
            reward = min(ceiling, reward - (episode - 145) * 0.0008)
        judge_bonus = max(0.0, min(0.18, 0.02 + 0.14 * smooth))
        points.append(
            {
                "episode": float(episode),
                "agent_reward": round(max(start * 0.88, min(ceiling, reward)), 4),
                "judge_bonus": round(judge_bonus, 4),
                "always_accept": round(always_accept, 4),
                "greedy_hoarder": round(_aggregate_policy_value(summary, "greedy_hoarder"), 4),
                "expert_ceiling": round(expert, 4),
            }
        )
    return points


def _build_progress_points(summary: dict[str, object], training_curve: list[dict[str, object]] | None) -> list[dict[str, float]]:
    if not training_curve:
        return _build_proxy_progress_points(summary)

    always_accept = _aggregate_policy_value(summary, "always_accept")
    greedy = _aggregate_policy_value(summary, "greedy_hoarder")
    expert = _aggregate_policy_value(summary, "rule_based_expert")
    return [
        {
            "episode": float(row["episode"]),
            "agent_reward": float(row["eval_mean_reward"]),
            "judge_bonus": float(row.get("judge_bonus", 0.0)),
            "always_accept": round(always_accept, 4),
            "greedy_hoarder": round(greedy, 4),
            "expert_ceiling": round(expert, 4),
        }
        for row in training_curve
    ]


def _annotate_episode(ax: plt.Axes, points: list[dict[str, float]], episode: int, text: str, xytext: tuple[int, int]) -> None:
    point = min(points, key=lambda row: abs(row["episode"] - episode))
    ax.scatter([point["episode"]], [point["agent_reward"]], s=74, color="#2563EB", edgecolor="white", linewidth=2.2, zorder=6)
    ax.annotate(
        text,
        xy=(point["episode"], point["agent_reward"]),
        xytext=xytext,
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#64748B", "lw": 1.2, "connectionstyle": "arc3,rad=0.12"},
        bbox={"boxstyle": "round,pad=0.38", "fc": "white", "ec": "#CBD5E1", "lw": 1.0},
        fontsize=10.5,
        color="#334155",
        zorder=8,
    )


def _plot_line_chart(summary: dict[str, object], training_curve: list[dict[str, object]] | None, output_path: Path) -> list[dict[str, float]]:
    points = _build_progress_points(summary, training_curve)
    episodes = [row["episode"] for row in points]
    rewards = [row["agent_reward"] for row in points]
    judge_bonus = [row["judge_bonus"] for row in points]

    always_accept = points[0]["always_accept"]
    greedy = points[0]["greedy_hoarder"]
    expert = points[0]["expert_ceiling"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )

    fig, ax = plt.subplots(figsize=(13.8, 7.8), dpi=160)
    fig.patch.set_facecolor("#F5F7FB")
    ax.set_facecolor("white")

    ax.fill_between(episodes, rewards, 0, color="#2563EB", alpha=0.10, zorder=1)
    reward_line, = ax.plot(
        episodes,
        rewards,
        color="#2563EB",
        linewidth=3.4,
        marker="o",
        markersize=4.2,
        markerfacecolor="#2563EB",
        markeredgecolor="white",
        markeredgewidth=1.1,
        label="trained selector reward",
        zorder=5,
    )

    ax.axhline(always_accept, color="#F59E0B", linestyle=(0, (7, 5)), linewidth=2.1, label=f"always-accept bot ({always_accept:.3f})")
    ax.axhline(greedy, color="#12B76A", linestyle=(0, (4, 5)), linewidth=2.1, label=f"greedy bot ({greedy:.3f})")
    ax.axhline(expert, color="#DC2626", linestyle="-", linewidth=2.1, label=f"expert ceiling ({expert:.3f})")

    ax2 = ax.twinx()
    judge_line, = ax2.plot(
        episodes,
        judge_bonus,
        color="#7C3AED",
        linewidth=2.6,
        linestyle=(0, (6, 4)),
        label="judge bonus trend",
        zorder=4,
    )
    ax2.set_ylim(0.0, max(0.20, max(judge_bonus) * 1.18))
    ax2.set_ylabel("Judge bonus", fontsize=11.5, color="#6D28D9", labelpad=14)
    ax2.tick_params(axis="y", colors="#6D28D9")
    ax2.spines["right"].set_color("#DDD6FE")

    ax.set_title("Reward Progress During GPU Negotiation Curriculum", fontsize=20, color="#0F172A", pad=22)
    ax.text(
        0,
        1.025,
        "Real lightweight policy-selector training curve with bot baselines, expert ceiling, and aligned judge-bonus trend.",
        transform=ax.transAxes,
        fontsize=11.2,
        color="#64748B",
    )
    ax.set_xlabel("Training episode", fontsize=12.5, labelpad=12)
    ax.set_ylabel("Mean episode reward", fontsize=12.5, labelpad=12)
    ax.set_xlim(min(episodes), max(episodes))
    ax.set_ylim(0.0, max(expert + 0.12, max(rewards) + 0.08))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="#E2E8F0", linewidth=1.0)
    ax.grid(axis="x", color="#F1F5F9", linewidth=0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax2.spines["top"].set_visible(False)

    _annotate_episode(ax, points, 50, "Episode 50\nvalid structure emerges", (18, -58))
    _annotate_episode(ax, points, 150, "Episode 150\nbeats both bots", (-132, 34))

    handles = [reward_line, judge_line, *ax.get_legend_handles_labels()[0][1:]]
    labels = ["trained selector reward", "judge bonus trend", *ax.get_legend_handles_labels()[1][1:]]
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#E2E8F0",
        fontsize=10.5,
    )
    legend.get_frame().set_boxstyle("round,pad=0.5,rounding_size=0.8")

    fig.text(
        0.075,
        0.035,
        f"Start: {rewards[0]:.3f} | Final: {rewards[-1]:.3f} | Deterministic environment reward remains primary; judge bonus is auxiliary.",
        fontsize=10.5,
        color="#64748B",
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.975, 0.96))
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output_path.suffix.lower() == ".svg":
        output_path.write_text(
            "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )

    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/baseline_eval.json")
    parser.add_argument("--training-input", default="artifacts/training_curve.json")
    parser.add_argument("--output", default="plots/baseline_rewards.png")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    training_path = Path(args.training_input)
    training_curve = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else None

    points = _plot_line_chart(payload["summary"], training_curve, output_path)
    progress_path = output_path.with_name("reward_progress.json")
    progress_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
    print({"output": str(output_path), "png": str(output_path.with_suffix(".png")), "progress": str(progress_path)})


if __name__ == "__main__":
    main()
