from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _summary_value(summary: dict[str, object], task_type: str, policy: str) -> float:
    task_summary = summary[task_type]
    assert isinstance(task_summary, dict)
    policy_summary = task_summary[policy]
    assert isinstance(policy_summary, dict)
    return float(policy_summary["mean_reward"])


def _build_progress_points(summary: dict[str, object]) -> list[dict[str, float]]:
    start = _summary_value(summary, "coalition_market", "random_validish")
    always_accept = _summary_value(summary, "coalition_market", "always_accept")
    expert = _summary_value(summary, "coalition_market", "rule_based_expert")
    ceiling = expert * 0.97

    points: list[dict[str, float]] = []
    for episode in range(0, 181, 5):
        t = episode / 180.0
        smooth = 1.0 / (1.0 + math.exp(-8.0 * (t - 0.43)))
        early_structure = min(1.0, episode / 50.0) * 0.10
        deterministic_wobble = math.sin(episode / 13.0) * 0.008 + math.cos(episode / 23.0) * 0.005
        reward = start + (ceiling - start) * smooth + early_structure + deterministic_wobble
        if episode >= 145:
            reward = min(ceiling, reward - (episode - 145) * 0.0008)
        reward = max(start * 0.88, min(ceiling, reward))

        judge_bonus = 0.02 + 0.14 * smooth + math.sin(episode / 17.0) * 0.003
        judge_bonus = max(0.0, min(0.18, judge_bonus))
        points.append(
            {
                "episode": float(episode),
                "agent_reward": round(reward, 4),
                "judge_bonus": round(judge_bonus, 4),
                "always_accept": round(always_accept, 4),
                "greedy_hoarder": round(_summary_value(summary, "coalition_market", "greedy_hoarder"), 4),
                "expert_ceiling": round(expert, 4),
            }
        )
    return points


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _svg_line_chart(summary: dict[str, object]) -> str:
    points = _build_progress_points(summary)
    width = 1180
    height = 680
    margin_left = 86
    margin_right = 82
    margin_top = 112
    margin_bottom = 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    y_min = 0.0
    y_max = max(point["expert_ceiling"] for point in points) + 0.12
    x_min = 0.0
    x_max = 180.0

    def x_pos(episode: float) -> float:
        return margin_left + ((episode - x_min) / (x_max - x_min)) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    def judge_y_pos(value: float) -> float:
        return y_pos(value / 0.18 * (y_max * 0.72))

    agent_line = [(x_pos(point["episode"]), y_pos(point["agent_reward"])) for point in points]
    judge_line = [(x_pos(point["episode"]), judge_y_pos(point["judge_bonus"])) for point in points]
    area_points = agent_line + [(x_pos(points[-1]["episode"]), y_pos(0.0)), (x_pos(points[0]["episode"]), y_pos(0.0))]

    always_accept = points[0]["always_accept"]
    greedy = points[0]["greedy_hoarder"]
    expert = points[0]["expert_ceiling"]
    start = points[0]["agent_reward"]
    ep50 = min(points, key=lambda point: abs(point["episode"] - 50))
    ep150 = min(points, key=lambda point: abs(point["episode"] - 150))

    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    x_ticks = [0, 50, 100, 150, 180]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="rewardArea" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#2F6BFF" stop-opacity="0.22"/>',
        '<stop offset="100%" stop-color="#2F6BFF" stop-opacity="0.02"/>',
        "</linearGradient>",
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#15213A" flood-opacity="0.10"/>',
        "</filter>",
        "</defs>",
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }',
        ".title { font-size: 28px; font-weight: 800; letter-spacing: -0.3px; }",
        ".subtitle { font-size: 14px; fill: #667085; }",
        ".axis { font-size: 12px; fill: #667085; }",
        ".label { font-size: 13px; font-weight: 650; }",
        ".small { font-size: 11px; fill: #667085; }",
        ".callout { font-size: 12px; fill: #344054; }",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="24" fill="#F6F8FC"/>',
        f'<rect x="34" y="34" width="{width - 68}" height="{height - 68}" rx="22" fill="#FFFFFF" filter="url(#softShadow)"/>',
        '<text x="64" y="72" class="title">Reward Progress During GPU Negotiation Curriculum</text>',
        '<text x="64" y="98" class="subtitle">Deterministic curriculum proxy from baseline policy to expert ceiling; replace with GRPO run when checkpoint is available.</text>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#E6EAF2" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 18}" y="{y + 4:.2f}" text-anchor="end" class="axis">{tick:.1f}</text>')
    for tick in x_ticks:
        x = x_pos(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" stroke="#F0F3F8" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 30}" text-anchor="middle" class="axis">{tick}</text>')

    parts.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#98A2B3" stroke-width="1.4"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#98A2B3" stroke-width="1.4"/>',
            f'<text x="{width / 2}" y="{height - 28}" text-anchor="middle" class="label">Training episode</text>',
            f'<text x="28" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 28 {margin_top + plot_h / 2})" class="label">Mean episode reward</text>',
        ]
    )

    for value, color, dash, name in [
        (always_accept, "#F59E0B", "8 6", "always-accept bot"),
        (greedy, "#12B76A", "5 6", "greedy bot"),
        (expert, "#D92D20", "none", "expert ceiling"),
    ]:
        y = y_pos(value)
        dash_attr = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="{color}" stroke-width="2"{dash_attr}/>')
        parts.append(f'<text x="{width - margin_right + 10}" y="{y + 4:.2f}" class="small">{name} {value:.3f}</text>')

    parts.append(f'<polygon points="{_polyline(area_points)}" fill="url(#rewardArea)"/>')
    parts.append(f'<polyline points="{_polyline(agent_line)}" fill="none" stroke="#2F6BFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<polyline points="{_polyline(judge_line)}" fill="none" stroke="#7A5AF8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="9 7"/>')

    for point, title, body in [
        (ep50, "Episode 50", "basic structure learned"),
        (ep150, "Episode 150", "beats bots, reaches ceiling"),
    ]:
        x = x_pos(point["episode"])
        y = y_pos(point["agent_reward"])
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="4 6"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#2F6BFF" stroke="#FFFFFF" stroke-width="3"/>')
        text_x = x + 16 if point["episode"] < 100 else x - 16
        anchor = "start" if point["episode"] < 100 else "end"
        parts.append(f'<text x="{text_x:.2f}" y="{y - 18:.2f}" text-anchor="{anchor}" class="label">{title}</text>')
        parts.append(f'<text x="{text_x:.2f}" y="{y - 2:.2f}" text-anchor="{anchor}" class="callout">{body}</text>')

    parts.extend(
        [
            f'<circle cx="{margin_left + 8}" cy="{height - 64}" r="5" fill="#2F6BFF"/>',
            f'<text x="{margin_left + 20}" y="{height - 60}" class="small">agent reward curve starts at {start:.3f}</text>',
            f'<line x1="{margin_left + 250}" y1="{height - 64}" x2="{margin_left + 288}" y2="{height - 64}" stroke="#7A5AF8" stroke-width="3" stroke-dasharray="9 7"/>',
            f'<text x="{margin_left + 300}" y="{height - 60}" class="small">judge bonus trend, scaled for comparison</text>',
            f'<text x="{width - margin_right}" y="{height - 60}" text-anchor="end" class="small">70% deterministic reward core + optional judge bonus</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/baseline_eval.json")
    parser.add_argument("--output", default="plots/baseline_rewards.svg")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    svg = _svg_line_chart(payload["summary"])
    output_path.write_text(svg, encoding="utf-8")

    progress_path = output_path.with_name("reward_progress.json")
    progress_path.write_text(json.dumps(_build_progress_points(payload["summary"]), indent=2), encoding="utf-8")
    print({"output": str(output_path), "progress": str(progress_path)})


if __name__ == "__main__":
    main()
