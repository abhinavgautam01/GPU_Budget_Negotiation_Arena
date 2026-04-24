from __future__ import annotations

import argparse
import json
from pathlib import Path


PALETTE = ["#5A7FFF", "#6CBF84", "#F0B34A", "#D96666", "#8B6FDB"]


def _svg_bar_chart(summary: dict[str, object]) -> str:
    task_types = list(summary.keys())
    policies = list(next(iter(summary.values())).keys())
    chart_width = 340
    chart_height = 240
    margin_left = 56
    margin_bottom = 48
    panel_gap = 28
    total_width = len(task_types) * chart_width + max(0, len(task_types) - 1) * panel_gap
    total_height = chart_height + 70

    all_means = [
        float(summary[task_type][policy]["mean_reward"])
        for task_type in task_types
        for policy in policies
    ]
    y_min = min(0.0, min(all_means) - 0.05)
    y_max = max(all_means) + 0.1
    y_span = max(0.25, y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1c2434; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.axis { font-size: 11px; }',
        '.panel-title { font-size: 13px; font-weight: 600; }',
        '.value { font-size: 10px; }',
        '</style>',
        f'<rect width="{total_width}" height="{total_height}" fill="#fbfcfe"/>',
        f'<text x="{total_width / 2}" y="24" text-anchor="middle" class="title">GPU Budget Negotiation Arena Baseline Comparison</text>',
    ]

    for panel_index, task_type in enumerate(task_types):
        x0 = panel_index * (chart_width + panel_gap)
        plot_x = x0 + margin_left
        plot_y = 44
        plot_w = chart_width - margin_left - 12
        plot_h = chart_height - margin_bottom
        zero_y = plot_y + plot_h - ((0 - y_min) / y_span) * plot_h

        parts.append(f'<rect x="{x0}" y="34" width="{chart_width}" height="{chart_height}" rx="6" fill="#ffffff" stroke="#d9dfeb"/>')
        parts.append(f'<text x="{x0 + chart_width / 2}" y="54" text-anchor="middle" class="panel-title">{task_type}</text>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#7a869c" stroke-width="1"/>')
        parts.append(f'<line x1="{plot_x}" y1="{zero_y}" x2="{plot_x + plot_w}" y2="{zero_y}" stroke="#7a869c" stroke-width="1"/>')

        bar_slot = plot_w / max(1, len(policies))
        bar_w = bar_slot * 0.58
        for idx, policy in enumerate(policies):
            value = float(summary[task_type][policy]["mean_reward"])
            color = PALETTE[idx % len(PALETTE)]
            bar_x = plot_x + idx * bar_slot + (bar_slot - bar_w) / 2
            value_y = plot_y + plot_h - ((value - y_min) / y_span) * plot_h
            rect_y = min(value_y, zero_y)
            rect_h = abs(zero_y - value_y)
            parts.append(f'<rect x="{bar_x}" y="{rect_y}" width="{bar_w}" height="{max(1, rect_h)}" fill="{color}" rx="4"/>')
            parts.append(f'<text x="{bar_x + bar_w / 2}" y="{plot_y + plot_h + 16}" text-anchor="middle" class="axis">{policy}</text>')
            label_y = rect_y - 6 if value >= 0 else rect_y + rect_h + 12
            parts.append(f'<text x="{bar_x + bar_w / 2}" y="{label_y}" text-anchor="middle" class="value">{value:.3f}</text>')

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
    svg = _svg_bar_chart(payload["summary"])
    output_path.write_text(svg, encoding="utf-8")
    print({"output": str(output_path)})


if __name__ == "__main__":
    main()
