"""Extract the real SFT training curve from `trainer_state.json`.

The actual run was split across two checkpoint folders because of an earlier
disconnect:

  * gpu_budget_negotiation_arena/artifacts/sft-checkpoint  → checkpoint-60, -120
  * gpu_budget_negotiation_arena/sft-checkpoint            → checkpoint-180 ... -500

Hugging Face's Trainer keeps the *full* `log_history` in every checkpoint, so
`checkpoint-500/trainer_state.json` already contains all 50 logged points
(every 10 steps from 1 → 500 over 13 epochs). We simply read it, normalise it,
and emit:

  artifacts/sft_training_curve.json   # machine-readable curve + summary
  plots/sft_loss_curve.svg            # static plot (vector, HF-Space safe)
"""
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


_DEFAULT_CANDIDATES = [
    Path("gpu_budget_negotiation_arena/sft-checkpoint/checkpoint-500/trainer_state.json"),
    Path("gpu_budget_negotiation_arena/artifacts/sft-checkpoint/checkpoint-120/trainer_state.json"),
]


def _pick_trainer_state(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"trainer_state.json not found at {explicit}")
        return explicit
    for candidate in _DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "No trainer_state.json found. Pass --trainer-state or copy a checkpoint dir."
    )


def _build_curve(state: dict) -> dict:
    rows = [r for r in state.get("log_history", []) if "loss" in r and "step" in r]
    rows.sort(key=lambda r: int(r["step"]))
    points = [
        {
            "step": int(r["step"]),
            "epoch": float(r.get("epoch", 0.0)),
            "loss": float(r["loss"]),
            "grad_norm": float(r.get("grad_norm", 0.0)),
            "learning_rate": float(r.get("learning_rate", 0.0)),
        }
        for r in rows
    ]
    losses = [p["loss"] for p in points]
    return {
        "summary": {
            "total_steps": int(state.get("global_step", points[-1]["step"] if points else 0)),
            "max_steps": int(state.get("max_steps", points[-1]["step"] if points else 0)),
            "num_train_epochs": int(state.get("num_train_epochs", 0)),
            "save_steps": int(state.get("save_steps", 0)),
            "logging_steps": int(state.get("logging_steps", 10)),
            "first_loss": round(losses[0], 6) if losses else 0.0,
            "final_loss": round(losses[-1], 6) if losses else 0.0,
            "min_loss": round(min(losses), 6) if losses else 0.0,
            "max_loss": round(max(losses), 6) if losses else 0.0,
            "loss_drop_pct": round(
                (losses[0] - losses[-1]) / losses[0] * 100.0, 2
            ) if losses else 0.0,
        },
        "points": points,
    }


def _plot_curve(curve: dict, output_path: Path) -> None:
    points = curve["points"]
    steps = [p["step"] for p in points]
    losses = [p["loss"] for p in points]
    lrs = [p["learning_rate"] for p in points]
    summary = curve["summary"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#0c0a08",
            "xtick.color": "#3a3530",
            "ytick.color": "#3a3530",
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 7.0), dpi=160)
    fig.patch.set_facecolor("#f1ebdb")
    ax.set_facecolor("#ebe3cf")

    ax.fill_between(steps, losses, 0, color="#ff3b30", alpha=0.10, zorder=1)
    ax.plot(
        steps,
        losses,
        color="#ff3b30",
        linewidth=3.0,
        marker="o",
        markersize=4.5,
        markerfacecolor="#ff3b30",
        markeredgecolor="#0c0a08",
        markeredgewidth=1.2,
        label="train loss",
        zorder=5,
    )

    ax.plot(
        [120, 120],
        [0, max(losses) * 1.05],
        color="#1234ff",
        linewidth=1.6,
        linestyle=(0, (5, 4)),
        zorder=3,
        label="resume from checkpoint-120",
    )

    ax2 = ax.twinx()
    ax2.plot(
        steps,
        lrs,
        color="#6a4cff",
        linewidth=1.8,
        linestyle=(0, (4, 4)),
        label="learning rate",
        zorder=4,
    )
    ax2.set_ylabel("Learning rate", color="#6a4cff", labelpad=12, fontsize=11.5)
    ax2.tick_params(axis="y", colors="#6a4cff")
    ax2.set_ylim(0, max(lrs) * 1.10 if lrs else 1)

    ax.set_title(
        f"SFT Training Loss · {summary['total_steps']} steps · {summary['num_train_epochs']} epochs",
        fontsize=20,
        color="#0c0a08",
        pad=22,
    )
    ax.text(
        0,
        1.025,
        f"first loss {summary['first_loss']:.4f}  →  final loss {summary['final_loss']:.4f}   "
        f"({summary['loss_drop_pct']:.1f}% drop)",
        transform=ax.transAxes,
        fontsize=11.2,
        color="#3a3530",
    )
    ax.set_xlabel("Training step", fontsize=12.5, labelpad=10)
    ax.set_ylabel("Loss", fontsize=12.5, labelpad=10)
    ax.set_xlim(0, max(steps) + 5)
    ax.set_ylim(0, max(losses) * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="rgba(12,10,8,.18)".replace("rgba(12,10,8,.18)", "#cfc7b3"), linewidth=1.0)
    ax.grid(axis="x", color="#dcd3bb", linewidth=0.7)

    for spine in ("top",):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom", "right"):
        ax.spines[spine].set_color("#0c0a08")
        ax.spines[spine].set_linewidth(1.3)
    ax2.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + h2,
        labels + l2,
        loc="upper right",
        frameon=True,
        facecolor="#f1ebdb",
        framealpha=0.96,
        edgecolor="#0c0a08",
        fontsize=10.5,
    )

    fig.tight_layout(rect=(0.035, 0.045, 0.975, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output_path.suffix.lower() == ".svg":
        output_path.write_text(
            "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines())
            + "\n",
            encoding="utf-8",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-state", type=Path, default=None)
    parser.add_argument(
        "--curve-output", type=Path, default=Path("artifacts/sft_training_curve.json")
    )
    parser.add_argument(
        "--plot-output", type=Path, default=Path("plots/sft_loss_curve.svg")
    )
    args = parser.parse_args()

    state_path = _pick_trainer_state(args.trainer_state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    curve = _build_curve(state)

    args.curve_output.parent.mkdir(parents=True, exist_ok=True)
    args.curve_output.write_text(
        json.dumps(curve, indent=2) + "\n", encoding="utf-8"
    )
    _plot_curve(curve, args.plot_output)
    print(
        json.dumps(
            {
                "source": str(state_path),
                "points": len(curve["points"]),
                "curve": str(args.curve_output),
                "plot": str(args.plot_output),
                "summary": curve["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
