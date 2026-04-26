"""GRPO training of an LLM against the live GPU Budget Negotiation environment.

This is the headline reward-improvement training loop. It uses TRL's
`GRPOTrainer` to fine-tune the SFT'd Llama-3.2-3B LoRA against the *actual*
environment reward — no proxy, no surrogate.

Training data shape
-------------------
Each training row corresponds to one observation reached by replaying the
rule-based expert for `K` rounds on a fixed `(task_type, seed)`. The model
sees the chat-formatted observation as its prompt, samples N completions,
and the reward function:

  1. Parses each completion as a `GpuNegotiationAction`.
  2. Replays the same `(task_type, seed)` up to round `K` with the rule
     expert (deterministic).
  3. Steps the env once with the model's action and uses
     `obs.reward + format_bonus` as the GRPO reward.

GRPO then ranks the N completions and updates the LoRA weights so that
high-reward completions become more likely. The reward curve is exported as
`artifacts/grpo_training_curve.json` and `plots/grpo_reward_curve.svg` for
the README and the front end.

CPU users
---------
Without CUDA the script prints `{"status":"skipped",...}` and exits 0 so the
notebook keeps moving.

Why "single-step + replay" instead of full multi-step rollouts?
---------------------------------------------------------------
TRL GRPO assumes a `(prompt, completion, reward)` interface. Multi-step
rollouts inside the reward function would block the trainer, double GPU
memory, and serialize an entire episode per completion. Single-step
evaluation on a *replayed* observation gives a fast, well-conditioned
reward signal that is still grounded in the real environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import rule_based_expert_policy
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.llm_policy import (
    SYSTEM_PROMPT,
    parse_action_text,
    render_user_prompt,
)
from gpu_budget_arena.models import (
    GpuNegotiationAction,
    GpuNegotiationObservation,
    ResetConfig,
)


DEFAULT_TASKS: tuple[str, ...] = ("single_trade", "market_round", "coalition_market")


@dataclass(frozen=True)
class PromptRecord:
    task_type: str
    seed: int
    round_index: int
    user_message: str


def _replay_to_round(task_type: str, seed: int, round_index: int) -> GpuNegotiationObservation | None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    for _ in range(round_index):
        if obs.done:
            return None
        obs = env.step(rule_based_expert_policy(obs))
    if obs.done:
        return None
    return obs


def build_prompt_records(
    tasks: list[str],
    seeds_per_task: int,
    rounds_per_seed: int,
    seed_offset: int = 0,
) -> list[PromptRecord]:
    """Generate (task, seed, round) triples and their rendered user prompts."""
    records: list[PromptRecord] = []
    for task in tasks:
        for seed in range(seed_offset, seed_offset + seeds_per_task):
            for round_index in range(rounds_per_seed):
                obs = _replay_to_round(task, seed, round_index)
                if obs is None:
                    continue
                records.append(
                    PromptRecord(
                        task_type=task,
                        seed=seed,
                        round_index=round_index,
                        user_message=render_user_prompt(obs),
                    )
                )
    return records


def _env_reward_for_completion(
    completion: str,
    task_type: str,
    seed: int,
    round_index: int,
    format_bonus: float,
    parse_penalty: float,
) -> tuple[float, dict[str, Any]]:
    obs = _replay_to_round(task_type, seed, round_index)
    if obs is None:
        return parse_penalty, {"reason": "replay_done", "parsed": False}
    action, raw = parse_action_text(completion)
    if action is None:
        return parse_penalty, {"reason": "parse_failure", "parsed": False, "raw": raw[:120]}
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type=task_type, seed=seed))
    target_obs = obs
    cur = env.reset(ResetConfig(task_type=task_type, seed=seed))
    for _ in range(round_index):
        cur = env.step(rule_based_expert_policy(cur))
    new_obs = env.step(action)
    reward = float(new_obs.reward) + format_bonus
    info = {
        "parsed": True,
        "action_type": action.action_type,
        "raw_reward": float(new_obs.reward),
        "format_bonus": format_bonus,
        "task_type": task_type,
        "seed": seed,
        "round_index": round_index,
        "valid_action": bool(new_obs.last_action_result and new_obs.last_action_result.ok),
    }
    return reward, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--sft-checkpoint",
        default="artifacts/sft-checkpoint",
        help="LoRA checkpoint to start GRPO from. Falls back to base model when missing.",
    )
    parser.add_argument("--output", default="artifacts/grpo-checkpoint")
    parser.add_argument("--drive-output", default="")
    parser.add_argument("--curve-output", default="artifacts/grpo_training_curve.json")
    parser.add_argument("--plot-output", default="plots/grpo_reward_curve.svg")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=list(DEFAULT_TASKS),
    )
    parser.add_argument("--seeds-per-task", type=int, default=8)
    parser.add_argument("--rounds-per-seed", type=int, default=4)
    parser.add_argument("--format-bonus", type=float, default=0.05)
    parser.add_argument("--parse-penalty", type=float, default=-0.5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--no-resume-from-sft",
        action="store_true",
        help="Train GRPO from the base model without the SFT LoRA warm start.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print({"status": "skipped", "reason": "PyTorch is not installed."})
        return

    if not torch.cuda.is_available():
        print(
            {
                "status": "skipped",
                "reason": (
                    "GRPO requires a CUDA GPU. In Colab, Runtime -> Change runtime type -> "
                    "T4 GPU and rerun."
                ),
                "torch": getattr(torch, "__version__", "unknown"),
            }
        )
        return

    import unsloth  # noqa: F401  # must come before TRL/transformers patches
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    sft_path = args.sft_checkpoint
    use_sft = (
        sft_path
        and not args.no_resume_from_sft
        and Path(sft_path).exists()
    )
    target = sft_path if use_sft else args.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=target,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    if not use_sft:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_records = build_prompt_records(
        args.tasks,
        seeds_per_task=args.seeds_per_task,
        rounds_per_seed=args.rounds_per_seed,
    )
    if not prompt_records:
        raise RuntimeError("No prompt records were generated. Check tasks / seeds / rounds.")

    chat_prompts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for record in prompt_records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.user_message},
        ]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(chat)
        metadata.append(
            {
                "task_type": record.task_type,
                "seed": record.seed,
                "round_index": record.round_index,
            }
        )
    dataset = Dataset.from_dict({"prompt": chat_prompts, "_meta": metadata})

    rolling_rewards: list[float] = []
    parse_failures: list[int] = []
    valid_action_flags: list[int] = []
    curve_points: list[dict[str, Any]] = []

    def env_reward(completions: list[str], **kwargs: Any) -> list[float]:
        meta_batch: list[dict[str, Any]] = kwargs.get("_meta") or []
        rewards: list[float] = []
        batch_parsed = 0
        batch_valid = 0
        for completion, meta in zip(completions, meta_batch, strict=False):
            reward, info = _env_reward_for_completion(
                completion=completion,
                task_type=meta["task_type"],
                seed=int(meta["seed"]),
                round_index=int(meta["round_index"]),
                format_bonus=args.format_bonus,
                parse_penalty=args.parse_penalty,
            )
            rewards.append(reward)
            if info.get("parsed"):
                batch_parsed += 1
            if info.get("valid_action"):
                batch_valid += 1
        if rewards:
            rolling_rewards.append(statistics.mean(rewards))
            parse_failures.append(len(rewards) - batch_parsed)
            valid_action_flags.append(batch_valid)
            curve_points.append(
                {
                    "step": len(curve_points) + 1,
                    "mean_reward": round(statistics.mean(rewards), 4),
                    "max_reward": round(max(rewards), 4),
                    "min_reward": round(min(rewards), 4),
                    "parse_failures": len(rewards) - batch_parsed,
                    "valid_actions": batch_valid,
                    "batch": len(rewards),
                }
            )
        return rewards

    grpo_config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=args.max_seq_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=max(50, args.max_steps // 2),
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "reward_funcs": [env_reward],
        "train_dataset": dataset,
    }
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output)

    summary = {
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "format_bonus": args.format_bonus,
        "parse_penalty": args.parse_penalty,
        "tasks": args.tasks,
        "seeds_per_task": args.seeds_per_task,
        "rounds_per_seed": args.rounds_per_seed,
        "prompts": len(dataset),
        "started_from": "sft_checkpoint" if use_sft else "base_model",
        "base_model": args.base_model,
        "sft_checkpoint": sft_path if use_sft else None,
        "first_step_mean_reward": round(curve_points[0]["mean_reward"], 4) if curve_points else None,
        "last_step_mean_reward": round(curve_points[-1]["mean_reward"], 4) if curve_points else None,
        "best_step_mean_reward": (
            round(max((p["mean_reward"] for p in curve_points), default=0.0), 4) if curve_points else None
        ),
    }
    curve_payload = {"summary": summary, "points": curve_points}
    Path(args.curve_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.curve_output).write_text(
        json.dumps(curve_payload, indent=2) + "\n", encoding="utf-8"
    )

    try:
        _plot_curve(curve_payload, Path(args.plot_output))
    except Exception as exc:  # plotting must not crash training
        print({"status": "warning", "plot_failed": str(exc)})

    if args.drive_output:
        drive_output = Path(args.drive_output)
        drive_output.mkdir(parents=True, exist_ok=True)
        target_dir = drive_output / Path(args.output).name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(args.output, target_dir)

    print(
        json.dumps(
            {
                "status": "ok",
                "output": args.output,
                "curve": args.curve_output,
                "plot": args.plot_output,
                "summary": summary,
            },
            indent=2,
        )
    )


def _plot_curve(payload: dict[str, Any], output_path: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    points = payload["points"]
    if not points:
        return
    steps = [p["step"] for p in points]
    rewards = [p["mean_reward"] for p in points]

    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=160)
    fig.patch.set_facecolor("#F5F7FB")
    ax.set_facecolor("white")

    ax.plot(steps, rewards, color="#7C3AED", linewidth=2.6, marker="o", markersize=3.5, label="batch mean env reward")

    if len(rewards) >= 8:
        window = max(3, len(rewards) // 12)
        smoothed = []
        for i in range(len(rewards)):
            lo = max(0, i - window + 1)
            smoothed.append(sum(rewards[lo : i + 1]) / (i - lo + 1))
        ax.plot(steps, smoothed, color="#1D4ED8", linewidth=2.4, linestyle="--", label=f"rolling mean (window={window})")

    ax.axhline(0.0, color="#94A3B8", linewidth=1.0, linestyle=":")
    ax.set_title("GRPO reward curve · Llama-3.2-3B against the live env reward", fontsize=15, color="#0F172A", pad=18)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Reward (env + format bonus)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="#E2E8F0", linewidth=1.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print({"status": "skipped", "reason": str(exc)})
        sys.exit(0)
