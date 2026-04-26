"""Evaluate a trained LLM (SFT or GRPO) as a policy in the negotiation arena.

This is the headline "trained-model-vs-baselines" evidence script. For every
seed in `--seeds` and every task in `--tasks` it:

  1. Loads the LoRA-adapted model from `--model-path` (with the `--base-model`
     as the foundation when needed) and wraps it as a policy.
  2. Rolls out a full episode using the LLM as the controlled lab.
  3. Records per-step rewards, final reward, parse failures, and chosen action
     types so we can plot it next to the scripted baselines later.
  4. Optionally re-runs the matched seed/task pairs with the base (untrained)
     model so the resulting JSON has a clean before/after column.

CPU users get a clean, deterministic skip with `status: skipped` in stdout —
the script never crashes the notebook even without `transformers` installed.

Output: `artifacts/trained_llm_eval.json` with shape::

    {
      "summary": { task_type: { policy_name: {mean_reward, stdev_reward, ...} } },
      "episodes": [ {task_type, seed, policy, steps, episode_reward, ...}, ... ],
      "parse_failure_rate": float,
      "model_path": str,
      "base_model": str,
      ...
    }

You can plot it with `scripts/plot_trained_vs_baselines.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import (
    GpuNegotiationAction,
    GpuNegotiationObservation,
    ResetConfig,
)


DEFAULT_TASKS: tuple[str, ...] = ("single_trade", "market_round", "coalition_market")


def _load_model_and_tokenizer(
    base_model: str,
    model_path: str | None,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    """Load model + tokenizer. Prefers Unsloth's FastLanguageModel when present.

    Falls back to `transformers + peft` if Unsloth is unavailable on the
    runtime, which lets evaluators reproduce the curve on a vanilla
    transformers stack.
    """
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except ImportError:
        FastLanguageModel = None  # type: ignore[assignment]

    if FastLanguageModel is not None:
        target = model_path or base_model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=target,
            max_seq_length=4096,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path or base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        load_in_4bit=load_in_4bit,
    )
    if model_path and model_path != base_model:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, tokenizer


def _build_policy(
    base_model: str,
    model_path: str | None,
    temperature: float,
    load_in_4bit: bool,
) -> tuple[Callable[[GpuNegotiationObservation], GpuNegotiationAction], dict[str, Any]]:
    from gpu_budget_arena.llm_policy import LlmPolicyConfig, make_llm_policy

    model, tokenizer = _load_model_and_tokenizer(base_model, model_path, load_in_4bit)
    cfg = LlmPolicyConfig(temperature=temperature, do_sample=temperature > 0.0)
    policy = make_llm_policy(model, tokenizer, cfg)
    info = {
        "model_path": model_path or base_model,
        "base_model": base_model,
        "temperature": temperature,
        "load_in_4bit": load_in_4bit,
        "device": str(getattr(model, "device", "auto")),
    }
    return policy, info


def run_episode(
    policy: Callable[[GpuNegotiationObservation], GpuNegotiationAction],
    task_type: str,
    seed: int,
    policy_name: str,
) -> dict[str, Any]:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    rewards: list[float] = []
    actions: list[str] = []
    parse_failures = 0
    steps = 0
    t0 = time.time()
    while not obs.done:
        action = policy(obs)
        actions.append(action.action_type)
        if action.action_type == "wait":
            parse_failures += 1
        obs = env.step(action)
        rewards.append(obs.reward)
        steps += 1
    return {
        "task_type": task_type,
        "seed": seed,
        "policy": policy_name,
        "steps": steps,
        "episode_reward": obs.cumulative_reward,
        "final_reward": obs.reward,
        "reward_trace": rewards,
        "action_types": actions,
        "wait_actions": parse_failures,
        "wall_seconds": round(time.time() - t0, 3),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        bucket = grouped.setdefault(record["task_type"], {})
        bucket.setdefault(record["policy"], []).append(float(record["episode_reward"]))

    summary: dict[str, Any] = {}
    for task_type, policies in grouped.items():
        summary[task_type] = {
            policy_name: {
                "mean_reward": round(statistics.mean(scores), 4),
                "stdev_reward": round(statistics.pstdev(scores), 4),
                "episodes": len(scores),
                "min_reward": round(min(scores), 4),
                "max_reward": round(max(scores), 4),
            }
            for policy_name, scores in sorted(policies.items())
        }
    return summary


def _action_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.get("action_types", []))
    return dict(counter.most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--model-path",
        default="artifacts/sft-checkpoint",
        help="LoRA adapter directory or HF repo id; falls back to base model if missing.",
    )
    parser.add_argument(
        "--policy-name",
        default="trained_llm",
        help="Label written to the JSON output for this run.",
    )
    parser.add_argument(
        "--include-base-model",
        action="store_true",
        help="Also evaluate the untrained base model for a same-seed before/after.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=DEFAULT_TASKS,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Use 4bit quantisation (recommended on Colab T4).",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4bit quantisation.",
    )
    parser.add_argument("--output", default="artifacts/trained_llm_eval.json")
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
                    "Trained-LLM evaluation requires a CUDA GPU. In Colab, "
                    "Runtime -> Change runtime type -> T4 GPU and rerun."
                ),
                "torch": getattr(torch, "__version__", "unknown"),
            }
        )
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if model_path and not Path(model_path).exists() and "/" not in model_path:
        print(
            {
                "status": "warning",
                "message": (
                    f"--model-path {model_path} not found locally. "
                    "Falling back to the base model so the eval still runs."
                ),
            }
        )
        model_path = None

    policy, info = _build_policy(
        args.base_model, model_path, args.temperature, args.load_in_4bit
    )
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        for task in args.tasks:
            print(
                json.dumps(
                    {"event": "rollout_start", "policy": args.policy_name, "task": task, "seed": seed}
                )
            )
            records.append(run_episode(policy, task, seed, args.policy_name))

    base_records: list[dict[str, Any]] = []
    if args.include_base_model:
        base_policy, base_info = _build_policy(
            args.base_model, None, args.temperature, args.load_in_4bit
        )
        for seed in range(args.seeds):
            for task in args.tasks:
                print(
                    json.dumps(
                        {
                            "event": "rollout_start",
                            "policy": "base_instruct_llm",
                            "task": task,
                            "seed": seed,
                        }
                    )
                )
                base_records.append(run_episode(base_policy, task, seed, "base_instruct_llm"))

    all_records = records + base_records
    total_actions = sum(rec["steps"] for rec in all_records)
    total_waits = sum(rec.get("wait_actions", 0) for rec in all_records)
    parse_failure_rate = round(total_waits / total_actions, 4) if total_actions else 0.0

    payload = {
        "summary": summarize(all_records),
        "episodes": all_records,
        "parse_failure_rate": parse_failure_rate,
        "action_distribution": {
            args.policy_name: _action_distribution(records),
        },
        "trained_policy_name": args.policy_name,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "temperature": args.temperature,
        **info,
    }
    if base_records:
        payload["action_distribution"]["base_instruct_llm"] = _action_distribution(base_records)
        payload["base_model_info"] = base_info

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "trained_episodes": len(records),
                "base_episodes": len(base_records),
                "parse_failure_rate": parse_failure_rate,
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
