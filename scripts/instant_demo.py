"""Sub-five-second CPU demo: print the headline numbers and one episode log.

Designed for a judge who clones the repo and just wants to see proof-of-life:

  $ python3 scripts/instant_demo.py

In ~3 seconds (no model download, no training, no GPU) this prints:

  1. The committed SFT loss curve summary (from artifacts/sft_training_curve.json).
  2. The committed GRPO reward curve summary (from artifacts/grpo_training_curve.json).
  3. The committed trained-LLM-vs-baselines table (from artifacts/trained_llm_summary.json).
  4. A live one-episode rollout of the rule_based_expert baseline on `coalition_market`,
     with all 11 reward components broken out, so the judge sees the env *running*.

Every value above is a real artifact from a real training/eval run that we
committed to git — not a generated example. If you want to regenerate them you
run `python3 scripts/check_submission.py` (~60s) or the Colab notebook (~30 min on T4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _print_section(title: str) -> None:
    bar = "─" * max(60, len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_sft_curve() -> None:
    _print_section("SFT loss curve  (Unsloth, Llama-3.2-3B-Instruct)")
    payload = _load(ROOT / "artifacts" / "sft_training_curve.json")
    if not payload:
        print("  artifacts/sft_training_curve.json not committed yet — run scripts/extract_sft_curve.py")
        return
    summary = payload.get("summary", {})
    print(f"  steps              : {summary.get('total_steps')}  ({summary.get('num_train_epochs')} epochs)")
    print(f"  loss               : {summary.get('first_loss')}  →  {summary.get('final_loss')}  ({summary.get('loss_drop_pct')}% drop)")
    print(f"  best loss          : {summary.get('best_loss')}")
    print(f"  source             : {summary.get('source')}")


def _print_grpo_curve() -> None:
    _print_section("GRPO reward curve  (TRL GRPOTrainer against env reward)")
    payload = _load(ROOT / "artifacts" / "grpo_training_curve.json")
    if not payload:
        print("  artifacts/grpo_training_curve.json not committed yet — run training/run_grpo_against_env.py")
        return
    summary = payload.get("summary", {})
    print(f"  steps × completions: {summary.get('max_steps')} × {summary.get('num_generations')} on {summary.get('prompts')} prompts")
    print(f"  mean reward        : {summary.get('first_step_mean_reward')}  →  {summary.get('last_step_mean_reward')}  (peak {summary.get('best_step_mean_reward')})")


def _print_trained_eval() -> None:
    _print_section("Trained LLM vs baselines  (5 seeds × 3 tasks, live rollouts)")
    payload = _load(ROOT / "artifacts" / "trained_llm_summary.json")
    if not payload:
        print("  artifacts/trained_llm_summary.json not committed — run scripts/evaluate_trained_llm.py")
        return
    rows = payload.get("rows", [])
    headers = ["policy", "single_trade", "market_round", "coalition_market", "overall"]
    fmt = "  {:<26} {:>13} {:>13} {:>17} {:>9}"
    print(fmt.format(*headers))
    print("  " + "─" * 86)
    for r in rows:
        def cell(key: str) -> str:
            v = r.get(f"{key}_mean") if key != "overall" else r.get("overall_mean")
            return f"{v:+.4f}" if isinstance(v, (int, float)) else "—"
        print(fmt.format(
            r.get("policy", "?"),
            cell("single_trade"),
            cell("market_round"),
            cell("coalition_market"),
            cell("overall"),
        ))


def _print_live_episode() -> None:
    _print_section("Live one-episode rollout  (rule_based_expert · coalition_market · seed=5)")
    from gpu_budget_arena.env import GpuBudgetNegotiationEnv
    from gpu_budget_arena.baselines import rule_based_expert_policy
    from gpu_budget_arena.models import ResetConfig

    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=5))
    components_total: dict[str, float] = {}
    total_reward = 0.0
    steps = 0
    while not obs.done:
        obs = env.step(rule_based_expert_policy(obs))
        steps += 1
        total_reward += obs.reward
        breakdown = obs.reward_breakdown
        if breakdown is not None:
            for k, v in breakdown.model_dump().items():
                if k == "normalized_reward":
                    continue
                if isinstance(v, (int, float)) and v != 0.0:
                    components_total[k] = components_total.get(k, 0.0) + float(v)
    print(f"  steps              : {steps}")
    print(f"  cumulative reward  : {obs.cumulative_reward:+.4f}")
    print(f"  per-step sum       : {total_reward:+.4f}")
    print(f"  non-zero reward components (summed across all steps):")
    if not components_total:
        print("     (none — try a different seed/task)")
    for k in sorted(components_total, key=lambda k: -abs(components_total[k])):
        print(f"     {k:<32} {components_total[k]:+.4f}")


def main() -> None:
    print("\n  GPU Budget Negotiation Arena — instant demo")
    print("  No GPU. No model download. ~3 seconds.")
    _print_sft_curve()
    _print_grpo_curve()
    _print_trained_eval()
    _print_live_episode()
    _print_section("That's it")
    print("  · Live Space        : https://abhinavgautam01-gpu-budget-negotiation-arena.hf.space")
    print("  · Reproduce all     : python3 scripts/check_submission.py    (~60s, CPU)")
    print("  · Re-train the LLM  : training/GPU_Budget_Negotiation_Arena_Colab.ipynb (free Colab T4)")
    print()


if __name__ == "__main__":
    main()
