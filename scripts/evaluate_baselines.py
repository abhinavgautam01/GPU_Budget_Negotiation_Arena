from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import (
    always_accept_policy,
    base_instruct_naive_policy,
    greedy_hoarder_policy,
    no_negotiation_allocator_policy,
    random_validish_policy,
    rule_based_expert_policy,
)
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


POLICIES = {
    "random_validish": random_validish_policy,
    "base_instruct_naive": base_instruct_naive_policy,
    "no_negotiation_allocator": no_negotiation_allocator_policy,
    "greedy_hoarder": greedy_hoarder_policy,
    "always_accept": always_accept_policy,
    "rule_based_expert": rule_based_expert_policy,
}


def run_episode(task_type: str, seed: int, policy_name: str) -> dict[str, object]:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    rewards = []
    steps = 0
    while not obs.done:
        obs = env.step(POLICIES[policy_name](obs))
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
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        grouped.setdefault(record["task_type"], {}).setdefault(record["policy"], []).append(float(record["episode_reward"]))
    summary: dict[str, object] = {}
    for task_type, policies in grouped.items():
        summary[task_type] = {
            policy: {
                "mean_reward": round(statistics.mean(scores), 4),
                "stdev_reward": round(statistics.pstdev(scores), 4),
                "episodes": len(scores),
            }
            for policy, scores in sorted(policies.items())
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="artifacts/baseline_eval.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(args.seeds):
        for task_type in ["single_trade", "market_round", "coalition_market"]:
            for policy_name in POLICIES:
                records.append(run_episode(task_type, seed, policy_name))
    payload = {"summary": summarize(records), "episodes": records}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"output": str(output), "episodes": len(records)})


if __name__ == "__main__":
    main()
