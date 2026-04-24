from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import always_accept_policy, greedy_hoarder_policy, random_validish_policy, rule_based_expert_policy
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


def run_policy(name: str, task_type: str, policy) -> float:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=7))
    while not obs.done:
        obs = env.step(policy(obs))
    print(f"{name:24s} task={task_type:16s} reward={obs.cumulative_reward:.3f}")
    return obs.cumulative_reward


if __name__ == "__main__":
    for task in ["single_trade", "market_round", "coalition_market"]:
        run_policy("random_validish", task, random_validish_policy)
        run_policy("greedy_hoarder", task, greedy_hoarder_policy)
        run_policy("always_accept", task, always_accept_policy)
        run_policy("rule_based_expert", task, rule_based_expert_policy)
