from __future__ import annotations

from gpu_budget_arena.baselines import random_validish_policy, rule_based_expert_policy
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


def run(task_type: str, policy, seed: int) -> float:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    while not obs.done:
        obs = env.step(policy(obs))
    return obs.cumulative_reward


def test_expert_beats_random_on_average() -> None:
    random_scores = []
    expert_scores = []
    for seed in range(40, 45):
        random_scores.append(run("market_round", random_validish_policy, seed))
        expert_scores.append(run("market_round", rule_based_expert_policy, seed))
    assert sum(expert_scores) / len(expert_scores) > sum(random_scores) / len(random_scores)

